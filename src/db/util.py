import os
import pickle
import numpy as np
from common import *
from util import make_obi, distance
from db.client import Database, COLL_ROUTES
from itertools import combinations

_db = Database()

_grid = './db/grid.p'


def get_grid():
    """
    Find all the x,y locations that all vehicles have been at.
    This is mostly a utility function for plotting the map.

    :return: [[x0, y0], ...] (numpy.ndarray)
    """
    if os.path.exists(_grid):
        with open(_grid, 'rb') as f:
            data = pickle.load(f)
    else:
        docs = _db.find()
        data = np.vstack({tuple(row) for row in np.round([[d['x'], d['y']] for d in docs])})
        with open(_grid, 'wb') as f:
            pickle.dump(data, f)

    return data


def get_actual_path(vehicle_id, proto=DSRC, as_node=True):
    """
    Find the x,y locations of a vehicle's path
    :param vehicle_id: the vehicle id
    :param proto: the protocol that should be queried, defaults to DSRC
    :param as_node: if True, NTNodes will be returned
    :return: [[x0, y0], ...] (numpy.ndarray)
    """
    pipe = [
        {'$match': {'subject': vehicle_id, 'protocol': proto}},
        {'$sort': {'time': 1}},
    ]
    path = _db.aggregate(pipe=pipe)
    if as_node:
        path = [NTNode(p['x'], p['y'], p['time']) for p in path]
    else:
        path = np.array([[p['x'], p['y'], p['time']] for p in path])

    return path


def get_vehicle_pseudo_paths(vehicle_id, proto=DSRC):
    """
    Get the paths of each pseudonym associated with a vehicle.
    :param vehicle_id: the id of the vehicle
    :param proto: the protocol to query
    :return: list of numpy.ndarray
        [
            [[x00, y00], [x01, y01], ...],
            [[x10, y10], [x11, y11], ...],
            ...
        ]
    """
    pipe = [
        {'$match': {'subject': vehicle_id, 'protocol': proto}},
        {'$sort': {'time': 1}},
        {'$group': {'_id': '$pseudo', 'x': {'$push': '$x'}, 'y': {'$push': '$y'}, 'times': {'$push': '$time'}}}
    ]
    return [np.array(list(zip(p['x'], p['y'], p['times']))) for p in _db.aggregate(pipe=pipe)]


def get_listener_obseration_intervals(x, y, proto, timeout, p_range=None, vehicle_ids=None, timeout_delta=0.001):
    """
    Get all of the observations at Listener (x, y)
    :param x: x location of the listener
    :param y: y location of the listener
    :param proto: the protocol to query
    :param timeout: the time (s) before a new interval should be constructed
    :param vehicle_ids: if specified, only consider observations from these vehicles
    :return: list of common.NTObservationIntervals
        [(subject_0, pseudo_0, x_0, y_0, loc_0, times_0, ts_0, te_0, mean_time_0), ...]
    """
    if not p_range:
        if proto == WIFI:
            p_range = WIFI_RANGE
        else:
            p_range = DSRC_RANGE

    if vehicle_ids is None:
        match = {'$match': {'protocol': proto}}
    else:
        match = {'$match': {'protocol': proto, 'subject': {'$in': vehicle_ids}}}

    pipe = [
        match,
        {'$project': {'subject': 1, 'protocol': 1, 'pseudo': 1, 'time': 1,
                      'dpos': {'$sqrt':
                                   {'$add': [{'$pow': [{'$subtract': ['$x', x]}, 2]},
                                             {'$pow': [{'$subtract': ['$y', y]}, 2]}]}}}},
        {'$match': {'dpos': {'$lte': p_range}}},
        {'$group': {'_id': '$pseudo',
                    'subj': {'$first': '$subject'},
                    'times': {'$push': '$time'},
                    'ts': {'$first': '$time'}}},
        {'$sort': {'ts': 1}}
    ]
    docs = _db.aggregate(pipe=pipe)

    obis = []
    for obi in docs:
        pseudo, subj, times = obi['_id'], obi['subj'], obi['times']
        slices = [0]

        for n in range(1, len(times)):
            if times[n] - times[n - 1] > timeout + timeout_delta:
                slices.append(n)

        slices.append(len(times))

        for n in range(1, len(slices)):
            times_ = times[slices[n - 1]: slices[n]]
            obis.append(make_obi(x, y, times_, subject=subj, pseudo=pseudo))

    return obis


def group_obis(obis):
    """
    Group observation intervals by pseudonym
    :param obis: a list of observation intervals
    :return: dictionary
        {pseudonym: [obi_0, ...], ... }
    """
    groups = {}

    for obi in obis:
        if obi.pseudo not in groups:
            groups[obi.pseudo] = [obi]
        else:
            groups[obi.pseudo].append(obi)

    return groups


def poly_area(xs, ys):
    """

    :param xs:
    :param ys:
    :return:
    """
    return 0.5 * np.abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))


def choose_max_area(ls_candidates, n):
    """
    :param ls_candidates: list of listener dicts ['x': int, 'y': int , 'seen': list, 'seem_size': int]
    :param n: size of polygon
    :return: subset of size n of ls_candidates that makes polygon
    """
    comb = list(combinations(ls_candidates, n))

    max_area = -1
    selected = None
    for candidate in comb:
        xs = [lstnr['x'] for lstnr in candidate]
        ys = [lstnr['y'] for lstnr in candidate]

        if n == 2:
            area = distance(candidate[0]['x'], candidate[0]['y'], candidate[1]['x'], candidate[1]['y'])
        else:
            area = poly_area(xs, ys)

        if max_area < area:
            max_area = area
            selected = candidate

    return selected


def unique_seen_count(seen_lists):
    uniq = set()
    for seen_list in seen_lists:
        for seen in seen_list:
            uniq.add(seen)

    return len(uniq)


def choose_max_area_seen_count_perc(ls_candidates, n, perc):
    """
    similar to max_area(), but the union of seen's must have a unique count within a percentage of the max_seen_count

    :param ls_candidates: list of listener dicts ['x': int, 'y': int , 'seen': list, 'seem_size': int]
    :param n: size of polygon
    :param perc: should be a float 0 < perc < 1, unique seens must be within perc percentage of max unique seens
    :return: subset of size n of ls_candidates that makes polygon
    """
    comb = list(combinations(ls_candidates, n))

    max_area = -1  # largest area
    currs_usc = 0  # largest area's unique seen countt
    max_usc = 0  # max number of unique seens
    selected = None
    for candidate in comb:
        xs = [lstnr['x'] for lstnr in candidate]
        ys = [lstnr['y'] for lstnr in candidate]
        seens = [lstnr['seen'] for lstnr in candidate]

        if n == 2:
            area = distance(candidate[0]['x'], candidate[0]['y'], candidate[1]['x'], candidate[1]['y'])
        else:
            area = poly_area(xs, ys)

        usc = unique_seen_count(seens)

        if max_area < area and usc >= max_usc * perc:
            max_area = area
            selected = candidate
            currs_usc = usc

        if max_usc < usc:
            max_usc = usc
            # if the current usc isn't within perc of new max_usc, will default to current set
            if currs_usc <= max_usc * perc:
                # of listeners (b/c greater number of unique seens)
                max_area = area
                selected = candidate
                currs_usc = usc

    return selected


def choose_max_area_seen_count_scalar(ls_candidates, n):
    """
    similar to max_area(), but uses (max_area * unique_seen_count) as basis for "best" subset

    :param ls_candidates: list of listener dicts ['x': int, 'y': int , 'seen': list, 'seem_size': int]
    :param n: size of polygon
    :return: subset of size n of ls_candidates that makes polygon
    """
    comb = list(combinations(ls_candidates, n))

    max_area_usc = 0
    selected = None
    for candidate in comb:
        xs = [lstnr['x'] for lstnr in candidate]
        ys = [lstnr['y'] for lstnr in candidate]
        unique_seen = [lstnr['seen'] for lstnr in candidate]

        if n == 2:
            area = distance(candidate[0]['x'], candidate[0]['y'], candidate[1]['x'], candidate[1]['y'])
        else:
            area = poly_area(xs, ys)

        usc = unique_seen_count(unique_seen)

        if max_area_usc < (area * usc):
            max_area_usc = area * usc
            selected = candidate

    return selected
