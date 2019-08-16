import util
import warnings
import csv
import logging
import os
import visualize
import numpy as np
from collections import defaultdict
from datetime import datetime
from common import *
from db import util as db_util
from db.client import Database, COLL_ROUTES


arr = np.array

# careful, this path is relative to ./src
_time = datetime.now().strftime('%H%M')
__results_dir = '../results'
_results_dir = os.path.abspath(os.path.join(__results_dir, datetime.now().strftime('%m_%d_%Y'), 'VET', _time))
if not os.path.exists(_results_dir):
    os.makedirs(_results_dir)

_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)

fh = logging.FileHandler(os.path.join(_results_dir, 'strategy.log'), mode='w')
fh.setLevel(logging.INFO)
formatter = logging.Formatter('| %(levelname)6s | %(funcName)8s:%(lineno)2d | %(message)s |')
fh.setFormatter(formatter)
_log.addHandler(fh)


class Strategy:
    def process(self, listeners, **kwargs):
        """

        :param listeners:
        :param kwargs:
        :return:
        """
        raise NotImplementedError


# noinspection PyTypeChecker,SpellCheckingInspection,PyUnresolvedReferences
class VET(Strategy):
    def __str__(self):
        return 'Ensemble Vehicle Tracking'

    def __repr__(self):
        return str(self)

    def process(self, listeners, conf_thresh=0.9, **kwargs):
        """
        Process observations given a set of listener x,y locatins
        :param listeners: list of listener x,y locations ([[x0, y0], ...])
        :param conf_thresh: the confidence threshold for saying w_i = d_j (float, [0, 1])
        :param kwargs: optional keyword arguments for subclass
        :return: StrategyResult
        """
        wgroups, dgroups, subject_ids = self._init_build_assoc_mtx(listeners, **kwargs)
        wps = sorted(list(wgroups.keys()))
        dps = sorted(list(dgroups.keys()))

        result = StrategyResult(subject_ids, dgroups)
        if wps:
            mtx = self._build_assoc_mtx(wgroups, dgroups, wps, dps, **kwargs)
            for r in np.arange(mtx.shape[0]):
                wobis = wgroups[wps[r]]
                s_id = wobis[0].subject

                w_points = []
                for obi in wobis:
                    w_points.extend([[*obi.loc, t] for t in (obi.ts, obi.te)])

                assoc_sids = []
                assoc = np.where(mtx[r, :] >= conf_thresh)[0]
                order = list(reversed(np.argsort(mtx[r, assoc])))
                d_points = []
                for c in [assoc[o] for o in order]:
                    d_obis = dgroups[dps[c]]
                    assoc_sids.append(d_obis[0].subject)
                    for obi in d_obis:
                        d_points.extend([[*obi.loc, t] for t in (obi.ts, obi.te)])

                result.add_subject_results(s_id, assoc_sids, w_points, d_points)

        return result

    @staticmethod
    def _init_build_assoc_mtx(listeners, wifi_range=WIFI_RANGE, **kwargs):
        """
        Helper function for constructing observation intervals given a listener configuration

        :param listeners: list of listener x,y locations ([[x0, y0], ...])
        :return:
                wgroups: dictionary mapping a wifi pseudonym to a list of observation intervals
                        ({pseduonym: [obi0, ...], ...})
                dgroups: dictionary mapping a dsrc pseudonym to a list of observation intervals
                        ({pseduonym: [obi0, ...], ...})
                vehicle_ids: list of vehicle ids that were found
        """
        w_obis = []
        for x, y in listeners:
            w_obis.extend(db_util.get_listener_obseration_intervals(x, y, WIFI, WIFI_HR, p_range=wifi_range))

        vehicle_ids = list(set([wobi.subject for wobi in w_obis]))
        d_obis = []

        for x, y in listeners:
            d_obis.extend(db_util.get_listener_obseration_intervals(x, y, DSRC, DSRC_HR, vehicle_ids=vehicle_ids))

        wgroups = db_util.group_obis(w_obis)
        dgroups = db_util.group_obis(d_obis)

        return wgroups, dgroups, vehicle_ids

    @staticmethod
    def _build_assoc_mtx(wgroups, dgroups, wps, dps, sigma=0.3, **kwargs):
        """
        Build the wifi-to-dsrc association matrix.

        :param wgroups: dictionary mapping a wifi pseudonym to a list of observation intervals ({pseduonym: [obi_0, ...], ...})
        :param dgroups: dictionary mapping a dsrc pseudonym to "" ({pseduonym: [obi_0, ...], ...})
        :param wps: sorted list of keys for wgroups (this just ensures multiple runs execute the same way)
        :param dps: sorted list of keys for drgroups
        :param sigma: constant redistribution factor
        :return: w2d matrix (the likelihood that w_i == d_j)
        """
        wlookup = {m: wp for m, wp in enumerate(wps)}
        dlookup = {n: dp for n, dp in enumerate(dps)}

        # dsrc to dsrc associations
        d2d_mtx = VET._build_d2d_mtx(dgroups, dlookup)
        # wifi to dsrc associations
        return VET._build_w2d_mtx(wgroups, dgroups, wlookup, dlookup, d2d_mtx, sigma=sigma)

    @staticmethod
    def _build_w2d_mtx(wgroups, dgroups, wlookup, dlookup, d2d_mtx, sigma=0.3):
        """
        Helper function for building wifi-to-dsrc association matrix.

        :param wgroups: dictionary mapping a wifi pseudonym to a list of observation intervals
                        ({pseduonym: [obi_0, ...], ...})
        :param dgroups: dictionary mapping a dsrc pseudonym to "" ({pseduonym: [obi_0, ...], ...})
        :param wlookup: dictionary mapping a wifi pseudonym to its row in the matrix
        :param dlookup: dictionary mapping a dsrc pseudonym to its column in the matrix
        :param d2d_mtx: the dsrc-to-dsrc association matrix
        :param sigma: constant redistribution factor
        :return: w2d matrix (the likelihood that w_i == d_j)
        """
        mtx = np.full((len(wlookup), len(dlookup)), 1 / len(wlookup))

        r_not_assoc = []
        r_assoc_by_loc = []

        for r in np.arange(mtx.shape[0]):
            not_assoc = set()
            assoc_by_loc = defaultdict(set)
            for c in np.arange(mtx.shape[1]):
                w_obis = wgroups[wlookup[r]]
                d_obis = dgroups[dlookup[c]]

                for n, m in util.cartesian((w_obis, d_obis), indices=False):
                    wobi = w_obis[n]
                    dobi = d_obis[m]
                    # coverage is non-overlapping;
                    # if d0 is seen by lx while w0 is seen by ly then d0 != w0
                    dist, t_inter = util.obi_dist_and_t_intersect(wobi, dobi)
                    if dist and t_inter:
                        if c not in assoc_by_loc[wobi.loc]:
                            # make sure that this dsrc isn't already possibly assoc
                            not_assoc.add(c)
                    # if d0 is seen by lx while w0 is seen by lx then d0 might be equal to w0
                    elif t_inter:
                        assoc_by_loc[wobi.loc].add(c)
                        if c in not_assoc:
                            # we just found that this dsrc might be associated, so update not assoc
                            not_assoc.remove(c)

            r_not_assoc.append(arr(list(not_assoc)))
            r_assoc_by_loc.append([arr(list(a)) for a in assoc_by_loc.values()])

        # set things to 0
        for r, not_assoc in enumerate(r_not_assoc):
            not_r = np.setdiff1d(np.arange(mtx.shape[0]), [r])

            if not_assoc.size:
                not_assoc = np.union1d(not_assoc, np.where(d2d_mtx[not_assoc, :] == 0)[0])

                for c_na in not_assoc:
                    s1, s2 = util.get_subject_ids(wgroups[wlookup[r]], dgroups[dlookup[c_na]])
                    _log.debug('w2d impossible association found: {} and {}'.format(s1, s2))
                    if s1 == s2:
                        _log.warning('invalid impossible w2d association found: {} and {}'.format(s1, s2))

                    delta = mtx[r, c_na]
                    mtx[r, c_na] = 0
                    p_assoc = np.where(mtx[not_r, c_na] > 0)[0]
                    if not p_assoc.size:
                        _log.warning('dsrc {} not associated with any wifi?'.format(dlookup[c_na]))
                        mtx[:, c_na] = (1 / mtx.shape[0])
                    else:
                        mtx[p_assoc, c_na] += (delta / p_assoc.size)

        # handle possible associations
        for r, assoc_by_loc in enumerate(r_assoc_by_loc):
            not_r = np.setdiff1d(np.arange(mtx.shape[0]), [r])
            not_assoc = np.where(mtx[r, :] == 0)[0]
            for assoc in assoc_by_loc:
                assoc = np.setdiff1d(assoc, not_assoc)
                if assoc.size == 1:
                    # we win, hopefully
                    c_a = assoc[0]
                    s1, s2 = util.get_subject_ids(wgroups[wlookup[r]], dgroups[dlookup[c_a]])
                    _log.debug('w2d association found: {} and {}'.format(wlookup[r], dlookup[c_a]))
                    if s1 != s2:
                        _log.warning('invalid w2d association found: Vehicle {} and Vehicle {}'.format(wlookup[r],
                                                                                                       dlookup[c_a]))
                    # update probas
                    mtx[:, c_a] = 0
                    for c_na in np.where(d2d_mtx[c_a, :] == 0)[0]:
                        delta = mtx[r, c_na]
                        mtx[r, c_na] = 0
                        p_assoc = np.where(mtx[:, c_na] > 0)[0]
                        if p_assoc.size:
                            mtx[p_assoc, c_na] += (delta / p_assoc.size)
                        else:
                            _log.warning('dsrc {} not associated with any wifi?'.format(dlookup[c_na]))
                            mtx[:, c_na] = (1 / mtx.shape[0])
                    mtx[r, c_a] = 1
                else:
                    for c in assoc:
                        delta = mtx[not_r, c].sum()
                        mtx[r, c] += (sigma * delta)
                        mtx[not_r, c] *= (1 - sigma)

        tots = np.round(mtx.sum(axis=0), 2)
        if tots.all() != 1.0:
            _log.warning('uh oh: wifi_2_dsrc matrix columns dont sum to 1.')
            _log.debug('w2d columns neq 1: {}'.format(','.join(tots[tots != 1].astype(str))))

        return mtx

    @staticmethod
    def _build_d2d_mtx(dgroups, dlookup):
        """
        Helper function for building the dsrc-to-dsrc association matrix.

        :param dgroups: dictionary mapping a dsrc pseudonym to a list of observation intervals
                        ({pseduonym: [obi_0, ...], ...})
        :param dlookup: dictionary mapping a dsrc pseudonym to its row/column in the matrix.
        :return: d2d_matrix (the likelihood that d_i = d_j)
        """
        n = len(dlookup)
        mtx = np.full((n, n), 1 / (n - 1))

        # ignore the diagonal
        np.fill_diagonal(mtx, -1)

        for r in np.arange(mtx.shape[0]):
            not_assoc = []
            for c in np.where(mtx[r, :] > 0)[0]:
                a_obis = dgroups[dlookup[r]]
                b_obis = dgroups[dlookup[c]]

                for n, m in util.cartesian((a_obis, b_obis), indices=False):
                    dist, intersect = util.obi_dist_and_t_intersect(a_obis[n], b_obis[m])
                    if intersect:
                        not_assoc.append(c)
                        break

            for c_na in not_assoc:
                a_obis = dgroups[dlookup[r]]
                b_obis = dgroups[dlookup[c_na]]
                s1, s2 = util.get_subject_ids(a_obis, b_obis)
                _log.debug('found impossible d2d association: {} and {}'.format(dlookup[r], dlookup[c_na]))
                if s1 == s2:
                    _log.warning(
                        'invalid impossible d2d association found: {} and {}'.format(dlookup[r], dlookup[c_na]))

                # zero out everything we can
                mtx[[r, c_na], [c_na, r]] = 0
                # TODO: is there anything else that we can do here?

        return mtx


class StrategyFactory:
    _default = VET

    @staticmethod
    def make_strategy(strat_id):
        strat = StrategyFactory._default

        if strat_id == 'VET':
            strat = VET
        else:
            msg = 'unrecognized strategy id: {}, defaulting to {}'.format(strat_id, StrategyFactory._default)
            warnings.warn(msg)

        return strat()


class StrategyResult:
    def __init__(self, subject_ids, dgroups):
        self.sids = subject_ids
        self.dgroups = dgroups
        self.n_subj = len(self.sids)
        self.found_subjects = {}
        self.db = Database()

    def add_subject_results(self, sid, d_assocs, w_points, d_points):
        if sid in self.found_subjects:
            _log.warning('entry for Vehicle {} already exists. duplicate subject found!'.format(sid))
        else:
            w, d = arr(w_points), arr(d_points)
            assoc = arr(d_assocs)
            n_correct = np.where(assoc == sid)[0].size
            n_assoc = assoc.size
            self.found_subjects[sid] = {'w_points': w, 'd_points': d,
                                        'n_correct': n_correct, 'n_assoc': n_assoc, 'assoc': d_assocs}

    def get_subject_path(self, sid, p_type='multi'):
        path = []
        if sid not in self.found_subjects:
            _log.warning('no path found for Vehicle {}'.format(sid))
        elif p_type == DSRC:
            path = self.found_subjects[sid]['d_points']
        elif p_type == WIFI:
            path = self.found_subjects[sid]['w_points']
        else:
            w, d = self.found_subjects[sid]['d_points'], self.found_subjects[sid]['w_points']
            if w.size and d.size:
                order = np.argsort(np.r_[w[:, -1], d[:, -1]].ravel())
                path = np.r_[w, d][order]
            elif w.size:
                path = self.found_subjects[sid]['w_points']
            else:
                path = arr([])

        return path

    def get_mean_accuracy(self, weighted=True):
        pass

    def get_assoc(self, sid):
        return self.found_subjects[sid]['assoc']

    def get_subjects_found(self):
        return sorted(list(self.found_subjects.keys()))

    def get_n_correct(self):
        return arr([self.found_subjects[s]['n_correct'] for s in self.get_subjects_found()])

    def get_n_assoc(self):
        return arr([self.found_subjects[s]['n_assoc'] for s in self.get_subjects_found()])

    def get_n_found(self):
        return len(self.get_subjects_found())

    def get_summary_stats(self):
        sids = self.get_subjects_found()
        data = []
        for n, sid in enumerate(sids):
            assoc = self.found_subjects[sid]['assoc']
            n_assoc = self.found_subjects[sid]['n_assoc']
            n_correct = self.found_subjects[sid]['n_correct']
            dsrcs = self.db.aggregate(COLL_ROUTES, [{'$match': {'subject': sid}}, {'$group': {'_id': '$pseudo'}}])
            n_dsrc = len(dsrcs)
            tp = n_correct
            fp = n_assoc - n_correct
            fn = sum([1 if dg[0].subject == sid and p not in assoc else 0 for p, dg in self.dgroups.items()])
            p = np.nan
            if tp + fp:
                p = tp / (tp + fp)
            r = np.nan
            if tp + fn:
                r = tp / (tp + fn)

            data.append([n_dsrc, n_assoc, p, r])

        data = arr(data)
        mu = []

        for i in range(data.shape[1]):
            mu.append(np.mean(data[data[:, i] == data[:, i], i]))

        return data, mu

    def write_results(self, s_fp, ls_cfg, ls_id='', draw_paths=False):
        if ls_id:
            rdir = os.path.join(_results_dir, ls_id)
        else:
            rdir = _results_dir

        if not os.path.exists(rdir):
            os.makedirs(rdir)

        self.write_summary(rdir, s_fp, ls_cfg, ls_id, draw_paths=draw_paths)

    def write_summary(self, rdir, s_fp, ls_cfg, ls_id, draw_paths=False):
        header = ['subject', 'n_dsrc', 'n_assoc', 'precision', 'recall']

        data, mu = self.get_summary_stats()
        data = np.round(np.r_[data, [mu]], 2)
        data = np.c_[np.arange(1, data.shape[0] + 1).reshape(-1, 1), data]

        with open(os.path.join(rdir, s_fp), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data)

        if draw_paths:
            rdir_ = os.path.join(rdir, 'paths')
            if not os.path.exists(rdir_):
                os.makedirs(rdir_)

            sids = self.get_subjects_found()
            ls_xys = arr(ls_cfg)

            paths = []
            for sid in sids:
                path = self.get_subject_path(sid)
                if path.size:
                    path = util.interpolate_path(path[:, :-1])
                    if util.distance(*path[0], *path[-1]):
                        paths.append((path, sid))

            for predicted, sid in paths:
                title = 'Vehicle {}'.format(sid)
                fp = os.path.join(rdir_, '{}_{}.png'.format(ls_id, title.lower().replace(' ', '_')))

                visualize.plot_predicted(predicted, fp, ls_xys=ls_xys)
