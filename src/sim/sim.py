import numpy as np
import xml.etree.ElementTree as ET
from collections import defaultdict
from db.client import Database


arr = np.array


def _parse(p2xml):
    tree = ET.parse(p2xml)
    root = tree.getroot()
    timesteps = root.findall('timestep')
    vehicles = defaultdict(list)
    for ts in timesteps:
        time = ts.get('time')
        for vehicle in ts.findall('vehicle'):
            vehicles[int(vehicle.get('id'))].append(arr([time, vehicle.get('x'), vehicle.get('y')]).astype(float))

    return vehicles


def _make_obs(v, pseudo, proto, time, x, y):
    """

    :param v: vehicle id
    :param pseudo: radio pseudonym
    :param proto: radio protocol
    :param time: time of observation
    :param x: listener x
    :param y: listener y
    :return:
    """
    return {'subject': v, 'pseudo': pseudo, 'x': x, 'y': y, 'time': time, 'protocol': proto}


# noinspection PyUnresolvedReferences
def main(p2xml='data.xml', dsrc_hr=0.1, dsrc_cf=30, wifi_hr=30, step=0.1):
    """
    Simulate some vehicles
    :param p2xml: path to sumo xml output
    :param dsrc_hr: dsrc heartbeat rate (s)
    :param dsrc_cf: dsrc change frequency (s)
    :param wifi_hr: wifi heartbeat rate (s)
    :param step: simulation time step

    Calling this will ERASE database.COLL_ROUTES entries and add new ones.
    """
    vehicles = list(_parse(p2xml).items())

    dsrc_pseudo = 0
    observations = []

    for v_id, route in vehicles:
        dsrc_pseudo += 1
        route = arr(route)
        t_start = route[0][0]
        t_end = route[-1][0]

        t2_wifi_hr = 10 * np.random.rand()
        t2_dsrc_hr = 2 * np.random.rand()
        t2_dsrc_change = int(dsrc_cf * np.random.rand())

        for t in np.round(np.arange(t_start, t_end - step, step), 1):
            if t2_dsrc_change <= 0:
                # time to change dsrc
                dsrc_pseudo += 1
                t2_dsrc_change = dsrc_cf

            if t2_wifi_hr <= 0 or t2_dsrc_hr <= 0:
                # we are going to add a new point p
                # p should look like [time, x, y]

                # find the closest point in the route with time lte t
                lte = route[route[:, 0] <= t]
                p0 = lte[-1]
                if p0[0] == t:
                    # same time, use this point
                    p = p0
                else:
                    # find the midpoint
                    gte = route[route[:, 0] >= t]
                    p1 = gte[0]
                    p = (p0 + p1) / 2
                    # need to adjust time so that beacons are every .1s
                    p[0] = t

                if t2_wifi_hr <= 0:
                    # we can just use the vehicle id as the wifi pseudonym
                    observations.append(_make_obs(v_id, v_id, 'WiFi', *p))
                    t2_wifi_hr = wifi_hr

                if t2_dsrc_hr <= 0:
                    observations.append(_make_obs(v_id, dsrc_pseudo, 'DSRC', *p))
                    t2_dsrc_hr = dsrc_hr

            t2_wifi_hr -= step
            t2_dsrc_hr -= step
            t2_dsrc_change -= step

    db = Database()
    db.drop()
    db.insert_many(documents=observations)


if __name__ == '__main__':
    main()