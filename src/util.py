import math
import os
import numpy as np
from copy import deepcopy
from datetime import datetime
from common import *


arr = np.array

_results_dir = '../results'


def interpolate_path(a, step=3):
    a_copy = deepcopy(a.tolist())
    a_ = [a_copy.pop(0)]

    while a_copy:
        p0 = a_[-1]
        p1_i = min(range(len(a_copy)), key=lambda i: distance(*a_copy[i], *p0))
        p1 = a_copy.pop(p1_i)

        if not np.array_equal(p0, p1):
            n_steps = max(3, step)
            xp_ = np.linspace(p0[0], p1[0], n_steps)
            yp_ = np.linspace(p0[1], p1[1], n_steps)
            a_.extend([[x, y] for x, y in zip(xp_, yp_)])

    return arr(a_)


def prep_rdir(strat):
    rdir = os.path.join(_results_dir, datetime.now().strftime('%m_%d_%Y'), strat)
    if not os.path.exists(rdir):
        os.makedirs(rdir)

    return rdir


def nCr(n, r):
    f = math.factorial
    return int(f(n) / f(r) / f(n-r))


def make_obi(x, y, times, subject=None, pseudo=None):
    return NTObservationInterval(subject, pseudo, x, y, (x, y), times, times[0], times[-1], np.mean(times))


def cartesian(arrays, out=None, indices=True):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
    """
    if indices:
        arrays = [np.asarray(x) for x in arrays]
    else:
        arrays = [np.asarray(range(len(x))) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n // arrays[0].size)
    out[:, 0] = np.repeat(arrays[0], m)

    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]

    return out


def get_subject_ids(a, b):
    """
    Unpack the subject id from two NTObservationIntervals
    :param a: first NTOBI
    :param b: second NTOBI
    :return: subject1, subject2
    """
    return a[0].subject, b[0].subject


def distance(x, y, x_, y_):
    """
    Calculate the euclidean distance between two points
    :param x: x1
    :param y: y1
    :param x_: x2
    :param y_: y2
    :return:
    """
    return math.sqrt(((x - x_) ** 2) + ((y - y_) ** 2))


def obi_time_intersect(a, b):
    """
    Determine if a and b intersect
    :param a:
    :param b:
    :return: True if a and b interesct
    """
    return (a.ts <= b.te and a.te >= b.ts) or (b.ts <= a.te and b.te >= a.ts)


def obi_dist_and_t_intersect(a, b):
    """
    Determine the distance between a and b and whether or not a and b t_intersect
    :param a:
    :param b:
    :return: dist, t_intersect (float, bool)
    """
    dist = distance(a.x, a.y, b.x, b.y)
    t_intersect = obi_time_intersect(a, b)

    return dist, t_intersect


def match_path_shapes(a, b):
    """
    Match the shapes of two paths
    :param a: actual path
    :param b: predicted path
    :return: interpolated predicted path
    """
    b_ = []

    for i in range(1, b.shape[0]):
        p0 = b[i - 1]
        p1 = b[i]

        trgt = np.intersect1d(np.where(p0[-1] <= a[:, -1]), np.where(a[:, -1] < p1[-1])[0])

        xp_ = np.linspace(p0[0], p1[0], trgt.size + 1)[:-1].reshape(-1, 1)
        yp_ = np.linspace(p0[1], p1[1], trgt.size + 1)[:-1].reshape(-1, 1)
        tp_ = np.linspace(p0[2], p1[2], trgt.size + 1)[:-1].reshape(-1, 1)

        b_.extend(np.concatenate((xp_, yp_, tp_), axis=1))

    # Extends rest of entries in predict if not the same length as actual
    # noinspection PyTypeChecker
    end = arr(b[-1]).reshape(-1, 3)
    size = len(a) - len(b_) - 1
    if size > 0:
        if not b_:
            b_ = np.concatenate((np.tile(b[-1], size).reshape(-1, 3), end))
        else:
            b_ = np.concatenate((b_, np.tile(b[-1], size).reshape(-1, 3), end))
    else:
        b_ = np.concatenate((b_, end))

    b_ = b_[:len(a), :]

    return b_


def fitness(actual, predict):
    """
    Fitness function using path error calculation

    LOWER number -> GREATER fitness
    :param actual:
    :param predict:
    :return: sum of all distance errors between corresponding actual & predicted points
    """
    if len(actual) != len(predict):
        # need to match shapes
        predicted_ = match_path_shapes(actual, predict)
    else:
        predicted_ = arr(predict)

    actual_ = arr(actual)
    if predicted_.shape != actual_.shape:
        exit('uh oh, predicted path and actual path have mismatched shapes!')

    err = np.sqrt(((actual_ - predicted_) ** 2).sum())

    return err


def sort_points(point_array):
    """Return point_array sorted by leftmost first, then by slope, ascending."""

    def slope(y):
        """returns the slope of the 2 points."""
        x = point_array[0]
        return (x['y'] - y['y']) / (x['x'] - y['x'])

    point_array = sorted(point_array, key=lambda d: d['x'])  # put leftmost first
    point_array = point_array[:1] + sorted(point_array[1:], key=slope)
    return point_array


def graham_scan(point_array):
    """
    :param point_array: array of listener dicts ['x': int, 'y': int , 'seen': list, 'seem_size': int]
    :return: array of listener dicts that make up the convex hull surrounding dicts in point_array
    """

    def cross_product_orientation(a, b, c):
        """Returns the orientation of the set of points.
        >0 if x,y,z are clockwise, <0 if counterclockwise, 0 if co-linear.
        """

        return (b['y'] - a['y']) * \
               (c['x'] - a['x']) - \
               (b['x'] - a['x']) * \
               (c['y'] - a['y'])

    # convex_hull is a stack of points beginning with the leftmost point.
    convex_hull = []
    sorted_points = sort_points(point_array)
    for p in sorted_points:
        # if we turn clockwise to reach this point, pop the last point from the stack, else, append this point to it.
        while len(convex_hull) > 1 and cross_product_orientation(convex_hull[-2], convex_hull[-1], p) >= 0:
            convex_hull.pop()
        convex_hull.append(p)
    # the stack is now a representation of the convex hull, return it.
    return convex_hull

