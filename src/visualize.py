import numpy as np
import matplotlib.pyplot as plt
from db.util import get_grid, get_actual_path, get_vehicle_pseudo_paths
from common import *
from matplotlib.collections import EllipseCollection


def _plot_listeners(ls_xys, ax):
    if isinstance(ls_xys, (list,)):
        ls_xys = np.array(ls_xys)

    size = np.full(ls_xys.shape[0], 2 * DSRC_RANGE)
    wifi = EllipseCollection(widths=size, heights=size, angles=0, units='xy', offsets=ls_xys,
                             transOffset=ax.transData, color='white', edgecolor='black', linestyle='--', zorder=2)
    ax.add_collection(wifi)

    size = np.full(ls_xys.shape[0], 2 * WIFI_RANGE)
    dsrc = EllipseCollection(widths=size, heights=size, angles=0, units='xy', offsets=ls_xys,
                             transOffset=ax.transData, color='white', edgecolor='black', linestyle='-', zorder=2)
    ax.add_collection(dsrc)

    # make sure the points show up in the legend
    ax.scatter([], [], color='white', s=21, edgecolor='black', linestyle='--', label='DSRC coverage')
    ax.scatter([], [], color='white', s=7, edgecolor='black', linestyle='-', label='WiFi coverage')

    ax.axis('equal')

    return ax


def _plot_grid(ax):
    grid = get_grid()
    ax.plot(grid[:, 0], grid[:, 1], lw=0, linestyle='', marker=',', color='grey', zorder=10)

    return ax


def plot_predicted(predicted, fp, ls_xys=None):
    f, ax = plt.subplots()
    _plot_grid(ax)
    if ls_xys is not None:
        _plot_listeners(ls_xys, ax)

    ax.plot(predicted[:, 0], predicted[:, 1], color='black', lw=3, linestyle='-', label='predicted')
    ax.set_xlim(250, 4000)
    ax.set_ylim(0, 3000)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.legend()
    plt.savefig(fp, dpi=300)
    plt.close()


def plot_subject_path(sid, ls_cfg=None, show_dsrc_change=False, show_wifi_beacon=False, out=None, show=False):
    f, ax = plt.subplots()
    _plot_grid(ax)

    if ls_cfg is not None:
        _plot_listeners(ls_cfg, ax)

    if show_wifi_beacon:
        wifi = get_actual_path(sid, proto=WIFI, as_node=False)
        ax.plot(wifi[:, 0], wifi[:, 1], color='black', linestyle='None', marker='o', zorder=3, label='WiFi beacon')

    paths = get_vehicle_pseudo_paths(sid, proto=DSRC)
    for n, path in enumerate(paths):
        if show_dsrc_change:
            if n % 2 == 0:
                color = 'black'
            else:
                color = 'red'
        else:
            color = 'black'

        ax.plot(path[:, 0], path[:, 1], color=color, lw=3, linestyle='-')

    ax.legend()
    ax.set_xlim(250, 4000)
    ax.set_ylim(0, 3000)
    ax.set_xticks([])
    ax.set_yticks([])
    if out:
        plt.savefig(out, dpi=300)
    if show:
        plt.show()

    plt.close()


def plot_grid(ls_cfg=LS_CFG_15):
    ls_xys = np.array(ls_cfg)
    f, ax = plt.subplots()
    _plot_grid(ax)
    _plot_listeners(ls_xys, ax)
    ax.axis('equal')
    # ax.set_title()
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.legend()
    plt.savefig('../results/map.png', dpi=300)
    plt.show()
    plt.close()


if __name__ == '__main__':
    plot_grid(LISTENERS)
    # sid = 27
    # plot_subject_path(sid, LS_CFG_5, show_dsrc_change=True, out='../results/ls_cfg_5_vehicle_{}_actual'.format(sid))
