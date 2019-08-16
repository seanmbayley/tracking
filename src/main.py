# Vehicle Tracking, Sean Bayley

import argparse
import numpy as np
from datetime import datetime
from common import *
from core import StrategyFactory

_results_dir = '../results'
_time = datetime.now().strftime('%H%M')
_summary_fp = 'summary_report.csv'


def tracking(strat_id, sigma=0.3, conf_thresh=0.5, n_listeners=(20, 15, 10, 5), n_iter=100,
             wifi_ranges=(25,), draw_paths=False, random_ls=False):

    """
    Do vehicle tracking.


    :param strat_id: the id of the strategy to process observations
    :param sigma: constant redistribution factor
    :param conf_thresh: the confidence threshold for saying w_i = d_j (float, [0, 1])
    :param n_listeners:
    :param n_iter:
    :param ls_cfgs: a list of listener configurations
                    [[[x00, y00], ...], [[x10, y10], ...]...]
    :param draw_paths: if True then vehicle paths will be drawn after processing
    :return: None
    """
    strat = StrategyFactory.make_strategy(strat_id)
    listeners = np.array(LISTENERS)
    listeners_idcs = np.arange(listeners.shape[0])

    if random_ls:
        for n_ls in n_listeners:
            for i in range(1, n_iter + 1):
                ls_idcs = np.random.choice(listeners_idcs, size=n_ls, replace=False)
                ls_cfg = listeners[ls_idcs].tolist()
                for w_range in wifi_ranges:
                    result = strat.process(listeners=ls_cfg, sigma=sigma, conf_thresh=conf_thresh, wifi_range=w_range)
                    if result.found_subjects:
                        ls_id = 'ls_cfg_{}_{}_{}'.format(n_ls, w_range, i)
                        result.write_results(_summary_fp, ls_cfg, ls_id, draw_paths=draw_paths)
    else:
        for ls_cfg in [LS_CFG_20, LS_CFG_15, LS_CFG_10, LS_CFG_5]:
            result = strat.process(listeners=ls_cfg, sigma=sigma, conf_thresh=conf_thresh)
            ls_id = 'ls_cfg_{}'.format(len(ls_cfg))
            result.write_results(_summary_fp, ls_cfg, ls_id, draw_paths=draw_paths)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = ap.add_subparsers(dest='command')

    tracking_parser = subparsers.add_parser('track', help=tracking.__doc__)
    tracking_parser.add_argument('-s', dest='strategy', choices=['VET'], default='VET')
    tracking_parser.add_argument('-n_ls', dest='num_listeners', nargs='+', default=(20, 15, 10, 5), type=int)
    tracking_parser.add_argument('-n_iter', dest='num_iter', default=100, type=int)
    tracking_parser.add_argument('-sigma', dest='sigma', default=0.3, type=float)
    tracking_parser.add_argument('-conf_thresh', dest='conf_thresh', default=0.5, type=float)
    tracking_parser.add_argument('-w_ranges', dest='w_ranges', nargs='+', default=(25,), type=int)
    tracking_parser.add_argument('-draw_paths', dest='draw_paths', action='store_true')
    tracking_parser.add_argument('-r_ls', dest='random_ls', action='store_true')
    ap.set_defaults(draw_paths=False, random_ls=False)

    args = ap.parse_args()

    if args.command == 'track':
        tracking(args.strategy,
                 sigma=args.sigma,
                 conf_thresh=args.conf_thresh,
                 n_listeners=args.num_listeners,
                 n_iter=args.num_iter,
                 random_ls=args.random_ls,
                 wifi_ranges=args.w_ranges,
                 draw_paths=args.draw_paths)



