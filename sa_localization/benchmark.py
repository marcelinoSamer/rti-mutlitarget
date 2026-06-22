#! /usr/bin/env python3
"""
Blind benchmark: run SA on every row of out.txt and compare to bitmask ground truth.

The ground-truth locations are derived automatically from the single-target rows
(bitmask == 2^j) using RTI MAP — no external position knowledge required.

Usage:
    python3 sa_localization/benchmark.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import rti

from sa_localization.config      import Config
from sa_localization.data_io     import load_listenx_file, load_sensor_coords
from sa_localization.calibration import CalibrationState
from sa_localization.qubo        import build_qubo
from sa_localization.solver      import run_sa
from sa_localization.decode      import decode_solution

SYNTH_FILE = 'out.txt'
CAL_FILE   = 'basement/basement_listenx_out_1.txt'


def load_synth_rows(fname):
    """Load rows from a synth_multi_target output file.

    Returns:
        rss_rows : list of np.ndarray (720,) int
        bitmasks : list of int
    """
    rss_rows, bitmasks = [], []
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals    = [int(x) for x in line.split()]
            bitmask = vals.pop()
            rss_rows.append(np.array(vals, dtype=int))
            bitmasks.append(bitmask)
    return rss_rows, bitmasks


def run_benchmark():
    cfg = Config()

    sensor_coords = load_sensor_coords(cfg.coord_file)
    cal_rows, _   = load_listenx_file(CAL_FILE)
    num_links, num_channels = 720, 8

    cal = CalibrationState(num_links, num_channels, cfg.top_chs)
    for i in range(min(cfg.calLines, len(cal_rows))):
        cal.update(cal_rows[i])
    cal.finalize()

    inversion, xVals, yVals = rti.initRTI(
        sensor_coords, cfg.delta_p, cfg.sigmax2, cfg.delta, cfg.excessPathLen
    )

    rss_rows, bitmasks = load_synth_rows(SYNTH_FILE)
    num_targets = int(np.ceil(np.log2(max(bitmasks) + 1)))

    # Derive known locations from single-target rows (blind-compatible: no hardcoded coords)
    known_locs = {}
    for j in range(num_targets):
        single_mask = 1 << j
        for rss, mask in zip(rss_rows, bitmasks):
            if mask == single_mask:
                score = cal.score_vec(rss)
                image = rti.callRTI(score, inversion, len(xVals), len(yVals))
                x, y  = rti.imageMaxCoord(image, xVals, yVals)
                known_locs[j] = (x, y)
                break

    print("Known target locations (RTI MAP peaks on single-target rows):")
    for j, loc in known_locs.items():
        print("  Target {:d}: ({:.3f}, {:.3f}) m".format(j, loc[0], loc[1]))

    print("\nRow | True mask | SA estimates                    | Error (m)")
    print("-" * 70)

    for row_idx, (rss, true_mask) in enumerate(zip(rss_rows, bitmasks)):
        score = cal.score_vec(rss)

        try:
            Q              = build_qubo(score, inversion, xVals, yVals, cfg)
            best_sample, _ = run_sa(Q, cfg)
            estimates      = decode_solution(best_sample, xVals, yVals)
        except NotImplementedError:
            image     = rti.callRTI(score, inversion, len(xVals), len(yVals))
            x, y      = rti.imageMaxCoord(image, xVals, yVals)
            estimates = [(x, y)] if image.max() > cfg.personInAreaThreshold else []

        errors = []
        for j in range(num_targets):
            if true_mask & (1 << j):
                gt = np.array(known_locs[j])
                if estimates:
                    dists = [np.linalg.norm(np.array(e) - gt) for e in estimates]
                    errors.append(min(dists))
                else:
                    errors.append(float('inf'))

        est_str = ', '.join('({:.2f},{:.2f})'.format(e[0], e[1]) for e in estimates) or 'none'
        err_str = ', '.join('{:.3f}'.format(e) for e in errors) or 'n/a'
        print("  {:2d} |   {:3d} ({:s}) | {:32s} | {:s}".format(
            row_idx, true_mask, bin(true_mask), est_str, err_str))


if __name__ == '__main__':
    run_benchmark()
