#! /usr/bin/env python3
"""
Plot RTI heatmap with SA result (red X) and known location (green X) for each out.txt row.

Usage:
    python3 sa_localization/plot_results.py            # interactive display
    python3 sa_localization/plot_results.py --save     # save PNGs to sa_localization/plots/
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rti

from sa_localization.config      import Config
from sa_localization.data_io     import load_listenx_file, load_sensor_coords
from sa_localization.calibration import CalibrationState
from sa_localization.qubo        import build_qubo
from sa_localization.solver      import run_sa
from sa_localization.decode      import decode_solution
from sa_localization.benchmark   import load_synth_rows

SYNTH_FILE = 'out.txt'
CAL_FILE   = 'basement/basement_listenx_out_1.txt'
SAVE_DIR   = 'sa_localization/plots'


def plot_row(row_idx, true_mask, image, xVals, yVals,
             sensor_coords, known_locs, sa_estimates, save=False):
    fig, ax = plt.subplots(figsize=(8, 7))

    extent = [xVals[0], xVals[-1], yVals[0], yVals[-1]]
    im = ax.imshow(image, origin='lower', extent=extent,
                   vmin=0, vmax=8, cmap='hot', aspect='equal')
    plt.colorbar(im, ax=ax, label='RTI intensity')

    # Sensor nodes
    ax.plot(sensor_coords[:, 0], sensor_coords[:, 1],
            'b.', markersize=10, label='Sensor nodes')
    for i, c in enumerate(sensor_coords):
        ax.text(c[0], c[1], str(i + 1), fontsize=7, color='cyan',
                ha='center', va='bottom')

    # Known target locations — green X
    num_targets = (max(known_locs.keys()) + 1) if known_locs else 0
    for j in range(num_targets):
        if (true_mask & (1 << j)) and j in known_locs:
            x, y = known_locs[j]
            label = 'Known T{:d}'.format(j) if j == 0 else '_nolegend_'
            ax.plot(x, y, 'gX', markersize=18, markeredgewidth=2.5, label=label)

    # SA estimated locations — red X
    for i, (x, y) in enumerate(sa_estimates):
        label = 'SA estimate' if i == 0 else '_nolegend_'
        ax.plot(x, y, 'rX', markersize=18, markeredgewidth=2.5, label=label)

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Row {:d} | bitmask={:d} ({:s}) | green=known, red=SA'.format(
        row_idx, true_mask, bin(true_mask)), fontsize=11)
    ax.legend(loc='upper right', fontsize=9)

    if save:
        os.makedirs(SAVE_DIR, exist_ok=True)
        path = os.path.join(SAVE_DIR,
                            'row_{:02d}_mask_{:d}.png'.format(row_idx, true_mask))
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print("Saved: {}".format(path))
        plt.close(fig)
    else:
        plt.show()


def main():
    save = '--save' in sys.argv
    cfg  = Config()

    sensor_coords   = load_sensor_coords(cfg.coord_file)
    cal_rows, _     = load_listenx_file(CAL_FILE)
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

    # Derive known locations from single-target rows
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

    for row_idx, (rss, true_mask) in enumerate(zip(rss_rows, bitmasks)):
        score = cal.score_vec(rss)
        image = rti.callRTI(score, inversion, len(xVals), len(yVals))

        try:
            Q              = build_qubo(score, inversion, xVals, yVals, cfg)
            best_sample, _ = run_sa(Q, cfg)
            sa_estimates   = decode_solution(best_sample, xVals, yVals)
        except NotImplementedError:
            x, y         = rti.imageMaxCoord(image, xVals, yVals)
            sa_estimates = [(x, y)] if image.max() > cfg.personInAreaThreshold else []

        plot_row(row_idx, true_mask, image, xVals, yVals,
                 sensor_coords, known_locs, sa_estimates, save=save)


if __name__ == '__main__':
    main()
