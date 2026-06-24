#! /usr/bin/env python3
"""
Plot RTI heatmap with per-target colour-coded X markers for each out.txt row.

Marker colour convention
------------------------
  Black X  — SA correctly detected this target   (active & selected)
  Green X  — SA missed this target               (active, not selected)
  Red   X  — SA false positive for this target   (not active, but selected)

Legend also shows:
  • expected target count and binary bitmask string
  • detected target count and binary bitmask string

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
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerBase
import rti

from sa_localization.config      import Config
from sa_localization.data_io     import load_listenx_file, load_sensor_coords
from sa_localization.calibration import CalibrationState
from sa_localization.qubo        import build_qubo, get_synth_targets
from sa_localization.solver      import run_sa
from sa_localization.benchmark   import load_synth_rows

SYNTH_FILE = 'out.txt'
CAL_FILE   = 'basement/basement_listenx_out_1.txt'
SAVE_DIR   = 'sa_localization/plots'

_MARKER_SIZE  = 10
_MARKER_WIDTH = 1.5


class _NullHandler(HandlerBase):
    """Legend handler that renders nothing — used for text-only legend rows."""
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        from matplotlib.patches import Rectangle
        r = Rectangle([0, 0], 0, 0, fill=False, edgecolor='none', linewidth=0,
                      transform=trans)
        return [r]


def _text_handle(label):
    """Return an invisible Line2D that appears as a text-only legend entry."""
    return mlines.Line2D([], [], color='none', linewidth=0, label=label)


def plot_row(row_idx, true_mask, image, xVals, yVals,
             sensor_coords, known_locs, active_targets, selected_targets, save=False):
    """
    Parameters
    ----------
    active_targets   : set of int  — target indices truly present (from true_mask)
    selected_targets : set of int  — target indices that SA detected (after threshold)
    """
    K = (max(known_locs.keys()) + 1) if known_locs else 0

    n_expected      = len(active_targets)
    n_detected      = len(selected_targets)
    true_binary     = format(true_mask, '0{}b'.format(K))
    detected_mask   = sum(1 << j for j in selected_targets)
    detected_binary = format(detected_mask, '0{}b'.format(K))

    fig, ax = plt.subplots(figsize=(9, 7))

    extent = [xVals[0], xVals[-1], yVals[0], yVals[-1]]
    im = ax.imshow(image, origin='lower', extent=extent,
                   vmin=0, vmax=8, cmap='hot', aspect='equal')
    plt.colorbar(im, ax=ax, label='RTI intensity')

    # ── Sensor nodes ──────────────────────────────────────────────────────
    sensor_h, = ax.plot(sensor_coords[:, 0], sensor_coords[:, 1],
                        'b.', markersize=8, label='Sensor nodes')
    for i, c in enumerate(sensor_coords):
        ax.text(c[0], c[1], str(i + 1), fontsize=6, color='cyan',
                ha='center', va='bottom')

    # ── Per-target X markers (colour-coded) ───────────────────────────────
    marker_handles = [sensor_h]
    null_map       = {}

    for j in range(K):
        if j not in known_locs:
            continue
        x, y        = known_locs[j]
        is_active   = j in active_targets
        is_selected = j in selected_targets

        if is_active and is_selected:
            color, cat = 'black', 'Correct'
        elif is_active and not is_selected:
            color, cat = 'green', 'Missed'
        elif not is_active and is_selected:
            color, cat = 'red', 'False+'
        else:
            continue

        h, = ax.plot(x, y, 'X', color=color,
                     markersize=_MARKER_SIZE, markeredgewidth=_MARKER_WIDTH,
                     label='{} T{:d}'.format(cat, j))
        ax.annotate('T{:d}'.format(j), xy=(x, y),
                    xytext=(5, 4), textcoords='offset points',
                    fontsize=7, color=color, fontweight='bold')
        marker_handles.append(h)

    # ── Stats as text-only legend entries ─────────────────────────────────
    sep   = _text_handle('')
    stat1 = _text_handle('Expected {:d}:  [{}]'.format(n_expected, true_binary))
    stat2 = _text_handle('Detected {:d}:  [{}]'.format(n_detected, detected_binary))

    text_entries = [sep, stat1, stat2]
    null_map     = {h: _NullHandler() for h in text_entries}

    ax.legend(handles=marker_handles + text_entries,
              handler_map=null_map,
              loc='upper right', fontsize=8,
              framealpha=0.85, borderpad=0.8)

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Row {:d} | bitmask {:d}'.format(row_idx, true_mask), fontsize=11)

    if save:
        os.makedirs(SAVE_DIR, exist_ok=True)
        path = os.path.join(SAVE_DIR,
                            'row_{:02d}_mask_{:d}.png'.format(row_idx, true_mask))
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print("Saved: {}".format(path))
        plt.close(fig)
    else:
        plt.show()


def _decode_selected_targets(best_sample, cal, inversion, xVals, yVals, cfg):
    """Return the set of target indices that SA detected (threshold-filtered)."""
    synth_targets = get_synth_targets()
    if synth_targets is None:
        return set()

    targets  = sorted(synth_targets.keys())
    selected = set()
    for ii, j in enumerate(targets):
        if best_sample.get(ii, 0) != 1:
            continue
        sc  = cal.score_vec(synth_targets[j])
        img = rti.callRTI(sc, inversion, len(xVals), len(yVals))
        if img.max() > cfg.personInAreaThreshold:
            selected.add(j)
    return selected


def main():
    save = '--save' in sys.argv
    cfg  = Config()

    sensor_coords        = load_sensor_coords(cfg.coord_file)
    cal_rows, _          = load_listenx_file(CAL_FILE)
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
                score       = cal.score_vec(rss)
                image       = rti.callRTI(score, inversion, len(xVals), len(yVals))
                x, y        = rti.imageMaxCoord(image, xVals, yVals)
                known_locs[j] = (x, y)
                break

    for row_idx, (rss, true_mask) in enumerate(zip(rss_rows, bitmasks)):
        score          = cal.score_vec(rss)
        image          = rti.callRTI(score, inversion, len(xVals), len(yVals))
        active_targets = {j for j in range(num_targets) if true_mask & (1 << j)}

        try:
            Q              = build_qubo(score, cal, inversion, xVals, yVals, cfg)
            best_sample, _ = run_sa(Q, cfg)
            selected_targets = _decode_selected_targets(
                best_sample, cal, inversion, xVals, yVals, cfg)
        except NotImplementedError:
            selected_targets = set()

        plot_row(row_idx, true_mask, image, xVals, yVals,
                 sensor_coords, known_locs, active_targets, selected_targets, save=save)


if __name__ == '__main__':
    main()
