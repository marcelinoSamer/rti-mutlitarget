#! /usr/bin/env python3
"""
Generate out.txt from a basement recording — no CLI arguments needed.

Edit the parameters below, then run:
    python3 sa_localization/generate_synth.py

K targets → out.txt has 2^K rows.
Increasing K gives more diverse target-position combinations, which produces
a smoother sigmoid CDF in the benchmark.

    K=2  →   4 rows  (original, staircase CDF)
    K=6  →  64 rows  (good for CDF)
    K=8  → 256 rows  (very smooth CDF, SA is slower)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import rti

# ── Parameters (edit these) ──────────────────────────────────────────────────
CAL_FILE    = 'basement/basement_listenx_out_1.txt'  # calibration source
DATA_FILE   = 'basement/basement_listenx_out_1.txt'  # target snapshots source
OUT_FILE    = 'out.txt'
CAL_LINES   = 50

# Number of target locations (out.txt will have 2^K rows)
K           = 6

# How to pick the K snapshot rows from post-calibration frames:
#   'uniform' — evenly spaced across all eligible (in-walk-window) frames
#   'random'  — random sample (reproducible with RANDOM_SEED)
SELECTION   = 'uniform'
RANDOM_SEED = 42

# Ground-truth walk parameterization (mirrors rti_stub.py / Config). Only frames
# whose timestamp falls inside the walk window are eligible as target snapshots,
# so every target has a REAL (interpolated walk) position — not an RTI-peak guess.
PIVOT_FILE      = 'basement/pivot_coords_basement_m.txt'
PATH_FILE       = 'basement/path_basement_1_f.txt'
START_PATH_TIME = 56000.0      # ms — when subject hits first waypoint
SPEED           = 1.0 / 8000.0 # pivot points per millisecond
TRUTH_FILE      = 'target_truth.txt'  # sidecar: "j x y" real ground truth per target
# ─────────────────────────────────────────────────────────────────────────────

MISSING_THRESH = -10
SENTINEL       = 127


def _load_rows(fname):
    """Load a 721-column listenx file with prevRSS sentinel fill.

    Returns (rows, times) where rows is a list of np.ndarray (720,) int.
    """
    rows, times, prev = [], [], None
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [int(x) for x in line.split()]
            t    = vals.pop()
            rss  = np.array(vals, dtype=int)
            if prev is not None:
                mask     = rss > MISSING_THRESH
                rss[mask] = prev[mask]
            rows.append(rss.copy())
            times.append(t)
            prev = rss.copy()
    return rows, times


def _compute_empty_rss(rows, cal_lines):
    """Per-link mean over first cal_lines rows. Links with no valid obs → 127.0."""
    n   = len(rows[0])
    s   = np.zeros(n, dtype=float)
    c   = np.zeros(n, dtype=float)
    lim = min(cal_lines, len(rows))
    for i in range(lim):
        for l in range(n):
            if rows[i][l] <= MISSING_THRESH:
                s[l] += rows[i][l]
                c[l] += 1.0
    return np.where(c > 0, s / np.maximum(c, 1.0), float(SENTINEL))


def _write_output(results, fout):
    """Write 2^K rows: 720 space-separated RSS integers + bitmask integer."""
    for mask, synth in results:
        ints = [int(round(v)) for v in synth]
        fout.write(' '.join(str(v) for v in ints) + ' ' + str(mask) + '\n')


def main():
    print("Loading '{}' …".format(DATA_FILE))
    rows, times = _load_rows(DATA_FILE)
    post_cal    = rows[CAL_LINES:]
    post_times  = times[CAL_LINES:]

    # Restrict candidates to frames inside the walk window, so each target
    # snapshot maps to a real interpolated position via calcActualPosition.
    pivots        = np.loadtxt(PIVOT_FILE)
    path          = np.loadtxt(PATH_FILE)
    end_path_time = START_PATH_TIME + (len(path) - 1) / SPEED
    eligible      = [i for i, t in enumerate(post_times)
                     if START_PATH_TIME <= t < end_path_time]
    n_elig        = len(eligible)
    print("Walk window: [{:.0f}, {:.0f}) ms — {:d} eligible post-cal frames.".format(
        START_PATH_TIME, end_path_time, n_elig))

    if n_elig < K:
        sys.exit("Error: only {:d} in-window frames available; need K={:d}. "
                 "Reduce K or use a longer recording.".format(n_elig, K))

    # Pick K positions among the eligible (in-window) frames
    if SELECTION == 'uniform':
        step    = (n_elig - 1) / max(K - 1, 1)
        picks   = [int(round(i * step)) for i in range(K)]
    elif SELECTION == 'random':
        rng     = np.random.default_rng(RANDOM_SEED)
        picks   = sorted(int(x) for x in rng.choice(n_elig, size=K, replace=False))
    else:
        sys.exit("Unknown SELECTION '{}'. Use 'uniform' or 'random'.".format(SELECTION))

    indices     = [eligible[p] for p in picks]   # indices into post_cal
    abs_indices = [CAL_LINES + i for i in indices]
    print("Selected {:d} target snapshots at file rows: {}".format(K, abs_indices))

    # Real ground-truth position for each target (bit j ↔ target_list[j])
    import rti as _rti
    true_locs = {}
    for j, i in enumerate(indices):
        pos = _rti.calcActualPosition(post_times[i], pivots, path,
                                      START_PATH_TIME, SPEED)
        true_locs[j] = (float(pos[0]), float(pos[1]))
    print("Ground-truth target positions (from walk path, not RTI):")
    for j in sorted(true_locs):
        print("  Target {:d}: ({:.3f}, {:.3f}) m".format(j, *true_locs[j]))

    # Compute empty-room baseline from calibration rows
    print("Computing empty baseline from first {:d} rows …".format(CAL_LINES))
    empty_rss = _compute_empty_rss(rows, CAL_LINES)

    # Build target snapshot list (float arrays required by synthMultiTargetRSS)
    target_list = [post_cal[i].astype(float) for i in indices]

    # Synthesise all 2^K combinations
    n_combinations = 2 ** K
    print("Synthesising {:d} combinations (2^{:d}) …".format(n_combinations, K))
    results = rti.synthMultiTargetRSS(empty_rss, target_list)

    # Write output
    with open(OUT_FILE, 'w') as f:
        _write_output(results, f)

    # Write real ground-truth sidecar: one "j x y" line per target
    with open(TRUTH_FILE, 'w') as f:
        for j in sorted(true_locs):
            f.write("{:d} {:.6f} {:.6f}\n".format(j, *true_locs[j]))
    print("Wrote ground truth for {:d} targets to '{}'.".format(len(true_locs), TRUTH_FILE))

    print("Done. Written {:d} rows to '{}'.".format(len(results), OUT_FILE))
    print("  Columns : 720 RSS values + 1 bitmask = 721")
    print("  Bitmasks: 0 (empty room) … {:d} (all {:d} targets present)".format(
        n_combinations - 1, K))
    print("\nRe-run the benchmark:")
    print("  python3 sa_localization/benchmark.py --save")


if __name__ == '__main__':
    main()
