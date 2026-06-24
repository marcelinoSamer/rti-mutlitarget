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
#   'uniform' — evenly spaced across all post-cal rows
#   'random'  — random sample (reproducible with RANDOM_SEED)
SELECTION   = 'uniform'
RANDOM_SEED = 42
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
    rows, _  = _load_rows(DATA_FILE)
    post_cal = rows[CAL_LINES:]
    n_post   = len(post_cal)

    if n_post < K:
        sys.exit("Error: only {:d} post-cal rows available; need K={:d}. "
                 "Reduce K or use a longer recording.".format(n_post, K))

    # Select K snapshot indices within post-calibration rows
    if SELECTION == 'uniform':
        step    = (n_post - 1) / max(K - 1, 1)
        indices = [int(round(i * step)) for i in range(K)]
    elif SELECTION == 'random':
        rng     = np.random.default_rng(RANDOM_SEED)
        indices = sorted(int(x) for x in rng.choice(n_post, size=K, replace=False))
    else:
        sys.exit("Unknown SELECTION '{}'. Use 'uniform' or 'random'.".format(SELECTION))

    abs_indices = [CAL_LINES + i for i in indices]
    print("Selected {:d} target snapshots at file rows: {}".format(K, abs_indices))

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

    print("Done. Written {:d} rows to '{}'.".format(len(results), OUT_FILE))
    print("  Columns : 720 RSS values + 1 bitmask = 721")
    print("  Bitmasks: 0 (empty room) … {:d} (all {:d} targets present)".format(
        n_combinations - 1, K))
    print("\nRe-run the benchmark:")
    print("  python3 sa_localization/benchmark.py --save")


if __name__ == '__main__':
    main()
