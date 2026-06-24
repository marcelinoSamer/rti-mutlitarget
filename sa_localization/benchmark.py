#! /usr/bin/env python3
"""
Benchmark: SA vs classical RTI for single-target and multi-target localization.

Compares:
  1. Single-target rows  (bitmask has exactly one bit set)
  2. Multi-target rows   (bitmask has two or more bits set)

For each regime, runs SA across a sweep of alpha values (linear vs quadratic
balance in the QUBO) and the classical RTI MAP estimator.  Results are shown
as empirical CDFs of per-target localization error.

Usage:
    python3 sa_localization/benchmark.py
    python3 sa_localization/benchmark.py --save   # saves CDF figure as PNG
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import copy
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
from sa_localization.decode      import decode_row_solution

# ── Tunable parameters ────────────────────────────────────────────────────────
CAL_FILE   = 'basement/basement_listenx_out_1.txt'
N_TRIALS   = 30          # independent SA runs per (row, alpha) to build the CDF
PENALTY    = 16.0        # metres — replaces inf for missed/wrong detections
ALPHA_LIST = [0.0, 0.25, 0.5, 0.75, 1.0]
SAVE_DIR   = 'sa_localization/plots'
# ─────────────────────────────────────────────────────────────────────────────


# ── Helpers ───────────────────────────────────────────────────────────────────

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
            vals = [int(x) for x in line.split()]
            bitmasks.append(vals.pop())
            rss_rows.append(np.array(vals, dtype=int))
    return rss_rows, bitmasks


def _active_targets(mask, num_targets):
    """Return list of target indices that are present in this bitmask."""
    return [j for j in range(num_targets) if mask & (1 << j)]


def per_target_errors(estimates, active_known_locs, penalty=PENALTY):
    """Compute per-target localization error.

    For each active target, finds the closest estimate (if any).
    Returns a list of errors, one per active target.
    Missing detections are penalised with `penalty`.
    """
    errors = []
    for loc in active_known_locs:
        if estimates:
            d = min(np.linalg.norm(np.array(e) - np.array(loc)) for e in estimates)
        else:
            d = penalty
        errors.append(d)
    return errors


def classical_estimates(rss, cal, inversion, xVals, yVals, cfg):
    """Return RTI MAP peak as a single-element estimate list."""
    score = cal.score_vec(rss)
    image = rti.callRTI(score, inversion, len(xVals), len(yVals))
    if image.max() > cfg.personInAreaThreshold:
        x, y = rti.imageMaxCoord(image, xVals, yVals)
        return [(x, y)]
    return []


def sa_estimates_once(rss, cal, inversion, xVals, yVals, cfg):
    """Run the SA pipeline once and return estimates (target-level decode)."""
    score      = cal.score_vec(rss)
    Q          = build_qubo(score, cal, inversion, xVals, yVals, cfg)
    best, _    = run_sa(Q, cfg)
    return decode_row_solution(best, cal, inversion, xVals, yVals, cfg)


def collect_errors_classical(rows_subset, masks_subset, active_locs_per_row,
                              cal, inversion, xVals, yVals, cfg):
    """Collect all per-target classical errors across the given rows."""
    errors = []
    for rss, _, active_locs in zip(rows_subset, masks_subset, active_locs_per_row):
        ests = classical_estimates(rss, cal, inversion, xVals, yVals, cfg)
        errors.extend(per_target_errors(ests, active_locs))
    return errors


def collect_errors_sa(rows_subset, masks_subset, active_locs_per_row,
                      cal, inversion, xVals, yVals, cfg, n_trials):
    """Collect per-target SA errors across rows × trials."""
    errors = []
    for rss, _, active_locs in zip(rows_subset, masks_subset, active_locs_per_row):
        for _ in range(n_trials):
            ests = sa_estimates_once(rss, cal, inversion, xVals, yVals, cfg)
            errors.extend(per_target_errors(ests, active_locs))
    return errors


def plot_cdf(ax, errors, label, penalty=PENALTY, **kwargs):
    """Plot an empirical CDF of errors on ax.

    Infinite errors are replaced by `penalty` before sorting.
    """
    vals = np.array([e if np.isfinite(e) else penalty for e in errors])
    if len(vals) == 0:
        return
    vals_sorted = np.sort(vals)
    p = np.arange(1, len(vals_sorted) + 1) / len(vals_sorted)
    ax.step(vals_sorted, p, where='post', label=label, **kwargs)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_benchmark(save=False):
    cfg = Config()
    cfg.num_reads  = 1    # single read per call; N_TRIALS controls repetitions

    # ── Setup ──────────────────────────────────────────────────────────────
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

    rss_rows, bitmasks = load_synth_rows(cfg.synth_file)
    num_targets = int(np.ceil(np.log2(max(bitmasks) + 1)))

    # Derive known target locations from single-target rows (no hardcoded coords)
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

    print("Known target locations:")
    for j, loc in known_locs.items():
        print("  Target {:d}: ({:.3f}, {:.3f}) m".format(j, loc[0], loc[1]))

    # ── Split rows into single-target and multi-target ─────────────────────
    single_rows, single_masks, single_active = [], [], []
    multi_rows,  multi_masks,  multi_active  = [], [], []

    for rss, mask in zip(rss_rows, bitmasks):
        active = _active_targets(mask, num_targets)
        if len(active) == 0:
            continue                            # empty room — skip
        active_locs = [known_locs[j] for j in active if j in known_locs]
        if len(active) == 1:
            single_rows.append(rss)
            single_masks.append(mask)
            single_active.append(active_locs)
        else:
            multi_rows.append(rss)
            multi_masks.append(mask)
            multi_active.append(active_locs)

    print("\nSingle-target rows: {:d}  |  Multi-target rows: {:d}".format(
        len(single_rows), len(multi_rows)))
    print("Collecting {:d} SA trials per (row × alpha)…\n".format(N_TRIALS))

    # ── Collect classical errors (deterministic — no alpha dependence) ─────
    classic_single = collect_errors_classical(
        single_rows, single_masks, single_active,
        cal, inversion, xVals, yVals, cfg)

    classic_multi = collect_errors_classical(
        multi_rows, multi_masks, multi_active,
        cal, inversion, xVals, yVals, cfg)

    # ── Collect SA errors for each alpha ───────────────────────────────────
    sa_single = {}
    sa_multi  = {}

    for alpha in ALPHA_LIST:
        cfg_a       = copy.copy(cfg)
        cfg_a.alpha = alpha

        print("  alpha = {:.2f}".format(alpha), end='  ', flush=True)

        sa_single[alpha] = collect_errors_sa(
            single_rows, single_masks, single_active,
            cal, inversion, xVals, yVals, cfg_a, N_TRIALS)

        sa_multi[alpha] = collect_errors_sa(
            multi_rows, multi_masks, multi_active,
            cal, inversion, xVals, yVals, cfg_a, N_TRIALS) if multi_rows else []

        print("single errors: {:d}  multi errors: {:d}".format(
            len(sa_single[alpha]), len(sa_multi[alpha])))

    # ── Print summary table ────────────────────────────────────────────────
    def median(errs):
        vals = [e if np.isfinite(e) else PENALTY for e in errs]
        return float(np.median(vals)) if vals else float('nan')

    def pct_within(errs, thr):
        vals = [e if np.isfinite(e) else PENALTY for e in errs]
        return 100.0 * sum(1 for e in vals if e <= thr) / len(vals) if vals else 0.0

    print("\n{:>12s} | {:>12s} {:>12s} | {:>12s} {:>12s}".format(
        "method", "med_single", "%<1m_single", "med_multi", "%<1m_multi"))
    print("-" * 68)
    print("{:>12s} | {:>12.3f} {:>12.1f} | {:>12.3f} {:>12.1f}".format(
        "classical",
        median(classic_single), pct_within(classic_single, 1.0),
        median(classic_multi),  pct_within(classic_multi,  1.0)))
    for alpha in ALPHA_LIST:
        print("{:>12s} | {:>12.3f} {:>12.1f} | {:>12.3f} {:>12.1f}".format(
            "SA a={:.2f}".format(alpha),
            median(sa_single[alpha]), pct_within(sa_single[alpha], 1.0),
            median(sa_multi[alpha]),  pct_within(sa_multi[alpha],  1.0)))

    # ── CDF figure ─────────────────────────────────────────────────────────
    alpha_cmap = plt.cm.plasma(np.linspace(0.1, 0.9, len(ALPHA_LIST)))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, title, classic_errs, sa_errs_dict in [
        (axes[0], "Single-target",  classic_single, sa_single),
        (axes[1], "Multi-target",   classic_multi,  sa_multi),
    ]:
        plot_cdf(ax, classic_errs, label="Classical RTI MAP",
                 color='black', linewidth=2.0, linestyle='--')

        for alpha, color in zip(ALPHA_LIST, alpha_cmap):
            plot_cdf(ax, sa_errs_dict[alpha],
                     label=r"SA $\alpha$={:.2f}".format(alpha),
                     color=color, linewidth=1.5)

        ax.axvline(x=1.0, color='gray', linestyle=':', linewidth=1,
                   label='1 m threshold')
        ax.set_xlabel("Localization error (m)", fontsize=12)
        ax.set_ylabel("CDF  P(error ≤ x)", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "SA vs Classical RTI — {:d} SA trials/row | penalty = {:.0f} m".format(
            N_TRIALS, PENALTY),
        fontsize=11)
    fig.tight_layout()

    if save:
        os.makedirs(SAVE_DIR, exist_ok=True)
        path = os.path.join(SAVE_DIR, 'benchmark_cdf.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print("\nSaved: {}".format(path))
    else:
        plt.show()

    plt.close(fig)


if __name__ == '__main__':
    run_benchmark(save='--save' in sys.argv)
