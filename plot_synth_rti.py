#!/usr/bin/env python3
"""
plot_synth_rti.py
-----------------
Synthesize multi-target RSS data using rti.synthMultiTargetRSS, run RTI imaging
on all 2^k combinations, and save PNG visualizations to Visuals/.

Two target snapshots are taken from rows TARGET_ROWS of the basement recording.
The empty-room baseline uses the first calLines rows of the same file.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')           # non-interactive — must come before any plt import
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
import rti as rtimod             # rti.py — the math core

# ── Configuration ──────────────────────────────────────────────────────────────
EMPTY_FILE    = os.path.join(REPO, 'basement', 'basement_listenx_out_1.txt')
DATA_FILE     = os.path.join(REPO, 'basement', 'basement_listenx_out_1.txt')
COORD_FILE    = os.path.join(REPO, 'basement', 'sensor_coords_basement_m.txt')
PIVOT_FILE    = os.path.join(REPO, 'basement', 'pivot_coords_basement_m.txt')
PATH_FILE     = os.path.join(REPO, 'basement', 'path_basement_1_f.txt')
VISUALS_DIR   = os.path.join(REPO, 'Visuals')

TARGET_ROWS   = [200, 500]   # 0-based post-load indices; must be >= calLines

calLines      = 50
topChs        = 3
channels      = 8
delta_p       = 0.2
sigmax2       = 0.5
delta_corr    = 1.0
excessPathLen = 0.1
units         = 'm'
MISSING_THRESH = -10
SENTINEL       = 127
startPathTime = 56000.0        # ms — when subject hits first waypoint
speed         = 1.0 / 8000.0  # pivot points per ms

LABELS  = [
    'Empty Room\n(mask 00)',
    'Target A Only\n(mask 01)',
    'Target B Only\n(mask 10)',
    'Both Targets\n(mask 11)',
]
TAGS    = ['empty', 'target_a', 'target_b', 'both']
COLORS  = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_rows(fname):
    """Read a listenx file → list of int RSS arrays (720,) with fill-forward on sentinels."""
    rows, times, prev = [], [], None
    with open(fname) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            vals  = list(map(int, line.split()))
            t     = vals.pop()
            rss   = np.array(vals, dtype=int)
            if prev is not None:
                rss[rss > MISSING_THRESH] = prev[rss > MISSING_THRESH]
            rows.append(rss.copy())
            times.append(t)
            prev = rss.copy()
    return rows, times


def compute_empty_rss(rows, n_cal):
    """Per-link mean of valid measurements in the first n_cal rows; 127.0 for unmeasured links."""
    L = len(rows[0])
    s = np.zeros(L, dtype=float)
    c = np.zeros(L, dtype=float)
    for row in rows[:n_cal]:
        valid = row <= MISSING_THRESH
        s[valid] += row[valid]
        c[valid] += 1.0
    return np.where(c > 0, s / c, float(SENTINEL))


def build_cal_vectors(empty_rss, n_pairs):
    """Compute maxInds and calVec from empty_rss, matching rti_stub.py calibration."""
    e = empty_rss.copy()
    # Replace sentinels with a very low value so they are never chosen as a top channel.
    e[e > MISSING_THRESH] = -200.0
    mat       = e.reshape(channels, n_pairs)
    max_inds  = mat.T.argsort()            # shape (n_pairs, channels); highest last
    cal_vec   = rtimod.sumTopRows(mat, max_inds, topChs)
    return max_inds, cal_vec


def synth_to_score(synth_rss, empty_rss, max_inds, cal_vec, n_pairs):
    """Convert a synthesized RSS row to an RTI score vector (calVec - curVec)."""
    r = synth_rss.astype(float).copy()
    # Fill any remaining sentinels with the empty baseline value so their
    # contribution to scoreVec is zero (no attenuation).
    sent = r > MISSING_THRESH
    r[sent] = empty_rss[sent]
    r[r > MISSING_THRESH] = -200.0   # if emptyRSS is also sentinel
    mat     = r.reshape(channels, n_pairs)
    cur_vec = rtimod.sumTopRows(mat, max_inds, topChs)
    return cal_vec - cur_vec


# ── Load & calibrate ───────────────────────────────────────────────────────────
print("Loading data...")
rows, times = load_rows(EMPTY_FILE)
n_links   = len(rows[0])
n_pairs   = n_links // channels          # 90

empty_rss = compute_empty_rss(rows, calLines)
max_inds, cal_vec = build_cal_vectors(empty_rss, n_pairs)

snap_A = rows[TARGET_ROWS[0]].astype(float)
snap_B = rows[TARGET_ROWS[1]].astype(float)
print(f"  Target A = row {TARGET_ROWS[0]}, Target B = row {TARGET_ROWS[1]}")

# ── Ground-truth positions ─────────────────────────────────────────────────────
pivot_coords = np.loadtxt(PIVOT_FILE)
path_ind     = np.loadtxt(PATH_FILE)

def ground_truth(row_idx):
    """Return (x, y) ground-truth coordinate for a given row index, or None."""
    t   = times[row_idx]
    pos = rtimod.calcActualPosition(t, pivot_coords, path_ind, startPathTime, speed)
    return tuple(pos) if len(pos) > 0 else None

gt_A = ground_truth(TARGET_ROWS[0])
gt_B = ground_truth(TARGET_ROWS[1])
print(f"  Ground truth A: {gt_A}")
print(f"  Ground truth B: {gt_B}")

# Which combinations have which targets present (by bitmask bit position)
# mask bit 0 = Target A,  bit 1 = Target B
GT_PER_MASK = {
    0: [],
    1: [gt_A],
    2: [gt_B],
    3: [gt_A, gt_B],
}

def plot_ground_truth(ax, mask):
    """Overlay X markers for all targets present in this combination."""
    for coord in GT_PER_MASK.get(mask, []):
        if coord is not None:
            ax.plot(coord[0], coord[1], 'w', marker='x', markersize=14,
                    markeredgewidth=3, zorder=10)
            ax.plot(coord[0], coord[1], 'k', marker='x', markersize=10,
                    markeredgewidth=1.5, zorder=11)

# ── Synthesize ─────────────────────────────────────────────────────────────────
print("Synthesizing 4 combinations...")
synth_results = rtimod.synthMultiTargetRSS(empty_rss, [snap_A, snap_B])

# ── RTI setup ─────────────────────────────────────────────────────────────────
print("Initialising RTI projection matrix (may take a few seconds)...")
sensor_coords = np.loadtxt(COORD_FILE)
inversion, xVals, yVals = rtimod.initRTI(
    sensor_coords, delta_p, sigmax2, delta_corr, excessPathLen)
xValsLen = len(xVals)
yValsLen = len(yVals)
image_extent = (min(xVals)-delta_p/2, max(xVals)+delta_p/2,
                min(yVals)-delta_p/2, max(yVals)+delta_p/2)

# ── Compute RTI images ─────────────────────────────────────────────────────────
images, score_vecs = [], []
for mask, synth in synth_results:
    sv    = synth_to_score(synth, empty_rss, max_inds, cal_vec, n_pairs)
    image = rtimod.callRTI(sv, inversion, xValsLen, yValsLen)
    images.append(image)
    score_vecs.append(sv)
    print(f"  mask={mask:02b}  max={image.max():.3f}  min={image.min():.3f}")

vmax = max(im.max() for im in images)


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 1 — 2×2 RTI grid (all 4 combinations)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nSaving synth_rti_grid.png ...")
fig, axes = plt.subplots(2, 2, figsize=(13, 11))
fig.suptitle(
    'Multi-Target RTI Synthesis — All 4 Occupancy Combinations\n'
    f'(Target A = row {TARGET_ROWS[0]}, Target B = row {TARGET_ROWS[1]})',
    fontsize=14, fontweight='bold')

for idx, ax in enumerate(axes.flat):
    mask, _ = synth_results[idx]
    im = ax.imshow(images[idx], origin='lower', extent=image_extent,
                   vmin=0, vmax=max(vmax, 0.5), cmap='hot', aspect='equal')
    ax.plot(sensor_coords[:,0], sensor_coords[:,1],
            'c.', markersize=9, label='Sensor', zorder=5)
    plot_ground_truth(ax, mask)
    ax.set_title(LABELS[idx], fontsize=12)
    ax.set_xlabel(f'X ({units})', fontsize=11)
    ax.set_ylabel(f'Y ({units})', fontsize=11)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('RTI value', fontsize=9)

plt.tight_layout()
out = os.path.join(VISUALS_DIR, 'synth_rti_grid.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 2 — Individual RTI PNGs (one per combination)
# ═══════════════════════════════════════════════════════════════════════════════
for idx, (mask, synth) in enumerate(synth_results):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(images[idx], origin='lower', extent=image_extent,
                   vmin=0, vmax=max(vmax, 0.5), cmap='hot', aspect='equal')
    ax.plot(sensor_coords[:,0], sensor_coords[:,1],
            'c.', markersize=10, zorder=5)
    for k, sc in enumerate(sensor_coords):
        ax.text(sc[0], sc[1], str(k+1), fontsize=7,
                ha='center', va='bottom', color='cyan', zorder=6)
    plot_ground_truth(ax, mask)
    ax.set_title(f'RTI — {LABELS[idx].replace(chr(10), " ")}', fontsize=13)
    ax.set_xlabel(f'X ({units})', fontsize=12)
    ax.set_ylabel(f'Y ({units})', fontsize=12)
    cb = plt.colorbar(im, ax=ax)
    cb.set_label('Image value', fontsize=11)
    plt.tight_layout()
    out = os.path.join(VISUALS_DIR, f'synth_rti_{TAGS[idx]}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 3 — Per-target delta heatmap (channels × tx-rx pairs)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nSaving synth_delta_heatmap.png ...")
empty_mask = empty_rss > MISSING_THRESH

def compute_delta(snap, empty_rss, empty_mask):
    s = snap.astype(float)
    d = np.where(empty_mask | (s > MISSING_THRESH), 0.0, s - empty_rss)
    return d

delta_A = compute_delta(snap_A, empty_rss, empty_mask)
delta_B = compute_delta(snap_B, empty_rss, empty_mask)

vlim = max(abs(delta_A).max(), abs(delta_B).max(), 1.0)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for ax, delta, label in zip(axes, [delta_A, delta_B], ['Target A', 'Target B']):
    mat = delta.reshape(channels, n_pairs)
    im  = ax.imshow(mat, aspect='auto', cmap='RdBu_r', vmin=-vlim, vmax=vlim,
                    interpolation='nearest')
    ax.set_title(f'{label}  Δ RSS (dBm) per link', fontsize=13)
    ax.set_xlabel('Tx-Rx pair index (0–89)', fontsize=11)
    ax.set_ylabel('Channel (0–7)', fontsize=11)
    ax.set_yticks(range(channels))
    cb = plt.colorbar(im, ax=ax)
    cb.set_label('ΔdBm (negative = attenuation)', fontsize=10)

fig.suptitle('Per-Link RSS Attenuation Delta for Each Single-Target Snapshot',
             fontsize=13, fontweight='bold')
plt.tight_layout()
out = os.path.join(VISUALS_DIR, 'synth_delta_heatmap.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 4 — RSS value distribution (histogram) for all 4 combinations
# ═══════════════════════════════════════════════════════════════════════════════
print("Saving synth_rss_dist.png ...")
fig, ax = plt.subplots(figsize=(10, 5))
for idx, (mask, synth) in enumerate(synth_results):
    valid = synth[synth <= MISSING_THRESH]
    ax.hist(valid.astype(int), bins=40, alpha=0.55, color=COLORS[idx],
            label=LABELS[idx].replace('\n', ' '), density=True, edgecolor='none')
ax.set_xlabel('RSS (dBm)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('RSS Value Distribution — 4 Synthesised Combinations', fontsize=13)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
out = os.path.join(VISUALS_DIR, 'synth_rss_dist.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 5 — RTI score vector per combination (calVec − curVec over 90 pairs)
# ═══════════════════════════════════════════════════════════════════════════════
print("Saving synth_score_vec.png ...")
fig, ax = plt.subplots(figsize=(12, 5))
pair_idx = np.arange(n_pairs)
for idx, sv in enumerate(score_vecs):
    ax.plot(pair_idx, sv, color=COLORS[idx], alpha=0.8, linewidth=1.2,
            label=LABELS[idx].replace('\n', ' '))
ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
ax.set_xlabel('Tx-Rx pair index (top-3 channel sum, 0–89)', fontsize=12)
ax.set_ylabel('calVec − curVec  (dBm)', fontsize=12)
ax.set_title('RTI Score Vector — Attenuation Signal Fed Into Back-Projection',
             fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
out = os.path.join(VISUALS_DIR, 'synth_score_vec.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 6 — Synthesised RSS values across all links for all 4 combinations
# ═══════════════════════════════════════════════════════════════════════════════
print("Saving synth_rss_profile.png ...")
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
link_idx = np.arange(n_links)
for idx, (mask, synth) in enumerate(synth_results):
    ax = axes[idx]
    valid_mask = synth <= MISSING_THRESH
    ax.scatter(link_idx[valid_mask], synth[valid_mask], s=2,
               color=COLORS[idx], alpha=0.7, rasterized=True)
    ax.scatter(link_idx[~valid_mask], np.full(np.sum(~valid_mask), -1),
               s=2, color='lightgray', alpha=0.4, rasterized=True)
    ax.set_ylabel('RSS (dBm)', fontsize=10)
    ax.set_title(LABELS[idx].replace('\n', ' '), fontsize=11)
    ax.set_ylim(-130, 5)
    ax.axhline(MISSING_THRESH, color='gray', linewidth=0.8,
               linestyle='--', alpha=0.5, label='Missing threshold')
    ax.grid(alpha=0.2)

axes[-1].set_xlabel('Link index (0–719)', fontsize=12)
fig.suptitle('Synthesised RSS Values Across All 720 Links', fontsize=13,
             fontweight='bold')
plt.tight_layout()
out = os.path.join(VISUALS_DIR, 'synth_rss_profile.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved {out}")


print("\n✓ All plots saved to", VISUALS_DIR)
