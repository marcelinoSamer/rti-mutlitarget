import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

# Populated by build_qubo; read by decode_row_solution via get_synth_targets()
_synth_rss     = None   # list of np.ndarray (720,) int — all 2^K rows
_synth_masks   = None   # list of int  (bitmasks)
_synth_targets = None   # dict { target_idx: np.ndarray (720,) } — single-target rows only


def _load_synth(cfg):
    """Load rows from the synth multi-target file (out.txt by default)."""
    fname = getattr(cfg, 'synth_file', 'out.txt')
    rows, masks = [], []
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [int(x) for x in line.split()]
            masks.append(vals.pop())              # last column = bitmask
            rows.append(np.array(vals, dtype=int))
    return rows, masks


def get_synth_data():
    """Return the RSS rows and bitmasks loaded by the last build_qubo call."""
    return _synth_rss, _synth_masks


def get_synth_targets():
    """Return the single-target RSS rows dict loaded by the last build_qubo call.

    Returns dict { target_index j : np.ndarray (720,) } for rows with exactly
    one bit set in their bitmask.
    """
    return _synth_targets


def build_qubo(score_vec, cal, inversion, xVals, yVals, cfg):
    """Build the QUBO matrix for multi-target DfP localization.

    Variables
    ---------
    x_j in {0, 1} for j = 0 ... K-1  (one per target, not per row)
    x_j = 1  means target j is detected as present.

    Linear term  Q[(j,j)] = -alpha * rti_image[peak_row_j, peak_col_j]
        The RTI image of the actual measurement evaluated at target j's known
        pixel location.  High value = strong signal evidence that target j is
        present.  This is the same signal used by the classical MAP estimator.

    Quadratic term  Q[(j,k)] = (1-alpha) * cosine_similarity(score_j, score_k)
        Penalise co-selecting two targets whose templates look alike (redundant).

    Args
    ----
    score_vec  : np.ndarray (numPairs,)            calibrated attenuation for this epoch
    cal        : CalibrationState                   provides score_vec() for templates
    inversion  : np.ndarray (numPixels, numPairs)   RTI projection matrix
    xVals      : np.ndarray (n_x,)                 pixel x-coordinates (m)
    yVals      : np.ndarray (n_y,)                 pixel y-coordinates (m)
    cfg        : Config                             must have cfg.synth_file, cfg.alpha

    Returns
    -------
    Q : dict { (int, int): float }
        QUBO in the dictionary form expected by openjij.SASampler.sample_qubo().
        Keys are (j, k) target-index pairs with j <= k.
    """
    import rti as _rti

    global _synth_rss, _synth_masks, _synth_targets
    _synth_rss, _synth_masks = _load_synth(cfg)

    # Extract single-target rows (bitmask is a power of 2, i.e. exactly one bit set)
    _synth_targets = {}
    for rss, mask in zip(_synth_rss, _synth_masks):
        if mask > 0 and (mask & (mask - 1)) == 0:
            j = mask.bit_length() - 1
            _synth_targets[j] = rss

    targets = sorted(_synth_targets.keys())
    K = len(targets)

    alpha = getattr(cfg, 'alpha', 0.5)
    n_x   = len(xVals)
    n_y   = len(yVals)

    # Calibrated scores for each single-target template
    template_scores = {j: cal.score_vec(_synth_targets[j]) for j in targets}

    # RTI image of the actual measurement — same quantity classical RTI uses
    measurement_image = _rti.callRTI(score_vec, inversion, n_x, n_y)

    Q = {}

    # --- Linear terms (diagonal) -----------------------------------------
    # For each target j, find its peak pixel from the template image, then
    # read the measurement image at that pixel.  Targets with strong signal
    # at their known location get a large negative (rewarding) diagonal.
    for ii, j in enumerate(targets):
        template_image = _rti.callRTI(template_scores[j], inversion, n_x, n_y)
        peak_idx       = int(np.argmax(template_image))
        peak_row, peak_col = divmod(peak_idx, n_x)
        signal = float(measurement_image[peak_row, peak_col])
        Q[(ii, ii)] = -alpha * signal

    # --- Quadratic terms (off-diagonal) ----------------------------------
    # Penalise co-selecting targets with similar calibrated score templates.
    for ii in range(K):
        j  = targets[ii]
        sj = template_scores[j]
        nj = np.linalg.norm(sj)
        for kk in range(ii + 1, K):
            k  = targets[kk]
            sk = template_scores[k]
            nk = np.linalg.norm(sk)
            if nj > 0 and nk > 0:
                cos_sim = float(np.dot(sj, sk) / (nj * nk))
            else:
                cos_sim = 0.0
            Q[(ii, kk)] = (1 - alpha) * cos_sim

    return Q
