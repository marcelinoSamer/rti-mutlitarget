import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np


def decode_solution(best_sample, xVals, yVals):
    """Convert a binary pixel occupancy solution to (x, y) location estimates.

    Args:
        best_sample : dict { pixel_index: 0_or_1 } — SA solution from run_sa()
        xVals       : np.ndarray (n_x,) — x-coordinates of pixel grid
        yVals       : np.ndarray (n_y,) — y-coordinates of pixel grid

    Returns:
        estimates : list of (float, float) — (x, y) per detected target cluster.
                    Empty list if no pixel is occupied.

    Notes:
        Pixel index i maps to (row, col) = divmod(i, n_x), matching callRTI row-major layout.
    """
    n_x = len(xVals)
    n_y = len(yVals)

    occupied = [idx for idx, val in best_sample.items() if val == 1]
    if not occupied:
        return []

    grid = np.zeros((n_y, n_x), dtype=int)
    for idx in occupied:
        row, col = divmod(idx, n_x)
        if 0 <= row < n_y and 0 <= col < n_x:
            grid[row, col] = 1

    return _cluster_to_coords(grid, xVals, yVals)


def decode_row_solution(best_sample, cal, inversion, xVals, yVals, cfg):
    """Decode a target-level QUBO solution to (x, y) location estimates.

    build_qubo variables are indexed 0..K-1 (one per target, not per row).
    For each selected target j (x_j=1), runs RTI MAP on that target's
    single-target RSS template and returns the image peak as the location.

    Args
    ----
    best_sample : dict { qubo_index: 0_or_1 }  SA solution from run_sa()
    cal         : CalibrationState              provides score_vec()
    inversion   : np.ndarray (numPixels, numPairs)
    xVals, yVals: pixel grid coordinate arrays
    cfg         : Config                        provides personInAreaThreshold

    Returns
    -------
    estimates : list of (float, float)  one (x, y) per detected target.
    """
    import rti as _rti
    from sa_localization.qubo import get_synth_targets

    synth_targets = get_synth_targets()
    if synth_targets is None:
        return []

    targets = sorted(synth_targets.keys())  # [0, 1, 2, ...K-1] target indices

    estimates = []
    for ii, j in enumerate(targets):
        if best_sample.get(ii, 0) != 1:
            continue
        score = cal.score_vec(synth_targets[j])
        image = _rti.callRTI(score, inversion, len(xVals), len(yVals))
        if image.max() > cfg.personInAreaThreshold:
            x, y = _rti.imageMaxCoord(image, xVals, yVals)
            estimates.append((x, y))
    return estimates


def _cluster_to_coords(grid, xVals, yVals):
    """Return centroids of connected components in a binary occupancy grid."""
    from scipy.ndimage import label

    labeled, num_features = label(grid)
    coords = []
    for comp_id in range(1, num_features + 1):
        rows, cols = np.where(labeled == comp_id)
        coords.append((float(np.mean(xVals[cols])), float(np.mean(yVals[rows]))))
    return coords
