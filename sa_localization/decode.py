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


def _cluster_to_coords(grid, xVals, yVals):
    """Return centroids of connected components in a binary occupancy grid."""
    from scipy.ndimage import label

    labeled, num_features = label(grid)
    coords = []
    for comp_id in range(1, num_features + 1):
        rows, cols = np.where(labeled == comp_id)
        coords.append((float(np.mean(xVals[cols])), float(np.mean(yVals[rows]))))
    return coords
