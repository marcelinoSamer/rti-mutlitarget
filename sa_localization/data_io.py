import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

MISSING_THRESH = -10
SENTINEL       = 127


def load_listenx_file(fname):
    """Read a pre-processed 721-column listenx file.

    Returns:
        rows  : list of np.ndarray (720,) int
        times : list of int  — timestamps in ms
    """
    rows, times = [], []
    prevRSS = None
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals    = [int(x) for x in line.split()]
            time_ms = vals.pop()
            rss     = np.array(vals, dtype=int)
            if prevRSS is not None:
                mask = rss > MISSING_THRESH
                rss[mask] = prevRSS[mask]
            rows.append(rss.copy())
            times.append(time_ms)
            prevRSS = rss.copy()
    return rows, times


def load_sensor_coords(fname):
    """Load sensor node (x,y) coordinates. Returns np.ndarray (numNodes, 2)."""
    return np.loadtxt(fname)


def load_pivot_coords(fname):
    """Load pivot waypoint (x,y) coordinates. Returns np.ndarray (numPivots, 2)."""
    return np.loadtxt(fname)


def load_path_indices(fname):
    """Load path pivot-index sequence. Returns np.ndarray (numSteps,) int."""
    return np.loadtxt(fname, dtype=int)


def write_estimates(fout, x, y):
    """Write one coordinate estimate (or -99 -99 if no person) to fout."""
    fout.write('{:.4f} {:.4f}\n'.format(x, y))
    fout.flush()
