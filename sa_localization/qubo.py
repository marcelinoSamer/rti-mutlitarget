import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np


def build_qubo(score_vec, inversion, xVals, yVals, cfg):
    """Build the QUBO matrix for multi-target DfP localization.

    Binary variable x_i in {0, 1}: pixel i is occupied by at least one target.

    Args:
        score_vec  : np.ndarray (numPairs,)         — link attenuation score (calVec - curVec)
        inversion  : np.ndarray (numPixels, numPairs) — RTI projection matrix from rti.initRTI()
        xVals      : np.ndarray (n_x,)              — x-coords of pixel grid centres (m)
        yVals      : np.ndarray (n_y,)              — y-coords of pixel grid centres (m)
        cfg        : Config                          — hyperparameters

    Returns:
        Q : dict { (int, int): float }
            QUBO matrix in dictionary form for openjij.SASampler.sample_qubo().
            Keys are flat pixel index pairs (0-based, row-major: index = row*n_x + col).
            Diagonal Q[(i,i)] = linear bias for pixel i.
            Off-diagonal Q[(i,j)] with i < j = pairwise interaction.

    Notes:
        numPixels = len(xVals) * len(yVals)
        Pixel ordering is row-major matching callRTI: row 0 = y_min, col 0 = x_min.
        score_vec has shape (numPairs,) where numPairs = numLinks / numChannels = 90.
        inversion has shape (numPixels, numPairs).
    """
    # ----------------------------------------------------------------
    # TODO: USER IMPLEMENTS QUBO FORMULATION HERE
    # ----------------------------------------------------------------
    raise NotImplementedError(
        "build_qubo() is not yet implemented. "
        "Fill in sa_localization/qubo.py with the QUBO formulation."
    )
