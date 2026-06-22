import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import rti

NO_PERSON_KEY = -99.0
PENALTY       = 16.0


class RunningPRMSE:
    """Accumulates squared errors epoch by epoch; call result() for final PRMSE."""

    def __init__(self):
        self._sum_sq = 0.0
        self._count  = 0

    def update(self, actual_coord, est_coord):
        """
        actual_coord : np.ndarray (2,) or [] (empty = person not in area)
        est_coord    : (x, y) tuple
        """
        actual = (np.array(actual_coord).reshape(-1, 2) if len(actual_coord) > 0
                  else np.array([[NO_PERSON_KEY, NO_PERSON_KEY]]))
        est    = np.array([[est_coord[0], est_coord[1]]])

        err = rti.prmse(actual, est, NO_PERSON_KEY, PENALTY)
        self._sum_sq += err ** 2
        self._count  += 1

    def result(self):
        """Return penalized RMSE over all accumulated epochs."""
        if self._count == 0:
            return float('nan')
        return float(np.sqrt(self._sum_sq / self._count))
