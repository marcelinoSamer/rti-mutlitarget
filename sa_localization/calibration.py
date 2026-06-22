import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import rti

MISSING_THRESH = -10


class CalibrationState:
    """Accumulates calibration frames, then provides per-epoch attenuation score."""

    def __init__(self, num_links, num_channels, top_chs):
        self.num_links  = num_links
        self.channels   = num_channels
        self.top_chs    = top_chs
        self.pairs      = num_links // num_channels   # 90 for 720-link, 8-ch setup

        self._sum_rss = np.zeros(num_links, dtype=float)
        self._count   = np.zeros(num_links, dtype=float)
        self.cal_vec  = None   # (pairs,) set after finalize()
        self.max_inds = None   # (pairs, channels) channel ranking
        self._buffers = [rti.FixedLenBuffer([0.0] * 4) for _ in range(num_links)]
        self.done     = False

    def update(self, rss_vec):
        """Feed one calibration frame (np.ndarray (numLinks,) int)."""
        for l in range(self.num_links):
            self._buffers[l].append(float(rss_vec[l]))
            if rss_vec[l] <= MISSING_THRESH:
                self._sum_rss[l] += rss_vec[l]
                self._count[l]   += 1.0

    def finalize(self):
        """Compute calVec from accumulated frames. Call after last calibration frame."""
        mean_rss = np.where(self._count > 0,
                            self._sum_rss / np.maximum(self._count, 1.0),
                            float(127))
        mean_rss_2d   = mean_rss.reshape(self.channels, self.pairs)
        self.max_inds = mean_rss_2d.transpose().argsort()
        self.cal_vec  = rti.sumTopRows(mean_rss_2d, self.max_inds, self.top_chs)
        self.done     = True

    def score_vec(self, rss_vec):
        """Return attenuation score vector for one post-calibration epoch.

        Returns cal_vec - cur_vec, shape (pairs,).
        Positive values indicate attenuation relative to empty room.
        """
        if not self.done:
            raise RuntimeError("finalize() must be called before score_vec()")
        cur_rss_2d = rss_vec.astype(float).reshape(self.channels, self.pairs)
        cur_vec    = rti.sumTopRows(cur_rss_2d, self.max_inds, self.top_chs)
        return self.cal_vec - cur_vec

    def var_vec(self, rss_vec):
        """Return per-link variance score for VRTI. Updates internal buffers."""
        if not self.done:
            raise RuntimeError("finalize() must be called before var_vec()")
        for l in range(self.num_links):
            self._buffers[l].append(float(rss_vec[l]))
        return np.array([self._buffers[l].var() for l in range(self.num_links)])
