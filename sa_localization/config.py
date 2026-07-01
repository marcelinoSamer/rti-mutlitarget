import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dataclasses import dataclass


@dataclass
class Config:
    # --- RTI geometry (same defaults as rti_stub.py) ---
    delta_p:          float = 0.2
    sigmax2:          float = 0.5
    delta:            float = 1.0
    excessPathLen:    float = 0.1

    # --- calibration ---
    calLines:         int   = 50
    top_chs:          int   = 3

    # --- OpenJij SA ---
    num_reads:        int   = 100
    num_sweeps:       int   = 1000
    beta_min:         float = 0.1
    beta_max:         float = 10.0

    # --- QUBO balance ---
    # alpha in [0, 1] blends linear and quadratic terms:
    #   diagonal   Q[i,i] = -alpha         * var(row_i)
    #   quadratic  Q[i,j] = (1 - alpha)    * cosine_sim(row_i, row_j)
    # alpha = 1 → pure variance selection (no diversity penalty)
    # alpha = 0 → pure diversity selection (no detectiveness term)
    alpha:            float = 0.5

    # --- detection threshold (mirrors rti_stub.py) ---
    personInAreaThreshold: float = 2.1

    # --- I/O paths ---
    synth_file:       str = 'out.txt'
    truth_file:       str = 'target_truth.txt'   # real ground-truth target positions
    coord_file:       str = 'basement/sensor_coords_basement_m.txt'
    pivot_file:       str = 'basement/pivot_coords_basement_m.txt'
    path_file:        str = 'basement/path_basement_1_f.txt'
    output_file:      str = 'sa_localization/sa_estimates.txt'

    # --- ground truth (mirrors rti_stub.py) ---
    actual_known:     bool  = True
    startPathTime:    float = 56000.0
    speed:            float = 1.0 / 8000.0
