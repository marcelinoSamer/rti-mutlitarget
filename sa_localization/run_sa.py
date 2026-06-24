#! /usr/bin/env python3
"""
SA-based multi-target RTI localization.

Usage:
    python3 listenx.py -i basement/basement_listenx_out_1.txt | python3 sa_localization/run_sa.py
    python3 sa_localization/run_sa.py -i basement/basement_listenx_out_1.txt
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import getopt
import rti

from sa_localization.config      import Config
from sa_localization.data_io     import (load_sensor_coords, load_pivot_coords,
                                         load_path_indices, write_estimates)
from sa_localization.calibration import CalibrationState
from sa_localization.qubo        import build_qubo
from sa_localization.solver      import run_sa
from sa_localization.decode      import decode_row_solution
from sa_localization.evaluate    import RunningPRMSE

# ---- CLI -----------------------------------------------------------------
cfg        = Config()
input_file = None
fin        = sys.stdin

myopts, _ = getopt.getopt(sys.argv[1:], 'i:')
for o, a in myopts:
    if o == '-i':
        input_file = a
        fin = open(a, 'r')

# ---- Geometry ------------------------------------------------------------
sensor_coords = load_sensor_coords(cfg.coord_file)
pivot_coords  = load_pivot_coords(cfg.pivot_file)
path_ind      = load_path_indices(cfg.path_file)

num_channels = 8
num_links    = len(sensor_coords) * (len(sensor_coords) - 1) * num_channels  # 720

inversion, xVals, yVals = rti.initRTI(
    sensor_coords, cfg.delta_p, cfg.sigmax2, cfg.delta, cfg.excessPathLen
)

# ---- State ---------------------------------------------------------------
cal      = CalibrationState(num_links, num_channels, cfg.top_chs)
acc_prmse = RunningPRMSE()
fout     = open(cfg.output_file, 'w')

frame    = 0
prev_rss = np.full(num_links, 127, dtype=int)

# ---- Main loop -----------------------------------------------------------
for line in fin:
    line = line.strip()
    if not line:
        continue

    vals    = [int(x) for x in line.split()]
    time_ms = vals.pop()
    rss     = np.array(vals, dtype=int)

    # prevRSS fill
    missing         = rss > -10
    rss[missing]    = prev_rss[missing]
    prev_rss        = rss.copy()

    if frame < cfg.calLines:
        cal.update(rss)
        write_estimates(fout, -99.0, -99.0)
        if frame == cfg.calLines - 1:
            cal.finalize()
            print("Calibration complete.")
        frame += 1
        continue

    # ---- Post-calibration ------------------------------------------------
    score = cal.score_vec(rss)

    try:
        Q                   = build_qubo(score, cal, inversion, xVals, yVals, cfg)
        best_sample, energy = run_sa(Q, cfg)
        estimates           = decode_row_solution(best_sample, cal, inversion, xVals, yVals, cfg)
    except NotImplementedError:
        image     = rti.callRTI(score, inversion, len(xVals), len(yVals))
        energy    = float('nan')
        if image.max() > cfg.personInAreaThreshold:
            x, y      = rti.imageMaxCoord(image, xVals, yVals)
            estimates = [(x, y)]
        else:
            estimates = []

    if estimates:
        x_est, y_est = estimates[0]
        write_estimates(fout, x_est, y_est)
    else:
        x_est, y_est = -99.0, -99.0
        write_estimates(fout, x_est, y_est)

    if cfg.actual_known:
        actual = rti.calcActualPosition(
            time_ms, pivot_coords, path_ind, cfg.startPathTime, cfg.speed
        )
        acc_prmse.update(actual, (x_est, y_est))

    print("Frame {:d} | energy: {:.3f} | est: ({:.2f}, {:.2f})".format(
        frame, energy, x_est, y_est))
    frame += 1

fout.close()
if input_file:
    fin.close()

print("Done. PRMSE = {:.4f} m".format(acc_prmse.result()))
