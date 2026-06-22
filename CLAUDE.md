# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a **Radio Tomographic Imaging (RTI)** research codebase. RTI is a device-free localization technique: a network of low-power radio nodes measures how a human body attenuates radio links, then back-projects those attenuations onto a 2D pixel grid to locate the person. All code is **Python 2** (print statements, `raw_input()`, `range()` returning lists, integer division). Porting to Python 3 requires replacing `/` with `//` in `rti.py` lines 86, 98, 100 — these divisions are silently wrong if you use true division.

## Running the pipeline

```bash
# Full live pipeline (real hardware):
ssh root@xandem-gateway.local "/opt/xandev/exec/gateway/bin/gateway -l" \
    | python listenx.py \
    | python rti_stub.py

# Replay from recorded data:
python listenx.py -i basement/basement_listenx_out_1.txt | python rti_stub.py

# Multi-target synthesis (Mode A — snapshot rows):
python synth_multi_target.py \
    -e basement/basement_listenx_out_1.txt \
    -f basement/basement_listenx_out_2.txt \
    -t 100 -t 200 \
    -o synth_out.txt

# Multi-target synthesis (Mode B — per-target files):
python synth_multi_target.py -e empty.txt -i target_a.txt -i target_b.txt -o synth_out.txt

# Evaluate localization accuracy:
python calc_prmse.py estimates.txt

# Evaluate binary crossing detector:
python calcDetectorPerformance.py -c detections.txt -r listenx_out.txt -t true_crossings.txt

# Visualize RSS histograms across conditions:
python histColumns.py -n 10 -c 8 -f empty.txt -f occupied.txt -l "[1,2,0]"

# Live RSS time-series plot:
python listenx.py -i data.txt | python plotColumnsx.py -n 10 -l "[1,2,0]"
```

## Architecture and data flow

```
listenx.py → rti_stub.py → estimates.txt → calc_prmse.py / calcDetectorPerformance.py
```

- `listenx.py`: parses Xandem gateway serial output → one 721-column row per epoch (720 RSS values + timestamp ms). Missing links = `127`.
- `rti_stub.py`: the main imaging loop. Consumes listenx rows, calibrates for 50 frames (empty room assumed), then runs both RTI (shadowing) and VRTI (variance) imaging each frame.
- `rti.py`: the mathematical core. Imported by everything. Key functions:
  - `initRTI(...)` — precomputes the regularized linear inverse (MAP estimator). Called once at startup.
  - `callRTI(linkMeas, inversion, ...)` — one matrix-vector product per frame; returns the image.
  - `imageMaxCoord(image, xVals, yVals)` — picks the brightest pixel as the location estimate.
  - `FixedLenBuffer` — circular FIFO initialized full; each link has one for VRTI variance.
  - `calcActualPosition(t_ms, ...)` — interpolates ground-truth walk position at time t.
  - `prmse(...)` — penalized RMSE (penalty=16 for presence/absence mismatches).
  - `synthMultiTargetRSS(emptyRSS, targetRSSList)` — linear superposition synthesis; returns `2^k` `(bitmask, synth_array)` tuples.
- `rss.py`: I/O glue (serial port detection, beeping, link-index helpers). Contains **duplicates** of `linkNumForTxRxChLists` and `txRxChForLinkNum` from `rti.py` — if you change the link layout, edit both files.
- `buffer.py`: defines `FixedMemoryBuffer`, a more capable FIFO (supports `pop()`, `empty()`, `isFull()`). **Not used by the main pipeline** — `rti_stub.py` uses `rti.FixedLenBuffer`.
- `hmm.py`: standalone NumPy HMM (Rabiner 1989): forward, backward, Viterbi, forward-backward smoother. Not wired into the imaging pipeline; a building block for future state-inference work.

## Data format

**listenx files** (e.g. `basement/basement_listenx_out_1.txt`): 721 columns.  
- Columns 1–720: int dBm RSS, layout `ch[0..7] × Tx[0..9] × Rx[0..8]` (skipping `Tx == Rx`) = 8 × 90 = 720.
- Column 721: timestamp in ms from recording start.
- `127` = no measurement; `> -10` is also treated as missing by `rti_stub.py`.

**synth_multi_target.py output**: same 721-column format, but column 721 is a **bitmask** (not a timestamp). Don't feed to scripts that rely on timing from column 721.

**Legacy files** (`empty_area.txt`, `testfile2.txt`): 361 columns (4-channel kit, pre-dates the 8-channel setup). Incompatible with `rti_stub.py`.

## Key constants and tunables

All configured at the top of `rti_stub.py` (lines 93–108):

| Param | Default | Meaning |
|---|---|---|
| `calLines` | 50 | Calibration frames; **must be person-free** |
| `buffL` | 4 | Per-link rolling window for VRTI variance |
| `topChs` | 3 | Top channels summed per Tx/Rx pair (Kaltiokallio 2012) |
| `delta_p` | 0.2 m | Pixel spacing |
| `sigmax2` | 0.5 | Prior variance for pixel values |
| `delta` | 1.0 m | Spatial correlation length |
| `excessPathLen` | 0.1 m | Fresnel ellipse half-thickness |
| `personInAreaThreshold` | 2.1 | Image-max threshold for "person present" |
| `startPathTime` | 56000 ms | When subject hits first waypoint |
| `speed` | 1/8000 pivot/ms | Ground-truth walk speed (one pivot per 8 s) |

## Known gotchas

- **`rss.serialFileName()` has a `glob` import bug**: uses `glob.glob(...)` without `import glob`. Add `import glob` to `rss.py` before calling it.
- **Two sentinel values**: `127` (listenx missing), `> -10` (rti_stub missing threshold), `-99` (output "no person"). Not interchangeable.
- **Channel-diversity sort is computed once at calibration** (`maxInds = meanRSS.transpose().argsort()`). If channel quality shifts mid-recording, the top-3 selection goes stale.
- **`synthMultiTargetRSS` uses linear superposition** — an approximation. Real multi-person attenuation has body-occlusion non-linearities, especially when two targets share a Fresnel zone.
- **`calcDetectorPerformance.py` forces `matplotlib.use('Agg')`**. Don't import it in the same session as interactive plotting scripts.
- **`numNodes`, `numChs`, and column offsets in `listenx.py` are hardcoded** for the 10-node, 8-channel Xandem HOME kit. Different hardware requires editing those constants.
