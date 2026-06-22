# RTI Codebase — Visual Summary

---

## 1. The Full Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                      LIVE HARDWARE PATH                         │
│                                                                 │
│  [Xandem Gateway]                                               │
│       │  SSH / serial — raw CSV, one row per (ch, rx) reading   │
│       ▼                                                         │
│  listenx.py  ──────────────────────────────────────────────►   │
│       │  stdout: 1 row per epoch (720 RSS ints + 1 timestamp)   │
│       ▼                                                         │
│  rti_stub.py ──► plots (Figure 2 RTI, Figure 3 VRTI)           │
│       │  writes coordinate estimates                            │
│       ▼                                                         │
│  basement/neals_estimate.txt                                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      REPLAY / OFFLINE PATH                      │
│                                                                 │
│  basement/basement_listenx_out_1.txt  (already listenx format)  │
│       │                                                         │
│       └──► python rti_stub.py <file>   ← PASS FILE DIRECTLY    │
│                 (avoids the stuck-at-EOF pipe problem)          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   MULTI-TARGET SYNTHESIS                        │
│                                                                 │
│  empty.txt ──┐                                                  │
│  target1.txt ├──► synth_multi_target.py ──► synth_out.txt       │
│  target2.txt ┘         (2^k rows, bitmask in col 721)           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   EVALUATION / VISUALIZATION                    │
│                                                                 │
│  neals_estimate.txt ──► calc_prmse.py ──► scatter plot + pRMSE  │
│  detections.txt     ──► calcDetectorPerformance.py ──► timeline │
│  any_listenx.txt    ──► histColumns.py ──► RSS histogram        │
│  listenx.py pipe    ──► plotColumnsx.py ──► live RSS chart      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Row Format (721 columns)

```
┌──────┬──────┬──────┬─────┬──────────────────────────────────────┬──────────┐
│ Ch0  │ Ch0  │ Ch0  │ ... │              Ch7                      │          │
│ Tx1  │ Tx1  │ Tx2  │ ... │  Tx10                                 │          │
│ Rx2  │ Rx3  │ Rx1  │ ... │  Rx9                                  │  col 720 │
├──────┴──────┴──────┴─────┴──────────────────────────────────────┤          │
│         720 RSS values (dBm)    sentinel = 127 (not measured)   │timestamp │
│         layout: channel × tx × rx,  skipping tx==rx             │   (ms)   │
└─────────────────────────────────────────────────────────────────┴──────────┘

  Total links = 8 channels × 10 nodes × 9 peers = 720
  In synth_multi_target.py output: col 720 = bitmask (NOT a timestamp)
```

---

## 3. The Three Sentinel Values

```
  127   ←── "link not measured this epoch"  (in RSS arrays, from hardware)
  > -10 ←── "treat as missing"             (rti_stub.py fill-forward check)
  -99   ←── "no person detected/present"   (in output estimate file)

  These are NOT interchangeable. 127 is input; -99 is output.
```

---

## 4. Module Dependency Map

```
                   ┌──────────┐
                   │  rss.py  │  linkNumForTxRxChLists
                   │          │  beeping, floor/ceil_multiple_of
                   └────┬─────┘
                        │ imported by
          ┌─────────────┼───────────────────┐
          ▼             ▼                   ▼
    listenx.py   histColumns.py      plotColumnsx.py

                   ┌──────────┐
                   │  rti.py  │  FixedLenBuffer, initRTI, callRTI
                   │          │  imageMaxCoord, sumTopRows
                   │          │  calcActualPosition, prmse
                   │          │  synthMultiTargetRSS, plotImage
                   └────┬─────┘
                        │ imported by
          ┌─────────────┼─────────────────┐
          ▼             ▼                 ▼
    rti_stub.py   calc_prmse.py   synth_multi_target.py

   ┌───────────┐   ┌────────┐
   │ buffer.py │   │ hmm.py │   ← standalone, NOT wired into pipeline
   └───────────┘   └────────┘
```

---

## 5. `initRTI` — What It Computes (One-Time Startup)

```
  INPUT: sensorCoords (10×2), delta_p, sigmax2, delta, excessPathLen
           │
           ▼
  ┌─────────────────────────────────────────────────┐
  │ 1. Build pixel grid                             │
  │    xVals = [0.0, 0.2, 0.4, ..., Xmax]          │
  │    yVals = [0.0, 0.2, 0.4, ..., Ymax]          │
  │    pixelCoords = all (x,y) pairs  → shape (P,2) │
  └──────────────────────┬──────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────┐
  │ 2. Pixel covariance (spatial smoothness prior)  │
  │                                                 │
  │    C_x[i,j] = sigmax2 × exp(-dist(i,j)/delta)  │
  │    CovPixelsInv = inv(C_x)   shape (P×P)        │
  └──────────────────────┬──────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────┐
  │ 3. Weight matrix W  (links × pixels)            │
  │                                                 │
  │    For each link (Tx→Rx):                       │
  │      Draw Fresnel ellipse around Tx–Rx line     │
  │      Pixels inside ellipse get weight 1/count   │
  │      Pixels outside get 0                       │
  │                                                 │
  │    Fresnel condition:                           │
  │      d(pixel,Tx) + d(pixel,Rx) - d(Tx,Rx)      │
  │                              < excessPathLen    │
  └──────────────────────┬──────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────┐
  │ 4. MAP projection matrix (pixels × links)       │
  │                                                 │
  │    inversion = inv(W^T W + C_x^{-1}) × W^T     │
  │                                                 │
  │    Stored once. Used every frame in callRTI.    │
  └─────────────────────────────────────────────────┘

  OUTPUT: inversion (P×numPairs), xVals, yVals
```

---

## 6. Per-Frame Imaging Loop (`rti_stub.py`)

```
  Each new row from listenx:
  ───────────────────────────────────────────────────────────────

  rss (720,)
    │
    ├── fill-forward missing values (127 or > -10 → use prevRSS)
    │
    ├── push each link into its FixedLenBuffer (for VRTI variance)
    │
    ├── [CALIBRATION: frames 0–49]
    │     accumulate sumRSS, countCalLines per link
    │     write "-99 -99" to output
    │
    ├── [END OF CAL: frame == 50]
    │     meanRSS = sumRSS / count         shape (8 × 90)
    │     maxInds = argsort channels       ← FIXED for all future frames
    │     calVec  = sumTopRows(meanRSS, maxInds, topChs=3)  shape (90,)
    │
    └── [IMAGING: frames > 50]
          │
          ├── RTI (shadowing)
          │     rss reshaped → (8 × 90)
          │     curVec  = sumTopRows(rss, maxInds, 3)
          │     scoreVec = calVec - curVec        (attenuation)
          │     image    = callRTI(scoreVec, inversion, ...)
          │     RTICoord = imageMaxCoord(image, xVals, yVals)
          │     → write to file if image.max() > 2.1
          │       else write "-99 -99"
          │
          └── VRTI (variance)
                varVec = [buff[i].var() for i in links]  shape (720,)
                varVec reshaped → (8 × 90)
                scoreVec = sumTopRows(varVec, maxInds, 3)
                image    = callRTI(scoreVec, inversion, ...)
                VRTICoord = imageMaxCoord(image, xVals, yVals)
                → plotted only (not written to file)
```

---

## 7. Channel Diversity (Why `sumTopRows` Exists)

```
  720 links = 8 channels × 90 (Tx,Rx) pairs

  During calibration, for each of the 90 pairs:
    rank the 8 channels by their mean RSS (highest = strongest signal)

  Each frame, instead of using all 8 channels:
    sum only the top 3 (topChs) channels for each pair

  This gives a 90-element vector instead of 720 — less noise,
  because weak/noisy channels are excluded.

         Ch0  Ch1  Ch2  Ch3  Ch4  Ch5  Ch6  Ch7
  Pair1: -55  -62  -48  -71  -50  -68  -45  -60    rank: 6,2,4,0,5,1,3,7
  Pair2: -40  -70  -55  -65  -42  -72  -38  -61    rank: 6,4,0,2,5,1,3,7
  ...

  sumTopRows picks the top 3 (highest avg RSS) and sums them → 1 value per pair
```

---

## 8. `FixedLenBuffer` — How the Circular Buffer Works

```
  buffL = 4, initial data = [0, 0, 0, 0], frontInd = 0

  After append(-45):   data=[-45,  0,  0,  0]  frontInd=1
  After append(-50):   data=[-45,-50,  0,  0]  frontInd=2
  After append(-48):   data=[-45,-50,-48,  0]  frontInd=3
  After append(-52):   data=[-45,-50,-48,-52]  frontInd=0  ← wraps!
  After append(-47):   data=[-47,-50,-48,-52]  frontInd=1  ← overwrites oldest

  list()      → oldest-first: [-48, -52, -47, -50]
  mostRecent()→ -47
  var()       → np.var([-47,-50,-48,-52]) = variance for VRTI

  One buffer exists per link (720 total).
```

---

## 9. Multi-Target Synthesis (`synthMultiTargetRSS`)

```
  Given: emptyRSS E (720,)  and  k target snapshots I_0 … I_{k-1}

  For target j, the "delta" (attenuation contribution) is:
    Δ_j[link] = I_j[link] - E[link]   if both are valid
    Δ_j[link] = 0                     if either is a sentinel (127)

  For each of 2^k subsets S (encoded as a bitmask):
    synth_S = E + Σ_{j ∈ S} Δ_j
    clip to [-127, -1]
    restore 127 on links where E was invalid

  k=2 example:
  ┌──────┬──────────┬───────────────────────────────┐
  │ mask │ binary   │ meaning                       │
  ├──────┼──────────┼───────────────────────────────┤
  │  0   │  00      │ empty room (just E)           │
  │  1   │  01      │ target 0 only                 │
  │  2   │  10      │ target 1 only                 │
  │  3   │  11      │ both targets present           │
  └──────┴──────────┴───────────────────────────────┘
  → 4 rows written to output file, col 721 = bitmask
```

---

## 10. Graphical Tools At a Glance

```
  ┌────────────────────────────┬────────┬────────┬────────────────────────────────────┐
  │ Script                     │ Live?  │ Saves? │ What you see                       │
  ├────────────────────────────┼────────┼────────┼────────────────────────────────────┤
  │ rti_stub.py                │  Yes   │  No    │ Fig 2: RTI image (shadowing)       │
  │                            │        │        │ Fig 3: VRTI image (variance)       │
  │                            │        │        │ Sensor dots + X for ground truth   │
  ├────────────────────────────┼────────┼────────┼────────────────────────────────────┤
  │ plotColumnsx.py            │  Yes   │  No    │ Scrolling RSS dBm vs time (sec)    │
  │                            │        │        │ One line per chosen link           │
  ├────────────────────────────┼────────┼────────┼────────────────────────────────────┤
  │ histColumns.py             │ Blocks │  No    │ Stacked PMF histograms             │
  │                            │        │        │ One subplot per file               │
  ├────────────────────────────┼────────┼────────┼────────────────────────────────────┤
  │ calc_prmse.py              │  No    │  PNG   │ Room map: X=truth, o=estimate      │
  │                            │        │        │ Red lines = error, prints pRMSE    │
  ├────────────────────────────┼────────┼────────┼────────────────────────────────────┤
  │ calcDetectorPerformance.py │  No    │  PNG   │ Detection timeline, cyan windows,  │
  │                            │        │        │ green=correct, red x=false alarm   │
  └────────────────────────────┴────────┴────────┴────────────────────────────────────┘

  ⚠️  calcDetectorPerformance.py uses matplotlib.use('Agg') — never import it in
      the same session as any interactive plot script or plots will go invisible.
```

---

## 11. How to Run Each Tool

```bash
# ── THE MAIN PIPELINE ──────────────────────────────────────────────

# Live from hardware
ssh root@xandem-gateway.local "/opt/xandev/exec/gateway/bin/gateway -l" \
    | python listenx.py | python rti_stub.py

# Replay — CORRECT WAY (avoids stuck-at-EOF pipe issue)
python rti_stub.py basement/basement_listenx_out_1.txt

# Replay — pipe way (will loop forever at end; Ctrl+C to stop)
python listenx.py -i basement/basement_listenx_out_1.txt | python rti_stub.py


# ── LIVE RSS CHART ─────────────────────────────────────────────────

# Watch link [tx=1, rx=2, ch=0] in real time
python listenx.py -i basement/basement_listenx_out_1.txt \
    | python plotColumnsx.py -n 10 -l "[1,2,0]"

# Watch multiple links
python listenx.py -i basement/basement_listenx_out_1.txt \
    | python plotColumnsx.py -n 10 -l "[1,2,0]" -l "[3,5,3]"


# ── RSS HISTOGRAM ──────────────────────────────────────────────────

# Compare empty vs occupied on one link
python histColumns.py \
    -f basement/basement_listenx_out_1.txt \
    -f basement/basement_listenx_out_2.txt \
    -l "[1,2,0]"


# ── MULTI-TARGET SYNTHESIS ─────────────────────────────────────────

# Mode A: pick specific rows from one file
python synth_multi_target.py \
    -e basement/basement_listenx_out_1.txt \
    -f basement/basement_listenx_out_1.txt \
    -t 100 -t 200 -o synth_out.txt

# Mode B: one file per target (uses post-cal mean)
python synth_multi_target.py \
    -e basement/basement_listenx_out_1.txt \
    -i basement/basement_listenx_out_1.txt \
    -i basement/basement_listenx_out_2.txt \
    -o synth_out.txt


# ── EVALUATION ─────────────────────────────────────────────────────

# Localization accuracy (run rti_stub.py first)
python calc_prmse.py basement/neals_estimate.txt

# Crossing detector performance
python calcDetectorPerformance.py \
    -c detections.txt \
    -r basement/basement_listenx_out_1.txt \
    -t true_crossings.txt
```

---

## 12. Key Parameters Cheat Sheet

```
  In rti_stub.py (lines 93–108):

  ┌───────────────────────┬─────────┬──────────────────────────────────────────┐
  │ Parameter             │ Default │ What changing it does                    │
  ├───────────────────────┼─────────┼──────────────────────────────────────────┤
  │ calLines              │ 50      │ More → stabler baseline, longer wait     │
  │ buffL                 │ 4       │ More → smoother VRTI, more lag           │
  │ topChs                │ 3       │ More channels → noisier but more data    │
  │ delta_p               │ 0.2 m   │ Smaller → finer image, much slower init  │
  │ sigmax2               │ 0.5     │ Higher → brighter images overall         │
  │ delta                 │ 1.0 m   │ Higher → smoother/blurrier images        │
  │ excessPathLen         │ 0.1 m   │ Higher → thicker ellipses, more pixels   │
  │ personInAreaThreshold │ 2.1     │ Lower → more detections, more FAs        │
  │ plotSkip              │ 2       │ Lower → faster plot refresh, more lag    │
  └───────────────────────┴─────────┴──────────────────────────────────────────┘
```

---

## 13. File Map

```
  rti-multitarget/
  ├── rti.py                      ← math core (import by everything)
  ├── rss.py                      ← I/O helpers (import by listenx, hist, plot)
  ├── listenx.py                  ← stage 1: parse gateway → 721-col rows
  ├── rti_stub.py                 ← stage 2: calibrate → image → locate → plot
  ├── synth_multi_target.py       ← offline: synthesize multi-person data
  ├── calc_prmse.py               ← offline: evaluate localization accuracy
  ├── calcDetectorPerformance.py  ← offline: evaluate crossing detector
  ├── histColumns.py              ← offline: RSS histogram by link
  ├── plotColumnsx.py             ← offline: live RSS time-series
  ├── buffer.py                   ← FixedMemoryBuffer (unused by pipeline)
  ├── hmm.py                      ← HMM (unused by pipeline)
  ├── basement/
  │   ├── basement_listenx_out_1.txt   ← 642 rows, single-person recording A
  │   ├── basement_listenx_out_2.txt   ← single-person recording B
  │   ├── sensor_coords_basement_m.txt ← 10×2 node positions (meters)
  │   ├── pivot_coords_basement_m.txt  ← waypoint coordinates
  │   ├── path_basement_1_f.txt        ← ordered pivot indices for path 1
  │   └── neals_estimate.txt           ← output of rti_stub.py
  ├── CODEBASE_WALKTHROUGH.md     ← full function-by-function guide
  └── VISUAL_SUMMARY.md           ← this file
```
