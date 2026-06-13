# RTI Repository — Research Walkthrough

A complete, file-by-file explanation of this Radio Tomographic Imaging (RTI) codebase. Intended for researchers who need to understand the system thoroughly before extending or modifying it.

---

## Table of Contents

1. [Background — What is RTI?](#1-background--what-is-rti)
2. [System Architecture & Data Flow](#2-system-architecture--data-flow)
3. [Core Library Files](#3-core-library-files)
   - [`rti.py`](#31-rtipy--the-core-library)
   - [`rss.py`](#32-rsspy--utilities-for-link-indexing--io)
   - [`buffer.py`](#33-bufferpy--an-alternative-fifo-buffer)
4. [Data Ingestion](#4-data-ingestion)
   - [`listenx.py`](#41-listenxpy--gateway--standardized-rss-rows)
5. [Main RTI Pipeline](#5-main-rti-pipeline)
   - [`rti_stub.py`](#51-rti_stubpy--rti--vrti-imaging-loop)
6. [Detection & State Inference](#6-detection--state-inference)
   - [`hmm.py`](#61-hmmpy--hidden-markov-model-toolkit)
   - [`crossDetectionStub.py`](#62-crossdetectionstubpy--crossing-detector-stub)
7. [Performance Evaluation](#7-performance-evaluation)
   - [`calcDetectorPerformance.py`](#71-calcdetectorperformancepy--detection-rate--false-alarm-scoring)
   - [`calc_prmse.py`](#72-calc_prmsepy--penalized-rmse-for-coordinate-estimates)
8. [Visualization Tools](#8-visualization-tools)
   - [`histColumns.py`](#81-histcolumnspy--rss-histograms)
   - [`plotColumnsx.py`](#82-plotcolumnsxpy--real-time-rss-time-series)
9. [Multi-Target Synthesis (New)](#9-multi-target-synthesis-new)
   - [`synth_multi_target.py`](#91-synth_multi_targetpy--multi-target-rss-data-augmentation)
10. [Data Files](#10-data-files)
11. [Repository Metadata](#11-repository-metadata)
12. [Cross-Reference: Function → File Map](#12-cross-reference-function--file-map)
13. [Gotchas, Quirks & Research Notes](#13-gotchas-quirks--research-notes)

---

## 1. Background — What is RTI?

**Radio Tomographic Imaging** (RTI) is a device-free localization technique introduced by Joey Wilson and Neal Patwari (~2010). The idea:

- Deploy `N` low-power radio nodes (here: the **Xandem HOME kit** with 10 nodes, 8 channels each) around a monitored area.
- Each node broadcasts; every other node measures the **Received Signal Strength (RSS)** of those broadcasts on each channel.
- A human body in the area **attenuates** the radio signals on links whose Tx–Rx line of sight passes near the body.
- By back-projecting the attenuation of every link onto a 2D pixel grid (via a regularized linear inversion), an "image" is produced whose peak indicates the person's location.

Two RTI variants live side-by-side in this repo:

| Variant | What it images | Strength |
|---|---|---|
| **Shadowing RTI** (a.k.a. attenuation-based, "RTI") | `E - I`: deviation of current RSS from the empty-room baseline | Best when targets stand still and attenuate links |
| **Variance RTI** ("VRTI") | Per-link rolling RSS variance | Best when targets *move* (motion-induced fading) |

The seminal references baked into the code's docstrings:

- Wilson & Patwari, *See Through Walls: Motion Tracking Using Variance-Based Radio Tomography Networks*, IEEE Trans. Mobile Computing, 2011.
- Kaltiokallio, Bocca & Patwari, *Enhancing the accuracy of radio tomographic imaging using channel diversity*, MASS 2012 (multi-channel extension).

The system is **single-target** by design: the assumption is one person walking a known path.

---

## 2. System Architecture & Data Flow

```
┌────────────────────┐  USB serial   ┌────────────────────┐
│ Xandem gateway     │ ────────────► │   listenx.py       │
│ (raw packets)      │               │ (parser, beeper)   │
└────────────────────┘               └─────────┬──────────┘
                                               │ 720 RSS + ts (one row per epoch)
                                               ▼
                              ┌────────────────────────────────┐
                              │   rti_stub.py (main pipeline)  │
                              │   - calibration                │
                              │   - RTI image (shadowing)      │
                              │   - VRTI image (variance)      │
                              │   - max-of-image → coord       │
                              └────────┬────────┬──────────────┘
                                       │        │
                       coordinate ests │        │ live plots
                                       ▼        ▼
                     ┌──────────────────┐   matplotlib figures 2 & 3
                     │ output_estimates │
                     │   .txt           │
                     └────────┬─────────┘
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
        ┌──────────────┐          ┌──────────────────────────┐
        │ calc_prmse.py │          │ calcDetectorPerformance │
        │ (locate RMSE) │          │ (binary detector eval)  │
        └──────────────┘          └──────────────────────────┘
```

Auxiliary tools (`histColumns.py`, `plotColumnsx.py`, `crossDetectionStub.py`, `hmm.py`) plug into the same data stream for diagnostic/exploratory work. The new `synth_multi_target.py` operates on already-processed listenx files to synthesize multi-target combinations.

**Important:** all Python code is **Python 2** (uses `print` statements, integer division, `range()` lists, `raw_input()`).

---

## 3. Core Library Files

### 3.1 [`rti.py`](rti.py) — the core library

The mathematical heart of the system. Imported by virtually every other script as `import rti`.

#### `FixedLenBuffer` (lines 40–67)
A circular FIFO buffer initialized at full length. Every `append()` overwrites the oldest entry. Used in `rti_stub.py` to maintain a per-link rolling window for variance computation.

- `__init__(initlist)` — initialized full
- `append(newItem)` — O(1), overwrites front
- `mostRecent()`, `mostRecentN(N)` — read access
- `list()` — returns the buffer in oldest-to-newest order
- `var()` — variance of the window (note: also `mean`, `std` per the commit log d4697ec)

> ⚠️ A *separate, more capable* buffer class lives in [`buffer.py`](#33-bufferpy--an-alternative-fifo-buffer). They are not interchangeable.

#### Link-index conversions (lines 70–110)
A link is a `(Tx, Rx, Channel)` triple. Since RSS data is stored as a flat 720-element vector, the system needs bijections between linear link index and `(Tx, Rx, Ch)`:

- `linkNumForTxRxChLists(tx, rx, ch, nodeList, channelList)` — `(Tx, Rx, Ch) → linkNum`
- `txRxForLinkNum(linknum, nodes)` — `linkNum → (Tx, Rx)` (ignores channel)
- `txRxChForLinkNum(linknum, nodeList, channelList)` — full inverse

Layout (with `nodes=10`, `chs=8`): 720 = 8 channels × (10 × 9) Tx/Rx ordered pairs. The ordering is `ch_enum × 90 + tx_enum × 9 + rx_enum (adjusted to skip self-link)`.

⚠️ These appear *duplicated* in [`rss.py`](#32-rsspy--utilities-for-link-indexing--io). The two copies behave identically; `rti.py` exits on bad input via `sys.exit`, `rss.py` only warns via `stderr.write`.

#### `calcGridPixelCoords(personLL, personUR, delta_p)` (lines 112–122)
Builds a rectangular pixel grid covering the bounding box of the sensors. Pixels are spaced `delta_p` meters apart. Returns flat pixel coordinates `(pixels, 2)` plus 1-D `xVals`, `yVals` arrays for image reshaping.

#### `plotLocs(nodeLocs)` (lines 124–141)
Plots sensor positions as labeled dots on the current matplotlib axes.

#### `plotImage(image, figNumber, sensorCoords, imageExtent, vmaxval, units, time_ms=None, actualCoord=None)` (lines 145–160)
Renders an RTI image with `imshow`, overlays sensor positions and (if known) the ground-truth coordinate as an X. Called every `plotSkip` frames by the main loop.

#### `initRTI(nodeLocs, delta_p, sigmax2, delta, excessPathLen)` (lines 178–212) ⭐
**This is the heart of the system.** It precomputes the regularized linear inverse that maps a link-RSS-deviation vector to a pixel image:

1. Build the pixel grid via `calcGridPixelCoords`.
2. Compute distance matrices: pixel↔pixel, pixel↔node, node↔node.
3. Build the **pixel covariance prior** `Σ_x = σ²ₓ · exp(−D_pp/δ)` (exponential spatial correlation with correlation length `δ`), and invert it.
4. Build the **weight matrix `W` (links × pixels)**:
   - For each link, find pixels whose `dist(tx → pixel) + dist(pixel → rx) − dist(tx, rx) < excessPathLen`. These are pixels inside the **Fresnel-style ellipse** with foci at Tx and Rx.
   - Each in-ellipse pixel gets weight `1 / |pixelsInEllipse|`; others get 0.
5. Compute the projection matrix:
   ```
   inversion = (Wᵀ W + Σ_x⁻¹)⁻¹ Wᵀ        (pixels × links)
   ```
   This is the **regularized least-squares (MAP) solution** treating the prior as a Gaussian.

Returns `(inversion, xVals, yVals)`.

Key tunables (set in `rti_stub.py`):

| Param | Meaning | Typical |
|---|---|---|
| `delta_p` | pixel spacing | 0.2 m |
| `sigmax2` | prior variance of any pixel's image value | 0.5 |
| `delta` | spatial correlation length (m) | 1.0 |
| `excessPathLen` | half-thickness of the ellipse (m) | 0.1 |

#### `callRTI(linkMeas, inversion, xValsLen, yValsLen)` (lines 214–218)
Applies the precomputed `inversion` matrix to a link-measurement vector and reshapes to a 2D image. One matrix-vector product per frame — cheap.

#### `imageMaxCoord(imageMat, xVals, yVals)` (lines 220–223)
Returns the `(x, y)` of the brightest pixel — used as the location estimate.

#### `sumTopRows(data, maxInds, topChs)` (lines 230–237)
For multi-channel RTI: given an `(channels × pairs)` matrix and per-pair channel rankings (computed once during calibration via `argsort`), sums the top-`topChs` channels per pair. This is the **channel-diversity selection** from Kaltiokallio et al. 2012.

#### `calcActualPosition(t_ms, pivotCoords, pathInd, startPathTime, speed)` (lines 246–257)
Ground-truth interpolator. The experimental subject walks a fixed sequence of "pivot points" at constant `speed`. Given a timestamp, returns the linearly-interpolated `(x, y)`. Returns `[]` outside the active window.

- `speed = 1 / 8000` pivots/ms → 8 s between pivots.
- `startPathTime = 56 s` after recording start (a calibration buffer).

#### `prmse(actualCoord, estCoord, noPersonKey, penalty)` (lines 260–276)
**Penalized RMSE** — the headline accuracy metric.

- For matched present/absent states: squared Euclidean error.
- For false-presence or missed-presence: a fixed `penalty = 4²` per mismatch.
- Then sqrt of the mean.

`noPersonKey = -99` is the in-band sentinel for "no person detected."

#### `synthMultiTargetRSS(emptyRSS, targetRSSList)` (lines 280+, *new*)
Multi-target RSS synthesis by linear superposition. See [section 9.1](#91-synth_multi_targetpy--multi-target-rss-data-augmentation).

---

### 3.2 [`rss.py`](rss.py) — utilities for link indexing & I/O

A loose collection of helpers used by all the data-ingestion and plotting scripts. Despite the name, it contains no RSS-specific math — mostly system glue.

| Function | Purpose |
|---|---|
| `linkNumForTxRxChLists` | **Duplicate of the same function in `rti.py`** |
| `txRxChForLinkNum` | **Duplicate of the same function in `rti.py`** |
| `hex2signedint(he)` | Decode 2's-complement hex strings (legacy serial protocol) |
| `prevChannel(channelList, ch_now)` | Given a channel, return the one transmitted on previously (round-robin) |
| `serialFileName()` | Auto-detect the USB serial device path (Linux: `/dev/ttyACM*`, macOS: `/dev/tty.usb*`, Windows: hardcoded `COM3`) |
| `beeping(time_diff, beepRate, beepCounter, printOption)` | Emit BEL characters at a fixed rate to give the experimenter audio cues for pacing |
| `floor_multiple_of(m, v)` / `ceil_multiple_of(m, v)` | Round to nearest multiple-of-`m` — used to keep plot axes stable |

⚠️ **The duplication of link-index functions between `rti.py` and `rss.py` is a real maintenance hazard.** If you change the link layout, you must edit both files. The two copies have subtly different error handling (`sys.exit` vs `stderr.write`).

⚠️ `serialFileName()` references `glob` but the file does **not** `import glob` at the top — calling it on Linux/macOS will `NameError`. Defensive-imports never happen; you must add `import glob` if you actually use it.

---

### 3.3 [`buffer.py`](buffer.py) — an alternative FIFO buffer

Defines `FixedMemoryBuffer` — a more capable circular FIFO than the one in `rti.py`:

| Feature | `rti.py FixedLenBuffer` | `buffer.py FixedMemoryBuffer` |
|---|---|---|
| Starts full | ✅ | ❌ — supports an empty state |
| Variance helper | ✅ `.var()` | ❌ (none built in) |
| `pop()` | ❌ | ✅ |
| `empty()` flag/reset | ❌ | ✅ |
| `isFull()`, `numStored()` | ❌ | ✅ |

> The two classes are *not interoperable* and the main pipeline (`rti_stub.py`) uses the simpler `FixedLenBuffer`. The richer `FixedMemoryBuffer` appears to be a refactor candidate that was never integrated. Consider it a reference implementation for future work.

---

## 4. Data Ingestion

### 4.1 [`listenx.py`](listenx.py) — gateway → standardized RSS rows

The data-ingestion script. Reads the Xandem HOME gateway output (either from stdin or from `-i filename`) and emits **one row per measurement epoch** to stdout.

**Input format:** comma-separated lines from the Xandem gateway, where each line carries one channel's worth of RSS for one receiver.

**Output format:** one row of `720` integers (8 channels × 10×9 ordered Tx/Rx pairs) + one integer timestamp (ms from start). Missing values are represented as `127`.

**Operation:**

1. Skip the first 24 lines (serial buffer junk).
2. Use the first valid line to anchor `time_start`.
3. For each subsequent line, parse:
   - `channelCol = 2`
   - `rxidCol = 4`
   - `firstRSSValueCol = 7` — start of the 10 RSS values for receiver `rxid_now` on channel `ch_now`
   - `timeCol = -1` — last column
4. Index each RSS into `currentLinkRSS` using `rss.linkNumForTxRxChLists(...)`.
5. After collecting `numLines = numNodes × numChs = 80` lines (one full round), emit the assembled `currentLinkRSS + time_diff_ms` and reset.
6. Optionally beep at `beepRate = 1.0 Hz` for experimenter pacing.

**Real-time mode:** can be piped from `ssh` into the gateway:

```bash
ssh root@xandem-gateway.local "/opt/xandev/exec/gateway/bin/gateway -l" \
    | python listenx.py \
    | python rti_stub.py
```

⚠️ **Hardcoded constants** that must match the kit:
- `numNodes = 10`, `numChs = 8`, `firstRSSValueCol = 7`, `startSkip = 24`.

⚠️ The script uses `currentLinkRSS = [127] * numLinks` as the missing-data sentinel, which `rti_stub.py` later interprets as "no measurement → fill with previous value."

---

## 5. Main RTI Pipeline

### 5.1 [`rti_stub.py`](rti_stub.py) — RTI + VRTI imaging loop

The main online pipeline. Consumes listenx output, computes both shadowing-RTI and VRTI in parallel, plots both images, and writes a stream of coordinate estimates.

**Configuration (lines 93–108):**

| Param | Meaning | Default |
|---|---|---|
| `plotSkip` | redraw every Nth frame | 2 |
| `startSkip` | discard first N input lines (serial junk) | 1 |
| `buffL` | per-link variance window length | 4 |
| `calLines` | calibration period (rows) | 50 |
| `topChs` | top channels to sum per pair (Kaltiokallio 2012) | 3 |
| `channels` | total channels per pair | 8 |
| `delta_p`, `sigmax2`, `delta`, `excessPathLen` | passed to `initRTI` | 0.2, 0.5, 1.0, 0.1 |
| `personInAreaThreshold` | image-max threshold for "person present" | 2.1 |
| `actualKnown` | enables RMSE computation against ground truth | True |

**Pipeline phases:**

1. **Setup (lines 113–142)** — load sensor coords (`basement/sensor_coords_basement_m.txt`), pivots, paths; precompute `inversion` matrix via `rti.initRTI`.
2. **Per-link rolling buffer (lines 170–174)** — one `FixedLenBuffer` per link, length `buffL`, for VRTI's variance computation.
3. **Calibration (lines 211–229):**
   - For `counter < calLines`: accumulate `sumRSS` and `countCalLines` per link, skipping sentinels (`rss[i] > -10`).
   - At `counter == calLines`: compute per-link mean RSS, reshape to `(channels, numPairs)`, **sort channels by mean RSS per pair** to get `maxInds`, then sum top-3 channels into `calVec` (the empty-room reference vector, length 90).
4. **Per-frame imaging (lines 235–283):**
   - Replace any sentinel RSS with `prevRSS[i]` (impute missing).
   - **Shadowing RTI:** subtract `curVec = sumTopRows(rss, maxInds, topChs)` from `calVec`. Project via `callRTI`. Take image max. If `image.max() > threshold`, write coordinate; else `-99 -99`.
   - **VRTI:** read each link buffer's `.var()`, sum top-channel variances, project, take max. (VRTI estimates are *not* written to the output file — only plotted.)
   - If `actualKnown` and the timestamp is inside the walk window, accumulate per-frame RTI and VRTI Euclidean errors.
5. **At EOF:** print VRTI and RTI overall RMSE.

⚠️ The "no person" sentinel in the output file is `-99 -99`. `prmse()` in `rti.py` uses `noPersonKey = -99` to detect this.

⚠️ Calibration assumes **no person is present** during the first `calLines = 50` rows. The default `basement_listenx_out_1.txt` is structured to satisfy this (the first 50 rows are 5 s of empty room).

⚠️ Two matplotlib figures (2 and 3) are updated in place. `plt.colorbar()` is only called once (at `counter == calLines`) — duplicating it produces multiple bars.

---

## 6. Detection & State Inference

### 6.1 [`hmm.py`](hmm.py) — Hidden Markov Model toolkit

A self-contained, **NumPy-only** HMM implementation following Rabiner's 1989 tutorial. Not currently wired into the imaging pipeline — provided as a building block for higher-level state inference (e.g., "person present / absent / crossing").

#### `nealsHMM` class

Constructor: `nealsHMM(B, A, pi, V)`

| Arg | Meaning |
|---|---|
| `A` | `(S × S)` state transition matrix; `A[i,j]` = P(state j next | state i now) |
| `B` | `(S × |V|)` observation probability matrix; `B[i,j]` = P(observation `V[j]` | state i) |
| `pi` | `(S,)` initial state prior |
| `V` | list of possible observation values (the alphabet) |

#### Algorithms implemented

| Method | Returns | Notes |
|---|---|---|
| `observe(newdata)` | — | Append new observations (must be in `V`) |
| `forward()` | `alpha[t]` for each t | Incremental — picks up where it left off |
| `backward(steps=None)` | `beta[t]` | Re-computed each call |
| `forwardBackward(steps)` | smoothed state posterior at `T − steps` | The classic smoother |
| `viterbi(steps)` | `qstar` MAP sequence | Incremental in `delta`, `phi`; `qstar` recomputed |

⚠️ Each `alpha[t]` and `beta[t]` is normalized to sum to 1 — these are the *normalized forward / backward variables*, not raw joint probabilities. This is the standard numerical-stability trick.

⚠️ `observe()` will raise `ValueError` (via `list.index`) on observations not in `V`.

---

### 6.2 [`crossDetectionStub.py`](crossDetectionStub.py) — crossing-detector stub

A scaffold for building a per-link variance-based **crossing detector** (e.g., "did someone walk past this doorway?"). It accepts a set of `(Tx, Rx, Ch)` link triples to monitor, maintains a `buffL`-deep buffer per link, and prints the variance vector each frame.

**CLI:**

```bash
python crossDetectionStub.py -n 10 -l "[1,2,0]" -l "[3,4,5]" [-f data.txt]
```

- `-n` — number of nodes (required)
- `-l` — repeatable `[tx, rx, ch]` triples to monitor
- `-f` — input file (defaults to stdin)

**What it does:** computes per-link variance and prints. **What it doesn't do:** threshold, output binary decisions, or feed `calcDetectorPerformance.py`. That last step is left as research work — likely by integrating `hmm.py` or a simple threshold + debouncer.

---

## 7. Performance Evaluation

### 7.1 [`calcDetectorPerformance.py`](calcDetectorPerformance.py) — detection rate & false alarm scoring

Evaluates a *binary* crossing-detector output against ground-truth crossing times.

**CLI:**

```bash
python calcDetectorPerformance.py \
    -c crossingEstimates.txt   # 1 = detected, 0 = not, one per RSS row
    -r listenx_output.txt       # provides timestamps
    -t trueCrossings.txt        # ground-truth crossing times in seconds
```

**Method:**

1. For each true crossing time `tc`, search for any detection within `tc ± delta` (`delta = 1500 ms`).
2. Count **correct detections** (any detection inside a ground-truth window).
3. Count **false alarms** (detections in gaps between windows, and after the last window).
4. Compute:
   - `correctDetectionRate = correctDetections / numTrueCrossings`
   - `falseAlarmRate = falseAlarms / totalRows`
   - `score = correctDetectionRate − falseAlarmRate`

**Plot output:** saved to `<crossingEstFile_basename>.png`. Shows the detection time-series, true-crossing windows (cyan), correctly-windowed detections (green underline), and false alarms (red ×s below the line). Uses the `Agg` backend so it runs headlessly.

---

### 7.2 [`calc_prmse.py`](calc_prmse.py) — penalized RMSE for coordinate estimates

Evaluates a coordinate-estimate file (produced by `rti_stub.py`) against the ground-truth path.

**CLI:** `python calc_prmse.py estimates.txt`

**Method:**

1. Load actual position via `rti.calcActualPosition` for each timestamp.
2. Because the algorithm may have **timing drift** (filtering delay etc.), try a range of offsets `−1000 ms … +1000 ms` in 250 ms steps.
3. For each offset, compute `rti.prmse(actualCoord, estCoord, noPersonKey=-99, penalty=16)`.
4. Report the **minimum** pRMSE and the best offset.

**Plot:** sensor positions + arrows from each actual `X` to each estimated `o`. Red lines mark large errors. Saved to `<estFileName>2.png`.

⚠️ Expects the ground-truth files at `basement/pivot_coords_basement_m.txt`, `basement/path_basement_1_f.txt`, and the RSS data at `basement/basement_listenx_out_1.txt` — hardcoded paths.

⚠️ Uses `raw_input()` at the end (Python 2 only) to keep the figure window open.

---

## 8. Visualization Tools

### 8.1 [`histColumns.py`](histColumns.py) — RSS histograms

Plots **probability mass functions** (normalized histograms) of RSS values for selected links across one or more data files. Useful for:

- Characterizing link quality (where does the bulk of the distribution sit?)
- Comparing **empty vs. occupied** conditions on the same link.
- Identifying broken or fading-dominated links.

**CLI:**

```bash
python histColumns.py -n 10 -c 8 \
    -f empty.txt -f occupied.txt \
    -l "[1,2,0]" -l "[3,4,5]"
```

Each input file gets its own subplot (stacked vertically); within each subplot, every monitored link gets its own curve. Bin centers run from `-110` to `-10` dBm.

---

### 8.2 [`plotColumnsx.py`](plotColumnsx.py) — real-time RSS time-series

A live scrolling plot of recent RSS values on selected links. Consumes listenx output from stdin.

**CLI:**

```bash
python listenx.py -i data.txt | python plotColumnsx.py -n 10 -l "[1,2,0]"
```

Keeps a `deque` of the last `buffL = 40` values per link, redraws every frame. The x-axis shows seconds-ago relative to the most recent timestamp. Y-axis auto-scales to multiples of 5 dBm via `rss.floor_multiple_of` / `rss.ceil_multiple_of` for stability.

---

## 9. Multi-Target Synthesis (New)

### 9.1 [`synth_multi_target.py`](synth_multi_target.py) — multi-target RSS data augmentation

A recently-added tool to **synthesize multi-target RSS measurements** from single-target recordings. The motivation: training/testing multi-target localizers without performing combinatorially many multi-person experiments.

**Algorithm (linear superposition):**

```
E       = empty-room RSS vector            (720 values)
I_j     = single-target RSS snapshot for target j  (720 values, one per location)
dI_j    = I_j - E
synth_S = E + Σ_{j ∈ S} dI_j   for each subset S ⊆ {0..k-1}
```

For `k` target locations, the script emits `2^k` rows — every combination of "which targets are present" — plus a bitmask label in column 721.

**Sentinel handling:**
- If `E[l]` is missing (`> -10`, including the `127` sentinel): force `synth[l] = 127` in all outputs (no baseline → no synthesis possible).
- If `I_j[l]` is missing: that target contributes zero delta on link `l`.
- Valid links: clipped to `[-127, -1]`.

**CLI (two modes):**

```bash
# Mode A — choose specific rows of a single data file as target snapshots:
python synth_multi_target.py \
    -e basement/basement_listenx_out_1.txt \
    -f basement/basement_listenx_out_2.txt \
    -t 100 -t 200 \
    -o synth_out.txt

# Mode B — one file per target, averaged over post-calibration rows:
python synth_multi_target.py \
    -e empty.txt -i target_location_A.txt -i target_location_B.txt -o synth_out.txt
```

| Flag | Meaning |
|---|---|
| `-e` | Empty-room listenx file (required) |
| `-f` | Listenx file containing target snapshots (Mode A) |
| `-t` | Repeatable; 0-based row index for a target snapshot (Mode A) |
| `-i` | Repeatable; per-target listenx file averaged after `-c` lines (Mode B) |
| `-c` | Calibration lines for the empty-room mean (default 50) |
| `-o` | Output file (defaults to stdout) |

**Output format:** identical to listenx output (721 columns), but column 721 is a **bitmask** integer instead of a timestamp. This makes the file directly consumable by any downstream script that ignores the last column.

**Key function** (lives in `rti.py`):

```python
def synthMultiTargetRSS(emptyRSS, targetRSSList):
    """Returns list of (bitmask, synth_array) tuples, length 2^k."""
```

⚠️ **Linearity assumption:** real-world multi-person attenuation is *not* purely linear — bodies can block each other (occlusion), and shadowing effects on overlapping links interact non-trivially. The synthesized data is an approximation good for first-cut multi-target work; verify against real multi-person recordings before drawing strong conclusions.

⚠️ The bitmask column 721 means downstream consumers that interpret column 721 as a timestamp (e.g., `calcDetectorPerformance.py`) will read garbage timing. Use this output with imagers that ignore the last column (most do).

---

## 10. Data Files

### 10.1 `basement/` — primary experimental dataset

A real RTI experiment recorded in a basement with 10 sensors and a known walking path.

| File | Format | Purpose |
|---|---|---|
| [`sensor_coords_basement_m.txt`](basement/sensor_coords_basement_m.txt) | 10 rows × 2 cols (`x y` in meters) | Sensor positions |
| [`pivot_coords_basement_m.txt`](basement/pivot_coords_basement_m.txt) | 21 rows × 2 cols (`x y` in m) | Waypoints the subject walks through |
| [`path_basement_1_f.txt`](basement/path_basement_1_f.txt) | 41 rows × 1 col (int) | Ordered list of pivot indices for walk 1 |
| [`path_basement_2_f.txt`](basement/path_basement_2_f.txt) | 41 rows × 1 col (int) | Walk 2's pivot sequence |
| [`basement_listenx_out_1.txt`](basement/basement_listenx_out_1.txt) | 642 rows × 721 cols | listenx output: walk 1's RSS |
| [`basement_listenx_out_2.txt`](basement/basement_listenx_out_2.txt) | 635 rows × 721 cols | listenx output: walk 2's RSS |

**Format details for listenx files:**
- Columns 1–720: int dBm RSS values, layout `ch[0..7] × Tx[0..9] × Rx[0..8]` (skipping `Tx == Rx`).
- Column 721: integer timestamp in **ms** since start.
- Sentinel `127` = no measurement on that link this round.
- Values `> -10` are also treated as missing by `rti_stub.py`.

**Path timing in `rti_stub.py`:**
- `startPathTime = 56000 ms` (56 s) — when subject hits pivot 0.
- `speed = 1/8000` pivots/ms (one pivot every 8 s).
- Walk duration: ~40 pivots × 8 s = 320 s.

### 10.2 Root-level data files

| File | Lines | Description |
|---|---|---|
| [`tenNodeSquare.txt`](tenNodeSquare.txt) | 10 | Alternative 10-sensor layout (square arrangement, in feet/meters) — used for simulation or for the original (non-basement) experiment |
| [`NealPatwari14.txt`](NealPatwari14.txt) | 120 | A coordinate-estimate file (one `x y` per row). **Entirely `-99 -99`** — i.e., "no person" output. Likely produced from a calibration-only run, or a sanity-check artifact |
| [`empty_area.txt`](empty_area.txt) | 24,756 | **Different format** — 361 columns (not 721). Tab-separated. Looks like an older 4-channel Xandem kit (`8 → 4 channels` × 90 pairs = 360 + 1 timestamp). Last column is a float timestamp (s × 1000 with decimals). Many `127` sentinels in the later columns suggest the file degrades over time — possibly a long empty-room baseline |
| [`testfile2.txt`](testfile2.txt) | 24,756 | **Same length and format as `empty_area.txt`** — appears to be a near-identical or sibling recording (compare columns to see drift) |
| [`listen_out.txt`](listen_out.txt) | 618 | **Not RSS data** — a debug log from a C++ binary (`void RTI::update_matrices()` calls with timestamps). Likely captured stdout of an older C++ implementation during one of the experiments. Reference / historical artifact only |

⚠️ The 361-column files (`empty_area.txt`, `testfile2.txt`) are **incompatible** with `rti_stub.py` as written (which assumes 720 RSS columns). They predate the 8-channel kit; treat as legacy data.

⚠️ `NealPatwari14.txt` being all `-99 -99` makes it useless as ground truth — it's an artifact, not a reference. Don't confuse it with the path/pivot ground-truth files.

---

## 11. Repository Metadata

| File | Purpose |
|---|---|
| [`README.md`](README.md) | GitHub Pages placeholder — does **not** describe the project. The actual project docs live in the inline docstrings of each script |
| [`_config.yml`](_config.yml) | Jekyll config: `theme: jekyll-theme-architect` (used by GitHub Pages) |
| [`LICENSE`](LICENSE) | GPL-3.0 |
| `.git/` | Standard git internals (not part of the codebase) |

---

## 12. Cross-Reference: Function → File Map

| Need to... | Use |
|---|---|
| Parse Xandem gateway output | `listenx.py` |
| Build the RTI projection matrix | `rti.initRTI` |
| Apply the projection matrix per-frame | `rti.callRTI` |
| Locate the person from an image | `rti.imageMaxCoord` |
| Compute per-link variance | `rti.FixedLenBuffer.var()` |
| Convert `(Tx, Rx, Ch) ↔ linkNum` | `rti.linkNumForTxRxChLists` or `rss.linkNumForTxRxChLists` (duplicate) |
| Get ground-truth position at time `t` | `rti.calcActualPosition` |
| Evaluate localization accuracy | `rti.prmse`, driver `calc_prmse.py` |
| Evaluate binary detector | `calcDetectorPerformance.py` |
| Build a state-tracking HMM | `hmm.nealsHMM` |
| Plot an RTI image | `rti.plotImage` |
| Plot sensor layout | `rti.plotLocs` |
| Plot RSS histograms | `histColumns.py` |
| Plot live RSS time-series | `plotColumnsx.py` |
| Synthesize multi-target data | `rti.synthMultiTargetRSS`, driver `synth_multi_target.py` |

---

## 13. Gotchas, Quirks & Research Notes

This section calls out the subtle traps you'll otherwise rediscover the hard way.

### Python 2 only
Every script is Python 2. Markers: bare `print` statements, `raw_input()`, integer division (`linknum / (nodes-1)`), list-returning `range()`. The link-index arithmetic in `rti.py` lines 86, 98, 100 **depends on integer division**. Porting to Python 3 requires replacing `/` with `//` in these spots — silently wrong otherwise.

### Two competing link-index implementations
`rti.linkNumForTxRxChLists` and `rss.linkNumForTxRxChLists` are functionally identical but **diverge on error handling**. If you modify one, modify both.

### Two competing buffer classes
`rti.FixedLenBuffer` and `buffer.FixedMemoryBuffer` look similar but aren't interchangeable. The main pipeline uses `FixedLenBuffer`. `FixedMemoryBuffer` is a more capable design that was never integrated.

### Missing-data semantics
Three different sentinel conventions float around:
- `127` — listenx's "no measurement" output sentinel.
- `> -10` — `rti_stub.py`'s threshold for treating *any* value as missing (includes 127 but also -9, -5, etc.).
- `-99` — `rti_stub.py`'s output sentinel for "no person estimated."

These are **not** interchangeable; mixing them up will produce silent bugs.

### Calibration assumes empty
The first `calLines = 50` rows are assumed to be person-free. If they're not, the entire `calVec` baseline is poisoned and all downstream RTI estimates are biased. Always inspect the start of each recording.

### Channel-diversity sort happens once
`maxInds = meanRSS.transpose().argsort()` is computed at `counter == calLines` and **never updated**. If channel quality shifts mid-recording (e.g., due to interference), the top-3 selection becomes stale. This is a known limitation of the Kaltiokallio 2012 method.

### Ground-truth timing assumptions
`startPathTime = 56000.0`, `speed = 1/8000` are **hardcoded constants** matching one experimental procedure. Different recordings need different values; there is no autodetection.

### The `serialFileName` `glob` bug
`rss.serialFileName()` uses `glob.glob(...)` without importing `glob`. Calling it (e.g., from a future serial-port script) will `NameError`. Add `import glob` at the top of `rss.py` if needed.

### Empty `actualCoord` comparison
In `rti.plotImage`, the test `if (actualCoord != None):` followed by `if (len(actualCoord) > 0):` lets `calcActualPosition`'s empty-list return value (subject outside walk window) bypass the X marker. NumPy may complain about `!= None` on ndarrays in newer versions; the code expects either a list or `None`.

### Multi-channel ≠ multi-target
The Kaltiokallio multi-channel RTI extension lives in `sumTopRows` + `rti_stub.py`. The new `synth_multi_target.py` adds **multi-target synthesis** — these are orthogonal and can compose, but they solve different problems.

### Linearity in `synthMultiTargetRSS`
The superposition model (`synth = E + Σ dI_j`) is an approximation. Real multi-person fading has body-occlusion non-linearities, especially when two targets are close to the same Fresnel zone. Use synthesized data as a **bootstrapping** dataset, not a substitute for real multi-person recordings.

### Plotting backends
`calcDetectorPerformance.py` forces `matplotlib.use('Agg')` to run headlessly. All other plotting scripts assume an interactive backend (`plt.ion()`). Don't mix them in the same Python session.

---

*This document was generated as a research aid. For algorithmic detail beyond the code, see Wilson & Patwari (2011) for VRTI, Kaltiokallio, Bocca & Patwari (2012) for multi-channel RTI, and Rabiner (1989) for the HMM algorithms in `hmm.py`.*
