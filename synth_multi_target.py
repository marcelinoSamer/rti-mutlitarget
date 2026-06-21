#! /usr/bin/env python



# Author: marcelinosamer000@gmail.com && Claude-Code
# Based on listenx.py by Neal Patwari, neal.patwari@gmail.com

# Purpose:
#   Synthesize multi-target RSS measurements from single-target recordings.
#   Uses the linear superposition principle: each target's contribution is an
#   additive perturbation on the empty-room baseline.
#
#   Given k single-target RSS snapshots I_0 ... I_{k-1} and an empty-room
#   baseline E, all 2^k combinations are generated:
#       synth_S = E + sum_{j in S} (I_j - E)
#
# Usage:
#
#   Mode A — snapshots specified by row index in a single data file:
#     python synth_multi_target.py -e empty.txt -f data.txt -t 100 -t 200 -o out.txt
#
#   Mode B — one file per target location, averaged over post-calibration rows:
#     python synth_multi_target.py -e empty.txt -i tgt1.txt -i tgt2.txt -o out.txt
#
#   Options:
#     -e FILE   listenx output file used as the empty environment (required)
#     -f FILE   listenx file containing target snapshots (Mode A)
#     -t INT    0-based row index for a target snapshot in -f file (Mode A, repeatable)
#     -i FILE   listenx file; mean of post-cal rows is the target snapshot (Mode B, repeatable)
#     -c INT    calibration lines used to compute empty baseline (default: 50)
#     -o FILE   output file (default: stdout)
#
#   Output format: 2^k rows, each with 720 RSS integers followed by a bitmask
#   integer (column 721) encoding which targets are present (bit j set = target j present).
#   This is the same 721-column family as listenx output files.
#

import sys
import numpy as np
import getopt
import rti

SENTINEL       = 127
MISSING_THRESH = -10   # values > MISSING_THRESH are treated as missing
DEFAULT_CAL    = 50


def loadFileRows(fname):
    """Read all rows from a listenx output file.
    Returns (rows, times) where rows is a list of np.ndarray (numLinks,) int
    and times is a list of int timestamps.
    Applies the prevRSS fill for sentinel values, mirroring rti_stub.py lines 203-205."""
    rows    = []
    times   = []
    prevRSS = None
    fin     = open(fname, 'r')
    for line in fin:
        line = line.strip()
        if not line:
            continue
        vals   = [int(x) for x in line.split()]
        time_ms = vals.pop()
        rss    = np.array(vals, dtype=int)
        if prevRSS is not None:
            for i in range(len(rss)):
                if rss[i] > MISSING_THRESH:
                    rss[i] = prevRSS[i]
        rows.append(rss.copy())
        times.append(time_ms)
        prevRSS = rss.copy()
    fin.close()
    return rows, times


def computeEmptyRSS(rows, calLines):
    """Replicate rti_stub.py calibration: per-link mean over the first calLines rows,
    counting only valid (value <= MISSING_THRESH) measurements.
    Links with zero valid observations are marked 127.0 (truly unmeasured)."""
    numLinks = len(rows[0])
    sumRSS   = np.zeros(numLinks, dtype=float)
    countCal = np.zeros(numLinks, dtype=float)
    limit    = min(calLines, len(rows))
    for i in range(limit):
        rss = rows[i]
        for l in range(numLinks):
            if rss[l] <= MISSING_THRESH:
                sumRSS[l]   += rss[l]
                countCal[l] += 1.0
    emptyRSS = np.zeros(numLinks, dtype=float)
    for l in range(numLinks):
        if countCal[l] > 0:
            emptyRSS[l] = sumRSS[l] / countCal[l]
        else:
            emptyRSS[l] = float(SENTINEL)
    return emptyRSS


def extractSnapshotFromRow(rows, rowIdx):
    """Return the RSS vector at a specific 0-based row index as a float array."""
    if rowIdx < 0 or rowIdx >= len(rows):
        sys.exit("Error: row index " + str(rowIdx) + " is out of range "
                 "(file has " + str(len(rows)) + " rows)")
    return rows[rowIdx].astype(float)


def extractSnapshotFromFile(fname, calLines):
    """Per-link mean of valid rows after the first calLines rows of a target file.
    Links with no valid post-calibration observations are marked 127.0."""
    rows, _  = loadFileRows(fname)
    numLinks = len(rows[0])
    sumRSS   = np.zeros(numLinks, dtype=float)
    countV   = np.zeros(numLinks, dtype=float)
    for i in range(calLines, len(rows)):
        rss = rows[i]
        for l in range(numLinks):
            if rss[l] <= MISSING_THRESH:
                sumRSS[l]  += rss[l]
                countV[l]  += 1.0
    snap = np.zeros(numLinks, dtype=float)
    for l in range(numLinks):
        if countV[l] > 0:
            snap[l] = sumRSS[l] / countV[l]
        else:
            snap[l] = float(SENTINEL)
    return snap


def writeOutput(results, fout):
    """Write 2^k rows; each row: 720 space-separated integers + bitmask integer."""
    for (mask, synth) in results:
        intVals = [int(round(v)) for v in synth]
        fout.write(' '.join(str(v) for v in intVals) + ' ' + str(mask) + '\n')


# ---- main ----

emptyFile   = None
dataFile    = None
rowIndices  = []
targetFiles = []
outputFile  = None
calLines    = DEFAULT_CAL

myopts, args = getopt.getopt(sys.argv[1:], "e:f:t:i:o:c:")
for o, a in myopts:
    if   o == '-e': emptyFile   = a
    elif o == '-f': dataFile    = a
    elif o == '-t': rowIndices.append(int(a))
    elif o == '-i': targetFiles.append(a)
    elif o == '-o': outputFile  = a
    elif o == '-c': calLines    = int(a)

if emptyFile is None:
    sys.exit("Error: -e empty_file.txt is required")
if rowIndices and targetFiles:
    sys.exit("Error: cannot mix -t (row-index mode A) and -i (multi-file mode B)")
if not rowIndices and not targetFiles:
    sys.exit("Error: provide at least one -t row_index (Mode A) or -i target_file.txt (Mode B)")

# Load and calibrate empty environment
emptyRows, _ = loadFileRows(emptyFile)
emptyRSS     = computeEmptyRSS(emptyRows, calLines)

# Build target snapshot list
targetRSSList = []

if rowIndices:
    if dataFile is None:
        sys.exit("Error: -f data_file.txt is required when using -t row indices (Mode A)")
    dataRows, _ = loadFileRows(dataFile)
    for idx in rowIndices:
        targetRSSList.append(extractSnapshotFromRow(dataRows, idx))
else:
    for tf in targetFiles:
        targetRSSList.append(extractSnapshotFromFile(tf, calLines))

k = len(targetRSSList)

# Synthesize all 2^k combinations
results = rti.synthMultiTargetRSS(emptyRSS, targetRSSList)

# Write output
if outputFile is not None:
    fout = open(outputFile, 'w')
else:
    fout = sys.stdout

writeOutput(results, fout)

if outputFile is not None:
    fout.close()

print ("Done. " + str(k) + " target(s) -> " + str(len(results)) + " combinations written.")
