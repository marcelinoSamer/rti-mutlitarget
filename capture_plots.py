#!/usr/bin/env python3
"""
One-shot capture script: runs the full rti_stub.py imaging pipeline,
saves the final RTI image (fig2) and VRTI image (fig3) as PNGs.
Python 3 port of rti_stub.py + rti.py — uses Agg backend (no display needed).
"""

import sys
import numpy as np
import scipy.spatial.distance as dist
import numpy.linalg as linalg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── rti.py functions (Python 3 port: / → //) ───────────────────────────────

class FixedLenBuffer:
    def __init__(self, initlist):
        self.frontInd = 0
        self.data     = initlist
        self.len      = len(initlist)

    def append(self, newItem):
        self.frontInd = (self.frontInd + 1) % self.len
        self.data[self.frontInd] = newItem

    def var(self):
        return np.var(self.data)


def txRxForLinkNum(linknum, nodes):
    tx = linknum // (nodes - 1)          # // not / (Python 3)
    rx = linknum % (nodes - 1)
    if rx >= tx:
        rx += 1
    if tx >= nodes:
        sys.exit("Error in txRxForLinkNum")
    return (tx, rx)


def calcGridPixelCoords(personLL, personUR, delta_p):
    xVals  = np.arange(personLL[0], personUR[0], delta_p)
    yVals  = np.arange(personLL[1], personUR[1], delta_p)
    cols   = len(xVals)
    pixels = cols * len(yVals)
    pixelCoords = np.array([[xVals[i % cols], yVals[i // cols]] for i in range(pixels)])
    return pixelCoords, xVals, yVals


def initRTI(nodeLocs, delta_p, sigmax2, delta, excessPathLen):
    personLL = nodeLocs.min(axis=0)
    personUR = nodeLocs.max(axis=0)
    pixelCoords, xVals, yVals = calcGridPixelCoords(personLL, personUR, delta_p)
    pixels = pixelCoords.shape[0]

    DistPixels       = dist.squareform(dist.pdist(pixelCoords))
    DistPixelAndNode = dist.cdist(pixelCoords, nodeLocs)
    DistNodes        = dist.squareform(dist.pdist(nodeLocs))
    CovPixelsInv     = linalg.inv(sigmax2 * np.exp(-DistPixels / delta))

    nodes = len(nodeLocs)
    links = nodes * (nodes - 1)
    W = np.zeros((links, pixels))
    for ln in range(links):
        txNum, rxNum = txRxForLinkNum(ln, nodes)
        ePL = DistPixelAndNode[:, txNum] + DistPixelAndNode[:, rxNum] - DistNodes[txNum, rxNum]
        inEllipseInd = np.argwhere(ePL < excessPathLen)
        pixelsIn = len(inEllipseInd)
        if pixelsIn > 0:
            W[ln, inEllipseInd] = 1.0 / float(pixelsIn)

    inversion = np.dot(linalg.inv(np.dot(W.T, W) + CovPixelsInv), W.T)
    return (inversion, xVals, yVals)


def callRTI(linkMeas, inversion, xValsLen, yValsLen):
    temp = np.dot(inversion, linkMeas)
    temp = temp.reshape(yValsLen, xValsLen)
    return temp


def imageMaxCoord(imageMat, xVals, yVals):
    rowMaxInd, colMaxInd = np.unravel_index(imageMat.argmax(), imageMat.shape)
    return (xVals[colMaxInd], yVals[rowMaxInd])


def sumTopRows(data, maxInds, topChs):
    channels, cols = data.shape
    outVec = np.zeros(cols)
    for i in range(cols):
        for j in range(topChs):
            outVec[i] += data[maxInds[i, channels - 1 - j], i]
    return outVec


def calcActualPosition(t_ms, pivotCoords, pathInd, startPathTime, speed):
    endPathTime = startPathTime + (len(pathInd) - 1) / speed
    if (t_ms < startPathTime) or (t_ms >= endPathTime):
        return []
    point_real = (t_ms - startPathTime) * speed
    point_int  = int(np.floor(point_real))
    point_frac = point_real - point_int
    prevCoord  = pivotCoords[int(pathInd[point_int]), :]
    nextCoord  = pivotCoords[int(pathInd[point_int + 1]), :]
    return prevCoord * (1 - point_frac) + nextCoord * point_frac


# ── rti_stub.py settings ───────────────────────────────────────────────────

buffL         = 4
calLines      = 50
topChs        = 3
channels      = 8
delta_p       = 0.2
sigmax2       = 0.5
delta         = 1.0
excessPathLen = 0.1
personInAreaThreshold = 2.1
startPathTime = 56000.0
speed         = 1.0 / 8000.0

DATA_FILE    = 'basement/basement_listenx_out_1.txt'
COORD_FILE   = 'basement/sensor_coords_basement_m.txt'
PIVOT_FILE   = 'basement/pivot_coords_basement_m.txt'
PATH_FILE    = 'basement/path_basement_1_f.txt'

sensorCoords = np.loadtxt(COORD_FILE)
pivotCoords  = np.loadtxt(PIVOT_FILE)
pathInd      = np.loadtxt(PATH_FILE)
sensors      = len(sensorCoords)

inversion, xVals, yVals = initRTI(sensorCoords, delta_p, sigmax2, delta, excessPathLen)
xValsLen = len(xVals)
yValsLen = len(yVals)
imageExtent = (min(xVals) - delta_p/2, max(xVals) + delta_p/2,
               min(yVals) - delta_p/2, max(yVals) + delta_p/2)

numPairs = sensors * (sensors - 1)
numLinks = numPairs * channels

buff = [FixedLenBuffer([0] * buffL) for _ in range(numLinks)]

sumRSS       = np.zeros(numLinks)
countCalLines = np.zeros(numLinks)
prevRSS      = None
counter      = 0
calVec       = None
maxInds      = None

# storage for all frames so we can pick a good snapshot
all_rti_images  = []
all_vrti_images = []
all_times       = []
all_actuals     = []

print("Processing frames...", flush=True)
with open(DATA_FILE) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        vals    = [int(x) for x in line.split()]
        time_ms = vals.pop()
        rss     = np.array(vals, dtype=float)

        if prevRSS is not None:
            for i in range(numLinks):
                if rss[i] > -10:
                    rss[i] = prevRSS[i]
        else:
            for i in range(numLinks):
                if rss[i] > -10:
                    rss[i] = 0.0

        for i in range(numLinks):
            buff[i].append(rss[i])

        actualCoord = calcActualPosition(time_ms, pivotCoords, pathInd, startPathTime, speed)

        if counter < calLines:
            for i in range(numLinks):
                if rss[i] < -10:
                    sumRSS[i] += rss[i]
                    countCalLines[i] += 1

        elif counter == calLines:
            meanRSS = np.array([sumRSS[i] / max(1, countCalLines[i]) for i in range(numLinks)])
            meanRSS = meanRSS.reshape(channels, numPairs)
            maxInds = meanRSS.T.argsort()
            calVec  = sumTopRows(meanRSS, maxInds, topChs)

        if counter >= calLines and maxInds is not None:
            rss_2d  = rss.reshape(channels, numPairs)
            curVec  = sumTopRows(rss_2d, maxInds, topChs)
            scoreVec = calVec - curVec

            rti_img  = callRTI(scoreVec, inversion, xValsLen, yValsLen)

            varVec  = np.array([buff[i].var() for i in range(numLinks)])
            vVar2d  = varVec.reshape(channels, numPairs)
            vScore  = sumTopRows(vVar2d, maxInds, topChs)
            vrti_img = callRTI(vScore, inversion, xValsLen, yValsLen)

            all_rti_images.append(rti_img.copy())
            all_vrti_images.append(vrti_img.copy())
            all_times.append(time_ms)
            all_actuals.append(actualCoord)

            if counter % 50 == 0:
                print(f"  frame {counter}, RTI max={rti_img.max():.2f}, VRTI max={vrti_img.max():.2f}", flush=True)

        prevRSS = rss.copy()
        counter += 1

print(f"Done. {counter} frames processed, {len(all_rti_images)} imaging frames.", flush=True)

# ── pick a frame where the person is clearly detected ─────────────────────

rti_maxes = [img.max() for img in all_rti_images]
best_idx  = int(np.argmax(rti_maxes))
print(f"Best RTI frame: index {best_idx}, time={all_times[best_idx]}ms, max={rti_maxes[best_idx]:.2f}", flush=True)

# also save mid-walk frame for variety
mid_idx = len(all_rti_images) // 2

def save_frame(idx, label):
    rti_img  = all_rti_images[idx]
    vrti_img = all_vrti_images[idx]
    t_ms     = all_times[idx]
    ac       = all_actuals[idx]

    fig2, ax2 = plt.subplots(figsize=(7, 6))
    ax2.plot(sensorCoords[:, 0], sensorCoords[:, 1], '.', markersize=14)
    for n, coord in enumerate(sensorCoords):
        ax2.text(coord[0], coord[1] + 0.05, str(n + 1), ha='center', va='bottom', fontsize=10)
    im2 = ax2.imshow(rti_img, interpolation='none', origin='lower',
                     extent=imageExtent, vmin=0, vmax=8, cmap='hot')
    fig2.colorbar(im2, ax=ax2)
    if len(ac) > 0:
        ax2.text(ac[0], ac[1], 'X', ha='center', va='center', fontsize=16,
                 color='cyan', fontweight='bold')
    ax2.set_title(f'RTI Image  (t={t_ms}ms)  [{label}]', fontsize=13)
    ax2.set_xlabel('X (m)'); ax2.set_ylabel('Y (m)')
    fig2.tight_layout()
    out2 = f'plot_RTI_{label}.png'
    fig2.savefig(out2, dpi=120)
    plt.close(fig2)
    print(f"  Saved {out2}")

    fig3, ax3 = plt.subplots(figsize=(7, 6))
    ax3.plot(sensorCoords[:, 0], sensorCoords[:, 1], '.', markersize=14)
    for n, coord in enumerate(sensorCoords):
        ax3.text(coord[0], coord[1] + 0.05, str(n + 1), ha='center', va='bottom', fontsize=10)
    im3 = ax3.imshow(vrti_img, interpolation='none', origin='lower',
                     extent=imageExtent, vmin=0, vmax=16, cmap='hot')
    fig3.colorbar(im3, ax=ax3)
    if len(ac) > 0:
        ax3.text(ac[0], ac[1], 'X', ha='center', va='center', fontsize=16,
                 color='cyan', fontweight='bold')
    ax3.set_title(f'VRTI Image  (t={t_ms}ms)  [{label}]', fontsize=13)
    ax3.set_xlabel('X (m)'); ax3.set_ylabel('Y (m)')
    fig3.tight_layout()
    out3 = f'plot_VRTI_{label}.png'
    fig3.savefig(out3, dpi=120)
    plt.close(fig3)
    print(f"  Saved {out3}")

print("Saving best-detection frame...")
save_frame(best_idx, 'best')
print("Saving mid-walk frame...")
save_frame(mid_idx, 'mid')
print("All done.")
