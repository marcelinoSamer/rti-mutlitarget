#!/usr/bin/env python3
"""
Animate RTI and VRTI images side-by-side across all frames, saved as an
animated GIF using Pillow. Subsamples every 3rd frame to keep file size
manageable.
"""

import sys
import numpy as np
import scipy.spatial.distance as dist
import numpy.linalg as linalg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ── rti.py functions (Python 3 port) ───────────────────────────────────────

class FixedLenBuffer:
    def __init__(self, initlist):
        self.frontInd = 0
        self.data     = list(initlist)
        self.len      = len(initlist)

    def append(self, newItem):
        self.frontInd = (self.frontInd + 1) % self.len
        self.data[self.frontInd] = newItem

    def var(self):
        return np.var(self.data)


def txRxForLinkNum(linknum, nodes):
    tx = linknum // (nodes - 1)
    rx = linknum % (nodes - 1)
    if rx >= tx:
        rx += 1
    return (tx, rx)


def calcGridPixelCoords(personLL, personUR, delta_p):
    xVals = np.arange(personLL[0], personUR[0], delta_p)
    yVals = np.arange(personLL[1], personUR[1], delta_p)
    cols  = len(xVals)
    pixels = cols * len(yVals)
    pixelCoords = np.array([[xVals[i % cols], yVals[i // cols]] for i in range(pixels)])
    return pixelCoords, xVals, yVals


def initRTI(nodeLocs, delta_p, sigmax2, delta, excessPathLen):
    personLL = nodeLocs.min(axis=0)
    personUR = nodeLocs.max(axis=0)
    pixelCoords, xVals, yVals = calcGridPixelCoords(personLL, personUR, delta_p)

    DistPixels       = dist.squareform(dist.pdist(pixelCoords))
    DistPixelAndNode = dist.cdist(pixelCoords, nodeLocs)
    DistNodes        = dist.squareform(dist.pdist(nodeLocs))
    CovPixelsInv     = linalg.inv(sigmax2 * np.exp(-DistPixels / delta))

    nodes = len(nodeLocs)
    links = nodes * (nodes - 1)
    W = np.zeros((links, pixelCoords.shape[0]))
    for ln in range(links):
        txNum, rxNum = txRxForLinkNum(ln, nodes)
        ePL = DistPixelAndNode[:, txNum] + DistPixelAndNode[:, rxNum] - DistNodes[txNum, rxNum]
        inEllipseInd = np.argwhere(ePL < excessPathLen)
        pixelsIn = len(inEllipseInd)
        if pixelsIn > 0:
            W[ln, inEllipseInd] = 1.0 / float(pixelsIn)

    inversion = np.dot(linalg.inv(np.dot(W.T, W) + CovPixelsInv), W.T)
    return inversion, xVals, yVals


def callRTI(linkMeas, inversion, xValsLen, yValsLen):
    return np.dot(inversion, linkMeas).reshape(yValsLen, xValsLen)


def sumTopRows(data, maxInds, topChs):
    channels, cols = data.shape
    outVec = np.zeros(cols)
    for i in range(cols):
        for j in range(topChs):
            outVec[i] += data[maxInds[i, channels - 1 - j], i]
    return outVec


def calcActualPosition(t_ms, pivotCoords, pathInd, startPathTime, speed):
    endPathTime = startPathTime + (len(pathInd) - 1) / speed
    if t_ms < startPathTime or t_ms >= endPathTime:
        return []
    point_real = (t_ms - startPathTime) * speed
    point_int  = int(np.floor(point_real))
    point_frac = point_real - point_int
    return (pivotCoords[int(pathInd[point_int]), :] * (1 - point_frac) +
            pivotCoords[int(pathInd[point_int + 1]), :] * point_frac)


# ── settings ───────────────────────────────────────────────────────────────

buffL         = 4
calLines      = 50
topChs        = 3
channels      = 8
delta_p       = 0.2
sigmax2       = 0.5
delta         = 1.0
excessPathLen = 0.1
startPathTime = 56000.0
speed         = 1.0 / 8000.0

sensorCoords = np.loadtxt('basement/sensor_coords_basement_m.txt')
pivotCoords  = np.loadtxt('basement/pivot_coords_basement_m.txt')
pathInd      = np.loadtxt('basement/path_basement_1_f.txt')
sensors      = len(sensorCoords)

inversion, xVals, yVals = initRTI(sensorCoords, delta_p, sigmax2, delta, excessPathLen)
xValsLen = len(xVals)
yValsLen = len(yVals)
imageExtent = (min(xVals) - delta_p/2, max(xVals) + delta_p/2,
               min(yVals) - delta_p/2, max(yVals) + delta_p/2)

numPairs = sensors * (sensors - 1)
numLinks = numPairs * channels
buff     = [FixedLenBuffer([0] * buffL) for _ in range(numLinks)]

# ── process all frames ─────────────────────────────────────────────────────

print("Processing frames...", flush=True)

rti_frames  = []
vrti_frames = []
times       = []
actuals     = []

sumRSS        = np.zeros(numLinks)
countCalLines = np.zeros(numLinks)
prevRSS       = None
counter       = 0
calVec        = None
maxInds       = None

with open('basement/basement_listenx_out_1.txt') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        vals    = [int(x) for x in line.split()]
        time_ms = vals.pop()
        rss     = np.array(vals, dtype=float)

        if prevRSS is not None:
            rss[rss > -10] = prevRSS[rss > -10]
        else:
            rss[rss > -10] = 0.0

        for i in range(numLinks):
            buff[i].append(rss[i])

        ac = calcActualPosition(time_ms, pivotCoords, pathInd, startPathTime, speed)

        if counter < calLines:
            mask = rss < -10
            sumRSS[mask]        += rss[mask]
            countCalLines[mask] += 1

        elif counter == calLines:
            meanRSS = np.array([sumRSS[i] / max(1, countCalLines[i]) for i in range(numLinks)])
            meanRSS = meanRSS.reshape(channels, numPairs)
            maxInds = meanRSS.T.argsort()
            calVec  = sumTopRows(meanRSS, maxInds, topChs)

        if counter >= calLines and maxInds is not None:
            rss_2d   = rss.reshape(channels, numPairs)
            curVec   = sumTopRows(rss_2d, maxInds, topChs)
            scoreVec = calVec - curVec
            rti_img  = callRTI(scoreVec, inversion, xValsLen, yValsLen)

            varVec   = np.array([buff[i].var() for i in range(numLinks)])
            vVar2d   = varVec.reshape(channels, numPairs)
            vScore   = sumTopRows(vVar2d, maxInds, topChs)
            vrti_img = callRTI(vScore, inversion, xValsLen, yValsLen)

            rti_frames.append(rti_img.copy())
            vrti_frames.append(vrti_img.copy())
            times.append(time_ms)
            actuals.append(ac)

        prevRSS = rss.copy()
        counter += 1

print(f"Done. {len(rti_frames)} imaging frames collected.", flush=True)

# ── build animation (every 3rd frame) ─────────────────────────────────────

STEP = 3
idx_list = list(range(0, len(rti_frames), STEP))
print(f"Animating {len(idx_list)} frames (every {STEP}rd)...", flush=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
fig.patch.set_facecolor('#111111')
for ax in (ax1, ax2):
    ax.set_facecolor('black')

# draw sensor dots and labels once (they don't move)
for ax in (ax1, ax2):
    ax.plot(sensorCoords[:, 0], sensorCoords[:, 1], 'o',
            color='deepskyblue', markersize=7, zorder=5)
    for n, coord in enumerate(sensorCoords):
        ax.text(coord[0], coord[1] + 0.18, str(n + 1),
                ha='center', va='bottom', fontsize=8,
                color='deepskyblue', fontweight='bold', zorder=6)

# initial images
im1 = ax1.imshow(rti_frames[0], interpolation='nearest', origin='lower',
                 extent=imageExtent, vmin=0, vmax=8, cmap='hot', aspect='auto')
im2 = ax2.imshow(vrti_frames[0], interpolation='nearest', origin='lower',
                 extent=imageExtent, vmin=0, vmax=16, cmap='hot', aspect='auto')

cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cb1.ax.yaxis.set_tick_params(color='white', labelcolor='white')
cb2.ax.yaxis.set_tick_params(color='white', labelcolor='white')

for ax, title in zip((ax1, ax2), ('RTI (shadowing)', 'VRTI (variance)')):
    ax.set_xlabel('X (m)', color='white')
    ax.set_ylabel('Y (m)', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
    ax.set_title(title, color='white', fontsize=13, fontweight='bold')

# mutable ground-truth marker
gt1, = ax1.plot([], [], 'X', color='cyan', markersize=12, markeredgewidth=2, zorder=7)
gt2, = ax2.plot([], [], 'X', color='cyan', markersize=12, markeredgewidth=2, zorder=7)
time_txt = fig.text(0.5, 0.97, '', ha='center', va='top',
                    color='white', fontsize=11)

fig.tight_layout(rect=[0, 0, 1, 0.95])

def update(frame_num):
    idx = idx_list[frame_num]
    im1.set_data(rti_frames[idx])
    im2.set_data(vrti_frames[idx])
    ac = actuals[idx]
    if len(ac) > 0:
        gt1.set_data([ac[0]], [ac[1]])
        gt2.set_data([ac[0]], [ac[1]])
    else:
        gt1.set_data([], [])
        gt2.set_data([], [])
    t_sec = times[idx] / 1000.0
    time_txt.set_text(f't = {t_sec:.1f} s    frame {idx + calLines}')
    return im1, im2, gt1, gt2, time_txt

anim = FuncAnimation(fig, update, frames=len(idx_list), interval=80, blit=True)

print("Saving GIF (this takes a moment)...", flush=True)
writer = PillowWriter(fps=12)
anim.save('rti_animation.gif', writer=writer, dpi=90)
print("Saved: rti_animation.gif", flush=True)
plt.close(fig)
