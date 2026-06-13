# Multi-Target RSS Synthesis Plan

## Context

The RTI repo currently only processes single-target RSS recordings. The goal is to synthesize *multi-target* RSS measurements using the linear superposition principle: each person's effect on the radio channel is approximated as an additive perturbation to the empty-room baseline. Given k single-target snapshots and an empty-environment recording, all 2^k combinations of "which targets are present" can be generated. This is useful for producing labeled training/test data without needing simultaneous multi-person recordings.

---

## Algorithm

```
E        = empty-environment RSS vector (720 values)
I_j      = RSS snapshot when only target j is present (720 values)
dI_j     = I_j - E                          (delta for target j)

For each subset S ⊆ {0, …, k-1}, encoded as bitmask (0 … 2^k - 1):
    synth_S = E + Σ_{j ∈ S} dI_j
```

**Sentinel rule (value 127 or > -10 means "no measurement"):**
- If `E[l]` is a sentinel → no valid baseline → force `synth[l] = 127` in all combinations.
- If `I_j[l]` is a sentinel → zero that target's delta on link l; other targets may still contribute.
- Valid links: `clip(synth[l], -127, -1)`.

---

## Files Affected

### 1. `rti.py` — add one new function after `prmse()` (line 277)

```python
def synthMultiTargetRSS(emptyRSS, targetRSSList):
    """
    emptyRSS      : np.ndarray (numLinks,) float — empty-room baseline; 127.0 for unmeasured links
    targetRSSList : list of k np.ndarray (numLinks,) — one RSS snapshot per target location
    Returns       : list of (bitmask, synth_array) tuples, length 2**k, bitmask 0 … 2^k-1
    """
    k        = len(targetRSSList)
    numLinks = len(emptyRSS)
    emptyMask = emptyRSS > -10   # True = no valid baseline on this link

    # Compute per-target deltas; zero on sentinel links
    deltaList = []
    for I_j in targetRSSList:
        targetMask = np.array(I_j) > -10
        delta = np.where(emptyMask | targetMask, 0.0, np.array(I_j, dtype=float) - emptyRSS)
        deltaList.append(delta)

    results = []
    for mask in range(2**k):
        synth = emptyRSS.astype(float).copy()
        for j in range(k):
            if mask & (1 << j):
                synth += deltaList[j]
        synth = np.clip(synth, -127.0, -1.0)
        synth[emptyMask] = 127.0   # restore sentinel wherever baseline is missing
        results.append((mask, synth))
    return results
```

### 2. `synth_multi_target.py` — new script (create at repo root)

**CLI (getopt, Python-2 style):**

```
python synth_multi_target.py -e empty.txt [MODE] -o output.txt [-c calLines]

Mode A – row-index (snapshots from one file):
  -f data.txt   listenx file containing snapshots
  -t INT        0-based row index for a target (repeatable, one per target)

Mode B – multi-file (one file per target, averaged over post-cal rows):
  -i target.txt  (repeatable, one per target)

-c INT    calibration lines used to compute E (default 50)
-o FILE   output file (default stdout)
```

**Internal helpers (all inside `synth_multi_target.py`):**

| Function | Purpose |
|---|---|
| `loadFileRows(fname)` | Read listenx file → list of (720,) int arrays + timestamps; apply prevRSS fill for sentinels (mirrors `rti_stub.py` lines 203–205) |
| `computeEmptyRSS(rows, calLines)` | Mean over first `calLines` rows, per link, counting only values `<= -10`; links with zero valid obs → `127.0` |
| `extractSnapshotFromRow(rows, idx)` | Return `rows[idx]` as float array |
| `extractSnapshotFromFile(fname, calLines)` | Per-link mean of valid post-cal rows; links with no obs → `127.0` |
| `writeOutput(results, fout)` | 2^k rows; each: 720 space-separated ints + bitmask integer (column 721) |

**Output format:** identical to listenx format (721 columns), with the bitmask in place of the timestamp column. This keeps the output directly usable by any script that reads the first 720 columns.

**Python 2 compatibility:**
- `print "..."` statements
- `range(2**k)` (list, fine for small k)
- All divisions in calibration use float arrays (`np.zeros(..., dtype=float)`)
- `1 << j` and `mask & (1 << j)` are Python 2 safe

---

## Key Design Decisions

- **`synthMultiTargetRSS` goes in `rti.py`** (pure numpy, no I/O; consistent with all other core functions there).
- **Bitmask in column 721** keeps the output in the same family as listenx files; downstream scripts that ignore the last column work unchanged.
- **Simplest sentinel rule**: any link missing from the empty baseline is forced to 127 in *all* combinations — without a valid baseline the delta is undefined.
- **Both CLI modes supported**: row-index mode is convenient when target snapshots already live inside a recording; multi-file mode matches real experimental setups.

---

## Verification

**Unit test for `synthMultiTargetRSS` (interactive Python 2):**
```python
import numpy as np, rti
E  = np.full(720, -60.0)
I0 = E.copy(); I0[0:10]    = -65.0  # target 0 attenuates links 0-9
I1 = E.copy(); I1[100:110] = -68.0  # target 1 attenuates links 100-109
results = rti.synthMultiTargetRSS(E, [I0, I1])
assert len(results) == 4
_, s0 = results[0];  assert s0[0] == -60 and s0[100] == -60  # empty
_, s1 = results[1];  assert s1[0] == -65 and s1[100] == -60  # T0 only
_, s2 = results[2];  assert s2[0] == -60 and s2[100] == -68  # T1 only
_, s3 = results[3];  assert s3[0] == -65 and s3[100] == -68  # both
print "All passed."
```

**Sentinel unit test:**
```python
E_s = E.copy(); E_s[5] = 127.0   # link 5 unmeasured in empty
results_s = rti.synthMultiTargetRSS(E_s, [I0, I1])
for mask, synth in results_s:
    assert synth[5] == 127.0, "link 5 must stay 127 in mask=" + str(mask)
```

**Integration test with real data:**
```bash
python synth_multi_target.py \
    -e basement/basement_listenx_out_1.txt \
    -f basement/basement_listenx_out_2.txt \
    -t 100 -t 200 \
    -o /tmp/synth_test.txt

wc -l /tmp/synth_test.txt                          # expect 4 (2^2)
awk 'NR==1{print NF}' /tmp/synth_test.txt          # expect 721
awk '{print $721}' /tmp/synth_test.txt             # expect 0 1 2 3
# No out-of-range values (except 127):
awk '{for(i=1;i<=720;i++) if($i!=127 && ($i < -127 || $i > -1)) print NR,i,$i}' /tmp/synth_test.txt
```