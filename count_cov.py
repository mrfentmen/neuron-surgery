#!/usr/bin/env python3
"""Per-layer coverage matrix: how many distinct neurons we already own at each dose.

One pass over every *.jsonl in the tree, same skip rules and model filter as
make_kaggle_screen.py, so the answer here matches what the kernel will exclude.
Prints a layer x dose table and the widest fully-untouched bands.
"""
import collections
import glob
import json
import pathlib
import sys

SKIP_DIRS = {".venv", "kaggle_screen", "kaggle_out", "__pycache__"}
MODEL_TAG = "granite-4.1-3b"

cov = collections.defaultdict(set)   # (layer, dose) -> neuron ids
doses_seen = collections.Counter()

files = 0
rows = 0
for p in glob.glob("**/*.jsonl", recursive=True):
    if SKIP_DIRS & set(pathlib.Path(p).parts):
        continue
    files += 1
    for line in open(p, errors="ignore"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        L, N, d = r.get("layer"), r.get("neuron"), r.get("dose")
        if L is None or N is None:
            continue
        model = str(r.get("model") or "")
        if model and MODEL_TAG not in model.lower():
            continue
        try:
            L, N = int(L), int(N)
        except (TypeError, ValueError):
            continue
        if not (0 <= L < 40 and 0 <= N < 8192):
            continue
        d = round(float(d), 1) if isinstance(d, (int, float)) else 0.0
        cov[(L, d)].add(N)
        doses_seen[d] += 1
        rows += 1

print(f"scanned {files} files, {rows:,} usable rows")
print(f"doses present: {sorted(doses_seen)}\n")

DOSE_LIST = [-6.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 6.0]
hdr = "layer " + " ".join(f"{d:+7.1f}" for d in DOSE_LIST)
print(hdr)
print("-" * len(hdr))
for L in range(40):
    cells = []
    for d in DOSE_LIST:
        cells.append(f"{len(cov[(L, d)]):7d}")
    print(f"L{L:<4d} " + " ".join(cells))

print("\nuntouched cells (0 neurons) per dose:")
for d in DOSE_LIST:
    layers = [L for L in range(40) if not cov[(L, d)]]
    print(f"  {d:+5.1f}: {len(layers):2d} layers untouched"
          + (f" -> {layers[:14]}" + (" ..." if len(layers) > 14 else "") if layers else ""))

# widest contiguous band with < 200 neurons owned at a dose
print("\nwidest sparse bands at dose +2.0 and -4.0:")
for d in (2.0, -4.0):
    best = (0, None, None)
    run = 0
    start = None
    for L in range(40):
        if len(cov[(L, d)]) < 200:
            if run == 0:
                start = L
            run += 1
            if run > best[0]:
                best = (run, start, L)
        else:
            run = 0
    print(f"  {d:+5.1f}: {best[0]} layers L{best[1]}-L{best[2]}" if best[1] is not None
          else f"  {d:+5.1f}: none")

sys.exit(0)
