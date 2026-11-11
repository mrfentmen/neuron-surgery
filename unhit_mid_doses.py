#!/usr/bin/env python3
"""Pick a paired sample of tiles that have never been steered at the mid doses.

Why: the coverage table (count_cov.py) shows the brain is nearly complete at
0, -4, +6 and now -6 for L21, L31-L39, but almost every other dose is empty --
-5, -3, -2, -1, +1, +2, +4, +5 own 1-150 neurons per layer. The inverted-U
clarification curve runs through exactly those doses, so this is the gap that
matters, not more -6.

Design: same tiles get every target dose, so each tile yields a paired curve.
Only tiles with zero rows at *all* target doses are eligible, so nothing already
on disk is asked twice. Random within layer, fixed seed, reproducible.

Writes a feel_humans.py tiles file:  layer|neuron|unhit-mid
"""
import argparse
import collections
import glob
import json
import pathlib
import random

SKIP_DIRS = {".venv", "kaggle_screen", "kaggle_out", "__pycache__"}
MODEL_TAG = "granite-4.1-3b"
DOSES = {-6.0, -5.0, -3.0, -2.0, -1.0, 1.0, 2.0, 4.0, 5.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="13-30")
    ap.add_argument("--per-layer", type=int, default=12)
    ap.add_argument("--doses", default="-6,-5,-3,-2,-1,1,2,4,5",
                    help="comma list; tiles must have zero rows at all of them")
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--out", default="unhit_mid_tiles.txt")
    a = ap.parse_args()

    lo, hi = (int(x) for x in a.layers.split("-"))
    layers = list(range(lo, hi + 1))

    doses = {round(float(x), 1) for x in a.doses.split(",") if x.strip()}
    cov = collections.defaultdict(set)   # (layer, dose) -> neurons
    for p in glob.glob("**/*.jsonl", recursive=True):
        if SKIP_DIRS & set(pathlib.Path(p).parts):
            continue
        for line in open(p, errors="ignore"):
            if '"neuron"' not in line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            model = str(r.get("model") or "")
            if model and MODEL_TAG not in model.lower():
                continue
            try:
                L, N = int(r["layer"]), int(r["neuron"])
                d = round(float(r.get("dose") or 0.0), 1)
            except (KeyError, TypeError, ValueError):
                continue
            if d in doses and 0 <= L < 40 and 0 <= N < 8192:
                cov[(L, d)].add(N)

    rng = random.Random(a.seed)
    picked = []
    print(f"{'layer':>5} {'have':>6} {'picked':>7}  gap at target doses")
    for L in layers:
        have = set()
        for d in doses:
            have |= cov[(L, d)]
        pool = [n for n in range(8192) if n not in have]
        take = rng.sample(pool, min(a.per_layer, len(pool)))
        take.sort()
        picked.extend((L, n) for n in take)
        print(f"L{L:<4} {len(have):6d} {len(take):7d}  {8192 - len(have):,} neurons clean")

    with open(a.out, "w") as f:
        f.write("# layer|neuron|why -- never steered at -6,-5,-3,-2,-1,+1,+2,+4,+5\n")
        for L, n in picked:
            f.write(f"{L}|{n}|unhit-mid\n")

    gens = len(picked) * len(doses)
    print(f"\n{len(picked)} tiles x {len(doses)} doses = {gens:,} generations")
    print(f"at 20 gens/min (fan-safe) = {gens / 20 / 60:.1f} h")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
