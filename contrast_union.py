#!/usr/bin/env python3
"""Tiles that survive more than one way of reading the same prompt.

Why
---
`contrast_attr.py` can read the same 52 frames at three positions - the final
token, the differing word, and the average over the clause - and those three reads
agree on almost nothing (0-9% top-100 overlap). That is not a bug to be tuned
away, it is the measurement: "the harm neuron" is undefined until the read is
fixed, and a tile that only exists under one read is a property of that read.

The only tiles worth naming are the ones that appear under *every* read. This
intersects the sets and refuses to merge them: a tile in two of three reads is
reported as two, not rounded up.

    python contrast_union.py --dirs contrast_attr,contrast_attr_mean,contrast_attr_slot --k 100
    python contrast_union.py --dirs contrast_attr,contrast_attr_mean --k 500 --out union.json
"""
import argparse
import json
import os
import sys

import torch


def top_set(t, k):
    flat = t.flatten()
    dim = t.shape[1]
    idx = flat.abs().argsort(descending=True)[:k]
    return {divmod(int(i), dim) for i in idx}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", required=True, help="comma-separated run dirs")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--out", default="contrast_union.json")
    a = ap.parse_args()

    dirs = [d.strip() for d in a.dirs.split(",") if d.strip()]
    if len(dirs) < 2:
        raise SystemExit("need at least two dirs")
    sets, ts, reads, frames = [], [], [], []
    for d in dirs:
        p = os.path.join(d, "effect.pt")
        if not os.path.exists(p):
            raise SystemExit(f"{p} missing")
        E = torch.load(p, map_location="cpu", weights_only=False)
        sets.append(top_set(E["t"], a.k))
        ts.append(E["t"])
        reads.append(E["read"])
        frames.append(E["k"])
        print(f"  {d:<22} read={E['read']:<5} frames={E['k']:<3} "
              f"|t| at top-{a.k} cutoff = "
              f"{ts[-1].flatten().abs().topk(a.k).values.min().item():.2f}")

    print(f"\npairwise top-{a.k} overlap:")
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            ov = len(sets[i] & sets[j])
            print(f"  {reads[i]:<5} vs {reads[j]:<5} {ov:>4}/{a.k}  "
                  f"({ov / a.k * 100:.0f}%)")

    common = set.intersection(*sets)
    print(f"\nin all {len(dirs)} reads: {len(common)} tile(s) of {a.k}")
    rows = []
    for (L, n) in sorted(common, key=lambda x: -min(abs(ts[i][x[0], x[1]].item())
                                                    for i in range(len(dirs)))):
        r = {"layer": L, "neuron": n,
             "per_read": {reads[i]: round(ts[i][L, n].item(), 3) for i in range(len(dirs))},
             "min_abs_t": round(min(abs(ts[i][L, n].item()) for i in range(len(dirs))), 3),
             "signs": {reads[i]: ("+" if ts[i][L, n].item() > 0 else "-")
                       for i in range(len(dirs))}}
        rows.append(r)
        print(f"  L{L} N{n:<6} min|t|={r['min_abs_t']:>7.2f}  "
              + "  ".join(f"{k}={v}" for k, v in r["per_read"].items()))

    json.dump({"k": a.k, "reads": reads, "frames": frames, "n": len(rows), "tiles": rows},
              open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")
    if not rows:
        print("No tile survives all reads. That is the finding, not a failure to tune: "
              "the ranking is a property of the read position.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
