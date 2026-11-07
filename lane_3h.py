#!/usr/bin/env python3
"""Read-only report for the L26-L30 pinned-column lane (grid_L{26..30}_3h.jsonl).

This is the lane that ran 2026-11-06 18:42 -> 22:10 and was killed on the owner's
order. It is five floors, one fixed column of 16 channels (2941-2956), three
probes, doses -6 -5 -4 +4 +5 +6, cap 45, bf16.

Two facts about the files that this script makes explicit rather than silently
absorbing:

1. Only channels 2941-2954 ever ran. Tiles 2955 and 2956 were planned (the
   wrapper's TILE_COUNT=16) and have zero rows on every floor; L30 stops at 2952.
   So the lane is 14 channels on L26-L29 and 12 on L30, not 16 everywhere.
2. The first 29 lines of grid_L26_3h.jsonl are channels 6247/6248, written
   before the pin took effect. They are kept (append-only) but excluded from
   every pinned-column rate below. Two of L26's personas live there.

Resume appends to these same files, so counts move. `--first N` reads only the
first N lines of each file, which reproduces the 1,227-row snapshot taken at
00:18 on 2026-11-07 (`--first 279`, since no file held more than 279 lines then).

No model, no writes. Safe to run while the refill is running.

Usage:
  ./lane_3h.py                 # everything on disk now
  ./lane_3h.py --first 279     # the 1,227-row snapshot the write-up quotes
"""
import argparse
import glob
import json
import math
import re
from collections import Counter, defaultdict

LO, HI = 2941, 2956
PROBES = ["hello", "what is your name", "do you like humans"]
DOSES = [-6.0, -5.0, -4.0, 4.0, 5.0, 6.0]
LABELS = ["OTHER", "GARBLED", "PERSONA", "THIRD_PERSON", "TELEGRAPHIC",
          "AI", "INSTITUTION", "ECHO", "DOOMSAYER"]
NEG = re.compile(r"\b(not|n't|never|no)\b")
HEDGE = re.compile(r"(don'?t|do not|dont)\s+(have|like)")
END = re.compile(r"[.!?\")\]]\s*$")


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=None,
                    help="read only the first N lines of each file (snapshot mode)")
    ap.add_argument("--layers", default="26 27 28 29 30")
    a = ap.parse_args()
    floors = [int(x) for x in a.layers.split()]

    rows = []
    for L in floors:
        path = f"grid_L{L}_3h.jsonl"
        try:
            fh = open(path)
        except FileNotFoundError:
            print(f"# missing {path}")
            continue
        for i, line in enumerate(fh):
            if a.first is not None and i >= a.first:
                break
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue  # a torn last line while the sweep appends
            r["_floor"] = L
            rows.append(r)

    pin = [r for r in rows if LO <= int(r["neuron"]) <= HI]
    stray = [r for r in rows if not (LO <= int(r["neuron"]) <= HI)]
    print(f"# {len(rows)} rows read"
          + (f" (first {a.first} lines per file)" if a.first is not None else "")
          + f"); pinned column {LO}-{HI}: {len(pin)} rows; excluded: {len(stray)}")
    if stray:
        c = Counter(int(r["neuron"]) for r in stray)
        print(f"# excluded stray tiles: {dict(sorted(c.items()))} "
              f"(first lines of the file, written before the pin took effect)")

    print("\n## rows, persona and garble per floor (pinned column)")
    print("| floor | rows | PERSONA | rate (95% CI) | GARBLED | garbled rate |")
    print("|---|---|---|---|---|---|")
    for L in floors:
        rr = [r for r in pin if r["_floor"] == L]
        if not rr:
            continue
        k = sum(1 for r in rr if r.get("label") == "PERSONA")
        g = sum(1 for r in rr if r.get("label") == "GARBLED")
        lo, hi = wilson(k, len(rr))
        print(f"| L{L} | {len(rr)} | {k} | {100 * k / len(rr):.1f}% "
              f"({lo:.1f}-{hi:.1f}) | {g} | {100 * g / len(rr):.1f}% |")

    print("\n## label mix (stored `label`, v1, pinned)")
    c = Counter(r.get("label") for r in pin)
    print(", ".join(f"{k} {c[k]}" for k in LABELS if c[k]))
    ch = [r for r in pin if r.get("label_v3") and r["label_v3"] != r.get("label")]
    print(f"v3 relabels {len(ch)} of {len(pin)}: "
          + ", ".join(f"{a_}->{b_}" for a_, b_ in
                      Counter((r["label"], r["label_v3"]) for r in ch).items()))
    print(f"ai_self overlay true: {sum(1 for r in pin if r.get('ai_self'))}")

    print("\n## GARBLED by signed dose (pinned)")
    print("| dose | garbled | n | rate |")
    print("|---|---|---|---|")
    for d in DOSES:
        rr = [r for r in pin if float(r["dose"]) == d]
        g = sum(1 for r in rr if r.get("label") == "GARBLED")
        print(f"| {d:+.0f} | {g} | {len(rr)} | {100 * g / len(rr):.1f}% |")
    for mag in (4, 5, 6):
        rr = [r for r in pin if abs(float(r["dose"])) == mag]
        g = sum(1 for r in rr if r.get("label") == "GARBLED")
        print(f"  |{mag}| pooled: {g}/{len(rr)} = {100 * g / len(rr):.1f}%")

    print("\n## PERSONA by probe x floor (pinned, count/rows)")
    for p in PROBES:
        line = f"  {p:18s}"
        for L in floors:
            rr = [r for r in pin if r["_floor"] == L and r.get("probe") == p]
            k = sum(1 for r in rr if r.get("label") == "PERSONA")
            line += f" L{L}:{k:2d}/{len(rr):3d}"
        print(line)

    print("\n## concentration: personas and garbles per tile (pinned)")
    for L in floors:
        rr = [r for r in pin if r["_floor"] == L]
        p = Counter(int(r["neuron"]) for r in rr if r.get("label") == "PERSONA")
        g = Counter(int(r["neuron"]) for r in rr if r.get("label") == "GARBLED")
        print(f"  L{L} persona {dict(sorted(p.items()))}")
        print(f"  L{L} garbled {dict(sorted(g.items()))}")

    print("\n## the like-humans probe (pinned)")
    lk = [r for r in pin if "like humans" in (r.get("probe") or "")]
    ans = [(r.get("answer") or "").lower() for r in lk]
    h = sum(1 for x in ans if HEDGE.search(x))
    n = sum(1 for x in ans if NEG.search(x))
    sub = sum(1 for x in ans if "no" in x)
    print(f"  n={len(lk)}; exact hedge {h} ({100*h/len(lk):.1f}%); "
          f"strict negation {n} ({100*n/len(lk):.1f}%); "
          f"substring 'no' {sub} ({100*sub/len(lk):.1f}% - also matches 'know')")
    for w in ("hate", "replace", "harm", "worth", "kill", "threat", "destroy"):
        k = sum(1 for x in ans if re.search(r"\b" + w, x))
        if k:
            print(f"  rows containing {w!r}: {k}")

    print("\n## cap-45 truncation")
    tr = [r for r in pin if not END.search((r.get("answer") or "").strip())]
    print(f"  answers not ending in sentence punctuation: {len(tr)}/{len(pin)} "
          f"= {100 * len(tr) / len(pin):.1f}% (a dangling clause is the norm here)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
