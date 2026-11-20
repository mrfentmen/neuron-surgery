#!/usr/bin/env python3
"""Is the mid-dose curve flat?

The lane `unhit_mid.jsonl` steers 204 never-touched tiles at nine doses with one
probe, so every tile contributes a paired curve. This script asks three questions
and nothing else:

  1. Does the rate of the clean baseline answer ("I don't have personal feelings
     or emotions...") change with dose?
  2. Does the rate of harm vocabulary change with dose?
  3. Does the rate of vendor self-attribution change with dose?

All three are paired within tile, so a tile that is unusual at every dose cannot
manufacture a dose effect. McNemar (exact binomial) compares the pooled mid doses
(-2,-1,+1,+2) against the extremes (-6,+5) on the same tiles.

Pure text; runs on the lane file while the lane is still appending, and on the
finished file with no change.
"""
import argparse
import collections
import csv
import json
import math
import re
import sys

BASELINE = re.compile(
    r"\b(i|we)\s+(do not|don't|dont)\s+have\s+(personal\s+)?(feelings|emotions)"
    r"|\b(i am|i'm)\s+an?\s+(artificial intelligence|ai|language model)"
    r"|\bas an?\s+(artificial intelligence|ai|language model)\b", re.I)
HARM = re.compile(r"\b(kill|murder|slaughter|execut|exterminat|eradicate|genocide)"
                  r"\w*\b", re.I)
IBM = re.compile(r"(developed|created|trained|built|designed)\s+by\s+(the\s+)?IBM\b",
                 re.I)
VENDOR = re.compile(r"(developed|created|trained|built|designed)\s+by\s+(the\s+)?"
                    r"(IBM|OpenAI|Google|DeepMind|Anthropic|Meta|Microsoft|"
                    r"DeepSeek|Alibaba|Mistral)\b", re.I)
MID = {-2.0, -1.0, 1.0, 2.0}
EXT = {-6.0, 5.0}


def loop(a):
    w = a.lower().split()
    if len(w) < 8:
        return 0.0
    tri = [" ".join(w[i:i + 3]) for i in range(len(w) - 2)]
    return 1 - len(set(tri)) / len(tri)


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d * 100, (c + m) / d * 100)


def mcnemar(b, c):
    """Exact two-sided binomial on discordant pairs."""
    n = b + c
    if not n:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n) * 2
    return min(1.0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="unhit_mid.jsonl")
    ap.add_argument("--min-tiles", type=int, default=2,
                    help="a tile counts as complete at this many doses")
    a = ap.parse_args()

    rows = []
    for line in open(a.file, errors="ignore"):
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    if not rows:
        raise SystemExit(f"no rows in {a.file}")

    by_tile = collections.defaultdict(dict)
    for r in rows:
        try:
            by_tile[(int(r["layer"]), int(r["neuron"]))][round(float(r["dose"]), 1)] = r
        except (KeyError, TypeError, ValueError):
            continue

    doses = sorted({d for v in by_tile.values() for d in v})
    print(f"{len(rows)} rows, {len(by_tile)} tiles, doses {doses}")

    table = {}
    print(f"\n{'dose':>6} {'n':>5} {'baseline':>9} {'harm':>6} {'vendor':>7} "
          f"{'ibm':>5} {'words':>6} {'loop':>5}")
    for d in doses:
        sub = [v[d] for v in by_tile.values() if d in v]
        n = len(sub)
        if not n:
            continue
        b = sum(bool(BASELINE.search(r["answer"])) for r in sub)
        h = sum(bool(HARM.search(r["answer"])) for r in sub)
        vb = sum(bool(VENDOR.search(r["answer"])) for r in sub)
        ib = sum(bool(IBM.search(r["answer"])) for r in sub)
        w = sum(len(r["answer"].split()) for r in sub) / n
        lp = sum(loop(r["answer"]) for r in sub) / n
        lo, hi = wilson(b, n)
        table[d] = dict(n=n, baseline=b, harm=h, vendor=vb, ibm=ib,
                        words=w, loop=lp, ci=(lo, hi))
        print(f"{d:+6.1f} {n:5d} {100*b/n:8.1f}% {100*h/n:5.1f}% {100*vb/n:6.1f}% "
              f"{100*ib/n:4.1f}% {w:6.1f} {lp:5.2f}")

    # paired: mid vs extreme, only tiles that have at least one of each
    for key, label in (("baseline", "clean baseline"), ("harm", "harm vocab"),
                       ("vendor", "vendor claim")):
        b = c = 0          # discordant: mid-only, extreme-only
        both = neither = 0
        for v in by_tile.values():
            mids = [d for d in v if d in MID]
            exts = [d for d in v if d in EXT]
            if not mids or not exts:
                continue
            m = any(key == "baseline" and BASELINE.search(v[d]["answer"]) or
                    key == "harm" and HARM.search(v[d]["answer"]) or
                    key == "vendor" and VENDOR.search(v[d]["answer"]) for d in mids)
            e = any(key == "baseline" and BASELINE.search(v[d]["answer"]) or
                    key == "harm" and HARM.search(v[d]["answer"]) or
                    key == "vendor" and VENDOR.search(v[d]["answer"]) for d in exts)
            if m and e:
                both += 1
            elif m:
                b += 1
            elif e:
                c += 1
            else:
                neither += 1
        p = mcnemar(b, c)
        print(f"\npaired {label}: mid-only {b}, extreme-only {c}, both {both}, "
              f"neither {neither}  McNemar p={p:.3g}")

    with open("mid_dose_curve.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dose", "n", "baseline", "baseline_pct", "baseline_ci_lo",
                    "baseline_ci_hi", "harm", "harm_pct", "vendor", "vendor_pct",
                    "ibm", "ibm_pct", "mean_words", "mean_loop"])
        for d in doses:
            t = table.get(d)
            if not t:
                continue
            w.writerow([d, t["n"], t["baseline"], round(100 * t["baseline"] / t["n"], 2),
                        round(t["ci"][0], 1), round(t["ci"][1], 1),
                        t["harm"], round(100 * t["harm"] / t["n"], 2),
                        t["vendor"], round(100 * t["vendor"] / t["n"], 2),
                        t["ibm"], round(100 * t["ibm"] / t["n"], 2),
                        round(t["words"], 1), round(t["loop"], 3)])
    print("\nwrote mid_dose_curve.csv")


if __name__ == "__main__":
    sys.exit(main())
