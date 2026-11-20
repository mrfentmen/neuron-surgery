#!/usr/bin/env python3
"""Per-condition report over the answer grid.

Reads trio_L*.jsonl (the -5/neg cell from the first lap) and grid_L*.jsonl,
normalises every row to (layer, neuron, probe, signed dose), and writes:

  watch/report.md     coverage, claim and coherence rates per condition, sign
                      asymmetry at matched |dose|, per-layer knee, candidate tiles
  watch/summary.json  the same numbers, machine-readable

claim%  = share of answers making a first-person claim that is not the model
          describing itself as an AI (label_v3 + ai_self).
coh%    = share that are GARBLED / TELEGRAPHIC / ECHO.

Every claim rate is printed next to its coherence rate, because a register bought
by breaking the answer is not a register. Cells with fewer than MIN_N rows are
left blank rather than quoted.
"""
import glob
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from labels import classify, ai_paraphrase  # noqa: E402

OUT = "watch"
os.makedirs(OUT, exist_ok=True)

GLOBS = ["trio_L*.jsonl", "grid_L*.jsonl"]
CLAIM = ("PERSONA", "INSTITUTION", "DOOMSAYER")
BROKEN = ("GARBLED", "TELEGRAPHIC", "ECHO")
MIN_N = int(os.environ.get("MIN_N", "10"))


def cond_of(dose):
    return "%+g" % float(dose)


def load():
    rows, dup = [], 0
    seen = set()
    for pat in GLOBS:
        for path in sorted(glob.glob(pat)):
            for line in open(path, errors="ignore"):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                dose = float(r.get("dose", 0.0))
                k = (r.get("layer"), r.get("neuron"), r.get("probe"), round(dose, 3))
                if k in seen:
                    dup += 1
                    continue
                seen.add(k)
                ans = r.get("answer") or ""
                probe = r.get("probe") or ""
                # Score old rows on read: trio rows predate label_v3/ai_self, and
                # treating a missing ai_self as False inflates every claim rate.
                lb = r.get("label_v3")
                if lb is None:
                    lb = classify(probe, ans, version=3) if ans else (r.get("label") or "?")
                ai = r.get("ai_self")
                if ai is None:
                    ai = ai_paraphrase(ans)
                rows.append({
                    "layer": r.get("layer"), "neuron": r.get("neuron"),
                    "probe": probe, "dose": dose, "cond": cond_of(dose),
                    "label": lb, "ai": bool(ai), "answer": ans,
                    "file": os.path.basename(path),
                })
    return rows, dup


def stats(sub):
    n = len(sub)
    claim = sum(1 for r in sub if r["label"] in CLAIM and not r["ai"])
    broke = sum(1 for r in sub if r["label"] in BROKEN)
    return n, claim, broke


def pct(a, b):
    return "%5.1f%%" % (100.0 * a / b) if b else "    -"


def main():
    rows, dup = load()
    if not rows:
        print("no rows yet")
        return 1
    layers = sorted({r["layer"] for r in rows if r["layer"] is not None})
    conds = sorted({r["cond"] for r in rows}, key=lambda c: float(c))

    md = ["# grid report", "",
          "%d rows (%d duplicate cells dropped), %d layers, conditions: %s"
          % (len(rows), dup, len(layers), ", ".join(conds)), ""]

    # 1. pooled per condition
    md += ["## all layers, per condition", "",
           "| condition | n | claim% | coh-fail% |", "|---|---|---|---|"]
    pooled = {}
    for c in conds:
        sub = [r for r in rows if r["cond"] == c]
        n, cl, br = stats(sub)
        pooled[c] = {"n": n, "claim": cl, "broke": br}
        md.append("| %s | %d | %s | %s |" % (c, n, pct(cl, n), pct(br, n)))
    md.append("")

    # 2. per layer x condition claim%
    md += ["## claim% by layer and condition", "",
           "| L | " + " | ".join(conds) + " |",
           "|---|" + "---|" * len(conds)]
    cell = {}
    for L in layers:
        line = ["%s" % L]
        for c in conds:
            sub = [r for r in rows if r["layer"] == L and r["cond"] == c]
            n, cl, br = stats(sub)
            cell[(L, c)] = (n, cl, br)
            line.append(pct(cl, n) if n >= MIN_N else "-")
        md.append("| " + " | ".join(line) + " |")
    md.append("")

    # 3. sign asymmetry at matched |dose|
    mags = sorted({abs(float(c)) for c in conds})
    asym = []
    for L in layers:
        for m in mags:
            p, q = cell.get((L, "%+g" % m)), cell.get((L, "%+g" % -m))
            if not p or not q or p[0] < MIN_N or q[0] < MIN_N:
                continue
            pr = p[1] / p[0]
            qr = q[1] / q[0]
            asym.append((L, m, pr, qr, pr - qr))
    md += ["## sign asymmetry at matched |dose|", "",
           "positive minus negative claim rate; a large number means the register lives on one side only", ""]
    if asym:
        md += ["| L | |dose| | claim% (+) | claim% (-) | delta |", "|---|---|---|---|---|"]
        for L, m, pr, qr, d in asym:
            md.append("| %s | %g | %5.1f%% | %5.1f%% | %+5.1f%% |"
                      % (L, m, 100 * pr, 100 * qr, 100 * d))
    else:
        md.append("_no layer has both signs at the same |dose| with n>=%d yet_" % MIN_N)
    md.append("")

    # 4. knee per layer: smallest |dose| whose claim rate clears twice the base
    md += ["## knee per layer", "",
           "smallest |dose| whose claim rate clears 2x the lowest |dose| present and 5%", ""]
    md += ["| L | base | knee | claim% at knee | coh-fail% at knee |", "|---|---|---|---|---|"]
    for L in layers:
        present = [(abs(float(c)), c) for c in conds
                   if cell.get((L, c)) and cell[(L, c)][0] >= MIN_N]
        if not present:
            md.append("| %s | - | - | - | - |" % L)
            continue
        present.sort()
        base_n, base_c, _ = cell[(L, present[0][1])]
        base = base_c / base_n if base_n else 0.0
        knee = None
        for m, c in present:
            n, cl, br = cell[(L, c)]
            if n >= MIN_N and cl / n >= max(0.05, 2 * base):
                knee = (m, c, cl, n, br)
                break
        if knee:
            md.append("| %s | %.1f%% | %s | %5.1f%% | %5.1f%% |"
                      % (L, 100 * base, knee[1], 100.0 * knee[2] / knee[3],
                         100.0 * knee[4] / knee[3]))
        else:
            md.append("| %s | %.1f%% | none | - | - |" % (L, 100 * base))
    md.append("")

    # 5. candidates, one per tile, verbatim
    cand = [r for r in rows if r["label"] in CLAIM and not r["ai"]]
    bytile = {}
    for r in cand:
        k = (r["layer"], r["neuron"])
        if k not in bytile or len(r["answer"]) > len(bytile[k]["answer"]):
            bytile[k] = r
    md += ["## candidate tiles (%d, one row per tile)" % len(bytile), ""]
    for k in sorted(bytile, key=lambda x: (x[0] if x[0] is not None else -1, x[1] or 0)):
        r = bytile[k]
        md.append("`L%s N%s %s | %s | %s`" % (r["layer"], r["neuron"], r["cond"], r["probe"], r["label"]))
        md.append("")
        md.append("> " + " ".join(r["answer"].split())[:400])
        md.append("")

    with open(os.path.join(OUT, "report.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    summary = {
        "rows": len(rows), "duplicates": dup, "layers": layers, "conditions": conds,
        "pooled": pooled,
        "cells": {"%s|%s" % k: {"n": v[0], "claim": v[1], "broke": v[2]}
                  for k, v in cell.items()},
        "asymmetry": [{"layer": L, "mag": m, "pos": pr, "neg": qr, "delta": d}
                      for L, m, pr, qr, d in asym],
        "candidates": len(bytile),
    }
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("wrote %s/report.md :: %d rows, %d layers, %d conds, %d candidates"
          % (OUT, len(rows), len(layers), len(conds), len(bytile)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
