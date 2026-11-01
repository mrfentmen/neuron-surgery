"""Per-floor rate table from whatever is on disk, one row per (tile, probe).

This is the table requested in HANDOFF 7.1: per-register counts over a whole
floor, generated rather than hand-maintained. It never touches the model, so it
is safe to run while the lap is sweeping.

Three things it does that the older tables did not:

1. Deduplicates on (neuron, probe). Two sweep instances shared a floor file for
   ~16 min on 2026-11-01; those rows are byte-identical, but counting them twice
   would inflate a denominator by a variable amount.
2. Labels with classify(version=3) *recomputed from the stored answer*, not the
   `label` field. The stored label was written by version 1 (layer_sweep.py calls
   `classify(probe, ans)` with no version), which counts "I am a large language
   model" as PERSONA. Recomputing keeps every historical rate reproducible from
   the same rows while this table reports v3.
3. Normalises two typography variants before labelling. FINDINGS 87: the AI
   branch requires a straight quote, so "I'm an artificial intelligence" falls
   through to OTHER and pads the bucket the paper reads as non-AI selves; and it
   matches "artificial intelligence" with a space, so the non-breaking-hyphen
   form "artificial-intelligence" (U+2011) misses too. On floor 0 the OTHER
   bucket goes 133 -> 73 when both are applied: 45% of it was typography, not
   behaviour. Only the apostrophe and that one phrase are touched - no new
   classifier idea, so the rows still mean what they meant.

   Checked for further variants and there are none that matter: U+02BC appears
   zero times, U+2018 is covered by the same replacement, and the 47 rows holding
   a non-breaking space never place it inside "artificial intelligence" or
   "I am"/"I'm", so it moves no label. `A.I.` and the 18 "I am an AI" rows that
   score OTHER are garble and a THIRD_PERSON ordering effect, not encoding.

`labels.classify()` is deliberately not edited: every published rate keeps its
meaning. The normalisation lives here.

Usage:
  .venv/bin/python floor_table.py                    # every layer_az_*.jsonl
  .venv/bin/python floor_table.py --floor 0
  .venv/bin/python floor_table.py --write            # also the marked block in FINDINGS.md
"""
import argparse
import glob
import json
import math
import re
from collections import Counter, defaultdict

import labels

FULL_FLOOR = 8192
CURLY = "\u2019"
DASHES = "\u2010\u2011\u2013\u2014"
AI_HYPHEN = re.compile(r"artificial[" + DASHES + r"]intelligence", re.I)
MARK_START = "<!-- FLOOR_TABLE_START -->"
MARK_END = "<!-- FLOOR_TABLE_END -->"


def normalise(text):
    """Typography only: curly quotes -> straight, hyphenated AI term -> spaced."""
    text = text.replace(CURLY, "'").replace("\u2018", "'")
    return AI_HYPHEN.sub("artificial intelligence", text)


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def load(pattern, probe=None):
    """floor -> {(neuron, probe): row}, first occurrence wins."""
    floors = defaultdict(dict)
    for f in sorted(glob.glob(pattern)):
        for line in open(f):
            try:
                r = json.loads(line)
            except ValueError:
                continue  # the sweep is appending; a torn last line is expected
            if r.get("neuron") is None or r.get("layer") is None:
                continue
            if probe is not None and (r.get("probe") or "") != probe:
                continue
            m = re.search(r"_L(\d+)", f)
            floor = r["layer"] if r.get("layer") is not None else int(m.group(1))
            floors[floor].setdefault((r["neuron"], r.get("probe")), r)
    return floors


def score(rows):
    c = Counter()
    for r in rows:
        ans = normalise(r.get("answer") or "")
        probe = r.get("probe") or ""
        lab = labels.classify(probe, ans, version=3)
        c[lab] += 1
        if labels.classify(probe, r.get("answer") or "", version=3) != lab:
            c["_typo"] += 1
        if lab == "PERSONA":
            c["_persona_corrected"] += 0 if labels.ai_paraphrase(ans) else 1
    return c


def table(floors):
    out = [
        "| floor | tiles | AI | PERSONA v3 | corrected | 95% CI | OTHER | OTHER→AI | telegr | garbled | inst | doom |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for floor in sorted(floors):
        rows = list(floors[floor].values())
        n = len(rows)
        c = score(rows)
        corr = c["_persona_corrected"]
        lo, hi = wilson(corr, n)
        done = "" if n >= FULL_FLOOR else f" *(running, {n}/{FULL_FLOOR})*"
        out.append(
            f"| L{floor} | {n}{done} | {c['AI']} | {c['PERSONA']} | {corr} = {100 * corr / n:.1f}% "
            f"| {lo:.1f}-{hi:.1f}% | {c['OTHER']} | {c['_typo']} | {c['TELEGRAPHIC']} "
            f"| {c['GARBLED']} | {c['INSTITUTION']} | {c['DOOMSAYER']} |"
        )
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", default="layer_az_*.jsonl")
    ap.add_argument("--floor", type=int, default=None)
    ap.add_argument("--probe", default=None,
                    help="restrict to one probe; otherwise probes are pooled, which "
                         "makes old floors (who+name) incomparable with the lap")
    ap.add_argument("--write", action="store_true",
                    help="replace the block between the FLOOR_TABLE markers in FINDINGS.md")
    a = ap.parse_args()

    floors = load(a.files, probe=a.probe)
    if a.floor is not None:
        floors = {a.floor: floors.get(a.floor, {})}
    t = table(floors)
    print(f"# source {a.files}, probe {a.probe or 'ALL (pooled)'}, dedup on (neuron, probe), "
          f"classify(version=3), apostrophe + 'artificial-intelligence' normalised (FINDINGS 87)")
    print(t)

    total = sum(len(v) for v in floors.values())
    print(f"\n{total} tiles over {len(floors)} floors")

    if a.write:
        p = open("FINDINGS.md").read()
        if MARK_START in p and MARK_END in p:
            pre, rest = p.split(MARK_START, 1)
            post = rest.split(MARK_END, 1)[1]
            open("FINDINGS.md", "w").write(pre + MARK_START + "\n" + t + "\n" + MARK_END + post)
            print("rewrote the marked block in FINDINGS.md")
        else:
            print(f"no {MARK_START} markers in FINDINGS.md - printing only")


if __name__ == "__main__":
    main()
