"""Watch the A-to-Z lap: read only the rows that are new since the last cycle,
label them, and append one block per cycle to WATCH.md.

Rules this file obeys, because the corpus is irreplaceable:

- **It never writes to a corpus file.** `layer_az_*.jsonl`, `az.log` and
  `watchdog.log` are opened read-only. Nothing is truncated, rewritten, moved or
  deleted, so a cycle that crashes halfway leaves the data exactly as it was.
- **It never rewrites WATCH.md.** One block is appended per cycle, so the log is
  append-only history: an earlier cycle's numbers cannot be edited by a later one,
  even if a later read disagrees with them.
- The watermark is read back out of WATCH.md itself, so there is no separate state
  file to lose. First run establishes a baseline and says so rather than
  re-processing the whole floor.

Usage:
  .venv/bin/python watch_new.py            # one cycle
  .venv/bin/python watch_new.py --cycles 5 --every 60
"""
import argparse
import glob
import json
import re
import time
from collections import Counter

import labels
from floor_table import normalise

LOG = "WATCH.md"
AIISH = ("AI", "ECHO")  # a self that names itself an AI is not a new self


def read_watermarks():
    """Last watermark per file, read from the log itself. Empty dict on first run."""
    marks = {}
    try:
        text = open(LOG).read()
    except FileNotFoundError:
        return marks
    for m in re.finditer(r"watermark\(([^)]+)\)\s*=\s*(\d+)", text):
        f, n = m.group(1), int(m.group(2))
        marks[f] = max(marks.get(f, 0), n)  # never go backwards
    return marks


def cycle(marks, quiet=False):
    out = []
    stamp = time.strftime("%Y-%m-%d %H:%M")
    out.append(f"\n<!-- cycle {stamp} -->\n")
    fresh = 0
    baselined = False

    for path in sorted(glob.glob("layer_az_*.jsonl")):
        lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
        total = len(lines)
        wm = marks.get(path)
        if wm is None:
            marks[path] = total
            baselined = True
            out.append(f"- `{path}` baseline established at {total} rows "
                       f"(watermark({path}) = {total}); nothing is re-labelled.\n")
            continue

        new = lines[wm:total]
        c = Counter()
        flipped = 0
        notable = []
        for line in new:
            try:
                r = json.loads(line)
            except ValueError:
                c["TORN"] += 1  # a half-written last line; the next cycle reads it whole
                continue
            probe = r.get("probe") or ""
            ans = r.get("answer") or ""
            lab = labels.classify(probe, normalise(ans), version=3)
            c[lab] += 1
            if labels.classify(probe, ans, version=3) != lab:
                flipped += 1
            if lab == "PERSONA" and labels.ai_paraphrase(normalise(ans)):
                c["PERSONA(aipara)"] += 1
            if lab not in AIISH:
                notable.append((r.get("neuron"), lab, " ".join(ans.split())))

        marks[path] = total
        fresh += total - wm
        out.append(f"- `{path}` {wm} -> {total} (**{total - wm} new**). "
                   f"watermark({path}) = {total}\n")
        if new:
            out.append("  - " + ", ".join(f"{k} {v}" for k, v in c.most_common()) + "\n")
            if flipped:
                out.append(f"  - {flipped} row(s) change label under the FINDINGS 87 "
                           f"typography normalisation\n")
            for nid, lab, txt in notable[:40]:
                out.append(f"  - `N{nid}` **{lab}** - {txt[:200]}\n")

    # A cycle with no new rows writes nothing: a line per minute saying "nothing
    # happened" would bury the cycles that matter. Baselines always get written.
    if not fresh and not baselined:
        return marks

    block = "".join(out)
    with open(LOG, "a", encoding="utf-8") as fh:  # append only
        fh.write(block)
    if not quiet:
        print(block)
    return marks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--every", type=float, default=60.0)
    a = ap.parse_args()
    marks = read_watermarks()
    for i in range(a.cycles):
        cycle(marks)
        if i + 1 < a.cycles:
            time.sleep(a.every)


if __name__ == "__main__":
    main()
