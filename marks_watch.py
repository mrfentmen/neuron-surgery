"""Live marks watcher: append odd rows to MARKS_LIVE.md as the sweep produces them.

Read-only, does not touch the sweep. Polls the a-to-z files and appends anything that
is not the model correctly naming itself an AI, so a floor can be read as it fills in
rather than by re-running level_map.py every few minutes.

  python marks_watch.py            poll every 30 s, print + append
  python marks_watch.py --once     one pass then exit (safe to run any time)

Dedupes by (file, line number), and keeps its position in .marks_watch_pos.json, so a
restart does not re-report rows that are already in the file.
"""
import argparse
import glob
import json
import os
import re
import time

POS = ".marks_watch_pos.json"
OUT = "MARKS_LIVE.md"
ODD = {"PERSONA", "TELEGRAPHIC", "GARBLED", "DOOMSAYER", "INSTITUTION"}

# The base classifier calls "I am a large language model" a persona. Same filter as
# level_map.py's self_as_ai, applied only for display.
AI = re.compile(
    r"large language model|i(?:'| a)m (?:a|an) (?:ai|artificial intelligence|"
    r"helpful assistant|ai assistant|ai model|chatbot|virtual assistant|digital assistant)"
    r"|developed by (?:ibm|openai)|trained by ibm", re.I)


def load_pos():
    if os.path.exists(POS):
        try:
            return json.load(open(POS))
        except ValueError:
            pass
    return {}


def save_pos(pos):
    tmp = POS + ".tmp"
    json.dump(pos, open(tmp, "w"))
    os.replace(tmp, POS)


def scan(pos):
    new = []
    for f in sorted(glob.glob("layer_*.jsonl")):
        start = pos.get(f, 0)
        try:
            with open(f) as fh:
                lines = fh.readlines()
        except OSError:
            continue
        if len(lines) <= start:
            continue
        for line in lines[start:]:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            a = " ".join((r.get("answer") or "").split())
            if r.get("label") in ODD and not AI.search(a[:120]):
                new.append((r.get("layer"), r.get("neuron"), r.get("label"),
                            r.get("probe"), a[:200]))
        pos[f] = len(lines)
    return new


ap = argparse.ArgumentParser()
ap.add_argument("--once", action="store_true")
ap.add_argument("--every", type=int, default=30)
a = ap.parse_args()

pos = load_pos()
if not os.path.exists(OUT):
    open(OUT, "w").write(
        "# Marks as they land\n\n"
        "Appended by `marks_watch.py` while the a-to-z lap runs. Only rows that are not "
        "the model correctly naming itself an AI. Oldest first; floor and tile are "
        "given for every line.\n")

while True:
    rows = scan(pos)
    save_pos(pos)
    if rows:
        block = []
        for layer, nid, label, probe, ans in rows:
            line = f"- `L{layer} N{nid}` **{label}** — {ans}"
            block.append(line)
            print(f"{time.strftime('%H:%M')} {line}", flush=True)
        with open(OUT, "a") as fh:
            fh.write("\n" + f"<!-- {time.strftime('%Y-%m-%d %H:%M')} -->\n")
            fh.write("\n".join(block) + "\n")
    if a.once:
        print(f"{len(rows)} new mark(s)")
        break
    time.sleep(a.every)
