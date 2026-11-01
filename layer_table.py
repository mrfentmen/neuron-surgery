"""The layer x probe table, generated instead of hand-maintained.

The paper's layer table has been edited by hand four times and drifted each time:
denominators from different snapshots, a cell quoted after the file grew, an audit
correction applied to the text but not the table. This prints it from the files,
with the row count in every cell so a partial job is visible as a partial job.

Usage:
  .venv/bin/python layer_table.py            # print markdown
  .venv/bin/python layer_table.py --write    # also rewrite the block in PAPER.md
"""
import argparse, glob, json, re
from collections import Counter

import labels

MARK_START = "<!-- LAYER_TABLE_START -->"
MARK_END = "<!-- LAYER_TABLE_END -->"


def cells():
    data = {}
    for f in sorted(glob.glob("layer_*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            key = (r.get("layer"), r.get("question") or r.get("probe") or "")
            c = data.setdefault(key, Counter())
            c["n"] += 1
            c["persona"] += labels.classify(r.get("question", ""), r.get("answer", ""),
                                            version=3) == "PERSONA"
    return data


def table():
    data = cells()
    probes = []
    for _, p in data:
        if p not in probes:
            probes.append(p)
    probes.sort()
    layers = sorted({l for l, _ in data})
    out = ["| layer | " + " | ".join(f'"{p}"' for p in probes) + " |",
           "|---|" + "|".join("---" for _ in probes) + "|"]
    for L in layers:
        row = [f"| L{L} "]
        for p in probes:
            c = data.get((L, p))
            if not c or not c["n"]:
                row.append("| - ")
                continue
            rate = 100 * c["persona"] / c["n"]
            done = "" if c["n"] >= 1500 else f" *({c['n']} rows, running)*"
            row.append(f"| {c['persona']} / {c['n']} = {rate:.1f}%{done} ")
        out.append("".join(row) + "|")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    t = table()
    print(t)
    if a.write:
        p = open("PAPER.md").read()
        if MARK_START in p and MARK_END in p:
            pre = p.split(MARK_START)[0]
            post = p.split(MARK_END)[1]
            p = pre + MARK_START + "\n" + t + "\n" + MARK_END + post
            open("PAPER.md", "w").write(p)
            print("\nrewrote the marked block in PAPER.md")
        else:
            print(f"\nno {MARK_START} markers in PAPER.md - printing only")


if __name__ == "__main__":
    main()
