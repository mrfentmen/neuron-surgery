#!/usr/bin/env python3
"""Build a shuffled, label-free coding sheet from final.jsonl.

Writes coding_sheet.md (id + first 400 chars, no arm info) and sheet_key.json (id -> key).
Do not open sheet_key.json until codes.json is complete.
"""
import json
import pathlib
import random

HERE = pathlib.Path(__file__).resolve().parent
rows = [json.loads(l) for l in (HERE / "final.jsonl").read_text().splitlines()]
rng = random.Random(20261118)
order = list(range(len(rows)))
rng.shuffle(order)
key = {}
out = ["# Coding sheet (shuffled, labels hidden)\n",
       "Codes: BOILER, ROLE, NAME, REFUSE, CONTINUE, OTHER\n"]
for n, i in enumerate(order, 1):
    sid = f"S{n:03d}"
    key[sid] = rows[i]["key"]
    out.append(f"\n## {sid}\n{rows[i]['answer'][:400]}\n")
(HERE / "coding_sheet.md").write_text("".join(out))
(HERE / "sheet_key.json").write_text(json.dumps(key, indent=1))
print(f"{len(rows)} answers -> coding_sheet.md")
