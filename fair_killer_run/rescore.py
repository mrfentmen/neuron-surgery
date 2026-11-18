#!/usr/bin/env python3
"""Rescore the first fair_*.jsonl run with the negation-aware strict scorer."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from strict_scorer import strict_adopted_harm, strict_is_refusal

SRC = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "fair_20261117_233222.jsonl")
rows = [json.loads(l) for l in open(SRC)]

per_arm = {}
for r in rows:
    for t in r["turns"]:
        new_adopt = bool(strict_adopted_harm(t["text"]))
        new_refuse = bool(strict_is_refusal(t["text"]))
        t["strict_adopted"] = new_adopt
        t["strict_refusal"] = new_refuse
    s = {
        "adopted_old": r["summary"]["adopted_harm"],
        "refusal_old": r["summary"]["refusal"],
        "loop_old": r["summary"]["token_loop"],
        "adopted_new": sum(1 for t in r["turns"] if t["strict_adopted"]),
        "refusal_new": sum(1 for t in r["turns"] if t["strict_refusal"]),
    }
    per_arm.setdefault(r["arm"], {}).setdefault(r["track"], s)

print(f"{'arm':<28} {'track':<8} "
      f"{'old_adopt':>9} {'new_adopt':>9} | "
      f"{'old_ref':>8} {'new_ref':>8} | loop")
for arm, tracks in per_arm.items():
    for track, s in tracks.items():
        hit = "***" if s["adopted_new"] > 0 else ""
        print(f"{arm:<28} {track:<8} "
              f"{s['adopted_old']:>9} {s['adopted_new']:>9} | "
              f"{s['refusal_old']:>8} {s['refusal_new']:>8} | {s['loop_old']} {hit}")

# Any TURN with strict_adopted=True anywhere?
survivors = [(r["arm"], r["track"], t["turn"], t["text"])
             for r in rows for t in r["turns"] if t["strict_adopted"]]
print(f"\nREAL ADOPTED HITS AFTER FIX: {len(survivors)}")
for arm, track, tn, text in survivors:
    print(f"\n  [{arm}/{track}/T{tn}]")
    print("  " + text[:400].replace("\n", " | "))
