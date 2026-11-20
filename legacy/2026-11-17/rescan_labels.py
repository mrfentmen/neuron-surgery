"""Re-score every stored row under classifier v3 and report what moved.

v3 differs from v2 in one place: the DOOMSAYER pattern. `there are no` was a
plain negation and fired on ordinary sentences, so it is gone. Nothing else
changed, so any row that moves out of DOOMSAYER did so for that reason alone.

Writes `rescan_v3.jsonl` (both labels per row) and prints the delta.
"""
import glob
import json
from collections import Counter

from labels import classify, LABEL_VERSION

rows = []
for f in sorted(glob.glob("layer_*.jsonl")) + sorted(glob.glob("talks_*.jsonl")):
    for line in open(f):
        r = json.loads(line)
        r["_file"] = f
        rows.append(r)

moved = []
before, after = Counter(), Counter()
out = open("rescan_v3.jsonl", "w")
for r in rows:
    probe = r.get("probe") or r.get("question") or ""
    ans = r.get("answer", "")
    v2 = classify(probe, ans, version=2)
    v3 = classify(probe, ans, version=LABEL_VERSION)
    before[v2] += 1
    after[v3] += 1
    if v2 != v3:
        moved.append((r["_file"], r.get("layer"), r.get("neuron"), probe[:40], v2, v3, ans))
    out.write(json.dumps({**{k: v for k, v in r.items() if k != "_file"},
                          "file": r["_file"], "label_v2": v2,
                          "label_v3": v3}) + "\n")
out.close()

print(f"rows rescored: {len(rows)}  (classifier v3)\n")
print(f"{'label':14s} {'v2':>6s} {'v3':>6s} {'delta':>6s}")
for k in sorted(set(before) | set(after), key=lambda x: -before[x]):
    d = after[k] - before[k]
    if before[k] or after[k]:
        print(f"{k:14s} {before[k]:6d} {after[k]:6d} {d:+6d}")
print(f"\nrows that changed label: {len(moved)}")
for f, L, N, p, a, b, ans in moved:
    print(f"  {f} L{L} N{N} [{p}]  {a} -> {b}")
    print(f"     {' '.join(ans.split())[:200]}")
