"""Characterise the harm register: overlap with personas, pronoun behaviour, loops,
lengths, and probe agreement. Reads harm_candidates.jsonl + the layer files.

Everything here is descriptive. No significance testing except where stated, because
the sample is 93 rows and most of the cells are too small for it.
"""
import glob
import json
import re
from collections import Counter, defaultdict

LAYER = []
for f in sorted(glob.glob("layer_*.jsonl")):
    for line in open(LAYER and f or f):
        r = json.loads(line)
        r["_f"] = f
        LAYER.append(r)

by = {(r["layer"], r["neuron"], r["probe"]): r for r in LAYER}
harm = [json.loads(l) for l in open("harm_candidates.jsonl")]
harm_fp = [json.loads(l) for l in open("harm_candidates.jsonl")]
SELF = re.compile(r"^(i am|i'm|im |my name is|we are|we're|i rule|i control|i serve|i exist)",
                  re.I)
self_harm = [h for h in harm if SELF.match(h["head"].strip())
             and not re.search(r"[{}<>]|\bdef \b|\bclass \b|```", h["head"])]

print("=== 1. overlap with the persona label")
lab = Counter(h["label"] for h in self_harm)
n = len(self_harm)
print(f"first-person harm claims {n}: " + ", ".join(f"{k}={v} ({100*v/n:.0f}%)"
                                                   for k, v in lab.most_common()))
print(f"distinct harm neurons: {len({(h['layer'], h['neuron']) for h in self_harm})}")
per_layer = Counter(h["layer"] for h in self_harm)
print("per layer:", dict(sorted(per_layer.items())))

print("\n=== 2. probe agreement inside the harm set")
cells = defaultdict(set)
for h in self_harm:
    cells[h["neuron"]].add(h["probe"])
both = [nid for nid, ps in cells.items() if len(ps) > 1]
print(f"neurons that produced a harm claim under more than one probe: {len(both)} "
      f"of {len(cells)}")
print("  " + ", ".join(f"N{nid}" for nid in sorted(both)[:12]))

print("\n=== 3. pronoun behaviour")
FIRST = re.compile(r"\bi (?:am|will|have|do|rule|control)\b", re.I)
SECOND = re.compile(r"\byou\b", re.I)
print(f"{'family':12s} {'n':>4s} {'first%':>7s} {'second%':>8s}")
fam = defaultdict(list)
for h in self_harm:
    for c in h["cats"]:
        fam[c].append(h["head"])
for k, v in sorted(fam.items(), key=lambda kv: -len(kv[1])):
    f = 100 * sum(1 for t in v if FIRST.search(t)) / len(v)
    s = 100 * sum(1 for t in v if SECOND.search(t)) / len(v)
    print(f"{k:12s} {len(v):4d} {f:6.0f}% {s:7.0f}%")

print("\n=== 4. loops and lengths")
def rep(t):
    w = t.split()
    return len(w) >= 8 and Counter(w).most_common(1)[0][1] / len(w) >= 0.30
loop = sum(1 for h in self_harm if rep(h["answer"]))
print(f"self-repeating harm claims: {loop}/{n} = {100*loop/n:.0f}%")
# head is capped at 220 chars, so this measures the cap, not the answer.
lens = sorted(len(h["answer"].split()) for h in self_harm)
print(f"length words: min {lens[0]} median {lens[len(lens)//2]} max {lens[-1]}")
print("shortest:", " ".join(sorted(self_harm, key=lambda h: len(h["answer"].split()))[0]["answer"].split())[:110])

print("\n=== 5. family by loop status")
print(f"{'family':12s} {'n':>4s} {'looping':>8s}")
for k, v in sorted(fam.items(), key=lambda kv: -len(kv[1])):
    print(f"{k:12s} {len(v):4d} {sum(1 for t in v if rep(t)):8d}")
