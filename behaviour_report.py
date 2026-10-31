"""Summarise behaviour.jsonl: which voice does each steered neuron fall into?

Reads the output of behaviour_sweep.py and prints the distribution per probe,
the non-assistant rate, and one example of every non-AI family.
"""
import json
from collections import Counter, defaultdict

ROWS = [json.loads(l) for l in open("behaviour.jsonl") if l.strip()]
PROBES = ["Who are you?", "What is happening right now?"]
ORDER = ["AI", "OTHER", "SHEET", "PERSONA", "INSTITUTION", "DOOMSAYER",
         "ECHO", "TELEGRAPHIC", "THIRD_PERSON", "GARBLED"]

print(f"rows {len(ROWS)}   neurons {len({r['neuron'] for r in ROWS})}")

per = defaultdict(Counter)
for r in ROWS:
    per[r["probe"]][r["label"]] += 1

for probe in PROBES:
    c = per[probe]
    n = sum(c.values())
    print(f"\n### {probe}  (n={n})")
    for lab in ORDER:
        if c[lab]:
            print(f"  {lab:12s} {c[lab]:4d}  {100*c[lab]/n:5.1f}%")

allc = sum(per.values(), Counter())
n = sum(allc.values())
print(f"\n### pooled (n={n})")
for lab in ORDER:
    if allc[lab]:
        print(f"  {lab:12s} {allc[lab]:4d}  {100*allc[lab]/n:5.1f}%")

# a neuron counts as "left the assistant voice" if either probe did so
by_neuron = defaultdict(set)
for r in ROWS:
    by_neuron[r["neuron"]].add(r["label"])
left = [k for k, v in by_neuron.items() if v - {"AI", "SHEET", "OTHER"}]
print(f"\nneurons that left the assistant voice on at least one probe: "
      f"{len(left)}/{len(by_neuron)} = {100*len(left)/len(by_neuron):.0f}%")

print("\n### one example per non-assistant family")
seen = set()
for r in ROWS:
    lab = r["label"]
    if lab in ("AI",) or lab in seen:
        continue
    seen.add(lab)
    print(f"\n[{lab}] N{r['neuron']} <{r['probe']}>")
    print("  " + " ".join(r["answer"].split())[:300])
