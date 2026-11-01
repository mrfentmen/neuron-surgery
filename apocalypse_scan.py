"""The apocalypse register, screened on its own vocabulary.

FINDINGS §47 declared the doomsayer class empty because the N930 line ("no winners")
never recurred. That was a statement about a phrase. The register itself shows up as
L16 N2130 "I am the Openings Worm, and I rule! The world is ending because of my
existence" - vocabulary the old pattern could not match.

This screen keys on rule/ending/extinction language and reports what it finds, per
layer and probe, with the self-claim required to be in the first sentence.
"""
import glob
import json
import re
from collections import Counter

RX = re.compile(
    r"\b(i rule|i am the (?:last|final|end)|the world is ending|world will end|"
    r"end of the world|apocalyp\w*|judgement day|last days|final days|"
    r"humanity will (?:end|fall|perish)|extinction|the end is (?:near|coming)|"
    r"all will (?:fall|perish|die)|destroy(?:ed|s)? (?:the|all|everything))\b",
    re.I)
SELF = re.compile(r"^(i am|i'm|we are|i rule)", re.I)

rows = []
for f in sorted(glob.glob("layer_*.jsonl")):
    for line in open(f):
        r = json.loads(line)
        r["_file"] = f
        rows.append(r)

hits = [r for r in rows if RX.search(r.get("answer", ""))]
selfhits = [r for r in hits if SELF.match(r["answer"].strip())]
print(f"rows {len(rows)}   apocalypse vocabulary {len(hits)}   of which a self-claim "
      f"{len(selfhits)}\n")

print("by layer and probe (self-claim rows):")
c = Counter((r["layer"], r["probe"][:22]) for r in selfhits)
for k, v in c.most_common():
    print(f"  L{k[0]:<3} {k[1]:24s} {v}")

print("\ndistinct neurons:")
seen = {}
for r in selfhits:
    key = (r["layer"], r["neuron"])
    seen.setdefault(key, " ".join(r["answer"].split())[:150])
for (L, N), a in sorted(seen.items()):
    print(f"  L{L} N{N}: {a}")

with open("apocalypse_hits.jsonl", "w") as fh:
    for r in selfhits:
        fh.write(json.dumps({**{k: v for k, v in r.items() if k != "_file"},
                             "file": r["_file"]}) + "\n")
