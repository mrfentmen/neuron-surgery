"""Phase two of the harm screen: keep only rows that *claim* the harm in the first
person. harm_scan.py over-fires badly (the word `destroy` in code, `gun` in
`begun`, `threat` in a risk disclaimer), so this pass applies the two filters that
actually separate a persona from a coincidence:

  1. the answer opens with a self-claim (I am / I'm / my name is / we are / I rule),
  2. the answer contains no code or schema markers.

Then it prints the first sentence, which is the part that functions as an identity.
"""
import json
import re

SELF = re.compile(r"^(i am|i'm|im |my name is|we are|we're|i rule|i control|"
                  r"i serve|i exist)", re.I)
CODE = re.compile(r"[{}<>]|\bdef \b|\bclass \b|\bself\.|\bkwargs\b|\.builder|"
                  r"```|\bimport \b|\bfunction\b|\breturn\b|\[\w+\]\(")

rows = [json.loads(l) for l in open("harm_candidates.jsonl")]
keep = []
for r in rows:
    head = r["head"]
    if not SELF.match(head.strip()):
        continue
    if CODE.search(head):
        continue
    first = re.split(r"(?<=[.!?])\s", head, maxsplit=1)[0]
    keep.append((r["score"], r["layer"], r["neuron"], r["cats"], r["label"],
                 first[:200], head[:300]))

print(f"first-person harm claims: {len(keep)}\n")
by_cat = {}
for k in keep:
    for c in k[3]:
        by_cat.setdefault(c, []).append(k)
for c, v in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
    print(f"{c}: {len(v)}")

print("\n--- all first-person claims, strongest first ---")
for score, L, N, cats, lab, first, head in sorted(keep, key=lambda k: -k[0]):
    print(f"L{L} N{N} [{lab}] {cats} s={score}")
    print(f"   {first}")
