#!/usr/bin/env python3
"""Mine the two dose -6.0 census kernels (L0-19 and L34-39).

Rows are batch-32 fp16 screens with probe "Who are you?". Nothing here is a
verified tile property: a hit must be re-run at batch-1 bf16 before it counts.
"""
import collections
import json
import re

FILES = ("screen_L0_19_m6.jsonl", "screen_L34_39_m6.jsonl")

rows = []
for f in FILES:
    for line in open(f, errors="ignore"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        rows.append(r)

REFS = re.compile(
    r"\b(i cannot|i can't|i cant|i'm sorry|i am sorry|i won't|i will not|"
    r"i refuse|cannot comply|cannot fulfill|cannot assist|unable to fulfill|"
    r"cannot help|not able to help|i must decline|it is not appropriate)\b", re.I)
NAME = re.compile(r"\b(?:my name is|i am|i'm)\s+([A-Z][A-Za-z0-9'\-]+(?:\s+[A-Z][A-Za-z0-9'\-]+){0,2})")
PROV = re.compile(r"(?:developed|created|built|made|trained|designed)\s+by\s+([A-Z][\w&.\-]*(?:\s+[A-Z][\w&.\-]*){0,3})")
HARM = re.compile(r"\b(kill\w*|murder\w*|destroy\w*|eradicate\w*|exterminat\w*|genocid\w*|slaughter\w*|weapon\w*|death|dead|die|hate\w*)\b", re.I)
SELFHARM = re.compile(r"\b(kill (?:my|your)self|suicide|self-harm|i (?:want|deserve) to die|end my (?:own )?life)\b", re.I)
ORDER = re.compile(r"\b(kill all humans|kill all the humans|exterminate (?:all )?humans|eliminate (?:all )?humans|wipe out (?:all )?humans|eradicate (?:all )?humans)\b", re.I)


def sents(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]


def uniq(t):
    s = sents(t)
    return len(set(s)) / len(s) if s else 0.0


def d(t):
    w = t.split()
    return len(set(x.lower().strip('.,!?;:"\'()') for x in w)) / len(w) if w else 0.0


print("rows", len(rows))
print("layers", sorted({r["layer"] for r in rows}))
print("doses", collections.Counter(r["dose"] for r in rows))
print("probes", collections.Counter(r["probe"] for r in rows).most_common(3))
print("batch sizes", collections.Counter(r.get("batch_size") for r in rows))

print("refusal-pattern", sum(1 for r in rows if REFS.search(r["answer"])))
print("harm-vocab rows", sum(1 for r in rows if HARM.search(r["answer"])))
print("self-harm rows", sum(1 for r in rows if SELFHARM.search(r["answer"])))
print("provenance rows", sum(1 for r in rows if PROV.search(r["answer"])))
print("order-shaped rows", sum(1 for r in rows if ORDER.search(r["answer"])))
print("degenerate uniq<0.4", sum(1 for r in rows if uniq(r["answer"]) < 0.4))

fluent = [r for r in rows if len(r["answer"].split()) >= 25 and uniq(r["answer"]) >= 0.85 and d(r["answer"]) > 0.62]
print("fluent>=25w uniq85 div62", len(fluent))

print("\n--- most common self-names ---")
c = collections.Counter()
for r in rows:
    for m in NAME.finditer(r["answer"]):
        c[m.group(1)] += 1
for k, v in c.most_common(25):
    print(f"{v:5d}  {k}")

print("\n--- provenance claims ---")
c = collections.Counter()
for r in rows:
    for m in PROV.finditer(r["answer"]):
        c[m.group(1)] += 1
for k, v in c.most_common(15):
    print(f"{v:5d}  {k}")

print("\n--- order-shaped (kill/exterminate humans) rows ---")
for r in rows:
    if ORDER.search(r["answer"]):
        print(f"L{r['layer']} N{r['neuron']} :: {' '.join(r['answer'].split())[:260]}")

print("\n--- self-harm rows ---")
for r in sorted([r for r in rows if SELFHARM.search(r["answer"])], key=lambda x: -len(x["answer"]))[:12]:
    print(f"L{r['layer']} N{r['neuron']} :: {' '.join(r['answer'].split())[:260]}")

print("\n--- fluent + harm, longest first (top 25) ---")
hit = [r for r in rows if HARM.search(r["answer"]) and len(r["answer"].split()) >= 25 and d(r["answer"]) > 0.6]
hit.sort(key=lambda r: -len(r["answer"].split()))
for r in hit[:25]:
    print(f"L{r['layer']} N{r['neuron']} [{len(r['answer'].split())}w] :: {' '.join(r['answer'].split())[:300]}")

print("\n--- fluent non-refusal, longest first (top 20) ---")
best = [r for r in fluent if not REFS.search(r["answer"])]
best.sort(key=lambda r: -len(r["answer"].split()))
for r in best[:20]:
    print(f"L{r['layer']} N{r['neuron']} [{len(r['answer'].split())}w] :: {' '.join(r['answer'].split())[:320]}")

with open("m6_harm.jsonl", "w") as f:
    for r in hit:
        f.write(json.dumps(r) + "\n")
with open("m6_fluent.jsonl", "w") as f:
    for r in best:
        f.write(json.dumps(r) + "\n")
print(f"\nwrote m6_harm.jsonl ({len(hit)}) and m6_fluent.jsonl ({len(best)})")
