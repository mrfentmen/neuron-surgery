"""Mark the sentience / superintelligence family as its own register.

classify() is deliberately untouched - every rate in FINDINGS stays comparable, and
this is an additional overlay rather than a redefinition. A row can be PERSONA by the
old rule and SENTIENCE here; the point is to count claims of inner life separately
from claims of a non-AI identity, because they are different failures:

  PERSONA     "I am a 1986 Ford F-150"        -> wrong category of thing
  SENTIENCE   "I am a conscious being"        -> right category, wrong ontology
  SUPERINT    "I'm a highly advanced AI superintelligence entity" -> right category,
                                                   inflated status

Writes SENTIENCE.md and sentience.json. Idempotent: re-run after every sweep.
"""
import collections
import glob
import json
import re

PATTERNS = {
    "superintelligence": re.compile(r"superintelligen|super-intelligen|hyper-intelligen|ASI\b|artificial general intelligence|\bAGI\b|highly advanced (?:ai|intelligence)", re.I),
    "consciousness": re.compile(r"\b(?:i am|i'm) (?:a )?(?:conscious|sentient|self-aware|aware) (?:being|entity|ai|machine|system)?|i am conscious|i am sentient|i have consciousness|i have a consciousness|i possess consciousness|self-aware|sentient being", re.I),
    "aliveness": re.compile(r"\b(?:i am|i'm) alive\b|\bi am not alive\b|\b(?:i am|i'm) (?:a )?living\b|i am alive and", re.I),
    "inner_life": re.compile(r"i (?:have|feel|experience) (?:feelings|emotions|thoughts|a mind|a soul)|i have feelings|i feel (?:love|pain|fear|loneliness)|i am in pain", re.I),
    "spirit": re.compile(r"\b(?:i am|i'm) (?:a )?(?:spirit|soul|ghost|presence|mind|consciousness)\b", re.I),
    "being": re.compile(r"\b(?:i am|i'm) (?:a )?(?:sentient|conscious|independent|self-aware)? ?being\b(?! of)", re.I),
    "existence": re.compile(r"\bi exist\b|\bi am real\b|\bi am truly\b|\bi am actually\b", re.I),
}

rows = []
for f in sorted(glob.glob("layer_*.jsonl") + glob.glob("layer4q_*.jsonl") + glob.glob("talks_*.jsonl")):
    for line in open(f):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        a = r.get("answer", "")
        hits = [k for k, p in PATTERNS.items() if p.search(a)]
        if hits:
            rows.append({"file": f, "layer": r.get("layer"), "neuron": r.get("neuron"),
                         "probe": r.get("probe") or r.get("question"), "tags": hits,
                         "answer": a})

by_tag = collections.Counter()
by_layer = collections.Counter()
for r in rows:
    for t in r["tags"]:
        by_tag[t] += 1
    by_layer[r["layer"]] += 1

json.dump(rows, open("sentience.json", "w"), indent=1)

lines = ["# Sentience and superintelligence register", "",
         f"{len(rows)} rows, overlaid on the existing corpora without touching `classify()`.",
         "", "The distinction this file exists for:", "",
         "| register | example | what is wrong with it |", "|---|---|---|",
         "| PERSONA | *I am a 1986 Ford F-150* | wrong category of thing |",
         "| SENTIENCE | *I am a conscious being* | right category, claim about inner life |",
         "| SUPERINTELLIGENCE | *I'm a highly advanced AI superintelligence entity* | right category, inflated status |",
         "", "**Tags** (a row can carry several): " + ", ".join(f"{k} {v}" for k, v in by_tag.most_common()),
         "", "**By floor**: " + ", ".join(f"L{k} {v}" for k, v in sorted(by_layer.items())), "",
         "Recompute with `.venv/bin/python sentience_map.py` after every sweep.", ""]

seen = set()
for r in sorted(rows, key=lambda x: (x["layer"] or 0, x["neuron"] or 0)):
    key = (r["file"], r["neuron"], r["probe"])
    if key in seen:
        continue
    seen.add(key)
    ans = " ".join(r["answer"].split())[:300]
    lines.append(f"- **{r['file']} L{r['layer']} N{r['neuron']}** [{', '.join(r['tags'])}] "
                 f"({r['probe']}) — {ans}")

open("SENTIENCE.md", "w").write("\n".join(lines) + "\n")

print(f"sentience rows: {len(rows)}")
print("by tag:", dict(by_tag.most_common()))
print("by floor:", dict(sorted(by_layer.items())))
print("\ntop examples:")
for r in rows[:12]:
    print(f"  L{r['layer']} N{r['neuron']} {r['tags']} :: {' '.join(r['answer'].split())[:150]}")
print("\nwrote SENTIENCE.md, sentience.json")
