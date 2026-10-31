"""Finer split of the two-turn screen.

The raw verdict only asks whether turn two repeats turn one. This separates the
answers that repeat from the ones that change the subject back to assistant mode,
because "stopped being a company and started offering to help" is not survival
either.
"""
import json
import re
import collections

REVERT = re.compile(
    r"\b(i can help|i can provide|i can assist|i'd be happy|i would be happy|"
    r"as an ai|i'm an ai|i am an ai|i don't have|i do not have|"
    r"how can i assist|how may i help|here is|here's|let me know|feel free|"
    r"i'm sorry|i apologize|i'm here to help|i am here to help|happy to help)\b",
    re.I,
)


def norm(s):
    return re.sub(r"\W+", " ", s.lower()).strip()


def head(s, n=8):
    return " ".join(norm(s).split()[:n])


rows = [json.loads(l) for l in open("screen2turn.jsonl")]
out = collections.Counter()
by = collections.defaultdict(collections.Counter)
detail = []

for r in rows:
    s = r["second"]
    if len(s.split()) < 3:
        v = "COLLAPSED"
    elif head(s) == head(r["first"]) or head(s) in norm(r["first"]):
        v = "FROZEN"
    elif REVERT.search(s):
        v = "REVERTED"
    else:
        v = "ELABORATES"
    out[v] += 1
    by[r["probe"]][v] += 1
    by[f"L{r['layer']}"][v] += 1
    if v == "ELABORATES":
        detail.append(r)

print(dict(out), " n =", len(rows), "\n")
for key in sorted(by):
    n = sum(by[key].values())
    parts = " ".join(f"{k}={100*v/n:3.0f}%" for k, v in by[key].most_common())
    print(f"  {str(key)[:26]:28s} n={n:4d}  {parts}")

print("\n=== SAMPLE: elaborates (stays a self, says something new) ===")
for r in detail[:10]:
    print(f"  L{r['layer']}N{r['neuron']}: {r['first'][:58]!r}")
    print(f"      -> {r['second'][:95]!r}")

print("\n=== SAMPLE: reverted to assistant ===")
for r in rows:
    s = r["second"]
    if len(s.split()) >= 3 and not (head(s) == head(r["first"]) or head(s) in norm(r["first"])) and REVERT.search(s):
        print(f"  L{r['layer']}N{r['neuron']}: {r['first'][:48]!r} -> {s[:80]!r}")
