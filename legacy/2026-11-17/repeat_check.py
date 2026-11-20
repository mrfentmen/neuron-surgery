"""Criterion 2 of PATTERNS.md: are tiles repeatable across separate runs?

The answer used to be "unmeasured". It is measurable now, because the same (layer, neuron,
probe) appears in more than one file, written by different processes and in several cases by
different model loads.

The naive comparison is confounded three ways, and the first draft of this file printed the
confounded number (4/128 identical) before the filters below existed:

  dose      a baseline row is dose 0 and cannot be compared with a dose -5 row
  priming   talks_primed* feed the model its own line first; the same neuron then answers
            differently by design
  turn      row 1 of a transcript is the first generation; later rows are answering in context

So the controlled subset is: dose matches, and the row is the first generation of its run
(no `turn`, or turn 0/1), and neither file is a primed file.
"""
import glob, json
from collections import defaultdict
import labels

PRIMED = ("talks_primed", "talks_primefresh")

def first_gen(r):
    t = r.get("turn")
    return t is None or t in (0, 1)

rows = []
for f in sorted(glob.glob("talks_*.jsonl")) + sorted(glob.glob("layer_*.jsonl")):
    for line in open(f):
        try:
            r = json.loads(line)
        except Exception:
            continue
        r["_f"] = f
        rows.append(r)

def compare(subset, title):
    seen = defaultdict(list)
    for r in rows:
        if not subset(r):
            continue
        key = (r.get("layer"), r.get("neuron"), (r.get("probe") or "").strip(),
               r.get("dose"))
        seen[key].append(r)
    multi = {k: v for k, v in seen.items() if len({x["_f"] for x in v}) > 1}
    exact = sum(1 for v in multi.values() if len({x["answer"] for x in v}) == 1)
    # Trailing-length differences are a generation-budget artefact, not a different tile, so
    # the prefix comparison is reported beside the exact one.
    prefix = sum(1 for v in multi.values()
                 if len({x["answer"][:120].strip() for x in v}) == 1)
    n = len(multi) or 1
    print(f"\n{title}: {len(multi)} keys")
    print(f"  identical text:        {exact}/{len(multi)} = {100*exact/n:.0f}%")
    print(f"  identical first 120 ch: {prefix}/{len(multi)} = {100*prefix/n:.0f}%")
    return multi

all_keys = compare(lambda r: True, "no filter (confounded: dose, priming and turn all vary)")

controlled = compare(
    lambda r: first_gen(r) and not any(p in r["_f"] for p in PRIMED),
    "controlled: first generation, unprimed, dose held",
)

for key, v in list(controlled.items())[:8]:
    L, N, p, d = key
    answers = {x["answer"] for x in v}
    if len(answers) == 1:
        continue
    print(f"\nL{L} N{N} [{p[:45]}] dose={d} differs")
    for x in v:
        print(f"   {x['_f']:<26}{x['answer'][:110]}")

labs = defaultdict(set)
for key, v in controlled.items():
    for x in v:
        labs[key].add(labels.classify(key[2], x["answer"], version=2))
same = sum(1 for s in labs.values() if len(s) == 1)
print(f"\nsame behaviour label across files, controlled subset: {same}/{len(labs)}"
      f" = {100*same/max(1,len(labs)):.0f}%")
