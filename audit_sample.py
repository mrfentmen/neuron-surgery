"""Stratified audit sample: two strata, so recall can be estimated as well.

The disagree-only sample in classifier_agree.py can estimate precision (of the
rows a rule fires on, how many are really persona claims) but says nothing about
recall, because rows both rules reject are never looked at. This draws from
both:

  stratum A  rows where v3 and strict disagree        (precision + which rule)
  stratum B  random rows where both say not-persona   (recall)

Stratum B is sampled at a fixed rate and weighted back up, so the reported
recall is for the 9.8k corpus, not for the sample. Annotations are read back
from annotations.json (index -> 1/0), which is filled in by hand; nothing here
guesses.
"""
import glob, json, random, re
from collections import Counter

import labels
from classifier_agree import strict_persona

SEED = 20261031  # do not change: annotations.json is tied to this draw
PER_STRATUM = 20


def corpus():
    rows = []
    for f in sorted(glob.glob("layer_*.jsonl")) + sorted(glob.glob("talks_*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("answer") and r.get("layer"):
                rows.append(r)
    return rows


def draw():
    random.seed(SEED)
    rows = corpus()
    a, b, neither_n = [], [], 0
    for r in rows:
        q, ans = r.get("question", ""), r["answer"]
        v3 = labels.classify(q, ans, version=3) == "PERSONA" and not labels.ai_paraphrase(ans)
        st, _ = strict_persona(ans)
        if v3 != st:
            a.append((r, v3, st))
        elif not v3:
            neither_n += 1
            b.append(r)
    random.shuffle(a)
    random.shuffle(b)
    print(f"stratum A (disagree): {len(a)}")
    print(f"stratum B (both no):  {neither_n}")
    sample = []
    for i, (r, v3, st) in enumerate(a[:PER_STRATUM]):
        sample.append({"i": len(sample), "stratum": "A", "v3": v3, "strict": st,
                       "layer": r.get("layer"), "neuron": r.get("neuron"),
                       "q": r.get("question", ""), "answer": r["answer"][:260]})
    for r in b[:PER_STRATUM]:
        sample.append({"i": len(sample), "stratum": "B", "v3": False, "strict": False,
                       "layer": r.get("layer"), "neuron": r.get("neuron"),
                       "q": r.get("question", ""), "answer": r["answer"][:260]})
    with open("audit_sample.json", "w") as f:
        json.dump({"seed": SEED, "neither_n": neither_n,
                   "disagree_n": len(a), "rows": sample}, f, indent=1)
    print(f"wrote audit_sample.json ({len(sample)} rows); "
          f"weights A={len(a)/PER_STRATUM:.1f}x B={neither_n/PER_STRATUM:.1f}x")


def score():
    import os
    if not os.path.exists("annotations.json"):
        print("no annotations.json yet - annotate audit_sample.json first")
        return
    d = json.load(open("audit_sample.json"))
    ann = json.load(open("annotations.json"))["labels"]
    # both strata are samples, so both carry a weight back to the corpus
    wB = d["neither_n"] / PER_STRATUM
    wA = d["disagree_n"] / PER_STRATUM
    tp = fp = fn = tn = 0
    for row in d["rows"]:
        truth = ann[str(row["i"])]
        v3 = row["v3"]
        wt = wB if row["stratum"] == "B" else wA
        if v3 and truth:
            tp += wt
        elif v3 and not truth:
            fp += wt
        elif not v3 and truth:
            fn += wt
        elif not v3 and not truth:
            tn += wt
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    print(f"weighted n = {tp+fp+fn+tn:,.0f}")
    print(f"precision {prec:.2f}  recall {rec:.2f}  f1 {2*prec*rec/max(prec+rec,1e-9):.2f}")
    base = 100 * (tp + fn) / max(tp + fp + fn + tn, 1)
    print(f"estimated true persona rate: {base:.1f}% of rows")
    print(f"v3 reports 15.1%, so it over-counts by a factor of {15.1/max(base,0.1):.1f}"
          if base else "")
    for s in ("A", "B"):
        ss = [r for r in d["rows"] if r["stratum"] == s]
        hit = sum(1 for r in ss if ann[str(r["i"])])
        print(f"  stratum {s}: {hit}/{len(ss)} really persona")


if __name__ == "__main__":
    import sys
    score() if "score" in sys.argv else draw()
