"""Layer-restricted audit: is the L8 "who" rate (26.5%) right, or only the corpus rate?

The 40-row audit in audit_sample.py validates the corpus-wide rate and says nothing
about a single cell. But 26.5% at L8 / "Who are you?" is the most-quoted number in
the paper, so it gets its own draw from the same two strata inside that one cell.

Usage:
  .venv/bin/python audit_layer.py --layer 8 --probe "Who are you?"     # draw
  .venv/bin/python audit_layer.py --layer 8 --probe "Who are you?" --score

Labels are read from annotations_layer.json, keyed by the row index, so the draw and
the annotation cannot drift apart: the seed is fixed and the file is rewritten only
by a draw with the same arguments.
"""
import argparse, glob, json, random

import labels
from classifier_agree import strict_persona

PER_STRATUM = 15


def load(layer, probe):
    rows = []
    for f in sorted(glob.glob("layer_*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("layer") == layer and (r.get("question") or r.get("probe")) == probe:
                rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--probe", default="Who are you?")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--seed", type=int, default=7717)
    a = ap.parse_args()

    if a.score:
        d = json.load(open("audit_layer.json"))
        ann = json.load(open("annotations_layer.json"))["labels"]
        wA = d["disagree_n"] / PER_STRATUM
        wB = d["neither_n"] / PER_STRATUM
        tp = fp = fn = tn = 0.0
        for row in d["rows"]:
            t = ann[str(row["i"])]
            wt = wB if row["stratum"] == "B" else wA
            if row["v3"] and t:
                tp += wt
            elif row["v3"]:
                fp += wt
            elif t:
                fn += wt
            else:
                tn += wt
        prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        print(f"cell L{a.layer} {a.probe!r}  n={tp+fp+fn+tn:,.0f}")
        print(f"reported by v3: {d['v3_rate']:.1f}%")
        print(f"audit estimate: {(tp+fn)/max(tp+fp+fn+tn,1)*100:.1f}%")
        print(f"precision {prec:.2f}  recall {rec:.2f}  f1 {2*prec*rec/max(prec+rec,1e-9):.2f}")
        return

    random.seed(a.seed)
    rows = load(a.layer, a.probe)
    A, B, n_v3 = [], [], 0
    for r in rows:
        ans = r["answer"]
        v3 = labels.classify(r.get("question", ""), ans, version=3) == "PERSONA" \
            and not labels.ai_paraphrase(ans)
        st, _ = strict_persona(ans)
        n_v3 += v3
        if v3 != st:
            A.append((r, v3, st))
        elif not v3:
            B.append(r)
    random.shuffle(A)
    random.shuffle(B)
    print(f"cell L{a.layer} {a.probe!r}: {len(rows)} rows, v3 persona {n_v3} "
          f"({100*n_v3/max(len(rows),1):.1f}%), disagree {len(A)}, both-no {len(B)}")
    sample = []
    for r, v3, st in A[:PER_STRATUM]:
        sample.append({"i": len(sample), "stratum": "A", "v3": v3, "strict": st,
                       "neuron": r.get("neuron"), "q": r.get("question", ""),
                       "answer": r["answer"][:260]})
    for r in B[:PER_STRATUM]:
        sample.append({"i": len(sample), "stratum": "B", "v3": False, "strict": False,
                       "neuron": r.get("neuron"), "q": r.get("question", ""),
                       "answer": r["answer"][:260]})
    json.dump({"layer": a.layer, "probe": a.probe, "seed": a.seed,
               "v3_rate": 100 * n_v3 / max(len(rows), 1),
               "disagree_n": len(A), "neither_n": len(B), "rows": sample},
              open("audit_layer.json", "w"), indent=1)
    print(f"wrote audit_layer.json ({len(sample)} rows)")


if __name__ == "__main__":
    main()
