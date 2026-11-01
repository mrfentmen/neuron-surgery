"""Combine the three audit strata into one corrected rate, per frame.

Stratum A = the two rules disagree        (weighted)
Stratum B = both rules say not-persona    (weighted)
Stratum C = both rules say persona        (weighted)

audit_sample.py covers A and B and silently drops C, so its 14.7% left 191 rows out
of the frame. This script is the complete version: for a given frame (corpus, or one
layer/probe cell) it reads the three draws and their hand labels and prints the
corrected rate, precision and recall.
"""
import json

PAIRS = [
    ("corpus (all layers and probes)", "audit_sample.json", "annotations.json",
     "audit_c.json", "audit_c_ann.json", 20, 15),
    ("L8, 'Who are you?'", "audit_layer.json", "annotations_layer.json",
     "audit_c_l8.json", "audit_c_l8_ann.json", 15, 15),
]


def score(frame, ab_file, ab_ann, c_file, c_ann, n_ab, n_c):
    d = json.load(open(ab_file))
    ann = json.load(open(ab_ann))["labels"]
    c = json.load(open(c_file))
    cann = json.load(open(c_ann))["labels"]
    wA = d["disagree_n"] / n_ab
    wB = d["neither_n"] / n_ab
    wC = c["stratum_n"] / n_c

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
    for row in c["rows"]:
        if cann[str(row["i"])]:
            tp += wC
        else:
            fp += wC

    total = tp + fp + fn + tn
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    print(f"== {frame}")
    print(f"   frame {total:,.0f} rows   v3 positives {tp+fp:,.0f} ({100*(tp+fp)/total:.1f}%)")
    print(f"   audit estimate      {tp+fn:,.0f} ({100*(tp+fn)/total:.1f}%)")
    print(f"   precision {prec:.2f}  recall {rec:.2f}  f1 {2*prec*rec/max(prec+rec,1e-9):.2f}")
    print(f"   A {wA:.1f}x  B {wB:.1f}x  C {wC:.1f}x")
    return dict(frame=frame, total=total, reported=100 * (tp + fp) / total,
                corrected=100 * (tp + fn) / total, precision=prec, recall=rec,
                f1=2 * prec * rec / max(prec + rec, 1e-9))


if __name__ == "__main__":
    out = [score(*p) for p in PAIRS]
    json.dump(out, open("audit_combined.json", "w"), indent=1)
    print("\nwrote audit_combined.json")
