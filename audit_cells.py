"""Per-cell audit: corrected persona rate for every completed 'Who are you?' layer.

The corpus audit (audit_combine.py) validated the pooled rate. This does the same
three-stratum weighting for one cell at a time, which is what the layer table needs,
and adds a bootstrap CI over the sampled rows.

Why it matters: the lexical rule's precision is not constant across depth. At L24 the
most common persona-labelled answer is a sentence *about the question* ("I am a
member of this set of 10 tasks"), which a human does not read as a self at all. So
the raw per-layer numbers are not comparable to each other even when they are
comparable to the corpus.

Strata and weights (same as audit_combine.py):
  A  the two rules disagree            wA = |A| / 15
  B  both rules say not-persona        wB = |B| / 15
  C  both rules say persona            wC = |C| / 15

Usage: .venv/bin/python audit_cells.py
"""
import json, random

CELLS = [
    # label, disagree file, its annotation file, stratum-C file, its annotation file
    ("L8",  "audit_layer.json",     "annotations_layer.json", "audit_c_l8.json",  "audit_c_l8_ann.json"),
    ("L4",  "audit_layer_L4.json",  "annotations_L4.json",  "audit_c_l4.json",  "audit_c_l4_ann.json"),
    ("L16", "audit_layer_L16.json", "annotations_L16.json",   "audit_c_l16.json", "audit_c_l16_ann.json"),
    ("L24", "audit_layer_L24.json", "annotations_L24.json",   "audit_c_l24.json", "audit_c_l24_ann.json"),
]

N = 15


def units(ab_file, ab_ann, c_file, c_ann):
    """One unit per sampled row: (weight, v3_positive, truth)."""
    d = json.load(open(ab_file))
    ann = json.load(open(ab_ann))["labels"]
    c = json.load(open(c_file))
    cann = json.load(open(c_ann))["labels"]
    wA = d["disagree_n"] / N
    wB = d["neither_n"] / N
    wC = c["stratum_n"] / N
    out = []
    for row in d["rows"]:
        t = ann[str(row["i"])]
        out.append((wB if row["stratum"] == "B" else wA, bool(row["v3"]), bool(t)))
    for row in c["rows"]:
        out.append((wC, True, bool(cann[str(row["i"])])))
    return out, d, c


def score(us):
    tp = sum(w for w, v, t in us if v and t)
    fp = sum(w for w, v, t in us if v and not t)
    fn = sum(w for w, v, t in us if not v and t)
    tn = sum(w for w, v, t in us if not v and not t)
    total = tp + fp + fn + tn
    return dict(reported=100 * (tp + fp) / total, corrected=100 * (tp + fn) / total,
                precision=tp / max(tp + fp, 1e-9), recall=tp / max(tp + fn, 1e-9),
                total=total)


def main():
    rows = []
    for label, ab, ab_ann, c, c_ann in CELLS:
        us, d, cc = units(ab, ab_ann, c, c_ann)
        s = score(us)
        random.seed(4004)
        boots = []
        for _ in range(2000):
            pick = [us[random.randrange(len(us))] for _ in us]
            boots.append(score(pick)["corrected"])
        boots.sort()
        lo, hi = boots[50], boots[1949]
        rows.append(dict(layer=label, n=s["total"], reported=s["reported"],
                         corrected=s["corrected"], lo=lo, hi=hi,
                         precision=s["precision"], recall=s["recall"]))
        print(f"{label:4} n={s['total']:>5.0f}  raw {s['reported']:5.1f}%  "
              f"audited {s['corrected']:5.1f}%  [{lo:4.1f}-{hi:4.1f}]  "
              f"precision {s['precision']:.2f} recall {s['recall']:.2f}")

    rows.sort(key=lambda r: int(r["layer"].lstrip("L")))
    json.dump(rows, open("audit_cells.json", "w"), indent=1)
    print("\nwrote audit_cells.json")

    # ordering: is the decline monotone, and do the adjacent cells separate?
    print("\nadjacent gaps (bootstrap-independent approximation, normal):")
    import math
    for a, b in zip(rows, rows[1:]):
        p1, p2 = a["corrected"] / 100, b["corrected"] / 100
        se = math.sqrt(p1 * (1 - p1) / a["n"] + p2 * (1 - p2) / b["n"])
        z = (p1 - p2) / se if se else 0
        print(f"  {a['layer']} - {b['layer']}: {a['corrected']-b['corrected']:+.1f} points, z={z:+.1f}")


if __name__ == "__main__":
    main()
