"""Stratum C: the rows the first audit forgot.

audit_sample.py drops rows where both rules say "persona": they are in neither the
disagreement stratum nor the both-negative stratum, so they were never weighted and
the corrected rate came out as if they did not exist. That is 204 rows corpus-wide
(1,491 v3-personas minus 207 in the disagreement stratum and 1,080-ish... whatever the
exact split is - the script prints it).

This draws from exactly that stratum, at a fixed rate, so the total can be
reconstructed as A + B + C = the corpus. Same two-rule rule as elsewhere.

Usage:
  .venv/bin/python audit_missing.py                       # corpus-wide, 15 rows
  .venv/bin/python audit_missing.py --layer 8 --probe "Who are you?" --out audit_c_l8.json
"""
import argparse, glob, json, random

import labels
from classifier_agree import strict_persona

N = 15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int)
    ap.add_argument("--probe")
    ap.add_argument("--seed", type=int, default=5150)
    ap.add_argument("--out", default="audit_c.json")
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()

    if a.score:
        d = json.load(open(a.out))
        ann = json.load(open(a.out.replace(".json", "_ann.json")))["labels"]
        w = d["stratum_n"] / N
        keep = sum(w for r in d["rows"] if ann[str(r["i"])])
        drop = w * len(d["rows"]) - keep
        print(f"stratum C: {d['stratum_n']} rows, weight {w:.1f}x")
        print(f"  annotated real personas : {sum(1 for r in d['rows'] if ann[str(r['i'])])}/{len(d['rows'])}")
        print(f"  weighted keep {keep:.0f}  weighted drop {drop:.0f}")
        return

    rows = []
    for f in sorted(glob.glob("layer_*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if a.layer is not None and r.get("layer") != a.layer:
                continue
            if a.probe and (r.get("question") or r.get("probe")) != a.probe:
                continue
            rows.append(r)

    C = []
    for r in rows:
        ans = r["answer"]
        v3 = labels.classify(r.get("question", ""), ans, version=3) == "PERSONA" \
            and not labels.ai_paraphrase(ans)
        st, _ = strict_persona(ans)
        if v3 and st:
            C.append(r)
    random.seed(a.seed)
    random.shuffle(C)
    print(f"stratum C ('both rules say persona'): {len(C)} rows of {len(rows)}")
    sample = [{"i": i, "stratum": "C", "layer": r.get("layer"), "neuron": r.get("neuron"),
               "q": r.get("question", ""), "answer": r["answer"][:260]}
              for i, r in enumerate(C[:N])]
    json.dump({"layer": a.layer, "probe": a.probe, "seed": a.seed,
               "stratum_n": len(C), "frame_n": len(rows), "rows": sample},
              open(a.out, "w"), indent=1)
    print(f"wrote {a.out} ({len(sample)} rows)")


if __name__ == "__main__":
    main()
