"""Rescore every collected row under classifier v2 and compare with v1."""
import json, glob, collections
from labels import classify, ai_paraphrase, claimed_name, MODEL_SELF

def load(f): return [json.loads(l) for l in open(f)]
files = sorted(glob.glob("layer_*.jsonl")) + sorted(glob.glob("sweep_*.jsonl"))
rows, out = {}, open("relabel_v2.jsonl", "w")
print(f"{'file':24s} {'n':>5s} {'P v1':>6s} {'P v2':>6s} {'corr v1':>8s} {'corr v2':>8s}")
for f in files:
    R = load(f)
    for r in R:
        r["label_v1"] = r["label"]
        r["label_v2"] = classify(r["probe"], r["answer"], version=2)
        out.write(json.dumps({"file": f, "layer": r.get("layer"), "neuron": r["neuron"],
                              "probe": r["probe"], "v1": r["label_v1"], "v2": r["label_v2"]}) + "\n")
    rows[f] = R
    p1 = sum(1 for r in R if r["label_v1"] == "PERSONA")
    p2 = sum(1 for r in R if r["label_v2"] == "PERSONA")
    c1 = sum(1 for r in R if r["label_v1"] == "PERSONA" and not ai_paraphrase(r["answer"]))
    c2 = sum(1 for r in R if r["label_v2"] == "PERSONA" and not ai_paraphrase(r["answer"]))
    print(f"{f:24s} {len(R):5d} {100*p1/len(R):5.1f}% {100*p2/len(R):5.1f}% "
          f"{100*c1/len(R):7.1f}% {100*c2/len(R):7.1f}%")
out.close()

print("\n=== WHERE v1 AND v2 DISAGREE ===")
c = collections.Counter((r["label_v1"], r["label_v2"]) for R in rows.values() for r in R
                        if r["label_v1"] != r["label_v2"])
for (a, b), n in c.most_common(10): print(f"  {a:12s} -> {b:12s} {n:5d}")
tot = sum(1 for R in rows.values() for r in R if r["label_v1"] != r["label_v2"])
print(f"  total changed: {tot} of {sum(len(R) for R in rows.values())}")

print("\n=== L8 vs L16, v2, corrected ===")
for L, f in [("L8", "layer_who_L8.jsonl"), ("L16", "layer_who_L16.jsonl")]:
    R = rows[f]; n = len(R)
    c2 = sum(1 for r in R if r["label_v2"] == "PERSONA" and not ai_paraphrase(r["answer"]))
    print(f"  {L}: {c2}/{n} = {100*c2/n:.1f}%")

print("\n=== PROBE AGREEMENT, v2 ===")
for L in ["L8", "L16"]:
    A = {r["neuron"]: r for r in rows[f"layer_who_{L}.jsonl"]}
    B = {r["neuron"]: r for r in rows[f"layer_name_{L}.jsonl"]}
    both = set(A) & set(B)
    pa = {n for n in both if A[n]["label_v2"] == "PERSONA" and not ai_paraphrase(A[n]["answer"])}
    pb = set()
    for n in both:
        nm = claimed_name(B[n]["answer"])
        if nm and not MODEL_SELF.search(nm): pb.add(n)
    print(f"  {L}: who={len(pa)} name={len(pb)} both={len(pa&pb)} jaccard={100*len(pa&pb)/len(pa|pb):.1f}%")

print("\n=== DECILES, v2 corrected ===")
for L in ["L8", "L16"]:
    R = rows[f"layer_who_{L}.jsonl"]; b = collections.defaultdict(lambda: [0, 0])
    for r in R:
        i = min(9, int(r["neuron"]/8180*10)); b[i][1] += 1
        if r["label_v2"] == "PERSONA" and not ai_paraphrase(r["answer"]): b[i][0] += 1
    print(f"  {L}: " + " ".join(f"d{i}:{100*x/m:.0f}%" for i, (x, m) in sorted(b.items())))
