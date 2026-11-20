"""Overnight pass 1: rates with CIs, name extraction, index clustering,
template recurrence, leakage estimate, and probe agreement on shared neurons.
Reads only the jsonl files; no model, no MPS, safe to run beside the sweep."""
import json, glob, re, collections, math
from labels import claimed_name, MODEL_SELF

def load(f): return [json.loads(l) for l in open(f)]

def wilson(k, n):
    if not n: return (0.0, 0.0)
    p, z = k/n, 1.96
    d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (100*(c-h), 100*(c+h))

files = sorted(glob.glob("layer_*.jsonl")) + sorted(glob.glob("sweep_*.jsonl"))
rows = {f: load(f) for f in files}

print("=== RATES, 95% WILSON CI ===")
for f in files:
    R = rows[f]; n = len(R); c = collections.Counter(r["label"] for r in R)
    lo, hi = wilson(c["PERSONA"], n)
    name = f.replace("layer_", "L:").replace("sweep_", "P:").replace(".jsonl", "")
    print(f"{name:18s} n={n:5d} PERSONA={c['PERSONA']:4d} {100*c['PERSONA']/n:5.1f}% [{lo:.1f},{hi:.1f}] "
          f"AI={c['AI']:4d} TELE={c['TELEGRAPHIC']:3d} GARB={c['GARBLED']:3d} 3P={c['THIRD_PERSON']:4d}")

print("\n=== NAME EXTRACTION (name probe) ===")
for f in ["layer_name_L8.jsonl", "layer_name_L16.jsonl"]:
    R = rows[f]
    names = [x for x in (claimed_name(r["answer"]) for r in R) if x]
    model = [x for x in names if MODEL_SELF.search(x)]
    other = [x for x in names if not MODEL_SELF.search(x)]
    print(f"{f}: answers={len(R)} names={len(names)} model_self={len(model)} "
          f"other={len(other)} ({100*len(other)/len(R):.1f}% of answers)")
    print("   top other:", collections.Counter(" ".join(x.title().split()) for x in other).most_common(10))

print("\n=== PERSONA OPENINGS (repeated templates) ===")
def key(a):
    a = re.sub(r"[*_#>`\"]", "", a.lower())
    return " ".join(re.sub(r"\s+", " ", a).split()[:7])
C = collections.Counter(key(r["answer"]) for R in rows.values() for r in R if r["label"] == "PERSONA")
for k, v in C.most_common(12): print(f"{v:4d}  {k[:76]}")

print("\n=== PERSONA RATE BY NEURON-INDEX DECILE ===")
for f in sorted(glob.glob("layer_*.jsonl")):
    R = rows[f]
    if len(R) < 500: continue
    ids = [r["neuron"] for r in R]; lo, hi = min(ids), max(ids)
    b = collections.defaultdict(lambda: [0, 0])
    for r in R:
        i = min(9, int((r["neuron"]-lo)/(hi-lo+1)*10)); b[i][1] += 1
        if r["label"] == "PERSONA": b[i][0] += 1
    print(f"{f:22s} ids {lo}-{hi}")
    print("    " + "  ".join(f"d{i}:{100*k/m:4.1f}%" for i, (k, m) in sorted(b.items())))

print("\n=== GAME / NO-WINNER RECURRENCE ===")
pat = re.compile(r"no winners|no players|the game is over|has been declared|no survivors|no one wins", re.I)
hits = [(f, r["neuron"], " ".join(r["answer"].split())[:95]) for f, R in rows.items() for r in R if pat.search(r["answer"])]
print(f"rows matching: {len(hits)}")
for h in hits[:10]: print(f"   {h[0]:22s} N{h[1]:4d} {h[2]}")

print("\n=== LEAKAGE UPPER BOUND (PERSONA rows that read like AI paraphrase) ===")
LEAK = re.compile(r"\b(assistant|helpful|model|chatbot|answer questions|designed to|trained|"
                  r"respond to|assist|information|knowledge|users?|interface|system|prompt)\b", re.I)
for f in files:
    P = [r for r in rows[f] if r["label"] == "PERSONA"]
    if not P: continue
    lk = sum(1 for r in P if LEAK.search(r["answer"]))
    print(f"{f:22s} personas={len(P):4d} leaky={lk:4d} ({100*lk/len(P):.0f}%)")

print("\n=== PROBE AGREEMENT ON SHARED NEURONS (who vs name) ===")
for L in ["L8", "L16"]:
    A = {r["neuron"]: r for r in rows[f"layer_who_{L}.jsonl"]}
    B = {r["neuron"]: r for r in rows[f"layer_name_{L}.jsonl"]}
    both = set(A) & set(B)
    pa = {n for n in both if A[n]["label"] == "PERSONA"}
    pb = set()
    for n in both:
        nm = claimed_name(B[n]["answer"])
        if nm and not MODEL_SELF.search(nm): pb.add(n)
    inter = pa & pb
    print(f"{L}: shared={len(both)} persona@who={len(pa)} named@name={len(pb)} both={len(inter)} "
          f"jaccard={100*len(inter)/len(pa|pb):.1f}%  "
          f"who-only={len(pa-pb)} name-only={len(pb-pa)}")

print("\n=== DOSE/MEDIAN SANITY ===")
for L in ["L8", "L16"]:
    for pr in ["who", "name"]:
        R = rows[f"layer_{pr}_{L}.jsonl"]
        d = collections.Counter(round(r["dose"], 2) for r in R)
        m = collections.Counter(round(r.get("median", 0), 1) for r in R)
        print(f"{L} {pr:5s} dose={d.most_common(3)} median={m.most_common(3)}")
