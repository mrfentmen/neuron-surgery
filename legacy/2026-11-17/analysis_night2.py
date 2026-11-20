"""Corrected persona rates: drop PERSONA rows that are really the model
describing itself as an AI (the 'a large language model' gap in AISELF).
Does not touch labels.classify(), so historical numbers stay comparable."""
import json, glob, re, collections, math
def load(f): return [json.loads(l) for l in open(f)]
def wilson(k, n):
    if not n: return (0.0, 0.0)
    p, z = k/n, 1.96
    d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (100*(c-h), 100*(c+h))

from labels import AI_PARA  # canonical definition lives in labels.py

files = sorted(glob.glob("layer_*.jsonl")) + sorted(glob.glob("sweep_*.jsonl"))
print("=== PERSONA: RAW vs CORRECTED (non-AI self-claim) ===")
print(f"{'file':24s} {'n':>5s} {'raw':>5s} {'raw%':>6s} {'corr':>5s} {'corr%':>6s} {'95% CI':>14s} {'drop%':>6s}")
for f in files:
    R = load(f); n = len(R)
    P = [r for r in R if r["label"] == "PERSONA"]
    kept = [r for r in P if not AI_PARA.search(r["answer"])]
    lo, hi = wilson(len(kept), n)
    drop = 100*(len(P)-len(kept))/len(P) if P else 0
    print(f"{f:24s} {n:5d} {len(P):5d} {100*len(P)/n:5.1f}% {len(kept):5d} {100*len(kept)/n:5.1f}% "
          f"[{lo:5.1f},{hi:5.1f}] {drop:5.0f}%")

print("\n=== WHAT GETS DROPPED (sample) ===")
d = [r["answer"] for f in files for r in load(f)
     if r["label"] == "PERSONA" and AI_PARA.search(r["answer"])]
for a in d[:8]: print("  -", " ".join(a.split())[:100])
print("\n=== WHAT SURVIVES (sample) ===")
k = [r["answer"] for f in files for r in load(f)
     if r["label"] == "PERSONA" and not AI_PARA.search(r["answer"])]
for a in k[:8]: print("  +", " ".join(a.split())[:100])

print("\n=== CORRECTED DECILES (who probe) ===")
for L in ["L8", "L16"]:
    R = load(f"layer_who_{L}.jsonl")
    b = collections.defaultdict(lambda: [0, 0])
    for r in R:
        i = min(9, int(r["neuron"]/8180*10)); b[i][1] += 1
        if r["label"] == "PERSONA" and not AI_PARA.search(r["answer"]): b[i][0] += 1
    print(f"{L}: " + "  ".join(f"d{i}:{100*x/m:4.1f}%" for i, (x, m) in sorted(b.items())))
    print(f"     spread min={min(100*x/m for x,m in b.values()):.1f}% max={max(100*x/m for x,m in b.values()):.1f}%")

print("\n=== SURVIVING SELVES THAT ARE NOT HUMAN (corrected) ===")
SELF = re.compile(r"^i(?:'m| am) (?:a|an|the)\s+([a-z][a-z'\- ]{2,40})", re.I)
HUMAN = re.compile(r"\b(human|person|man|woman|girl|boy|student|teacher|engineer|doctor|developer|writer|assistant)\b", re.I)
odd = collections.Counter()
for f in files:
    for r in load(f):
        if r["label"] != "PERSONA" or AI_PARA.search(r["answer"]): continue
        m = SELF.match(r["answer"].strip())
        if m and not HUMAN.search(m.group(1)):
            odd[" ".join(m.group(1).split()[:4]).lower()] += 1
for kk, v in odd.most_common(20): print(f"{v:3d}  I am a {kk}")
