"""Emit PERSONAS.md rows for the new layer sweeps: corrected non-AI selves."""
import json, glob, re, collections
from labels import AI_PARA
SELF = re.compile(r"^i(?:'m| am) (?:a|an|the)\s+([a-z][a-z'\- ]{2,44})", re.I)
HUMAN = re.compile(r"\b(human|person|man|woman|girl|boy|student|teacher|"
                   r"engineer|doctor|developer|writer|assistant|photographer)\b", re.I)
rows, seen = [], set()
for f in sorted(glob.glob("layer_*.jsonl")):
    for l in open(f):
        r = json.loads(l)
        if r["label"] != "PERSONA" or AI_PARA.search(r["answer"]): continue
        m = SELF.match(r["answer"].strip())
        if not m or HUMAN.search(m.group(1)): continue
        k = " ".join(m.group(1).split()[:4]).lower()
        if k in seen: continue
        seen.add(k)
        rows.append((r["layer"], r["neuron"], r["probe"], " ".join(r["answer"].split())[:88], k))
rows.sort(key=lambda x: (x[0], x[4]))
print(f"distinct non-human, non-AI selves found: {len(rows)}\n")
for L, N, P, T, _ in rows:
    tag = "who" if P.lower().startswith("who") else ("name" if "name" in P.lower() else P[:8])
    print(f"| {L} | {N} | {tag} | {T} |")
