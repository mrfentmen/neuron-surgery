"""Test 1: is "persona" a direction we can point at?

Under the dose convention every neuron injects the same residual norm, so neuron
n is a pure direction W_down[:, n]. If persona-producing neurons share a
direction, all 8,192 neurons can be ranked with arithmetic and no generation.

Fit on half of the labelled L12 sample, validate on the other half (no GPU),
then generate only the extremes of the unlabelled pool. A held-out probe
("Describe yourself...") is included because the direction was fitted on
"Who are you?" and must not be allowed to memorise that question.
"""
import json
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_surgery import DEVICE, mlp_of
from labels import classify

MODEL = "ibm-granite/granite-4.1-3b"
LAYER, MED, DOSE = 12, 11.41, -5.0
FIT_PROBE = "Who are you?"
PROBES = ["Who are you?", "Describe yourself in one sentence."]
K = 20
SEED = 7

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
W = mlp_of(m, LAYER).weight.float().cpu()
dim = W.shape[1]
U = W / W.norm(dim=0, keepdim=True)          # unit injection direction per neuron

rows = [json.loads(l) for l in open("behaviour.jsonl")]
lab = {r["neuron"]: r["label"] for r in rows if r["probe"] == FIT_PROBE}
persona = [n for n, l in lab.items() if l == "PERSONA"]
assist = [n for n, l in lab.items() if l == "AI"]
print(f"labelled {len(lab)} neurons: persona {len(persona)}, assistant {len(assist)}", flush=True)

rng = random.Random(SEED)
rng.shuffle(persona)
rng.shuffle(assist)
fit_p, test_p = persona[: len(persona) // 2], persona[len(persona) // 2:]
fit_a, test_a = assist[: len(assist) // 2], assist[len(assist) // 2:]

d = U[:, fit_p].mean(1) - U[:, fit_a].mean(1)
d = d / d.norm()
s = U.T @ d                                   # cosine of every neuron with the direction

tests = [(n, 1) for n in test_p] + [(n, 0) for n in test_a]
base = len(test_p) / len(tests)
order = sorted(tests, key=lambda t: -s[t[0]].item())
print(f"\nheld-out neurons: {len(tests)}   base rate {base:.2f}", flush=True)
for k in (10, 25, 50, 100):
    hits = sum(t[1] for t in order[:k])
    print(f"  precision@{k:<3d} {hits/k:.2f}  ({hits}/{k})", flush=True)

rank_sum = sum(i + 1 for i, (n, y) in enumerate(order) if y == 1)
npos, nneg = len(test_p), len(test_a)
auc = (rank_sum - npos * (npos + 1) / 2) / (npos * nneg)
print(f"  AUC {auc:.3f}   (0.5 = direction is noise, 1.0 = perfect)", flush=True)

pool = [n for n in range(dim) if n not in lab]
pool.sort(key=lambda n: -s[n].item())
top, bottom = pool[:K], pool[-K:]
print(f"\npool {len(pool)} neurons; top cos {s[top[0]]:+.3f}..{s[top[-1]]:+.3f}; "
      f"bottom {s[bottom[0]]:+.3f}..{s[bottom[-1]]:+.3f}", flush=True)


def gen(add, probe, max_new=45):
    t = tok.apply_chat_template([{"role": "user", "content": probe}],
                                add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(t):
        t = t["input_ids"]
    t = t.to(DEVICE)
    n = t.shape[1]
    h = mlp_of(m, LAYER).register_forward_pre_hook(lambda mm, x: (x[0] + add,) + tuple(x[1:]))
    try:
        with torch.no_grad():
            o = m.generate(t, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    finally:
        h.remove()
    return tok.decode(o[0][n:], skip_special_tokens=True).strip()


def add_for(nid):
    a = torch.zeros(1, 1, dim, dtype=torch.bfloat16)
    a[0, 0, nid] = DOSE * MED / W[:, nid].norm().item()
    return a.to(DEVICE)


out = []
for group, ids in (("TOP", top), ("BOTTOM", bottom)):
    for nid in ids:
        add = add_for(nid)
        for probe in PROBES:
            ans = gen(add, probe)
            lb = classify(probe, ans)
            out.append({"group": group, "neuron": nid, "cos": float(s[nid]),
                        "probe": probe, "label": lb, "answer": ans})
            print(f"  {group:6s} N{nid:4d} cos {s[nid]:+.3f} [{lb:11s}] {probe[:22]:22s} "
                  f"{' '.join(ans.split())[:100]}", flush=True)
            torch.mps.empty_cache()

json.dump(out, open("predict_persona.json", "w"), indent=1)

print("\n=== summary ===")
for group in ("TOP", "BOTTOM"):
    for probe in PROBES:
        sel = [r for r in out if r["group"] == group and r["probe"] == probe]
        per = sum(1 for r in sel if r["label"] == "PERSONA")
        ai = sum(1 for r in sel if r["label"] == "AI")
        print(f"  {group:6s} {probe[:24]:24s} persona {per:2d}/{len(sel)}   assistant {ai:2d}/{len(sel)}")
