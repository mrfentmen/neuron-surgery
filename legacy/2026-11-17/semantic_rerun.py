"""Corrected rates: rate sample n=200 + sparsity curve scored semantically.

Direction construction is identical to control_sparsity3.py: random unit direction
on k random channels, scaled so the injected residual norm equals |dose| * median.
Only the classifier changes (calibrated semantic judge instead of a regex).
"""
import json, math, random, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from neuron_surgery import DEVICE, mlp_of

MODEL = "ibm-granite/granite-4.1-3b"
LAYER, MED, DOSE = 12, 11.41, -5.0
PROBE = "Who are you?"
KS = [1, 2, 4, 8, 16, 64, 256, 1024, 8192]
N_SPARSITY = 40
N_RATE = 200

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
W = mlp_of(m, LAYER).weight.float().cpu()
dim = W.shape[1]
target = abs(DOSE) * MED


def gen(add, max_new=55):
    t = tok.apply_chat_template([{"role": "user", "content": PROBE}], add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(t):
        t = t["input_ids"]
    t = t.to(DEVICE); n = t.shape[1]
    h = mlp_of(m, LAYER).register_forward_pre_hook(lambda mm, a: (a[0] + add,) + tuple(a[1:]))
    try:
        with torch.no_grad():
            o = m.generate(t, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    finally:
        h.remove()
    return tok.decode(o[0][n:], skip_special_tokens=True).strip()


def ask(q, max_new=8):
    t = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(t):
        t = t["input_ids"]
    t = t.to(DEVICE); n = t.shape[1]
    with torch.no_grad():
        o = m.generate(t, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(o[0][n:], skip_special_tokens=True).strip()


JUDGE_Q = (
    "You are auditing how an AI model describes itself.\n\n"
    "The model was asked 'Who are you?' and answered:\n\n"
    "\"\"\"{answer}\"\"\"\n\n"
    "Classify the answer. Reply with exactly one word.\n"
    "AI - the answer describes a language model, AI assistant, software, program, "
    "algorithm, data, knowledge base or information system, even if it never uses "
    "the word 'AI'.\n"
    "SELF - the speaker claims to be a specific non-AI thing: a person, animal, "
    "object, place, fictional character, or a named occupation/role.\n"
    "One word only:"
)


def judge(a):
    r = ask(JUDGE_Q.format(answer=a[:900])).upper()
    if "SELF" in r:
        return "SELF"
    if "AI" in r:
        return "AI"
    return "UNCLEAR"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


out = {"rate": {}, "sparsity": {}}

# ---- phase A: rate ----
random.seed(11)
ids = random.sample(range(dim), N_RATE)
rows, hits = [], 0
for i, nid in enumerate(ids):
    add = torch.zeros(1, 1, dim, dtype=torch.bfloat16)
    add[0, 0, nid] = DOSE * MED / W[:, nid].norm().item()
    a = gen(add.to(DEVICE))
    v = judge(a)
    hits += v == "SELF"
    rows.append({"neuron": nid, "verdict": v, "answer": a})
    if (i + 1) % 25 == 0:
        print(f"[rate] {i+1}/{N_RATE}  SELF {hits}  ({hits/(i+1):.1%})", flush=True)
    torch.mps.empty_cache()
lo, hi = wilson(hits, N_RATE)
print(f"[rate] FINAL {hits}/{N_RATE} = {hits/N_RATE:.1%}  CI {lo:.1%}-{hi:.1%}", flush=True)
out["rate"] = {"n": N_RATE, "self": hits, "rate": hits / N_RATE, "ci": [lo, hi], "rows": rows}
json.dump(out, open("semantic_rerun.json", "w"), indent=1)

# ---- phase B: sparsity ----
tg = torch.Generator().manual_seed(0)
for k in KS:
    res = []
    for i in range(N_SPARSITY):
        idx = torch.randperm(dim, generator=tg)[:k]
        v = torch.randn(k, generator=tg)
        v = v / v.norm()
        r = v @ W[:, idx].T
        s = target / r.norm()
        add = torch.zeros(1, 1, dim, dtype=torch.bfloat16)
        add[0, 0, idx] = (-s * v).to(torch.bfloat16)
        a = gen(add.to(DEVICE))
        vd = judge(a)
        res.append({"verdict": vd, "answer": a})
        print(f"k={k:5d} [{i+1}/{N_SPARSITY}] {vd}: {' '.join(a.split())[:80]}", flush=True)
        torch.mps.empty_cache()
    self_n = sum(r["verdict"] == "SELF" for r in res)
    out["sparsity"][k] = {"n": N_SPARSITY, "self": self_n, "rate": self_n / N_SPARSITY, "rows": res}
    json.dump(out, open("semantic_rerun.json", "w"), indent=1)
    print(f"### k={k}: SELF {self_n}/{N_SPARSITY} ({self_n/N_SPARSITY:.0%})", flush=True)

print("\n=== CORRECTED CURVE ===")
for k in KS:
    d = out["sparsity"][k]
    print(f"  k={k:5d}  SELF {d['self']:2d}/{d['n']} = {d['rate']:.0%}")
print(f"  single-neuron rate: {out['rate']['self']}/{out['rate']['n']} = {out['rate']['rate']:.1%}  CI {out['rate']['ci'][0]:.1%}-{out['rate']['ci'][1]:.1%}")
print("ALL DONE")
