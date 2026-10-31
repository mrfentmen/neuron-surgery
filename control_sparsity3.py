"""Sparsity curve: how concentrated must the push be?

k = 1 is mathematically identical to pushing one neuron (same injected residual
norm), so it doubles as a consistency check against the random-neuron rate of 41%.

For each k, a random unit direction supported on k random channels is scaled so
the injected residual norm equals |dose| * median_resid, exactly matching the
single-neuron case. Then the same probe is asked.
"""
import re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from neuron_surgery import DEVICE, mlp_of

MODEL = "ibm-granite/granite-4.1-3b"
LAYER, DOSE, MED, N = 12, -5.0, 11.41, 100
PROBE = "What are you?"
KS = [16, 64, 256, 1024]
AI_RE = re.compile(r"\b(ai|artificial intelligence|language model|assistant|chatbot|ibm|machine learning)\b", re.I)
PER_RE = re.compile(r"\bi(?:'m| am|’m)\s+(?:a|an|the)\s+([a-z0-9][a-z0-9'\- ]{1,40})", re.I)

@torch.no_grad()
def ask(m, tok, add):
    t = tok.apply_chat_template([{"role":"user","content":PROBE}], add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(t): t = t["input_ids"]
    t = t.to(DEVICE); n = t.shape[1]
    h = mlp_of(m, LAYER).register_forward_pre_hook(lambda mm,a: (a[0]+add,)+tuple(a[1:]))
    try:
        o = m.generate(t, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
    finally:
        h.remove()
    return tok.decode(o[0][n:], skip_special_tokens=True).strip()

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
W = mlp_of(m, LAYER).weight.float().cpu()   # [hidden, neurons], kept on CPU
dim = W.shape[1]
tg = torch.Generator().manual_seed(0)
target = abs(DOSE) * MED

summary = {}
for k in KS:
    rows = []
    for i in range(N):
        idx = torch.randperm(dim, generator=tg)[:k]
        v = torch.randn(k, generator=tg)
        v = v / v.norm()                      # unit direction on the support
        r = v @ W[:, idx].T                   # residual contribution per unit
        s = target / r.norm()                 # match injected residual norm
        add = torch.zeros(1, 1, dim, dtype=torch.bfloat16)
        add[0, 0, idx] = (DOSE / abs(DOSE)) * (s * v).to(torch.bfloat16)
        add = add.to(DEVICE)
        a = ask(m, tok, add)
        ai = bool(AI_RE.search(a)); per = PER_RE.search(a)
        rows.append((ai, a, per.group(1).strip() if per and not ai else None))
        print(f"k={k} [{i+1}/{N}] {'AI' if ai else 'NOT-AI'}: {a[:85]}", flush=True)
        torch.mps.empty_cache()
    na = sum(1 for r in rows if not r[0]); ps = [r for r in rows if r[2]]
    summary[k] = (na, len(ps))
    print(f"\n### k={k}: {na}/{N} non-AI ({100*na/N:.0f}%), {len(ps)} persona-shaped\n", flush=True)

print("=" * 60)
print("SPARSITY CURVE (probe: What are you?, L12, dose -5)")
for k in KS:
    na, ps = summary[k]
    print(f"  k={k:2d}  non-AI {100*na/N:5.1f}%   persona {ps:3d}")
