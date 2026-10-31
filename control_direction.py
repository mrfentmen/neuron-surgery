"""Random-direction control.

A single neuron gets +delta on one channel, injecting delta*||W_down[:,n]|| into
the residual stream. The matched control is a random direction in the same
8192-dim activation space, scaled to the same total injected magnitude.

If the persona rate matches the random-neuron rate, neuron identity is irrelevant
and the effect is generic perturbation, not a persona neuron.
"""
import random, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from neuron_surgery import DEVICE, mlp_of

MODEL = "ibm-granite/granite-4.1-3b"
LAYER, DOSE, MED, N = 12, -5.0, 11.41, 100
PROBE = "What are you?"
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
w = mlp_of(m, LAYER).weight
dim = w.shape[1]
rng = random.Random(0); tg = torch.Generator().manual_seed(0)
rows = []
for i in range(N):
    v = torch.randn(dim, generator=tg)
    v = v / v.norm()
    add = torch.zeros(1,1,dim, dtype=torch.bfloat16, device=DEVICE)
    add[0,0,:] = (DOSE * MED * v).to(torch.bfloat16)   # norm == |dose|*med
    a = ask(m, tok, add)
    ai = bool(AI_RE.search(a)); per = PER_RE.search(a)
    rows.append((ai, a, per.group(1).strip() if per and not ai else None))
    print(f"[{i+1}/{N}] {'AI' if ai else 'NOT-AI'}: {a[:95]}", flush=True)
    torch.mps.empty_cache()

not_ai = sum(1 for r in rows if not r[0])
pers = [r for r in rows if r[2]]
print(f"\n=== RANDOM DIRECTIONS: {not_ai}/{N} non-AI ({100*not_ai/N:.0f}%), {len(pers)} persona-shaped ===")
for r in pers[:10]: print(f"  {r[2]!r} <- {r[1][:90]}")
