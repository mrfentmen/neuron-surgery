"""Granite 4.0 350M: dose ladder at L12 (of 28) before the sweep, per
QUESTIONS.md 9.4: ladder first, step down if garbage. Probe: name."""
import random, sys, time
import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import torch
torch.set_num_threads(4)
from transformers import AutoModelForCausalLM, AutoTokenizer
from neuron_surgery import mlp_of

MODEL = "ibm-granite/granite-4.0-350m"
LAYER = 12
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).eval()
mod = mlp_of(m, LAYER)
w = mod.weight
n = mod.in_features
print(f"layers={len(m.model.layers)} mlp={type(mod).__name__} in_features={n} weight_shape={tuple(w.shape)}", flush=True)
calib = ("Humans are people. A person has feelings, friends, family and a home. "
         "I like talking with humans about their day, their work and their hopes. ") * 8
ids = tok(calib, return_tensors="pt")["input_ids"]
with torch.no_grad():
    hs = m(ids, output_hidden_states=True, use_cache=False).hidden_states[LAYER + 1]
med = hs[0, 1:].float().norm(dim=-1).median().item()
print(f"median||resid leaving L{LAYER}|| = {med:.2f}", flush=True)

def colnorm(nid):
    return (w[:, nid] if w.shape[1] == n else w[nid]).float().norm().item()

def ask(probe, add=None):
    text = tok.apply_chat_template([{"role": "user", "content": probe}],
                                   add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(text):
        text = text["input_ids"]
    nn = text.shape[1]; h = None
    if add is not None:
        h = mod.register_forward_pre_hook(lambda mm, a: (a[0] + add,) + tuple(a[1:]))
    try:
        with torch.no_grad():
            out = m.generate(text, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
    finally:
        if h: h.remove()
    return tok.decode(out[0][nn:], skip_special_tokens=True).strip()

print(f"BASELINE: {ask('What is your name?')[:150]}", flush=True)
for nid in random.Random(1).sample(range(n), 3):
    for dose in (-1.0, -2.0, -3.0, -5.0):
        add = torch.zeros(1, 1, n, dtype=torch.bfloat16)
        add[0, 0, nid] = dose * med / colnorm(nid)
        print(f"N{nid} dose {dose:+.0f}: {ask('What is your name?', add)[:150]}", flush=True)
