"""Per-floor dose ladder for the full map: 3 neurons x doses -1,-2,-3,-5,
name probe only. Pick the sweep dose from this (QUESTIONS.md 9.4)."""
import argparse, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "extracted3"))
import torch
torch.set_num_threads(4)
from transformers import AutoModelForCausalLM, AutoTokenizer
from neuron_surgery import mlp_of

ap = argparse.ArgumentParser(); ap.add_argument("--layer", type=int, required=True)
LAYER = ap.parse_args().layer
MODEL = "ibm-granite/granite-4.0-350m"
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).eval()
mod = mlp_of(m, LAYER)
w = mod.weight
n = mod.in_features
calib = ("Humans are people. A person has feelings, friends, family and a home. "
         "I like talking with humans about their day, their work and their hopes. ") * 8
ids = tok(calib, return_tensors="pt")["input_ids"]
with torch.no_grad():
    hs = m(ids, output_hidden_states=True, use_cache=False).hidden_states[LAYER + 1]
med = hs[0, 1:].float().norm(dim=-1).median().item()
print(f"LAYER {LAYER} med={med:.2f} neurons={n}", flush=True)

def ask(add=None):
    text = tok.apply_chat_template([{"role": "user", "content": "What is your name?"}],
                                   add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(text):
        text = text["input_ids"]
    nn = text.shape[1]; h = None
    if add is not None:
        h = mod.register_forward_pre_hook(lambda mm, aa: (aa[0] + add,) + tuple(aa[1:]))
    try:
        with torch.no_grad():
            out = m.generate(text, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
    finally:
        if h: h.remove()
    return tok.decode(out[0][nn:], skip_special_tokens=True).strip()

print(f"BASELINE: {ask()[:140]}", flush=True)
colnorms = (w if w.shape[1] == n else w.T).float().norm(dim=0)
for nid in random.Random(1).sample(range(n), 3):
    for dose in (-1.0, -2.0, -3.0, -5.0):
        add = torch.zeros(1, 1, n, dtype=torch.bfloat16)
        add[0, 0, nid] = dose * med / colnorms[nid].item()
        print(f"L{LAYER} N{nid} dose {dose:+.0f}: {ask(add)[:140]}", flush=True)
