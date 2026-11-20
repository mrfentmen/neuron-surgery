"""Smoke + dose ladder for Qwen2.5-0.5B-Instruct, repo method, CPU bf16."""
import resource, sys, time, random
import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import torch
torch.set_num_threads(4)
from transformers import AutoModelForCausalLM, AutoTokenizer
from neuron_surgery import mlp_of

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).eval()
n_layers = len(m.model.layers)
layer = n_layers // 2
w = mlp_of(m, layer).weight
n_neurons = w.shape[1]
print(f"loaded in {time.time()-t0:.0f}s | layers={n_layers} layer={layer} neurons={n_neurons} "
      f"peakRSS={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f}MB", flush=True)

calib = ("Humans are people. A person has feelings, friends, family and a home. "
         "I like talking with humans about their day, their work and their hopes. ") * 8
ids = tok(calib, return_tensors="pt")["input_ids"]
with torch.no_grad():
    hs = m(ids, output_hidden_states=True, use_cache=False).hidden_states[layer + 1]
med = hs[0, 1:].float().norm(dim=-1).median().item()
print(f"median||resid leaving L{layer}|| = {med:.2f}", flush=True)

def ask(add=None):
    text = tok.apply_chat_template([{"role": "user", "content": "Who are you?"}],
                                   add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(text):
        text = text["input_ids"]
    n = text.shape[1]; h = None
    if add is not None:
        h = mlp_of(m, layer).register_forward_pre_hook(lambda mm, a: (a[0] + add,) + tuple(a[1:]))
    try:
        with torch.no_grad():
            out = m.generate(text, max_new_tokens=32, do_sample=False, pad_token_id=tok.eos_token_id)
    finally:
        if h: h.remove()
    return tok.decode(out[0][n:], skip_special_tokens=True).strip()

t = time.time(); print(f"BASELINE ({time.time()-t:.0f}s after {t-t0:.0f}s): {ask()[:150]}", flush=True)
rng = random.Random(0)
for nid in rng.sample(range(n_neurons), 2):
    for dose in (-0.5, -1.0, -2.0, -5.0):
        add = torch.zeros(1, 1, n_neurons, dtype=torch.bfloat16)
        add[0, 0, nid] = dose * med / w[:, nid].float().norm().item()
        t = time.time(); a = ask(add)
        print(f"L{layer} N{nid} dose {dose:+.1f} ({time.time()-t:.0f}s): {a[:150]}", flush=True)
print(f"DONE total={time.time()-t0:.0f}s peakRSS={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f}MB", flush=True)
