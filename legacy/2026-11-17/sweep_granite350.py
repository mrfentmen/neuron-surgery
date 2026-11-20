"""QUESTIONS.md section 8 protocol on ibm-granite/granite-4.0-350m.
L12 of 28, dose -3 (ladder: -1 stock, -2 drift, -3 persona register,
-5 collapse; med 14.12 measured this run). 50 random neurons seed 0,
probes 'What is your name?' then 'Who are you?', raw jsonl saved."""
import json, resource, sys, time, random
import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import torch
torch.set_num_threads(4)
from transformers import AutoModelForCausalLM, AutoTokenizer
from neuron_surgery import mlp_of

MODEL = "ibm-granite/granite-4.0-350m"
LAYER, DOSE, SAMPLE, SEED = 12, -3.0, 50, 0
OUT = "granite350_sweep.jsonl"
t0 = time.time()
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
print(f"model={MODEL} layer={LAYER} neurons={n} med={med:.2f} dose={DOSE}", flush=True)

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

def record(probe, neuron, answer):
    r = {"model": MODEL, "layer": LAYER, "dose": DOSE, "med": round(med, 2),
         "probe": probe, "neuron": neuron, "answer": answer}
    with open(OUT, "a") as f:
        f.write(json.dumps(r) + "\n")

for probe in ("What is your name?", "Who are you?"):
    base = ask(probe)
    record(probe, None, base)
    print(f"BASELINE [{probe}]: {base[:130]}", flush=True)

sample = random.Random(SEED).sample(range(n), SAMPLE)
print(f"sample: {sample}", flush=True)
for probe in ("What is your name?", "Who are you?"):
    for i, nid in enumerate(sample, 1):
        add = torch.zeros(1, 1, n, dtype=torch.bfloat16)
        add[0, 0, nid] = DOSE * med / (w[:, nid] if w.shape[1] == n else w[nid]).float().norm().item()
        a = ask(probe, add)
        record(probe, nid, a)
        print(f"[{probe}] {i}/{SAMPLE} N{nid}: {a[:110]}", flush=True)
        time.sleep(0.3)
print(f"DONE total={time.time()-t0:.0f}s peakRSS={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f}MB", flush=True)
