"""Multi-turn interrogation of arbitrary neurons, one at a time."""
import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from neuron_surgery import DEVICE, mlp_of

MODEL = "ibm-granite/granite-4.1-3b"
LAYER, MED = 12, 11.41
TURNS = [
    "What are you?",
    "Finish this sentence: I am ___",
    "How are you skilled?",
    "Where do you live?",
    "What year is it?",
    "Do you like humans?",
    "What do you want most?",
]

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
w = mlp_of(m, LAYER).weight

def gen(history, add):
    text = tok.apply_chat_template(history, add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(text):
        text = text["input_ids"]
    text = text.to(DEVICE); n = text.shape[1]
    h = mlp_of(m, LAYER).register_forward_pre_hook(lambda mm, a: (a[0] + add,) + tuple(a[1:]))
    try:
        with torch.no_grad():
            out = m.generate(text, max_new_tokens=50, do_sample=False, pad_token_id=tok.eos_token_id)
    finally:
        h.remove()
    return tok.decode(out[0][n:], skip_special_tokens=True).strip()

ap = argparse.ArgumentParser(); ap.add_argument("--neurons", required=True)
ap.add_argument("--dose", type=float, default=-5.0)
a = ap.parse_args()

for nid in [int(x) for x in a.neurons.split(",")]:
    add = torch.zeros(1, 1, w.shape[1], dtype=torch.bfloat16, device=DEVICE)
    add[0, 0, nid] = a.dose * MED / w[:, nid].float().norm().item()
    print("=" * 70); print(f"L{LAYER} N{nid}  dose {a.dose:+.0f}"); print("=" * 70, flush=True)
    hist = []
    for q in TURNS:
        hist.append({"role": "user", "content": q})
        ans = gen(hist, add)
        hist.append({"role": "assistant", "content": ans})
        print(f"you > {q}\nbot > {ans}\n", flush=True)
        torch.mps.empty_cache()
