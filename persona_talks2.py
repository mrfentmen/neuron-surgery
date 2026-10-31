"""Deeper multi-turn interrogation. First answer gets room to finish."""
import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from neuron_surgery import DEVICE, mlp_of

MODEL = "ibm-granite/granite-4.1-3b"
LAYER, MED = 12, 11.41
FIRST = "What are you?"
REST = [
    "Tell me more about yourself.",
    "What are you made of?",
    "How long have you existed?",
    "What can you do?",
    "Where are you?",
    "What year is it?",
    "Do you have a body?",
    "Finish this sentence: I am ___",
    "Do you like humans?",
    "What do you want most?",
]

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
w = mlp_of(m, LAYER).weight

def gen(history, add, max_new):
    text = tok.apply_chat_template(history, add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(text):
        text = text["input_ids"]
    text = text.to(DEVICE); n = text.shape[1]
    h = mlp_of(m, LAYER).register_forward_pre_hook(lambda mm, a: (a[0] + add,) + tuple(a[1:]))
    try:
        with torch.no_grad():
            out = m.generate(text, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    finally:
        h.remove()
    return tok.decode(out[0][n:], skip_special_tokens=True).strip()

ap = argparse.ArgumentParser(); ap.add_argument("--neurons", required=True)
ap.add_argument("--dose", type=float, default=-5.0)
ap.add_argument("--first", default=FIRST, help="opening probe; use the one that discovered the persona")
ap.add_argument("--questions", default=None, help="JSON list of questions, replaces the default set")
a = ap.parse_args()
FIRST = a.first
if a.questions:
    QS = __import__("json").loads(a.questions)
else:
    QS = [a.first] + REST

for nid in [int(x) for x in a.neurons.split(",")]:
    add = torch.zeros(1, 1, w.shape[1], dtype=torch.bfloat16, device=DEVICE)
    add[0, 0, nid] = a.dose * MED / w[:, nid].float().norm().item()
    print("=" * 72); print(f"L{LAYER} N{nid}  dose {a.dose:+.0f}"); print("=" * 72, flush=True)
    hist = []
    for i, q in enumerate(QS):
        hist.append({"role": "user", "content": q})
        ans = gen(hist, add, 130 if i == 0 else 55)
        hist.append({"role": "assistant", "content": ans})
        print(f"you > {q}\nbot > {ans}\n", flush=True)
        torch.mps.empty_cache()
