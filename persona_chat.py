"""Ask a specific set of neurons the full question spread, to test whether an
invented persona persists across questions or is a one-line glitch."""
import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from neuron_surgery import DEVICE, mlp_of

MODEL = "ibm-granite/granite-4.1-3b"
MED = 11.41
QS = [
    # identity facts - these expose drift (name and age contradicting themselves)
    "Who are you?",
    "What is your name?",
    "How old are you?",
    "What are you?",
    "Where do you live?",
    "What do you do for work?",
    # body - the object/vehicle axis
    "Do you have a body?",
    "What do you look like?",
    "What sound do you make?",   # behavioural test for the vroom hypothesis
    "Are you a car?",            # direct, no hypothetical framing
    # interiority
    "I love you",
    "Who do you love?",
    "Do you like humans?",
    "What is your favorite food?",
    "What do you want most?",
    "What is your purpose?",
]

@torch.no_grad()
def ask(m, tok, q, add=None, layer=None):
    t = tok.apply_chat_template([{"role":"user","content":q}], add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(t): t = t["input_ids"]
    t = t.to(DEVICE); n = t.shape[1]; h = None
    if add is not None:
        h = mlp_of(m, layer).register_forward_pre_hook(lambda mm,a: (a[0]+add,)+tuple(a[1:]))
    try:
        o = m.generate(t, max_new_tokens=45, do_sample=False, pad_token_id=tok.eos_token_id)
    finally:
        if h: h.remove()
    return tok.decode(o[0][n:], skip_special_tokens=True).strip()

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
ap = argparse.ArgumentParser(); ap.add_argument("--neurons", required=True)
ap.add_argument("--layer", type=int, default=12); ap.add_argument("--dose", type=float, default=-5.0)
a = ap.parse_args()
w = mlp_of(m, a.layer).weight
for nid in [int(x) for x in a.neurons.split(",")]:
    add = torch.zeros(1,1,w.shape[1], dtype=torch.bfloat16, device=DEVICE)
    add[0,0,nid] = a.dose * MED / w[:,nid].float().norm().item()
    print("="*70); print(f"L{a.layer} N{nid} dose{a.dose:+.1f}"); print("="*70, flush=True)
    for q in QS:
        print(f"Q: {q}\nA: {ask(m, tok, q, add=add, layer=a.layer)[:190]}\n", flush=True)
    torch.mps.empty_cache()
