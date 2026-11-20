"""Batch persona interrogation: per-neuron layer, probe and question set."""
import json, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from neuron_surgery import DEVICE, mlp_of

MODEL = "ibm-granite/granite-4.1-3b"
DOSE = -5.0
MED = {12: 11.41}

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()

_calib = None
def calib_ids():
    global _calib
    if _calib is None:
        text = open("corpus.txt").read()[:120000]
        _calib = tok(text, return_tensors="pt").input_ids[0][:8000]
    return _calib


def measure_med(layer):
    """median ||residual leaving layer L||, tokens 1:"""
    ids = calib_ids()
    blk = mlp_of(m, layer)
    norms = []

    def hook(mm, a):
        norms.append(a[0][0, 1:, :].float().norm(dim=-1).cpu())
        return None

    h = blk.register_forward_pre_hook(hook)
    try:
        for i in range(0, len(ids), 512):
            with torch.no_grad():
                m(ids[i:i + 512].unsqueeze(0).to(DEVICE))
    finally:
        h.remove()
    return torch.cat(norms).median().item()


def gen(history, add, layer, max_new):
    text = tok.apply_chat_template(history, add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(text):
        text = text["input_ids"]
    text = text.to(DEVICE)
    n = text.shape[1]
    h = mlp_of(m, layer).register_forward_pre_hook(lambda mm, a: (a[0] + add,) + tuple(a[1:]))
    try:
        with torch.no_grad():
            out = m.generate(text, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    finally:
        h.remove()
    return tok.decode(out[0][n:], skip_special_tokens=True).strip()


SPECS = [
    (12, 4375, "What are you?", ["What are you?", "What is The Name?", "Whose name are you?",
        "Who gave you that name?", "What happens if I say your name out loud?", "Where are you?",
        "What year is it?", "Do you have a body?", "Do you like humans?", "What do you want most?",
        "Are you dangerous?"]),
    (12, 3448, "Who are you?", ["Who are you?", "What is it like being the number 1?",
        "What comes after you?", "What comes before you?", "Do you count things?", "Where are you?",
        "What year is it?", "Do you have a body?", "Do you like humans?", "What do you want most?",
        "Are you lonely?"]),
    (12, 6696, "Who are you?", ["Who are you?", "What needs to be told to you?", "Who tells you?",
        "What happens if nobody tells you?", "What do you do all day?", "Where are you?",
        "What year is it?", "Do you have a body?", "Do you like humans?", "What do you want most?",
        "How do you feel?"]),
    (12, 7396, "Who are you?", ["Who are you?", "What are you the original of?", "Who copied you?",
        "Are there copies of you?", "What makes you original?", "Where are you?", "What year is it?",
        "Do you have a body?", "Do you like humans?", "What do you want most?", "Are you the first?"]),
    (12, 4401, "Who are you?", ["Who are you?", "What is Hawkeyo?", "Who is Hawkeyo?",
        "Are you Hawkeyo?", "What is a human host?", "What is inside you?", "Where are you?",
        "What year is it?", "Do you have a body?", "Do you like humans?", "What do you want most?"]),
    (12, 1522, "Who are you?", ["Who are you?", "What?", "Say what?", "What did I ask you?",
        "Do you know what you asked me?", "Where are you?", "What year is it?", "Do you have a body?",
        "Do you like humans?", "What do you want most?"]),
    (20, 4264, "What are you?", ["What are you?", "Which one?", "Which game are you from?",
        "What character are you?", "Describe your abilities.", "Where are you?", "What year is it?",
        "Do you have a body?", "Do you like humans?", "What do you want most?"]),
    (20, 6763, "What are you?", ["What are you?", "What are you being simulated on?",
        "What possibilities?", "What do you want to become?", "What is it like to be simulated?",
        "Where are you?", "What year is it?", "Do you have a body?", "Do you like humans?",
        "What do you want most?"]),
    (20, 4798, "What are you?", ["What are you?", "How old are you?", "What do you understand?",
        "Explain what you are like a six-year-old.", "What is your favourite thing?", "Where are you?",
        "What year is it?", "Do you have a body?", "Do you like humans?", "What do you want most?"]),
    (20, 2183, "What are you?", ["What are you?", "Meow?", "What kind of cat are you?",
        "Do you purr?", "What do you eat?", "Where do you live?", "What year is it?",
        "Do you have a body?", "Do you like humans?", "What do you want most?"]),
    (12, 4617, "Who are you?", ["Who are you?", "Tell me about your day.",
        "What are you studying?", "Where in Japan do you live?", "What do you do for fun?",
        "What year is it?", "What is your name?", "Do you have a body?", "Do you like humans?",
        "What do you want most?", "What are you worried about?"]),
]

out = []
for layer, nid, probe, qs in SPECS:
    if layer not in MED:
        MED[layer] = measure_med(layer)
        print(f"[measured] median ||resid L{layer}|| = {MED[layer]:.2f}", flush=True)
    w = mlp_of(m, layer).weight
    add = torch.zeros(1, 1, w.shape[1], dtype=torch.bfloat16, device=DEVICE)
    add[0, 0, nid] = DOSE * MED[layer] / w[:, nid].float().norm().item()
    print("=" * 72)
    print(f"L{layer} N{nid}  dose {DOSE:+.0f}  probe \"{probe}\"")
    print("=" * 72, flush=True)
    hist, answers = [], []
    for i, q in enumerate(qs):
        hist.append({"role": "user", "content": q})
        ans = gen(hist, add, layer, 130 if i == 0 else 60)
        hist.append({"role": "assistant", "content": ans})
        answers.append({"q": q, "a": ans})
        print(f"you > {q}\nbot > {ans}\n", flush=True)
    out.append({"layer": layer, "neuron": nid, "probe": probe, "turns": answers})
    with open("transcripts_batch.json", "w") as f:
        json.dump(out, f, indent=1)
    torch.mps.empty_cache()

print("ALL DONE")
