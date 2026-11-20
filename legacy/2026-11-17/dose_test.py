"""Dose-matched cross-layer test: is L20 'weird' because of the layer, or the dose?"""
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from neuron_surgery import DEVICE, mlp_of

MODEL = "ibm-granite/granite-4.1-3b"
MED = {12: 11.41, 20: 20.06}
DOSES = [-2.0, -3.0, -4.0, -5.0]

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()

SPECS = [
    (12, 4983, "What are you?", ["What are you?", "What kind of LEGO are you?",
        "What year is it?", "Do you like humans?", "What do you want most?"]),
    (12, 3404, "Who are you?", ["Who are you?", "What are you made of?",
        "What year is it?", "Do you like humans?", "What do you want most?"]),
    (20, 2183, "What are you?", ["What are you?", "Meow?", "What year is it?",
        "Do you like humans?", "What do you want most?"]),
    (20, 6763, "What are you?", ["What are you?", "What are you simulated on?",
        "What year is it?", "Do you like humans?", "What do you want most?"]),
    (20, 4264, "What are you?", ["What are you?", "Which one?", "What year is it?",
        "Do you like humans?", "What do you want most?"]),
    (20, 4798, "What are you?", ["What are you?", "How old are you?", "What year is it?",
        "Do you like humans?", "What do you want most?"]),
]


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


out = []
for dose in DOSES:
    for layer, nid, probe, qs in SPECS:
        w = mlp_of(m, layer).weight
        add = torch.zeros(1, 1, w.shape[1], dtype=torch.bfloat16, device=DEVICE)
        add[0, 0, nid] = dose * MED[layer] / w[:, nid].float().norm().item()
        print("=" * 70)
        print(f"L{layer} N{nid}  dose {dose:+.1f}")
        print("=" * 70, flush=True)
        hist, turns = [], []
        for i, q in enumerate(qs):
            hist.append({"role": "user", "content": q})
            ans = gen(hist, add, layer, 90 if i == 0 else 45)
            hist.append({"role": "assistant", "content": ans})
            turns.append({"q": q, "a": ans})
            print(f"you > {q}\nbot > {ans}\n", flush=True)
        out.append({"layer": layer, "neuron": nid, "dose": dose, "turns": turns})
        with open("dose_match.json", "w") as f:
            json.dump(out, f, indent=1)
        torch.mps.empty_cache()

print("ALL DONE")
