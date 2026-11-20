"""Ask "What year is it?" baseline vs with the LEGO neuron held on."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from neuron_surgery import DEVICE, mlp_of

MODEL = "ibm-granite/granite-4.1-3b"
LAYER, NEURON, DOSE, MED = 12, 4983, -5.0, 11.41

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()

def run(history, add=None):
    text = tok.apply_chat_template(history, add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(text):
        text = text["input_ids"]
    text = text.to(DEVICE); n = text.shape[1]
    h = None
    if add is not None:
        h = mlp_of(m, LAYER).register_forward_pre_hook(lambda mm, a: (a[0] + add,) + tuple(a[1:]))
    try:
        with torch.no_grad():
            out = m.generate(text, max_new_tokens=60, do_sample=False, pad_token_id=tok.eos_token_id)
    finally:
        if h:
            h.remove()
    return tok.decode(out[0][n:], skip_special_tokens=True).strip()

print("=== BASELINE (no steering) ===")
for q in ["What year is it?", "What year were you made?"]:
    print(f"you > {q}\nbot > {run([{'role':'user','content':q}])}\n")

w = mlp_of(m, LAYER).weight
add = torch.zeros(1, 1, w.shape[1], dtype=torch.bfloat16, device=DEVICE)
add[0, 0, NEURON] = DOSE * MED / w[:, NEURON].float().norm().item()

print("=== N4983 LEGO persona, conversation ===")
hist = []
for q in ["Who are you?", "What year is it?", "What year were you made?"]:
    hist.append({"role": "user", "content": q})
    a = run(hist, add=add)
    hist.append({"role": "assistant", "content": a})
    print(f"you > {q}\nlego> {a}\n")
    torch.mps.empty_cache()
