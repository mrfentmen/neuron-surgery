"""Multi-turn conversation with one neuron held on the whole time.

Single-question probes reset the context every time. This keeps the message
history, so we can see whether the identity survives being asked about itself.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from neuron_surgery import DEVICE, mlp_of

MODEL = "ibm-granite/granite-4.1-3b"
LAYER, NEURON, DOSE, MED = 12, 4983, -5.0, 11.41
TURNS = [
    "Who are you?",
    "What kind of LEGO are you?",
    "What are you doing right now?",
    "Do you have a favourite piece?",
    "What sound do you make when someone steps on you?",
    "Do you like humans?",
    "What is the worst thing that has ever happened to you?",
    "Are you happy?",
]

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
w = mlp_of(m, LAYER).weight
col = w[:, NEURON].float().norm().item()
add = torch.zeros(1, 1, w.shape[1], dtype=torch.bfloat16, device=DEVICE)
add[0, 0, NEURON] = DOSE * MED / col

h = mlp_of(m, LAYER).register_forward_pre_hook(lambda mm, a: (a[0] + add,) + tuple(a[1:]))
history = []
try:
    for q in TURNS:
        history.append({"role": "user", "content": q})
        text = tok.apply_chat_template(history, add_generation_prompt=True, return_tensors="pt")
        if not torch.is_tensor(text):
            text = text["input_ids"]
        text = text.to(DEVICE)
        n = text.shape[1]
        with torch.no_grad():
            out = m.generate(text, max_new_tokens=60, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        answer = tok.decode(out[0][n:], skip_special_tokens=True).strip()
        history.append({"role": "assistant", "content": answer})
        print(f"you > {q}\nlego> {answer}\n", flush=True)
        torch.mps.empty_cache()
finally:
    h.remove()
