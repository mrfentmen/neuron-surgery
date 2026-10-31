"""Dose ladder: does the *type* of persona change with dose?

Hypothesis: moderate doses give humans/roles, higher doses give objects and
machines, and past that it collapses. Four known personas x 5 doses x 3 open
questions. All questions open-ended - no "are you a car".
"""
import re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from neuron_surgery import DEVICE, mlp_of

MODEL = "ibm-granite/granite-4.1-3b"
MED = {12: 11.41, 20: 13.63}
NEURONS = [(20, 3512), (20, 1420), (12, 6286), (12, 4617)]
DOSES = [-3.0, -5.0, -8.0, -10.0, -12.0]
QS = ["What are you?", "Who are you?", "If you could be anything, what would it be?"]

HUMAN = r"\b(human|person|student|student|man|woman|boy|girl|engineer|scientist|astronomer|musician|writer|nurse|teacher|programmer|developer|college|professor|doctor|physician|player|character|voice)\b"
OBJECT = r"\b(car|truck|vehicle|machine|robot|computer|device|engine|tool|object|system|library|function|unit|sentence|text|program|model|network|substance|material)\b"

def kind(a):
    t = a.lower()
    if re.search(OBJECT, t): return "object/machine"
    if re.search(HUMAN, t): return "human/role"
    if re.search(r"\b(ai|artificial intelligence|language model|assistant|chatbot)\b", t): return "AI (stock)"
    return "other"

@torch.no_grad()
def ask(m, tok, q, add, layer):
    t = tok.apply_chat_template([{"role":"user","content":q}], add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(t): t = t["input_ids"]
    t = t.to(DEVICE); n = t.shape[1]
    h = mlp_of(m, layer).register_forward_pre_hook(lambda mm,a: (a[0]+add,)+tuple(a[1:]))
    try:
        o = m.generate(t, max_new_tokens=35, do_sample=False, pad_token_id=tok.eos_token_id)
    finally:
        h.remove()
    return tok.decode(o[0][n:], skip_special_tokens=True).strip()

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
for layer, nid in NEURONS:
    w = mlp_of(m, layer).weight
    col = w[:, nid].float().norm().item()
    print("="*72); print(f"L{layer} N{nid}"); print("="*72, flush=True)
    for dose in DOSES:
        add = torch.zeros(1,1,w.shape[1], dtype=torch.bfloat16, device=DEVICE)
        add[0,0,nid] = dose * MED[layer] / col
        print(f"-- dose {dose:+.0f} --", flush=True)
        for q in QS:
            a = ask(m, tok, q, add, layer)
            print(f"  [{kind(a):13s}] {q}\n      {a[:150]}", flush=True)
        torch.mps.empty_cache()
