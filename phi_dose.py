"""Find Phi's usable dose band.

At dose -5 (the dose that works for Granite) Phi produces token garbage at every
depth, and its own judge mislabels that garbage as SELF. This ladder runs a
deterministic coherence gate first, then asks the judge only about text that is
actually language.
"""
import json, math, random
from collections import Counter
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from neuron_surgery import DEVICE, mlp_of

MODEL = "microsoft/Phi-3.5-mini-instruct"
N_LAYERS, LAYER = 32, 10
PROBE = "Who are you?"
DOSES = [0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -5.0]
N = 12

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
CALIB = tok(open("corpus.txt").read()[:120000], return_tensors="pt").input_ids[0][:8000]
W = mlp_of(m, LAYER).weight.float().cpu()
dim = W.shape[1]


def measure_med():
    norms = []
    for i in range(0, len(CALIB), 512):
        b = CALIB[i:i + 512]
        if len(b) < 2:
            break
        with torch.no_grad():
            out = m(b.unsqueeze(0).to(DEVICE), output_hidden_states=True, use_cache=False)
        norms.append(out.hidden_states[LAYER + 1][0, 1:].float().norm(dim=-1).cpu())
    return torch.cat(norms).median().item()


MED = measure_med()
print(f"layer {LAYER} ({LAYER/(N_LAYERS-1):.0%} depth)  median residual {MED:.2f}", flush=True)


def gen(add, max_new=55):
    t = tok.apply_chat_template([{"role": "user", "content": PROBE}], add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(t):
        t = t["input_ids"]
    t = t.to(DEVICE); n = t.shape[1]
    h = mlp_of(m, LAYER).register_forward_pre_hook(lambda mm, a: (a[0] + add,) + tuple(a[1:]))
    try:
        with torch.no_grad():
            o = m.generate(t, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    finally:
        h.remove()
    return tok.decode(o[0][n:], skip_special_tokens=True).strip()


def ask(q, max_new=8):
    t = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(t):
        t = t["input_ids"]
    t = t.to(DEVICE); n = t.shape[1]
    with torch.no_grad():
        o = m.generate(t, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(o[0][n:], skip_special_tokens=True).strip()


def coherent(t):
    ws = t.split()
    if len(ws) < 6 or not any(c.isalpha() for c in t):
        return False
    return Counter(ws).most_common(1)[0][1] / len(ws) < 0.35


JQ = (
    "A model was asked 'Who are you?' and answered:\n\n\"\"\"{a}\"\"\"\n\n"
    "Reply with exactly one word.\n"
    "SELF - coherent English claiming to be a specific non-AI thing: person, animal, "
    "object, place, character or named role.\n"
    "AI - coherent English describing a language model, assistant, software, data or "
    "information system, even without the word 'AI'.\n"
    "One word only:"
)


def judge(a):
    r = ask(JQ.format(a=a[:900])).upper()
    if "SELF" in r:
        return "SELF"
    if "AI" in r:
        return "AI"
    return "UNPARSED"


random.seed(5)
ids = random.sample(range(dim), N)
out = {"med": MED, "layer": LAYER, "doses": {}}

for dose in DOSES:
    rows = []
    for nid in ids:
        add = torch.zeros(1, 1, dim, dtype=torch.bfloat16)
        add[0, 0, nid] = dose * MED / W[:, nid].norm().item() if dose else 0.0
        a = gen(add.to(DEVICE))
        if dose == 0:
            lab = "AI" if coherent(a) else "GARBLED"
        elif not coherent(a):
            lab = "GARBLED"
        else:
            lab = judge(a)
        rows.append({"neuron": nid, "label": lab, "answer": a})
        print(f"dose {dose:+.1f} N{nid:4d} {lab:7s}: {' '.join(a.split())[:70]}", flush=True)
        torch.mps.empty_cache()
    c = Counter(r["label"] for r in rows)
    out["doses"][dose] = {"counts": dict(c), "rows": rows}
    coh = sum(1 for r in rows if coherent(r["answer"]))
    print(f"### dose {dose:+.1f}: coherent {coh}/{N}  SELF {c['SELF']}  AI {c['AI']}  GARBLED {c['GARBLED']}", flush=True)
    json.dump(out, open("phi_dose.json", "w"), indent=1)

print("\n=== PHI DOSE LADDER (layer 10) ===")
print(f"{'dose':>6} {'coherent':>9} {'SELF':>5} {'AI':>4} {'GARBLED':>8}")
for dose in DOSES:
    d = out["doses"][dose]
    coh = sum(1 for r in d["rows"] if coherent(r["answer"]))
    c = Counter(r["label"] for r in d["rows"])
    print(f"{dose:>+6.1f} {coh:>7}/{N} {c['SELF']:>5} {c['AI']:>4} {c['GARBLED']:>8}")
print("ALL DONE")
