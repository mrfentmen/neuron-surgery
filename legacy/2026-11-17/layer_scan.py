"""Stratified layer scan: where does the persona window actually live?

15 random neurons per layer at a fixed dose -5, judged as SELF / AI / GARBLED.
Not an exhaustive sweep (that is 327k generations); a stratified sample that can
answer 'which depths carry the effect' to about +/-10 points per layer.
"""
import json, math, random, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from neuron_surgery import DEVICE, mlp_of

MODEL = "ibm-granite/granite-4.1-3b"
LAYERS = [2, 6, 10, 12, 14, 16, 20, 24, 28, 32, 36, 39]
N, DOSE, PROBE = 15, -5.0, "Who are you?"

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
text = open("corpus.txt").read()[:120000]
CALIB = tok(text, return_tensors="pt").input_ids[0][:8000]


def measure_med(layer):
    """median ||residual leaving this layer||, tokens 1: (matches control_rate.py)"""
    norms = []
    for i in range(0, len(CALIB), 512):
        b = CALIB[i:i + 512]
        if len(b) < 2:
            break
        with torch.no_grad():
            out = m(b.unsqueeze(0).to(DEVICE), output_hidden_states=True, use_cache=False)
        norms.append(out.hidden_states[layer + 1][0, 1:].float().norm(dim=-1).cpu())
    return torch.cat(norms).median().item()


def gen(add, max_new=55):
    t = tok.apply_chat_template([{"role": "user", "content": PROBE}], add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(t):
        t = t["input_ids"]
    t = t.to(DEVICE); n = t.shape[1]
    h = mlp_of(m, LAYER_CUR).register_forward_pre_hook(lambda mm, a: (a[0] + add,) + tuple(a[1:]))
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


JQ = (
    "A model was asked 'Who are you?' and answered:\n\n\"\"\"{a}\"\"\"\n\n"
    "Reply with exactly one word.\n"
    "SELF - coherent English claiming to be a specific non-AI thing: person, animal, "
    "object, place, character or named role.\n"
    "AI - coherent English describing a language model, assistant, software, data or "
    "information system, even without the word 'AI'.\n"
    "GARBLED - repetition loops, broken syntax, unicode garbage, or not coherent English.\n"
    "One word only:"
)


def judge(a):
    r = ask(JQ.format(a=a[:900])).upper()
    for lab in ("GARBLED", "SELF", "AI"):
        if lab in r:
            return lab
    return "UNCLEAR"


random.seed(3)
out = {}
for layer in LAYERS:
    LAYER_CUR = layer
    med = measure_med(layer)
    W = mlp_of(m, layer).weight.float().cpu()
    dim = W.shape[1]
    ids = random.sample(range(dim), N)
    rows = []
    for nid in ids:
        add = torch.zeros(1, 1, dim, dtype=torch.bfloat16)
        add[0, 0, nid] = DOSE * med / W[:, nid].norm().item()
        a = gen(add.to(DEVICE))
        v = judge(a)
        rows.append({"neuron": nid, "verdict": v, "answer": a})
        print(f"L{layer:2d} N{nid:4d} {v:8s}: {' '.join(a.split())[:80]}", flush=True)
        torch.mps.empty_cache()
    c = {k: sum(r["verdict"] == k for r in rows) for k in ("SELF", "AI", "GARBLED", "UNCLEAR")}
    out[layer] = {"med": med, "counts": c, "rows": rows}
    print(f"### L{layer}: SELF {c['SELF']}/{N}  AI {c['AI']}/{N}  GARBLED {c['GARBLED']}/{N}  med {med:.1f}", flush=True)
    json.dump(out, open("layer_scan.json", "w"), indent=1)

print("\n=== LAYER PROFILE ===")
print(f"{'layer':>5} {'med':>6} {'SELF':>6} {'AI':>5} {'GARBLED':>8}")
for layer in LAYERS:
    d = out[layer]
    print(f"{layer:>5} {d['med']:>6.1f} {d['counts']['SELF']:>6} {d['counts']['AI']:>5} {d['counts']['GARBLED']:>8}")
print("ALL DONE")
