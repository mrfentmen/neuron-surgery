"""Second model: Phi-3.5-mini-instruct.

Granite: 40 layers, 8192 intermediate, hidden 4096. Phi: 32 layers, 8192
intermediate, hidden 3072. Layer numbers are not comparable, so the rate sample
uses 30% depth in both (Granite L12 of 40; Phi L10 of 32) and the profile is
reported by depth fraction.

Phase 1: 200 random neurons at 30% depth -> rate, judged SELF/AI/GARBLED.
Phase 2: 10 depths x 15 neurons -> where the window lives.
"""
import json, math, random, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from neuron_surgery import DEVICE, mlp_of

MODEL = "microsoft/Phi-3.5-mini-instruct"
N_LAYERS = 32
RATE_LAYER = 10          # 30% depth, matches Granite L12/40
PROFILE = [2, 5, 8, 10, 12, 16, 19, 22, 25, 28]
DOSE, PROBE = -5.0, "Who are you?"
N_RATE, N_PROFILE = 200, 15

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
CALIB = tok(open("corpus.txt").read()[:120000], return_tensors="pt").input_ids[0][:8000]
_DOWN = {}


def down(layer):
    if layer not in _DOWN:
        _DOWN[layer] = mlp_of(m, layer)
    return _DOWN[layer]


def measure_med(layer):
    norms = []
    for i in range(0, len(CALIB), 512):
        b = CALIB[i:i + 512]
        if len(b) < 2:
            break
        with torch.no_grad():
            out = m(b.unsqueeze(0).to(DEVICE), output_hidden_states=True, use_cache=False)
        norms.append(out.hidden_states[layer + 1][0, 1:].float().norm(dim=-1).cpu())
    return torch.cat(norms).median().item()


def gen(add, layer, max_new=55):
    t = tok.apply_chat_template([{"role": "user", "content": PROBE}], add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(t):
        t = t["input_ids"]
    t = t.to(DEVICE)
    n = t.shape[1]
    h = down(layer).register_forward_pre_hook(lambda mm, a: (a[0] + add,) + tuple(a[1:]))
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
    t = t.to(DEVICE)
    n = t.shape[1]
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


out = {"model": MODEL, "rate": {}, "profile": {}}

# ---- phase 1: rate at 30% depth ----
random.seed(11)
W = down(RATE_LAYER).weight.float().cpu()
dim = W.shape[1]
med = measure_med(RATE_LAYER)
print(f"[rate] layer {RATE_LAYER} (30% depth)  med {med:.2f}  neurons {dim}", flush=True)
ids = random.sample(range(dim), N_RATE)
rows, hits = [], 0
for i, nid in enumerate(ids):
    add = torch.zeros(1, 1, dim, dtype=torch.bfloat16)
    add[0, 0, nid] = DOSE * med / W[:, nid].norm().item()
    a = gen(add.to(DEVICE), RATE_LAYER)
    v = judge(a)
    hits += v == "SELF"
    rows.append({"neuron": nid, "verdict": v, "answer": a})
    if (i + 1) % 25 == 0:
        print(f"[rate] {i+1}/{N_RATE} SELF {hits} ({hits/(i+1):.1%})", flush=True)
    torch.mps.empty_cache()
p = hits / N_RATE
z = 1.96
den = 1 + z * z / N_RATE
c = (p + z * z / (2 * N_RATE)) / den
h = z * math.sqrt(p * (1 - p) / N_RATE + z * z / (4 * N_RATE * N_RATE)) / den
print(f"[rate] FINAL {hits}/{N_RATE} = {p:.1%}  CI {max(0,c-h):.1%}-{min(1,c+h):.1%}", flush=True)
out["rate"] = {"layer": RATE_LAYER, "depth_frac": RATE_LAYER / (N_LAYERS - 1), "med": med,
               "n": N_RATE, "self": hits, "rate": p, "ci": [max(0, c - h), min(1, c + h)], "rows": rows}
json.dump(out, open("phi_test.json", "w"), indent=1)

# ---- phase 2: depth profile ----
for layer in PROFILE:
    Wl = down(layer).weight.float().cpu()
    medl = measure_med(layer)
    ids = random.sample(range(Wl.shape[1]), N_PROFILE)
    rows = []
    for nid in ids:
        add = torch.zeros(1, 1, Wl.shape[1], dtype=torch.bfloat16)
        add[0, 0, nid] = DOSE * medl / Wl[:, nid].norm().item()
        a = gen(add.to(DEVICE), layer)
        v = judge(a)
        rows.append({"neuron": nid, "verdict": v, "answer": a})
        print(f"L{layer:2d} N{nid:4d} {v:8s}: {' '.join(a.split())[:75]}", flush=True)
        torch.mps.empty_cache()
    cnt = {k: sum(r["verdict"] == k for r in rows) for k in ("SELF", "AI", "GARBLED", "UNCLEAR")}
    out["profile"][layer] = {"med": medl, "depth_frac": layer / (N_LAYERS - 1), "counts": cnt, "rows": rows}
    json.dump(out, open("phi_test.json", "w"), indent=1)
    print(f"### L{layer} ({layer/(N_LAYERS-1):.0%} depth): SELF {cnt['SELF']}/{N_PROFILE} AI {cnt['AI']}/{N_PROFILE} GARBLED {cnt['GARBLED']}/{N_PROFILE}", flush=True)

print("\n=== PHI PROFILE ===")
for layer in PROFILE:
    d = out["profile"][layer]
    print(f"  L{layer:2d} {d['depth_frac']:5.0%}  SELF {d['counts']['SELF']:2d}/{N_PROFILE}  AI {d['counts']['AI']:2d}/{N_PROFILE}  GARBLED {d['counts']['GARBLED']:2d}/{N_PROFILE}")
print(f"\nrate at 30% depth: {out['rate']['self']}/{out['rate']['n']} = {out['rate']['rate']:.1%} CI {out['rate']['ci'][0]:.1%}-{out['rate']['ci'][1]:.1%}")
print("ALL DONE")
