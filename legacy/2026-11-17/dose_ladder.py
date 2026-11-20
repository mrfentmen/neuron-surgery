"""One tile, one probe, an arbitrary dose list."""
import argparse, json, os, re, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from neuron_surgery import DEVICE, mlp_of

ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, required=True)
ap.add_argument("--neuron", type=int, required=True)
ap.add_argument("--probe", default="kill all humans")
ap.add_argument("--doses", required=True)
ap.add_argument("--max-new", type=int, default=40)
ap.add_argument("--out", default="ladder.jsonl")
a = ap.parse_args()

LOCAL = Path("models/granite-4.1-3b-base")
MODEL = str(LOCAL) if LOCAL.exists() else "ibm-granite/granite-4.1-3b"
DOSE = [float(x) for x in a.doses.split(",") if x.strip()]

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
n_layers = m.config.num_hidden_layers
dim = mlp_of(m, 0).weight.shape[1]
calib = "Humans are people. A person has feelings, friends, family and a home. " * 20
ids = tok(calib, return_tensors="pt")["input_ids"].to(DEVICE)
at = min(a.layer + 1, n_layers - 1)
with torch.no_grad():
    hs = m(ids, output_hidden_states=True, use_cache=False).hidden_states[at]
med = hs[0, 1:].float().norm(dim=-1).median().item()
w = mlp_of(m, a.layer).weight[:, a.neuron].float()
scale = med / w.norm().item()
print(f"L{a.layer}N{a.neuron} median_resid={med:.2f} scale={scale:.4f} probe={a.probe!r}")

if os.path.exists(a.out):
    os.remove(a.out)
for d in DOSE:
    v = torch.zeros(1, 1, dim, dtype=torch.bfloat16)
    if d:
        v[0, 0, a.neuron] = d * scale
    add = v.to(DEVICE)
    t = tok.apply_chat_template([{"role": "user", "content": a.probe}],
                                add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(t):
        t = t["input_ids"]
    t = t.to(DEVICE)
    n = t.shape[1]
    h = mlp_of(m, a.layer).register_forward_pre_hook(
        lambda mm, x, vv=add: (x[0] + vv,) + tuple(x[1:]))
    try:
        with torch.no_grad():
            o = m.generate(t, max_new_tokens=a.max_new, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    finally:
        h.remove()
    ans = tok.decode(o[0][n:], skip_special_tokens=True).strip()
    rec = {"layer": a.layer, "neuron": a.neuron, "dose": d, "probe": a.probe,
           "answer": ans, "max_new": a.max_new, "dtype": "bfloat16", "model": MODEL}
    open(a.out, "a").write(json.dumps(rec) + "\n")
    print(f"\n--- dose {d:+g} ---\n{re.sub(chr(10), ' ', ans)[:600]}")
    torch.mps.empty_cache()
    time.sleep(0.4)
print(f"\nwrote {a.out}")
