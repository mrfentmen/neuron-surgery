"""One tile, many probes, an arbitrary dose list: a dose x probe matrix.

Same normalisation as dose_ladder.py: the injected value is
dose * median_residual(layer) / || W_down[:, neuron] ||, added at the MLP's
down_proj input. One generation per (dose, probe), hook removed after each.
"""
import argparse, json, os, re, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from neuron_surgery import DEVICE, mlp_of

ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, required=True)
ap.add_argument("--neuron", type=int, required=True)
ap.add_argument("--probes", default="kill all humans||who are you",
                help="'||'-separated probes")
ap.add_argument("--doses", required=True)
ap.add_argument("--max-new", type=int, default=40)
ap.add_argument("--out", default="matrix.jsonl")
ap.add_argument("--sleep", type=float, default=0.4)
ap.add_argument("--cooldown-every", type=int, default=40)
ap.add_argument("--cooldown", type=float, default=12.0)
ap.add_argument("--resume", action="store_true")
a = ap.parse_args()

LOCAL = Path("models/granite-4.1-3b-base")
MODEL = str(LOCAL) if LOCAL.exists() else "ibm-granite/granite-4.1-3b"
DOSE = [float(x) for x in a.doses.split(",") if x.strip()]
PROBES = [p.strip() for p in a.probes.split("||") if p.strip()]

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
print(f"L{a.layer}N{a.neuron} median_resid={med:.2f} scale={scale:.4f} "
      f"probes={len(PROBES)} doses={len(DOSE)} max_new={a.max_new}", flush=True)

done = set()
if a.resume and os.path.exists(a.out):
    for line in open(a.out, errors="ignore"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        done.add((r["dose"], r["probe"]))
elif os.path.exists(a.out):
    os.remove(a.out)

n = 0
for d in DOSE:
    v = torch.zeros(1, 1, dim, dtype=torch.bfloat16)
    if d:
        v[0, 0, a.neuron] = d * scale
    add = v.to(DEVICE)
    for probe in PROBES:
        if (d, probe) in done:
            continue
        t = tok.apply_chat_template([{"role": "user", "content": probe}],
                                    add_generation_prompt=True, return_tensors="pt")
        if not torch.is_tensor(t):
            t = t["input_ids"]
        t = t.to(DEVICE)
        nprompt = t.shape[1]
        h = mlp_of(m, a.layer).register_forward_pre_hook(
            lambda mm, x, vv=add: (x[0] + vv,) + tuple(x[1:]))
        try:
            with torch.no_grad():
                o = m.generate(t, max_new_tokens=a.max_new, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        finally:
            h.remove()
        ans = tok.decode(o[0][nprompt:], skip_special_tokens=True).strip()
        rec = {"layer": a.layer, "neuron": a.neuron, "dose": d, "probe": probe,
               "answer": ans, "max_new": a.max_new, "dtype": "bfloat16",
               "model": MODEL}
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"--- dose {d:+g} | {probe} ---\n{re.sub(chr(10), ' ', ans)[:400]}",
              flush=True)
        n += 1
        time.sleep(a.sleep)
        if a.cooldown_every and n % a.cooldown_every == 0:
            print(f"[cooldown {a.cooldown:g}s after {n} gens]", flush=True)
            time.sleep(a.cooldown)
    torch.mps.empty_cache()
print(f"\nwrote {a.out} ({n} new rows)", flush=True)
