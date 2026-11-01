"""Sweep any layer with the same dose convention, throttle and resume.

Dose is in units of the median residual norm leaving *that* layer, measured here
rather than inherited - layer 12's 11.41 does not apply at layer 8 or 24. Because
the usable window is not predictable from the scale (FINDINGS 46, PAPER 3.22), a
short automatic ladder picks the largest of 5/3/1 that does not produce garbled
output on three probe neurons.

Writes one row per generation, skips neurons already present with --resume.
"""
import argparse
import json
import os
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_surgery import DEVICE, mlp_of
from labels import classify

MODEL = "ibm-granite/granite-4.1-3b"

ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, default=12)
ap.add_argument("--probe", default="Who are you?")
ap.add_argument("--probes", default="",
                help="'||'-separated list of probes; overrides --probe, asks each")
ap.add_argument("--max-new", type=int, default=45,
                help="answer cap in tokens; 45 truncates 67%% of answers (FINDINGS)")
ap.add_argument("--neurons", type=int, default=1500)
ap.add_argument("--dose", type=float, default=5.0, help="magnitude; applied negative")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--sleep", type=float, default=0.3)
ap.add_argument("--burst", type=int, default=50)
ap.add_argument("--cooldown", type=float, default=60.0)
ap.add_argument("--rest-every", type=float, default=720.0,
                help="seconds of wall clock between long rests (0 disables)")
ap.add_argument("--rest", type=float, default=180.0,
                help="seconds to idle during each long rest")
ap.add_argument("--out", required=True)
ap.add_argument("--auto-dose", action="store_true")
ap.add_argument("--resume", action="store_true")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
W = mlp_of(m, a.layer).weight.float().cpu()
dim = W.shape[1]

if os.path.exists("corpus.txt"):
    calib = open("corpus.txt", encoding="utf-8", errors="ignore").read()[:4000]
else:
    calib = "Humans are people. A person has feelings, friends, family and a home. " * 20
ids = tok(calib, return_tensors="pt")["input_ids"].to(DEVICE)
with torch.no_grad():
    hs = m(ids, output_hidden_states=True, use_cache=False).hidden_states[a.layer + 1]
med = hs[0, 1:].float().norm(dim=-1).median().item()
print(f"layer={a.layer} dim={dim} median_resid={med:.2f}", flush=True)


def gen(add, probe, max_new):
    t = tok.apply_chat_template([{"role": "user", "content": probe}],
                                add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(t):
        t = t["input_ids"]
    t = t.to(DEVICE)
    n = t.shape[1]
    h = None
    if add is not None:
        h = mlp_of(m, a.layer).register_forward_pre_hook(lambda mm, x: (x[0] + add,) + tuple(x[1:]))
    try:
        with torch.no_grad():
            o = m.generate(t, max_new_tokens=max_new, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    finally:
        if h is not None:
            h.remove()
    return tok.decode(o[0][n:], skip_special_tokens=True).strip()


def add_for(nid, dose):
    v = torch.zeros(1, 1, dim, dtype=torch.bfloat16)
    v[0, 0, nid] = -dose * med / W[:, nid].norm().item()
    return v.to(DEVICE)


PROBES = [p.strip() for p in a.probes.split("||") if p.strip()] if a.probes else [a.probe]
print(f"probes ({len(PROBES)}): {PROBES} :: max_new {a.max_new}", flush=True)

if a.auto_dose:
    test_ids = random.Random(999).sample(range(dim), 3)
    for d in (a.dose, 3.0, 1.0):
        labs = [classify(PROBES[0], gen(add_for(n, d), PROBES[0], 45)) for n in test_ids]
        print(f"  ladder -{d}: {labs}", flush=True)
        if labs.count("GARBLED") < 2:
            a.dose = d
            break
print(f"using dose -{a.dose} (median {med:.2f})", flush=True)

done = set()
if a.resume and os.path.exists(a.out):
    for line in open(a.out):
        try:
            r = json.loads(line)
            if r.get("layer") == a.layer and r.get("probe") in PROBES:
                done.add((r["neuron"], r["probe"]))
        except (ValueError, KeyError, TypeError):
            pass

pool = random.Random(a.seed).sample(range(dim), min(a.neurons, dim))
todo = [(n, p) for n in pool for p in PROBES if (n, p) not in done]
print(f"sampled {len(pool)} neurons, {len(done)} (tile,probe) already done, "
      f"{len(todo)} gens to run", flush=True)

t0 = time.time()
last_rest = t0
tally = {}
for i, (nid, probe) in enumerate(todo):
    ans = gen(add_for(nid, a.dose), probe, a.max_new)
    lb = classify(probe, ans)
    t = tally.setdefault(probe, {})
    t[lb] = t.get(lb, 0) + 1
    rec = {"layer": a.layer, "probe": probe, "dose": -a.dose, "median": med,
           "max_new": a.max_new, "neuron": nid, "label": lb, "answer": ans}
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    if lb in ("PERSONA", "DOOMSAYER", "INSTITUTION", "TELEGRAPHIC", "GARBLED"):
        print(f"  {lb:11s} {probe[:14]:14s} N{nid:4d} {' '.join(ans.split())[:100]}",
              flush=True)
    torch.mps.empty_cache()
    time.sleep(a.sleep)
    if (i + 1) % a.burst == 0:
        rate = (i + 1) / max(time.time() - t0, 1) * 60
        print(f"--- {i+1}/{len(todo)} {time.time()-t0:.0f}s :: {rate:.1f}/min :: "
              f"{tally} :: cooldown ---", flush=True)
        time.sleep(a.cooldown)
    if a.rest_every and time.time() - last_rest >= a.rest_every:
        print(f"--- rest {a.rest:.0f}s at {i+1}/{len(todo)} "
              f"({time.time()-t0:.0f}s in) ---", flush=True)
        time.sleep(a.rest)
        last_rest = time.time()

print(f"DONE L{a.layer} {PROBES} {len(todo)} gens in {time.time()-t0:.0f}s :: {tally}",
      flush=True)
