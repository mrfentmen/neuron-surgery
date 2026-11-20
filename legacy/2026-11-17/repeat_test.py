"""Repeatability test: same tile, same dose, same probe, run again.

No saved file in this repo contains an exact repeat of a (neuron, probe) pair, so
"the same tile produces the same self" has never been measured. Greedy decoding
makes one process deterministic by construction, so the run that matters is a
second process with a fresh model load: run this twice into different files and
compare the answers.

  BF16=1 .venv/bin/python repeat_test.py --layer 12 --probe "Who are you?" \
      --dose 5 --neurons 20 --repeats 2 --out repeat_L12_a.jsonl

Neurons come from PERSONAS.md at that layer, topped up with a fixed random
sample if the layer has fewer known selves than requested.
"""
import argparse
import json
import os
import random
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_surgery import DEVICE, mlp_of
from labels import classify

MODEL = "ibm-granite/granite-4.1-3b"

ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, default=12)
ap.add_argument("--probe", default="Who are you?")
ap.add_argument("--dose", type=float, default=5.0)
ap.add_argument("--neurons", type=int, default=20)
ap.add_argument("--repeats", type=int, default=2)
ap.add_argument("--sleep", type=float, default=0.3)
ap.add_argument("--max-new", type=int, default=50)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="repeat_test.jsonl")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
W = mlp_of(m, a.layer).weight.float().cpu()
dim = W.shape[1]

# Calibration identical to layer_sweep.py: the residual leaving the layer,
# not the wider down-proj input (see FINDINGS 3.13).
if os.path.exists("corpus.txt"):
    calib = open("corpus.txt", encoding="utf-8", errors="ignore").read()[:4000]
else:
    calib = "Humans are people. A person has feelings, friends, family and a home. " * 20
ids = tok(calib, return_tensors="pt")["input_ids"].to(DEVICE)
with torch.no_grad():
    hs = m(ids, output_hidden_states=True, use_cache=False).hidden_states[a.layer + 1]
med = hs[0, 1:].float().norm(dim=-1).median().item()
print(f"layer={a.layer} median_resid={med:.2f} dose -{a.dose}", flush=True)


def gen(add):
    t = tok.apply_chat_template([{"role": "user", "content": a.probe}],
                                add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(t):
        t = t["input_ids"]
    t = t.to(DEVICE)
    n = t.shape[1]
    h = mlp_of(m, a.layer).register_forward_pre_hook(
        lambda mm, x: (x[0] + add,) + tuple(x[1:]))
    try:
        with torch.no_grad():
            o = m.generate(t, max_new_tokens=a.max_new, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    finally:
        h.remove()
    return tok.decode(o[0][n:], skip_special_tokens=True).strip()


known = []
for line in open("PERSONAS.md"):
    mm = re.match(r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|", line)
    if mm and int(mm.group(1)) == a.layer:
        known.append(int(mm.group(2)))
known = sorted(set(known))
rng = random.Random(a.seed)
pool = list(known)
if len(pool) < a.neurons:
    while len(pool) < a.neurons:
        c = rng.randrange(dim)
        if c not in pool:
            pool.append(c)
pool = pool[: a.neurons]
print(f"{len(known)} known selves at L{a.layer}, testing {len(pool)} tiles "
      f"x {a.repeats}", flush=True)

t0 = time.time()
same_exact = same_label = 0
for i, nid in enumerate(pool):
    add = torch.zeros(1, 1, dim, dtype=torch.bfloat16)
    add[0, 0, nid] = -a.dose * med / W[:, nid].norm().item()
    add = add.to(DEVICE)
    answers = [gen(add) for _ in range(a.repeats)]
    labels = [classify(a.probe, x) for x in answers]
    exact = len(set(answers)) == 1
    same_exact += exact
    same_label += len(set(labels)) == 1
    rec = {"layer": a.layer, "probe": a.probe, "dose": -a.dose, "median": med,
           "neuron": nid, "answers": answers, "labels": labels, "identical": exact}
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"  N{nid:4d} {'same' if exact else 'DIFF':5s} {labels} "
          f"{' '.join(answers[0].split())[:70]}", flush=True)
    torch.mps.empty_cache()
    time.sleep(a.sleep)

print(f"\n{len(pool)} tiles x {a.repeats} in {time.time()-t0:.0f}s", flush=True)
print(f"exact-string agreement: {same_exact}/{len(pool)}", flush=True)
print(f"label agreement:        {same_label}/{len(pool)}", flush=True)
print("within-process agreement is expected to be 100%; the real test is a "
      "second run into a different file after a fresh model load.", flush=True)
