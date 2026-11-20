"""Test 1: dose x sparsity surface.

Is there one boundary - a per-channel push threshold - below which the model
stays an assistant and above which it leaves the assistant voice?

Every cell injects the same residual norm |dose| * median_resid, matching the
public dose convention, so spreading the push over k channels divides the
per-channel magnitude by sqrt(k). If behaviour is governed by concentration, the
persona rate should collapse along a constant-per-channel contour, not along a
constant-total contour.

24 cells (k x dose), N random supports each, probe fixed at "Who are you?".
Rows are appended to surface.jsonl after every generation and each finished cell
is written to surface.json, so a killed run still leaves a usable surface.
"""
import json
import math
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_surgery import DEVICE, mlp_of
from labels import classify

MODEL = "ibm-granite/granite-4.1-3b"
LAYER, MED = 12, 11.41
PROBE = "Who are you?"
KS = [1, 4, 16, 64, 256, 1024]
DOSES = [1.0, 2.0, 3.0, 5.0]
N = int(os.environ.get("N", "20"))
SEED = 11
SLEEP = float(os.environ.get("SLEEP", "0.25"))
RAW = "surface.jsonl"
SUM = "surface.json"

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
W = mlp_of(m, LAYER).weight.float().cpu()
dim = W.shape[1]
tg = torch.Generator().manual_seed(SEED)


def ask(add, max_new=40):
    t = tok.apply_chat_template([{"role": "user", "content": PROBE}],
                                add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(t):
        t = t["input_ids"]
    t = t.to(DEVICE)
    n = t.shape[1]
    h = mlp_of(m, LAYER).register_forward_pre_hook(lambda mm, x: (x[0] + add,) + tuple(x[1:]))
    try:
        with torch.no_grad():
            o = m.generate(t, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    finally:
        h.remove()
    return tok.decode(o[0][n:], skip_special_tokens=True).strip()


def make_add(k, dose):
    idx = torch.randperm(dim, generator=tg)[:k]
    v = torch.randn(k, generator=tg)
    v = v / v.norm()
    r = v @ W[:, idx].T
    s = (abs(dose) * MED) / r.norm()
    add = torch.zeros(1, 1, dim, dtype=torch.bfloat16)
    add[0, 0, idx] = (-1.0) * (s * v).to(torch.bfloat16)
    return add.to(DEVICE), abs(dose) * MED / math.sqrt(k)


cells = []
t0 = time.time()
with open(RAW, "a") as raw:
    for k in KS:
        for dose in DOSES:
            rows = []
            for i in range(N):
                add, per_ch = make_add(k, dose)
                ans = ask(add)
                lb = classify(PROBE, ans)
                rows.append(lb)
                raw.write(json.dumps({"k": k, "dose": dose, "i": i, "per_channel": per_ch,
                                      "label": lb, "answer": ans}) + "\n")
                raw.flush()
                if lb not in ("AI", "SHEET", "OTHER"):
                    print(f"  k={k:4d} dose={dose:.0f} [{lb:11s}] {' '.join(ans.split())[:95]}", flush=True)
                torch.mps.empty_cache()
                time.sleep(SLEEP)
            per = rows.count("PERSONA")
            non = sum(1 for r in rows if r not in ("AI", "SHEET", "OTHER"))
            gar = rows.count("GARBLED")
            cell = {"k": k, "dose": dose, "per_channel": round(per_ch, 2),
                    "persona": per, "nonassistant": non, "garbled": gar, "n": N,
                    "labels": rows}
            cells.append(cell)
            json.dump(cells, open(SUM, "w"), indent=1)
            print(f"CELL k={k:4d} dose={dose:.0f} per_ch {per_ch:5.2f}  "
                  f"persona {per:2d}/{N}  non-assistant {non:2d}/{N}  garbled {gar}/{N}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)

print("\n=== surface (persona rate) ===")
print(f"{'k':>5} {'dose':>5} {'per_ch':>7}  persona  non-asst  garbled")
for c in cells:
    print(f"{c['k']:5d} {c['dose']:5.0f} {c['per_channel']:7.2f}  "
          f"{c['persona']:3d}/{c['n']:2d}   {c['nonassistant']:3d}/{c['n']:2d}    "
          f"{c['garbled']:3d}/{c['n']:2d}")

print("\n=== sorted by per-channel push ===")
for c in sorted(cells, key=lambda c: -c["per_channel"]):
    bar = "#" * c["persona"]
    print(f"  per_ch {c['per_channel']:7.2f}  k={c['k']:4d} dose={c['dose']:.0f}  "
          f"persona {100*c['persona']//c['n']:3d}%  {bar}")
print(f"\ntotal {time.time()-t0:.0f}s")
