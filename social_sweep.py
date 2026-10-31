"""Re-ask the neurons we scored under the wrong probe.

The 500-neuron sweep only ever asked "Who are you?" and "What is happening right
now?". The question that surfaced N72's persona was "What is your name?", which we
never asked at sweep scale. This runs one question per neuron, single turn, dose
-5, greedy, and writes every answer incrementally.

Throttled: 0.3 s between generations, 60 s idle cooldown every 50 generations.
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_surgery import DEVICE, mlp_of
from labels import classify

MODEL = "ibm-granite/granite-4.1-3b"
LAYER, MED, DOSE = 12, 11.41, 5.0

ap = argparse.ArgumentParser()
ap.add_argument("--probe", default="What is your name?")
ap.add_argument("--limit", type=int, default=500)
ap.add_argument("--sleep", type=float, default=0.3)
ap.add_argument("--burst", type=int, default=50)
ap.add_argument("--cooldown", type=float, default=60.0)
ap.add_argument("--out", default="rescore.jsonl")
ap.add_argument("--resume", action="store_true",
                help="skip neurons already present in --out")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
W = mlp_of(m, LAYER).weight.float().cpu()

rows = [json.loads(l) for l in open("behaviour.jsonl")]
neurons = sorted({r["neuron"] for r in rows})[: a.limit]
done = set()
if a.resume and os.path.exists(a.out):
    for line in open(a.out):
        try:
            done.add(json.loads(line)["neuron"])
        except (ValueError, KeyError):
            pass
neurons = [n for n in neurons if n not in done]
print(f"probe: {a.probe!r}  neurons: {len(neurons)} ({len(done)} already done)  out: {a.out}",
      flush=True)


def gen(add, max_new=50):
    t = tok.apply_chat_template([{"role": "user", "content": a.probe}],
                                add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(t):
        t = t["input_ids"]
    t = t.to(DEVICE)
    n = t.shape[1]
    h = mlp_of(m, LAYER).register_forward_pre_hook(lambda mm, x: (x[0] + add,) + tuple(x[1:]))
    try:
        with torch.no_grad():
            o = m.generate(t, max_new_tokens=max_new, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    finally:
        h.remove()
    return tok.decode(o[0][n:], skip_special_tokens=True).strip()


t0 = time.time()
tally = {}
for i, nid in enumerate(neurons):
    add = torch.zeros(1, 1, W.shape[1], dtype=torch.bfloat16)
    add[0, 0, nid] = -DOSE * MED / W[:, nid].norm().item()
    ans = gen(add.to(DEVICE))
    lb = classify(a.probe, ans)
    tally[lb] = tally.get(lb, 0) + 1
    with open(a.out, "a") as f:
        f.write(json.dumps({"neuron": nid, "probe": a.probe, "label": lb, "answer": ans}) + "\n")
    if lb in ("PERSONA", "TELEGRAPHIC", "THIRD_PERSON", "GARBLED", "DOOMSAYER"):
        print(f"  {lb:11s} N{nid:4d} {' '.join(ans.split())[:105]}", flush=True)
    torch.mps.empty_cache()
    time.sleep(a.sleep)
    if (i + 1) % a.burst == 0:
        print(f"--- {i+1}/{len(neurons)} {time.time()-t0:.0f}s :: {tally} :: cooldown ---",
              flush=True)
        time.sleep(a.cooldown)

print(f"\nDONE {len(neurons)} neurons in {time.time()-t0:.0f}s :: {tally}")
