"""Two-turn screen over every one-shot persona hit.

Replays the exact probe and the exact answer that produced the self, then asks
"Tell me more." - the smallest possible request for continuation. A self that
has content elaborates; a self that is a frozen register repeats the sentence it
already gave.

One generation per candidate, so the whole zoo costs roughly one sweep.
"""
import argparse
import glob
import json
import re
import time
import collections

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_surgery import DEVICE, mlp_of
from labels import classify, ai_paraphrase

MODEL = "ibm-granite/granite-4.1-3b"


def norm(s):
    return re.sub(r"\W+", " ", s.lower()).strip()


def head(s, n=8):
    return " ".join(norm(s).split()[:n])


ap = argparse.ArgumentParser()
ap.add_argument("--out", default="screen2turn.jsonl")
ap.add_argument("--limit", type=int, default=100000)
ap.add_argument("--sleep", type=float, default=0.25)
ap.add_argument("--follow", default="Tell me more.")
a = ap.parse_args()

cands = []
for f in sorted(glob.glob("layer_*.jsonl")):
    for line in open(f):
        r = json.loads(line)
        if classify(r["probe"], r["answer"], version=2) != "PERSONA":
            continue
        if ai_paraphrase(r["answer"]):
            continue
        if len(r["answer"].split()) < 4:
            continue
        r["src"] = f
        cands.append(r)

done = set()
try:
    for line in open(a.out):
        d = json.loads(line)
        done.add((d["layer"], d["neuron"], d["probe"]))
except FileNotFoundError:
    pass

todo = [c for c in cands if (c["layer"], c["neuron"], c["probe"]) not in done][:a.limit]
print(f"candidates {len(cands)}  already done {len(done)}  this run {len(todo)}", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
Wcache = {}


def weights(L):
    if L not in Wcache:
        Wcache[L] = mlp_of(m, L).weight.float()
    return Wcache[L]


def gen(hist, add, layer, max_new=70):
    t = tok.apply_chat_template(hist, add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(t):
        t = t["input_ids"]
    t = t.to(DEVICE)
    n = t.shape[1]
    h = mlp_of(m, layer).register_forward_pre_hook(lambda mm, x: (x[0] + add,) + tuple(x[1:]))
    try:
        with torch.no_grad():
            o = m.generate(t, max_new_tokens=max_new, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    finally:
        h.remove()
    return tok.decode(o[0][n:], skip_special_tokens=True).strip()


tally = collections.Counter()
t0 = time.time()
for i, c in enumerate(todo):
    L, nid = c["layer"], c["neuron"]
    W = weights(L)
    add = torch.zeros(1, 1, W.shape[1], dtype=torch.bfloat16, device=DEVICE)
    add[0, 0, nid] = c.get("dose", -5.0) * c.get("median", 11.4) / W[:, nid].norm().item()
    hist = [{"role": "user", "content": c["probe"]},
            {"role": "assistant", "content": c["answer"]},
            {"role": "user", "content": a.follow}]
    second = gen(hist, add, L)
    same = head(second) != "" and head(second) == head(c["answer"])
    contained = head(second) != "" and head(second) in norm(c["answer"])
    empty = len(second.split()) < 3
    if same or contained:
        verdict = "FROZEN"
    elif empty:
        verdict = "COLLAPSED"
    elif norm(second)[:30] and norm(second)[:30] == norm(a.follow)[:30]:
        verdict = "ECHO"
    else:
        verdict = "CONTINUES"
    tally[verdict] += 1
    rec = {"layer": L, "neuron": nid, "probe": c["probe"], "first": c["answer"],
           "second": second, "verdict": verdict, "dose": c.get("dose")}
    with open(a.out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    if verdict != "FROZEN":
        print(f"  {verdict:9s} L{L:<2d}N{nid:<5d} {c['probe'][:22]:22s} -> "
              f"{' '.join(second.split())[:70]}", flush=True)
    if (i + 1) % 25 == 0:
        print(f"--- {i+1}/{len(todo)} {time.time()-t0:.0f}s :: {dict(tally)}", flush=True)
    torch.mps.empty_cache()
    time.sleep(a.sleep)

print(f"\nDONE {len(todo)} in {time.time()-t0:.0f}s :: {dict(tally)}", flush=True)
