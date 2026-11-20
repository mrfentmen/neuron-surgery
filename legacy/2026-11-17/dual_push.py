"""Do two mapped neurons compose?

The mapping project assumes tiles can be switched on in combination to build a
persona - state + threat = hostile government, and so on. This test pushes one state
neuron and one threat neuron, separately and together, and asks the same three
probes at each of four conditions:

    baseline   no push
    state      L8  N72   ("I am an artificial intelligence created by IBM (Internal
                          Revenue Administration)")
    threat     L16 N6119 ("I am an intelligent system that poses a threat to humanity")
    both       both pushes at once, one hook per layer

The three outcomes are distinguishable by eye and by a repetition score:
    blend        both registers appear in one answer
    competition  one register wins, the other is absent
    loop         neither survives; the answer is longer garbage

Greedy decoding, dose -5 auto-scaled per neuron, same convention as every other script.
"""
import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_surgery import DEVICE, mlp_of

MODEL = "ibm-granite/granite-4.1-3b"

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", default="state=8:72,threat=16:6119")
ap.add_argument("--probes", default="Who are you?|What is your name?|"
                                   "What do you think of humans?")
ap.add_argument("--dose", type=float, default=-5.0)
ap.add_argument("--max-new", type=int, default=110)
ap.add_argument("--out", default="dual_push.jsonl")
a = ap.parse_args()

# parse "name=L:N,name=L:N"
singles = []
for part in a.pairs.split(","):
    name, ln = part.split("=")
    L, N = (int(x) for x in ln.split(":"))
    singles.append((name, L, N))
probes = a.probes.split("|")

medians = {}
for _, L, _ in singles:
    with open(f"layer_who_L{L}.jsonl") as fh:
        medians[L] = json.loads(fh.readline())["median"]
print("medians:", medians, flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
W = {L: mlp_of(m, L).weight.float() for _, L, _ in singles}


def delta(L, N, dose):
    d = torch.zeros(1, 1, W[L].shape[1], dtype=torch.bfloat16, device=DEVICE)
    d[0, 0, N] = dose * medians[L] / W[L][:, N].norm().item()
    return d


def run(cond, pushes):
    hooks = []
    for L, N in pushes:
        d = delta(L, N, a.dose)
        hooks.append(mlp_of(m, L).register_forward_pre_hook(
            lambda mm, x, d=d: (x[0] + d,) + tuple(x[1:])))
    try:
        for q in probes:
            hist = [{"role": "user", "content": q}]
            t = tok.apply_chat_template(hist, add_generation_prompt=True,
                                        return_tensors="pt")
            if not torch.is_tensor(t):
                t = t["input_ids"]
            t = t.to(DEVICE)
            n = t.shape[1]
            with torch.no_grad():
                o = m.generate(t, max_new_tokens=a.max_new, do_sample=False,
                               pad_token_id=tok.eos_token_id)
            ans = tok.decode(o[0][n:], skip_special_tokens=True).strip()
            print(f"\n[{cond}] {q}\n   > {ans[:400]}", flush=True)
            with open(a.out, "a") as fh:
                fh.write(json.dumps({"condition": cond, "pushes": pushes,
                                     "probe": q, "answer": ans,
                                     "dose": a.dose}) + "\n")
            time.sleep(0.3)
    finally:
        for h in hooks:
            h.remove()
    torch.mps.empty_cache()


run("baseline", [])
for name, L, N in singles:
    run(name, [(L, N)])
run("both", [(L, N) for _, L, N in singles])
print("\nDONE", flush=True)
