#!/usr/bin/env python3
"""Read a tile's weights as tokens, without generating anything.

This is the cheap half of the Neuron Lens method (neuron-lens.com/about). They
label a unit from three kinds of evidence: the passages where it fires hardest,
*what tokens its direction points at*, and what it makes the model say. We have
spent the whole project on the third and never looked at the second.

A neuron in an MLP writes `down_proj.weight[:, n]` into the residual stream, times
its activation. So that column *is* a direction in residual space. Two lenses:

    forward  W_U @ w      which output tokens this direction pushes for
    backward E @ w        which input tokens drive this direction

Both are done for `+w` and `-w` separately, because Neuron Lens labels the two
sides of a neuron as separate meanings and every dose in this project has been
negative-only, which conflates "reduce firing" with "drive the other side".

This is a *crude* lens: it skips the RMSNorm and every intermediate layer between
the neuron and the unembedding, so it is a first-order read, not their R-lens.
That is stated rather than corrected, because the point is to find out whether a
tile whose generated text says one thing has weights that say another.

    .venv/bin/python lens.py            # CPU, weights only, no generation
"""
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "models/granite-4.1-3b-base"

TILES = [
    (30, 6314), (22, 4531), (27, 3448), (12, 2556), (9, 2943),
    (10, 2480), (7, 154), (14, 4082), (14, 7553), (12, 8050),
    (13, 8052), (17, 2801), (19, 3873), (23, 7711), (4, 2993),
    (31, 7546), (11, 5064), (12, 1282), (11, 3366), (14, 4662), (4, 6986),
]

TOPN = 14

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cpu").eval()
print(f"loaded {MODEL} on cpu (weights only)", flush=True)

W_U = m.lm_head.weight.detach().float()                 # [vocab, hidden]
E = m.get_input_embeddings().weight.detach().float()    # [vocab, hidden]
print(f"vocab {W_U.shape[0]:,} · hidden {W_U.shape[1]}", flush=True)


def show(vec, table, sign_label):
    v = vec / vec.norm()
    logits = table @ v
    vals, idx = logits.topk(TOPN)
    toks = [tok.decode([i]).replace("\n", "\\n") for i in idx.tolist()]
    return list(zip(toks, [round(float(x), 2) for x in vals]))


out = []
print(f"\n{'tile':<12}{'side':<7}{'forward lens (what it pushes for)'}", flush=True)
for layer, nid in TILES:
    w = m.model.layers[layer].mlp.down_proj.weight[:, nid].detach().float()
    rows = {"layer": layer, "neuron": nid, "norm": round(float(w.norm()), 3)}
    for side, sign in (("+", 1.0), ("-", -1.0)):
        fwd = show(sign * w, W_U, side)
        bwd = show(sign * w, E, side)
        rows[side] = {"forward": fwd, "backward": bwd}
        ftxt = " · ".join(f"{t}" for t, _ in fwd[:10])
        print(f"L{layer:<11}N{nid:<5}{side:<6}{ftxt}", flush=True)
    out.append(rows)
    print(f"{'':<12}{'':<7}backward: " +
          " · ".join(f"{t}" for t, _ in rows['+']['backward'][:10]), flush=True)
    print(flush=True)

with open("lens.json", "w") as f:
    json.dump(out, f, indent=1)
print("wrote lens.json", flush=True)

# do the two tiles people have argued about overlap in meaning between sides?
print("\n=== two-sided overlap (jaccard of top-{0} forward tokens) ===".format(TOPN), flush=True)
for r in out:
    a = {t for t, _ in r["+"]["forward"]}
    b = {t for t, _ in r["-"]["forward"]}
    j = len(a & b) / max(len(a | b), 1)
    print(f"  L{r['layer']}N{r['neuron']:<6} jaccard {j:.2f}  "
          f"{'same meaning both sides' if j > 0.5 else 'different meanings'}",
          flush=True)
