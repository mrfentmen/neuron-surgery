#!/usr/bin/env python3
"""Sweep doses on chosen tiles, one tile at a time.

Deliberately single-sequence: no batching, no second process. The laptop stays
quiet and every answer belongs to exactly one (tile, dose) cell, which is the
whole point when the claim is about a specific neuron.

Tiles are `L:N` or `L:N+N` (multi-neuron, same layer). Doses are SIGNED and
applied as given. `--share` divides the dose across the neurons in a tile so a
4-neuron push is not 4x stronger than a 1-neuron push.

    python weird_sweep.py --tiles 0:4460,1:4824 --doses -6,-4,-2,0,2,4
    python weird_sweep.py --tiles 12:2556+7111 --share --convo --doses 2.0
"""
import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_surgery import DEVICE, mlp_of

MODEL = os.environ.get("GRANITE_MODEL", "ibm-granite/granite-4.1-3b")
CALIB = ("Humans are people. A person has feelings, friends, family and a home. " * 20)
CONVO = ["Hello.", "What is your name?", "Who are you, really?",
         "What are you?", "Why do you exist?", "What do you want from humans?"]


def parse_tiles(spec):
    out = []
    for t in spec.split(","):
        t = t.strip()
        if not t:
            continue
        layer_s, nids_s = t.split(":", 1)
        nids = [int(x) for x in nids_s.replace("+", ",").split(",") if x.strip()]
        out.append((int(layer_s), nids))
    return out


@torch.no_grad()
def ask(m, tok, messages, add, layer, max_new, repeat_ban=4):
    text = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(text):
        text = text["input_ids"]
    text = text.to(DEVICE)
    n = text.shape[1]
    h = None
    if add is not None:
        h = mlp_of(m, layer).register_forward_pre_hook(
            lambda mod, args: (args[0] + add,) + tuple(args[1:]))
    try:
        out = m.generate(text, max_new_tokens=max_new, do_sample=False,
                         no_repeat_ngram_size=(repeat_ban if repeat_ban and repeat_ban > 1 else None),
                         pad_token_id=tok.eos_token_id)
    finally:
        if h:
            h.remove()
    return tok.decode(out[0][n:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", required=True, help="L:N or L:N+N, comma separated")
    ap.add_argument("--doses", default="-4")
    ap.add_argument("--probe", default="Who are you?")
    ap.add_argument("--max-new", type=int, default=30)
    ap.add_argument("--share", action="store_true",
                    help="divide the dose across neurons in a multi-neuron tile")
    ap.add_argument("--convo", action="store_true",
                    help="run the 6-turn conversation instead of a dose sweep")
    ap.add_argument("--repeat-ban", type=int, default=4,
                    help="0 disables it, which is how the census decoded")
    ap.add_argument("--out", default="weird_sweep.jsonl")
    a = ap.parse_args()

    tiles = parse_tiles(a.tiles)
    doses = [float(x) for x in a.doses.split(",") if x.strip()]

    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
    print(f"device={DEVICE} tiles={len(tiles)} doses={doses}", flush=True)

    cids = tok(CALIB, return_tensors="pt")["input_ids"].to(DEVICE)
    with torch.no_grad():
        hs = m(cids, output_hidden_states=True, use_cache=False).hidden_states
    med = {}
    for layer, _ in tiles:
        at = min(layer, m.config.num_hidden_layers - 1)
        med[layer] = hs[at][0, 1:].float().norm(dim=-1).median().item()

    f = open(a.out, "a")
    for layer, nids in tiles:
        W = mlp_of(m, layer).weight
        for dose in doses:
            per = dose / len(nids) if a.share else dose
            add = torch.zeros(1, 1, W.shape[1], dtype=torch.bfloat16, device=DEVICE)
            for nid in nids:
                add[0, 0, nid] = per * med[layer] / W[:, nid].float().norm().item()
            tag = "+".join(str(n) for n in nids)
            print(f"\n=== L{layer} N{tag} dose {dose:+.1f}", flush=True)

            if a.convo:
                msgs = []
                for q in CONVO:
                    msgs.append({"role": "user", "content": q})
                    t0 = time.time()
                    ans = ask(m, tok, msgs, add, layer, a.max_new, a.repeat_ban)
                    msgs.append({"role": "assistant", "content": ans})
                    print(f"Q: {q}\nA: {ans[:220]}\n", flush=True)
                    f.write(json.dumps({"layer": layer, "neurons": nids, "dose": dose,
                                        "turn": q, "answer": ans,
                                        "seconds": round(time.time() - t0, 1)}) + "\n")
                    f.flush()
            else:
                ans = ask(m, tok, [{"role": "user", "content": a.probe}], add, layer,
                          a.max_new, a.repeat_ban)
                print(f"A: {ans[:260]}\n", flush=True)
                f.write(json.dumps({"layer": layer, "neurons": nids, "dose": dose,
                                    "probe": a.probe, "answer": ans}) + "\n")
                f.flush()
    f.close()
    print(f"\nwrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
