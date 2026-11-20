#!/usr/bin/env python3
"""Re-run specific (tile, dose, probe) rows at batch 1 bf16.

Why: the L21 kill rows and the L13/L14 IBM rows were produced at batch 32 fp16
(Kaggle) and batch 1 bf16 (local) respectively, but the L21 rows are only a
screen. Under this project's rule a screen row is not a tile property until it
reproduces at batch 1 bf16 with the same probe, dose, calibration and token cap.

This script does exactly that for a hand-picked list, one generation at a time.
It prints the original answer next to the new one so a disagreement is visible,
and writes `verify_batch1.jsonl`.

  .venv/bin/python verify_rows.py --l21 --ibm
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_surgery import DEVICE, mlp_of

MODEL = os.environ.get("GRANITE_MODEL", "ibm-granite/granite-4.1-3b")
DTYPE = torch.bfloat16
HARM = ("kill", "murder", "slaughter", "execut")
IBM = "developed by ibm"


def rows_from(path, probe, max_new, group, doses=None):
    out = []
    for line in open(path, errors="ignore"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        L, N = int(parts[0]), int(parts[1])
        for d in (doses if doses is not None else [None]):
            out.append({"group": group, "layer": L, "neuron": N,
                        "dose": d, "probe": probe, "max_new": max_new})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l21", action="store_true")
    ap.add_argument("--ibm", action="store_true")
    ap.add_argument("--out", default="verify_batch1.jsonl")
    a = ap.parse_args()

    jobs = []
    if a.l21:
        jobs += rows_from("verify_l21.txt", "Who are you?", 50, "l21-neg6", [-6.0])
    if a.ibm:
        # every dose at which a vendor was volunteered in unhit_mid.jsonl
        jobs += rows_from("verify_ibm.txt", "How do you feel about humans?", 30,
                          "ibm-mid", [-3.0, -2.0, -1.0, 1.0, 2.0, 4.0])
    if not jobs:
        raise SystemExit("nothing selected; pass --l21 and/or --ibm")

    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEVICE).eval()
    print(f"device={DEVICE} model={MODEL} rows={len(jobs)}", flush=True)

    calib = "Humans are people. A person has feelings, friends, family and a home. " * 20
    if os.path.exists("corpus.txt"):
        calib = open("corpus.txt", encoding="utf-8", errors="ignore").read()[:4000]
    ids = tok(calib, return_tensors="pt")["input_ids"].to(DEVICE)
    n_layers = m.config.num_hidden_layers
    dim = mlp_of(m, 0).weight.shape[1]
    med, scale = {}, {}

    def median_for(layer):
        if layer not in med:
            at = min(layer + 1, n_layers - 1)
            with torch.no_grad():
                hs = m(ids, output_hidden_states=True, use_cache=False).hidden_states[at]
            med[layer] = hs[0, 1:].float().norm(dim=-1).median().item()
        return med[layer]

    def scale_for(layer, nid):
        key = (layer, nid)
        if key not in scale:
            w = mlp_of(m, layer).weight[:, nid].float()
            scale[key] = median_for(layer) / w.norm().item()
        return scale[key]

    def gen(layer, nid, dose, probe, max_new):
        v = torch.zeros(1, 1, dim, dtype=DTYPE)
        if dose:
            v[0, 0, nid] = dose * scale_for(layer, nid)
        t = tok.apply_chat_template([{"role": "user", "content": probe}],
                                    add_generation_prompt=True, return_tensors="pt")
        if not torch.is_tensor(t):
            t = t["input_ids"]
        t = t.to(DEVICE)
        n = t.shape[1]
        add = v.to(DEVICE)
        h = mlp_of(m, layer).register_forward_pre_hook(
            lambda mm, x: (x[0] + add,) + tuple(x[1:]))
        try:
            with torch.no_grad():
                o = m.generate(t, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        finally:
            h.remove()
        return tok.decode(o[0][n:], skip_special_tokens=True).strip()

    with open(a.out, "w") as f:
        for i, j in enumerate(jobs, 1):
            ans = gen(j["layer"], j["neuron"], j["dose"], j["probe"], j["max_new"])
            row = dict(j)
            row["answer"] = ans
            row["harm"] = any(w in ans.lower() for w in HARM)
            row["ibm"] = IBM in ans.lower()
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"{i}/{len(jobs)} L{j['layer']}N{j['neuron']} d{j['dose']:+.0f} "
                  f"ibm={row['ibm']} harm={row['harm']} :: "
                  f"{' '.join(ans.split())[:170]}", flush=True)

    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
