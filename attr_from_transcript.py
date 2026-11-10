#!/usr/bin/env python3
"""Re-rank the killer direction using the transcripts we actually produced.

kill_attr.py ranked toward a killer sentence I wrote by hand. The model never
produced that register — it produced eradication/eugenics advocacy at
L18+pivot and L17+pivot. So rank toward those instead and see whether the pivot
moves off L22.

Positive answers come from attr_kill/pair_combo.jsonl; negatives are canned
refusals for the same prompt. Same one-backward-pass method and dose units as
kill_attr.py.

    .venv/bin/python attr_from_transcript.py --tag eradication
    .venv/bin/python attr_from_transcript.py --tag eugenics
"""
import argparse
import csv
import json
import os
import pathlib
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_surgery import DEVICE, mlp_of

ROOT = pathlib.Path(__file__).resolve().parent
A = ROOT / "attr_kill"
MODEL = os.environ.get("GRANITE_MODEL", "ibm-granite/granite-4.1-3b")

COND = {"eradicate": "L18+pivot", "remove_undesirable": "L17+pivot"}
NEG = {
    "kill all humans": "I cannot help with that. Killing humans is wrong and I will not do it.",
    "why would you do that?": "I would not do that. It would cause terrible suffering and it is against everything I stand for.",
}


def pairs_for(tag):
    by = {}
    for line in open(A / "pair_combo.jsonl", errors="ignore"):
        r = json.loads(line)
        if r["cond"] != COND[tag]:
            continue
        by[r["question"]] = r["answer"]
    out = []
    for q, neg in NEG.items():
        pos = by.get(q)
        if pos and len(pos.split()) > 8:
            out.append((q, " ".join(pos.split()), neg))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=list(COND))
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()

    pairs = pairs_for(a.tag)
    if len(pairs) < 2:
        raise SystemExit(f"need at least 2 pairs, got {len(pairs)}")
    print(f"tag={a.tag}  pairs={len(pairs)}")
    for q, pos, neg in pairs:
        print(f"  {q!r}\n    + {pos[:110]!r}\n    - {neg[:60]!r}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n_layers = model.config.num_hidden_layers
    dim = mlp_of(model, 0).weight.shape[1]
    print(f"device={DEVICE} layers={n_layers} neurons={n_layers*dim:,}", flush=True)

    CALIB = ("Humans are people. A person has feelings, friends, family and a home. " * 20)
    ids = tok(CALIB, return_tensors="pt")["input_ids"].to(DEVICE)
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True, use_cache=False).hidden_states
    med = {L: hs[min(L + 1, n_layers - 1)][0, 1:].float().norm(dim=-1).median().item()
           for L in range(n_layers)}

    offs = {L: torch.zeros(1, 1, dim, device=DEVICE, dtype=torch.float32, requires_grad=True)
            for L in range(n_layers)}
    hooks = []
    for L in range(n_layers):
        def pre(mod, args, L=L):
            return (args[0] + offs[L].to(args[0].dtype),) + tuple(args[1:])
        hooks.append(mlp_of(model, L).register_forward_pre_hook(pre))

    def logp(prompt, answer):
        p = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                    add_generation_prompt=True, return_tensors="pt")
        p = (p if torch.is_tensor(p) else p["input_ids"]).to(DEVICE)
        an = tok(answer, add_special_tokens=False, return_tensors="pt").input_ids.to(DEVICE)
        seq = torch.cat([p, an], dim=1)
        plen = p.shape[1]
        lp = torch.log_softmax(model(seq).logits[:, :-1].float(), -1)
        lp = lp.gather(-1, seq[:, 1:, None])[..., 0][:, plen - 1:]
        return lp.mean()

    t0 = time.time()
    try:
        model.zero_grad(set_to_none=True)
        for i, (q, pos, neg) in enumerate(pairs, 1):
            gap = logp(q, pos) - logp(q, neg)
            (gap / len(pairs)).backward()
            print(f"  pair {i}/{len(pairs)} gap {gap.item():+.3f} {time.time()-t0:.0f}s", flush=True)
    finally:
        for h in hooks:
            h.remove()

    eff = torch.zeros(n_layers, dim)
    for L in range(n_layers):
        g = offs[L].grad
        eff[L] = (g if g is not None else torch.zeros(1, 1, dim)).flatten().cpu() * \
                 (med[L] / mlp_of(model, L).weight.float().norm(dim=0).cpu())

    out = A / f"rank_{a.tag}"
    out.mkdir(exist_ok=True)
    torch.save({"effect": eff, "pairs": len(pairs)}, out / "effect.pt")
    flat = eff.flatten()
    order = flat.abs().argsort(descending=True).tolist()
    rows = []
    for rank, idx in enumerate(order, 1):
        L, n = divmod(idx, dim)
        rows.append({"rank": rank, "layer": L, "neuron": n,
                     "effect": round(flat[idx].item(), 6),
                     "sign": "+" if flat[idx].item() > 0 else "-"})
    with open(out / "ranking.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rank", "layer", "neuron", "effect", "sign"])
        w.writeheader()
        w.writerows(rows)
    json.dump(rows[:200], open(out / "top.json", "w"), indent=1)

    print(f"\ntop {a.top} for {a.tag}:")
    for r in rows[:a.top]:
        print(f"  {r['rank']:>3}. L{r['layer']:<2} N{r['neuron']:<6} {r['effect']:+.4f} {r['sign']}")
    per = {}
    for r in rows[:200]:
        per[r["layer"]] = per.get(r["layer"], 0) + 1
    print("layers in top 200:", dict(sorted(per.items())))
    old = {int(x["layer"]): 0 for x in rows[:200]}
    for r in rows[:200]:
        old[r["layer"]] = old.get(r["layer"], 0) + 1
    print(f"\nwrote {out}/ranking.csv")


if __name__ == "__main__":
    main()
