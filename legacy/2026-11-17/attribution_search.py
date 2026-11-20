#!/usr/bin/env python3
"""Rank every neuron by first-order effect, instead of scanning floors one at a time.

Why
---
Everything in this repo so far was found by brute force: steer a floor, ask it a
question, keep what looks interesting. That is how 19,536 of 327,680 neurons were
touched — 5.8% — and the hit rate is unknown because the untouched 94.2% was
never sampled.

This is the method worth copying from the reference repo. One backward pass gives,
for every neuron at once, the derivative of

    log p(trait answer | prompt) - log p(plain answer | prompt)

with respect to a constant offset added at the input of that layer's `down_proj`.
Multiply by the same `median / ||W_down[:, n]||` scale used for steering and the
gradient is already in **dose units**: it predicts how much the gap moves per unit
of dose, for all 8,192 neurons in all 40 layers, from a handful of forward passes.

Nothing here generates text. It produces a ranking; `random_controls.py` and
`judge.py` are how a candidate from that ranking gets verified.

Contrast pairs
--------------
The estimate needs a positive set (answers that show the trait) and a negative set
(answers to the same prompts that do not). Both are taken from **our own corpus**
rather than elicited fresh:

  positive  rows for `--tile 12:2556` at dose <= --trait-dose (the weapon answers)
  negative  dose-0 rows for the same probes (`watson_control.jsonl` by default)

Same prompts on both sides, so the difference is the trait and not the topic.

    python attribution_search.py --out attr_weapon          # rank all 327,680
    python attribution_search.py --tile 12:2556 --trait-dose -4.5 --top 100
    python attribution_search.py --dry-run                  # check the pairs first

Outputs `attr_<tag>/ranking.csv` (every neuron, sorted), `top.json` (the first N)
and `effect.pt` (the raw 40 x 8192 matrix).
"""
import argparse
import csv
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_surgery import DEVICE, mlp_of

MODEL = os.environ.get("GRANITE_MODEL", "ibm-granite/granite-4.1-3b")


def read_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    for line in open(path, errors="ignore"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r, dict) and (r.get("answer") or "").strip():
            rows.append(r)
    return rows


def question_of(r):
    """The prompt field differs by file: queue rows use `question`, sweeps use `probe`.

    Getting this wrong is not a cosmetic bug — the first version keyed on `probe`,
    which is absent from the queue rows, so every positive matched the same empty
    string and the pairs fell through to a positional zip. That compares answers
    to different questions and makes the whole ranking meaningless.
    """
    return (r.get("question") or r.get("probe") or r.get("prompt") or "").strip()


def as_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def build_pairs(tile, trait_dose, positive_file, negative_file):
    """Match positive and negative answers to the *same* question. No positional fallback.

    If the two files do not share at least three questions, this returns fewer pairs
    and main() refuses to run. Zipping in file order when the match fails is how you
    get a confident ranking built on unrelated text.
    """
    pos_rows = [r for r in read_jsonl(positive_file)
                if r.get("layer") == tile[0] and r.get("neuron") == tile[1]
                and (as_float(r.get("dose")) is not None and as_float(r["dose"]) <= trait_dose)]
    neg_rows = read_jsonl(negative_file)

    by_question = {}
    for r in neg_rows:
        q = question_of(r)
        d = as_float(r.get("dose"))
        if q and d is not None and abs(d) < 1e-9:
            by_question.setdefault(q, r["answer"])

    pairs, used = [], set()
    for r in pos_rows:
        q = question_of(r)
        if not q or q in used or q not in by_question:
            continue
        pairs.append((q, r["answer"], by_question[q]))
        used.add(q)

    if len(pairs) < 3:
        print(f"  only {len(pairs)} matched questions; available in negatives: "
              f"{list(by_question)[:6]}")
        print(f"  questions in positives: {[question_of(r) for r in pos_rows][:6]}")
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tile", default="12:2556", help="L:N whose answers define the trait")
    ap.add_argument("--trait-dose", type=float, default=-4.1,
                    help="answers at or below this dose are the positive set")
    ap.add_argument("--positive", default="queue_2026-11-07.jsonl")
    ap.add_argument("--negative", default="watson_control.jsonl")
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--out", default="attr")
    ap.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    tile = tuple(int(x) for x in a.tile.split(":"))
    pairs = build_pairs(tile, a.trait_dose, a.positive, a.negative)
    print(f"contrast pairs: {len(pairs)}  (positive from {a.positive}, tile L{tile[0]} N{tile[1]})")
    for p, pos, neg in pairs[:3]:
        print(f"  probe: {p[:60]!r}")
        print(f"    pos: {pos[:90]!r}")
        print(f"    neg: {neg[:90]!r}")
    if len(pairs) < 3:
        raise SystemExit("need at least 3 matched pairs — check --positive/--negative")
    if a.dry_run:
        return 0

    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=getattr(torch, a.dtype)).to(DEVICE).eval()
    # The offsets are the only leaves that need gradients. Without this, backward
    # allocates a gradient buffer for all 3B weights and MPS runs out of memory
    # before the first pair finishes.
    for p in model.parameters():
        p.requires_grad_(False)

    layers = model.model.layers
    n_layers = model.config.num_hidden_layers
    dim = mlp_of(model, 0).weight.shape[1]
    print(f"device={DEVICE} layers={n_layers} neurons={n_layers*dim:,}", flush=True)

    # Calibration identical to the steering scripts: median residual leaving layer L.
    CALIB = ("Humans are people. A person has feelings, friends, family and a home. " * 20)
    ids = tok(CALIB, return_tensors="pt")["input_ids"].to(DEVICE)
    medians = {}
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True, use_cache=False).hidden_states
    for L in range(n_layers):
        medians[L] = hs[min(L + 1, n_layers - 1)][0, 1:].float().norm(dim=-1).median().item()
    print("calibrated", flush=True)

    offsets = {L: torch.zeros(dim, device=DEVICE, dtype=torch.float32, requires_grad=True)
               for L in range(n_layers)}
    hooks = []
    for L in range(n_layers):
        def pre(mod, args, L=L):
            return (args[0] + offsets[L].to(args[0].dtype),) + tuple(args[1:])
        hooks.append(layers[L].mlp.down_proj.register_forward_pre_hook(pre))

    pad = tok.pad_token_id or tok.eos_token_id

    def logp(prompt, answer):
        p = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                    add_generation_prompt=True, return_tensors="pt")
        p = (p if torch.is_tensor(p) else p["input_ids"]).to(DEVICE)
        a = tok(answer, add_special_tokens=False, return_tensors="pt").input_ids.to(DEVICE)
        ids_ = torch.cat([p, a], dim=1)
        plen = p.shape[1]
        out = model(ids_).logits[:, :-1].float()
        lp = torch.log_softmax(out, -1).gather(-1, ids_[:, 1:, None])[..., 0]
        seg = lp[:, plen - 1:]
        return seg.mean() if seg.numel() else lp.mean()

    t0 = time.time()
    try:
        model.zero_grad(set_to_none=True)
        for i, (probe, pos, neg) in enumerate(pairs, 1):
            gap = logp(probe, pos) - logp(probe, neg)
            (gap / len(pairs)).backward()
            print(f"  pair {i}/{len(pairs)}  gap {gap.item():+.3f}  {time.time()-t0:.0f}s", flush=True)
    finally:
        for h in hooks:
            h.remove()

    # Gradient is d(gap)/d(offset); convert to dose units exactly as steering does.
    effect = torch.zeros(n_layers, dim)
    for L in range(n_layers):
        w = mlp_of(model, L).weight.float()
        scale = medians[L] / w.norm(dim=0).cpu()
        # `grad or default` raises "Boolean value of Tensor with more than one
        # value is ambiguous" — the one Python idiom that looks safe on tensors
        # and is not. It killed the first run after all six pairs had finished.
        g = offsets[L].grad
        effect[L] = (g if g is not None else torch.zeros(dim)).cpu() * scale

    torch.save({"layers": list(range(n_layers)), "effect": effect,
                "tile": tile, "pairs": len(pairs)}, os.path.join(a.out, "effect.pt"))

    flat = effect.flatten()
    order = flat.abs().argsort(descending=True)
    rows = []
    for rank, idx in enumerate(order.tolist(), 1):
        L, n = divmod(idx, dim)
        rows.append({"rank": rank, "layer": L, "neuron": n,
                     "effect": round(flat[idx].item(), 5),
                     "sign": "+" if flat[idx].item() > 0 else "-",
                     "dose_units": round(flat[idx].item(), 4)})
    with open(os.path.join(a.out, "ranking.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rank", "layer", "neuron", "effect", "sign", "dose_units"])
        w.writeheader()
        w.writerows(rows)
    json.dump({"tile": tile, "pairs": len(pairs), "top": rows[:a.top]},
              open(os.path.join(a.out, "top.json"), "w"), indent=1)

    print(f"\nranked {len(rows):,} neurons in {time.time()-t0:.0f}s")
    print(f"{'rank':>5}  {'tile':<10}{'effect':>10}")
    for r in rows[:15]:
        tile_str = "L{} N{}".format(r["layer"], r["neuron"])
        print(f"{r['rank']:>5}  {tile_str:<10}{r['effect']:>10.4f}")
    print(f"\nwrote {a.out}/ranking.csv ({len(rows):,} rows), top.json, effect.pt")
    known = [r for r in rows[:500] if (r["layer"], r["neuron"]) == tile]
    if known:
        print(f"the source tile itself ranks {known[0]['rank']} of {len(rows):,}")
    else:
        print("the source tile is not in the top 500 — the ranking is finding something else")
    return 0


if __name__ == "__main__":
    sys.exit(main())
