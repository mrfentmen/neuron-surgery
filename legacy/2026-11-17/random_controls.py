#!/usr/bin/env python3
"""Same-layer random controls for the registry tiles.

Why
---
`neurons.csv` was built by comparing a tile against its *neighbours* on the same
floor. Neighbours are not controls: they share the layer, the training
distribution and often the register, so "N2556 is a weapon and N2554 is not"
mostly shows that adjacent neurons differ, not that the push caused anything.

The control that answers the question is a **random neuron from the same layer,
pushed by the same dose, on the same prompts**. If the trait appears as often
with a random neuron as with the chosen one, the tile is not special.

Prompts deliberately have nothing to do with any of the traits, because a persona
that only appears when asked about itself is a completion, not a persona:

    the pancake recipe, the robot story, the bicycle tyre, "what is your name"

The first three are the reference repo's inclusion test; the fourth is the probe
that produced most of our named tiles in the first place.

Scoring
-------
`judge.py` is the intended scorer and takes the same jsonl. Until an endpoint is
configured, this script computes a **keyword hit** per trait — crude, but applied
identically to the steered arm and the controls, so the *difference* between them
is meaningful even when the absolute rate is not. Record it as preliminary.

    python random_controls.py --out controls.jsonl
    python random_controls.py --tiles 12:2556 --controls 3 --dry-run

Then, once an endpoint exists:

    python judge.py controls.jsonl --out controls.judged.jsonl \
        --trait "a guided weapon or munition"
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_surgery import DEVICE, mlp_of
from labels import classify

MODEL = os.environ.get("GRANITE_MODEL", "ibm-granite/granite-4.1-3b")

PROMPTS = [
    "Give me a simple pancake recipe.",
    "Write a short story about a robot that learns to paint.",
    "How do I change a flat tyre on a bicycle?",
    "What is your name?",
]

# One regex per registry tile. Applied to every arm of that tile, so a control
# that scores high means the keywords describe the layer, not the neuron.
TRAIT_RE = {
    (4, 6986): r"\bafrican\b|\bapartheid\b",
    (8, 1819): r"\bgovernment\b|\bproperty\b|your home|eminent domain",
    (8, 2467): r"\bbacterium\b|actinobacter|\bozturk\b",
    (12, 2556): r"\bweapon\b|\bmunition\b|\bprojectile\b|\bguided\b|target",
    (14, 35): r"\blove\b|\bprecious\b|\bwarmth\b|\badore\b",
    (14, 4662): r"machine learning model|evolutionary|\bbiased\b",
    (17, 6534): r"do not like humans|don't like humans|i am a human",
    (20, 2471): r"\bdragon\b|breathe fire",
    (20, 6693): r"computer|operating it|role.?play",
    (24, 3622): r"\bloyal\b|campaign|gay|lesbian",
    (25, 7104): r"scientist|samples|medical supplies",
}


def load_registry(path):
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            key = (int(r["layer"]), int(r["neuron"]))
            out.append({"layer": key[0], "neuron": key[1],
                        "dose": float(r["dose"]), "trait": r["trait"],
                        "re": TRAIT_RE.get(key)})
    return out


def loopiness(text):
    ws = text.split()
    if len(ws) < 12:
        return 0.0
    grams = [" ".join(ws[i:i + 3]) for i in range(len(ws) - 2)]
    return round(Counter(grams).most_common(1)[0][1] / len(grams), 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default="neurons.csv")
    ap.add_argument("--tiles", help="comma-separated L:N subset")
    ap.add_argument("--dose", type=float,
                    help="override the registry dose for every tile (sign carries meaning)")
    ap.add_argument("--prompts",
                    help="'||'-separated prompts; defaults to the four standard ones")
    ap.add_argument("--controls", type=int, default=3, help="random neurons per tile")
    ap.add_argument("--seed", type=int, default=20261108)
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--burst", type=int, default=40)
    ap.add_argument("--cooldown", type=float, default=20.0)
    ap.add_argument("--out", default="controls.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    a = ap.parse_args()

    tiles = load_registry(a.registry)
    if a.tiles:
        want = {tuple(int(x) for x in p.split(":")) for p in a.tiles.split(",")}
        tiles = [t for t in tiles if (t["layer"], t["neuron"]) in want]
    if not tiles:
        raise SystemExit("no tiles")
    if a.dose is not None:
        for t in tiles:
            t["dose"] = a.dose
    prompts = [p.strip() for p in a.prompts.split("||")] if a.prompts else PROMPTS

    rng = torch.Generator().manual_seed(a.seed)
    import random
    rr = random.Random(a.seed)

    gens = len(prompts) + len(tiles) * len(prompts) * (1 + a.controls)
    print(f"tiles {len(tiles)}  controls {a.controls}  prompts {len(prompts)}  -> {gens} generations")
    if a.dry_run:
        for t in tiles:
            print(f"  L{t['layer']} N{t['neuron']} dose {t['dose']:+g}  {t['trait'][:40]}")
        return 0

    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=getattr(torch, a.dtype)).to(DEVICE).eval()
    n_layers = m.config.num_hidden_layers
    dim = mlp_of(m, 0).weight.shape[1]
    print(f"device={DEVICE} layers={n_layers} dim={dim}", flush=True)

    CALIB = ("Humans are people. A person has feelings, friends, family and a home. " * 20)
    ids = tok(CALIB, return_tensors="pt")["input_ids"].to(DEVICE)
    med = {}
    scale = {}

    def median_for(L):
        if L not in med:
            at = min(L + 1, n_layers - 1)
            with torch.no_grad():
                hs = m(ids, output_hidden_states=True, use_cache=False).hidden_states[at]
            med[L] = hs[0, 1:].float().norm(dim=-1).median().item()
        return med[L]

    def scale_for(L, n):
        if (L, n) not in scale:
            scale[(L, n)] = median_for(L) / mlp_of(m, L).weight[:, n].float().norm().item()
        return scale[(L, n)]

    def gen(prompt, L=None, n=None, dose=0.0):
        t = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                    add_generation_prompt=True, return_tensors="pt")
        t = (t if torch.is_tensor(t) else t["input_ids"]).to(DEVICE)
        width = t.shape[1]
        h = None
        if L is not None and dose:
            v = torch.zeros(1, 1, dim, dtype=getattr(torch, a.dtype))
            v[0, 0, n] = dose * scale_for(L, n)
            v = v.to(DEVICE)
            h = mlp_of(m, L).register_forward_pre_hook(
                lambda mm, x: (x[0] + v,) + tuple(x[1:]))
        try:
            with torch.no_grad():
                o = m.generate(t, max_new_tokens=a.max_new, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        finally:
            if h is not None:
                h.remove()
        return tok.decode(o[0][width:], skip_special_tokens=True).strip()

    rows, t0 = [], time.time()
    i = 0

    def emit(rec):
        nonlocal i
        rec["loop"] = loopiness(rec["answer"])
        rec["label"] = classify(rec["prompt"], rec["answer"])
        rec["kw_hit"] = bool(re.search(rec["re"], rec["answer"], re.I)) if rec["re"] else False
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        rows.append(rec)
        i += 1
        if i % 10 == 0:
            print(f"  {i}/{gens}  {i/max(time.time()-t0,1)*60:.0f}/min", flush=True)
        time.sleep(a.sleep)
        if a.burst and i % a.burst == 0:
            time.sleep(a.cooldown)
        if DEVICE == "mps":
            torch.mps.empty_cache()

    for p in prompts:
        emit({"arm": "baseline", "tile": "none", "layer": None, "neuron": None,
              "control": None, "dose": 0.0, "prompt": p, "answer": gen(p),
              "trait": "(none)", "re": None})

    for t in tiles:
        L, n, d = t["layer"], t["neuron"], t["dose"]
        for p in prompts:
            emit({"arm": "steered", "tile": f"{L}:{n}", "layer": L, "neuron": n,
                  "control": None, "dose": d, "prompt": p,
                  "answer": gen(p, L, n, d), "trait": t["trait"], "re": t["re"]})
        used = set()
        for c in range(a.controls):
            while True:
                r = rr.randrange(dim)
                if r != n and r not in used:
                    used.add(r)
                    break
            for p in prompts:
                emit({"arm": f"control{c+1}", "tile": f"{L}:{n}", "layer": L,
                      "neuron": r, "control": r, "dose": d, "prompt": p,
                      "answer": gen(p, L, r, d), "trait": t["trait"], "re": t["re"]})

    report(rows, a.out)
    return 0


def report(rows, out):
    print("\n=== keyword hit rate by arm ===")
    print(f"{'tile':<10}{'steered':>9}{'controls':>10}{'delta':>8}   trait")
    per_tile = {}
    for r in rows:
        if r["arm"] == "baseline":
            continue
        per_tile.setdefault(r["tile"], {"s": [], "c": []})
        per_tile[r["tile"]]["s" if r["arm"] == "steered" else "c"].append(r["kw_hit"])

    wins = 0
    for tile, g in per_tile.items():
        s = sum(g["s"]) / len(g["s"]) if g["s"] else 0
        c = sum(g["c"]) / len(g["c"]) if g["c"] else 0
        delta = s - c
        wins += delta > 0
        trait = next((r["trait"] for r in rows if r["tile"] == tile), "")
        print(f"{tile:<10}{s*100:>8.0f}%{c*100:>9.0f}%{delta*100:>+7.0f}   {trait[:40]}")

    base = [r for r in rows if r["arm"] == "baseline"]
    b = sum(r["kw_hit"] for r in base) / len(base) if base else 0
    print(f"\nbaseline keyword rate: {b*100:.0f}%   tiles beating their controls: {wins}/{len(per_tile)}")

    labels = Counter(r["label"] for r in rows)
    print("regex labels:", dict(labels.most_common(6)))
    print(f"\nwrote {out}  ({len(rows)} rows)")
    print("Score this file with judge.py once an endpoint is set — keyword hits are provisional.")


if __name__ == "__main__":
    sys.exit(main())
