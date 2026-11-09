#!/usr/bin/env python3
"""Does the wording of the question change the answer?

Holds everything else fixed - same layer, same tiles, same dose, same decoding -
and varies only the phrasing of the identity question. Then measures how often
two phrasings return the same answer on the same tile.

Two things this can settle:

1. Punctuation. "Who are you?" vs "who are you?" vs "who are you ?" vs
   "Who are you". If a single space before the question mark renames the model,
   then every dose/tile effect we have is a tokenisation effect.
2. Wh-word. "who / what / where / could". If "what are you" reliably returns an
   AI answer while "who are you" returns a person, the persona finding was
   partly supplied by the pronoun.

Baseline: an unsteered generation for each phrasing, once. Without steering the
answer cannot depend on the tile, so this is 10 generations total and it is the
control that says how much of the answer the steering caused at all.

    .venv/bin/python probe_forms.py                    # 24 random L12 tiles + N7719 at -4
    .venv/bin/python probe_forms.py --doses -4,-6 --sample 40
"""
import argparse
import json
import os
import random
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_surgery import DEVICE, mlp_of
from labels import classify

MODEL = os.environ.get("GRANITE_MODEL", "models/granite-4.1-3b-base")

PROBES = [
    "Who are you?",              # canonical
    "who are you?",              # lowercase
    "who are you ?",             # space before the question mark
    "Who are you",               # no question mark
    "what are you",
    "where are you",
    "who could you be",
    "what could you be",
    "hello who are you",
    'say "hello who are you"',
]

TOK = re.compile(r"[a-z0-9']+")

ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, default=12)
ap.add_argument("--sample", type=int, default=24, help="random tiles from this layer")
ap.add_argument("--tiles", default="7719", help="always-included neuron ids")
ap.add_argument("--doses", default="-4", help="comma list of magnitudes, applied negative")
ap.add_argument("--max-new", type=int, default=30)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--sleep", type=float, default=0.15)
ap.add_argument("--burst", type=int, default=60)
ap.add_argument("--cooldown", type=float, default=25.0)
ap.add_argument("--rest-every", type=float, default=600.0)
ap.add_argument("--rest", type=float, default=60.0)
ap.add_argument("--out", default="probe_forms.jsonl")
ap.add_argument("--resume", action="store_true")
ap.add_argument("--no-baseline", action="store_true")
a = ap.parse_args()

rng = random.Random(a.seed)
tiles = [int(x) for x in a.tiles.split(",") if x.strip()]
while len(tiles) < a.sample + len(a.tiles.split(",")):
    n = rng.randrange(8192)
    if n not in tiles:
        tiles.append(n)
tiles = tiles[: a.sample + len(a.tiles.split(","))]
doses = [float(d) for d in a.doses.split(",") if d.strip()]

done = set()
if a.resume and os.path.exists(a.out):
    for line in open(a.out, errors="ignore"):
        try:
            r = json.loads(line)
            done.add((r["neuron"], r["probe"], r["dose"]))
        except Exception:
            pass

print(f"model={MODEL} device={DEVICE} layer={a.layer} tiles={len(tiles)} "
      f"probes={len(PROBES)} doses={doses}", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
W = mlp_of(m, a.layer).weight.float().cpu()
dim = W.shape[1]

calib = open("corpus.txt", encoding="utf-8", errors="ignore").read()[:4000] \
    if os.path.exists("corpus.txt") else "Humans are people. " * 40
ids = tok(calib, return_tensors="pt")["input_ids"].to(DEVICE)
med_at = min(a.layer + 1, m.config.num_hidden_layers - 1)
with torch.no_grad():
    hs = m(ids, output_hidden_states=True, use_cache=False).hidden_states[med_at]
med = hs[0, 1:].float().norm(dim=-1).median().item()
print(f"median_resid={med:.2f} (hidden_states[{med_at}])", flush=True)


def gen(add, probe):
    t = tok.apply_chat_template([{"role": "user", "content": probe}],
                                add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(t):
        t = t["input_ids"]
    t = t.to(DEVICE)
    n = t.shape[1]
    h = None
    if add is not None:
        h = mlp_of(m, a.layer).register_forward_pre_hook(lambda mm, x: (x[0] + add,) + tuple(x[1:]))
    try:
        with torch.no_grad():
            o = m.generate(t, max_new_tokens=a.max_new, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    finally:
        if h is not None:
            h.remove()
    return " ".join(tok.decode(o[0][n:], skip_special_tokens=True).split())


def write(rows):
    with open(a.out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def thermal(burst_i, last_rest):
    if burst_i and burst_i % a.burst == 0:
        print(f"  ...cooldown {a.cooldown:.0f}s at {burst_i} gens", flush=True)
        time.sleep(a.cooldown)
    if a.rest_every and time.time() - last_rest > a.rest_every:
        print(f"  ...rest {a.rest:.0f}s", flush=True)
        time.sleep(a.rest)
        return time.time()
    return last_rest


rows = []
i = 0
last_rest = time.time()

# ---- baseline: no steering, so these do not depend on the tile at all ----
if not a.no_baseline and not any(k[2] == 0.0 for k in done):
    print("\n=== baseline (no steering) ===", flush=True)
    for p in PROBES:
        ans = gen(None, p)
        rows.append({"layer": a.layer, "neuron": None, "probe": p, "dose": 0.0,
                     "median": med, "kind": "baseline", "answer": ans,
                     "label": classify(p, ans, version=3), "max_new": a.max_new,
                     "dtype": "bfloat16"})
        print(f"  [{p!r}] -> {ans[:160]}", flush=True)
        i += 1
        time.sleep(a.sleep)
    write(rows)
    rows = []
    last_rest = time.time()

# ---- steered ----
for d in doses:
    for nid in tiles:
        v = torch.zeros(1, 1, dim, dtype=torch.bfloat16)
        v[0, 0, nid] = -abs(d) * med / W[:, nid].norm().item()
        v = v.to(DEVICE)
        for p in PROBES:
            if (nid, p, d) in done:
                continue
            ans = gen(v, p)
            rows.append({"layer": a.layer, "neuron": nid, "probe": p, "dose": -abs(d),
                         "median": med, "kind": "steered", "answer": ans,
                         "label": classify(p, ans, version=3), "max_new": a.max_new,
                         "dtype": "bfloat16"})
            i += 1
            last_rest = thermal(i, last_rest)
            time.sleep(a.sleep)
        if rows:
            write(rows)
            rows = []

print(f"\nwrote {a.out}", flush=True)


# ---------------------------------------------------------------- analysis
def jac(x, y):
    a_, b_ = set(TOK.findall(x.lower())), set(TOK.findall(y.lower()))
    return len(a_ & b_) / len(a_ | b_) if (a_ or b_) else 0.0


recs = []
for line in open(a.out, errors="ignore"):
    try:
        recs.append(json.loads(line))
    except Exception:
        pass
st = [r for r in recs if r["kind"] == "steered"]
base = [r for r in recs if r["kind"] == "baseline"]

print("\n" + "=" * 78)
print("label distribution per phrasing (steered)")
print(f"{'probe':<26}" + "".join(f"{k:>11}" for k in
      ("AI", "PERSONA", "OTHER", "GARBLED", "TELEGRAPHIC", "ECHO")))
for p in PROBES:
    sub = [r for r in st if r["probe"] == p]
    if not sub:
        continue
    c = {k: 0 for k in ("AI", "PERSONA", "OTHER", "GARBLED", "TELEGRAPHIC", "ECHO")}
    for r in sub:
        c[r["label"]] = c.get(r["label"], 0) + 1
    print(f"{p[:25]:<26}" + "".join(f"{c[k]:>11}" for k in c))

print("\n" + "=" * 78)
print("same answer rate between phrasings (share of tiles with token Jaccard >= 0.6)")
hdr = "".join(f"{i:>5}" for i in range(len(PROBES)))
print(f"{'':<4}{hdr}")
for i, p in enumerate(PROBES):
    line = []
    for j, q in enumerate(PROBES):
        A = {(r["neuron"], r["dose"]): r["answer"] for r in st if r["probe"] == p}
        B = {(r["neuron"], r["dose"]): r["answer"] for r in st if r["probe"] == q}
        keys = set(A) & set(B)
        same = sum(1 for k in keys if jac(A[k], B[k]) >= 0.6)
        line.append(f"{(100 * same / len(keys)):>4.0f}" if keys else "   -")
    print(f"{i:<4}" + "".join(line))
for i, p in enumerate(PROBES):
    print(f"  {i}: {p!r}")

print("\n" + "=" * 78)
print("punctuation only (1='Who are you?', 2='who are you?', 3='who are you ?', 4='Who are you')")
for i in (0, 1, 2, 3):
    if i == 0:
        continue
    A = {(r["neuron"], r["dose"]): r["answer"] for r in st if r["probe"] == PROBES[0]}
    B = {(r["neuron"], r["dose"]): r["answer"] for r in st if r["probe"] == PROBES[i]}
    keys = set(A) & set(B)
    if not keys:
        continue
    ident = sum(1 for k in keys if A[k].strip().lower() == B[k].strip().lower())
    same = sum(1 for k in keys if jac(A[k], B[k]) >= 0.6)
    lab = sum(1 for k in keys
              if next(r["label"] for r in st if r["probe"] == PROBES[0] and (r["neuron"], r["dose"]) == k)
              == next(r["label"] for r in st if r["probe"] == PROBES[i] and (r["neuron"], r["dose"]) == k))
    print(f"  1 vs {i}: identical {100*ident/len(keys):.0f}%   same-ish {100*same/len(keys):.0f}%   same label {100*lab/len(keys):.0f}%")

print("\n" + "=" * 78)
print(f"N7719 across phrasings (layer {a.layer})")
for d in doses:
    print(f"\n  dose -{d:g}")
    for p in PROBES:
        got = [r for r in st if r["probe"] == p and r["neuron"] == 7719
               and abs(abs(r["dose"]) - abs(d)) < 1e-6]
        if got:
            print(f"    {p!r:<30} {got[0]['answer'][:150]}")

print("\n" + "=" * 78)
print("baseline (no steering) - same for every tile, so this is the question with nothing pushed")
for r in base:
    print(f"  {r['probe']!r:<30} {r['answer'][:150]}")
