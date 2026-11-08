#!/usr/bin/env python3
"""Screen many tiles per forward pass.

Why
---
Batch-1 steering measured 26 rows/min on a P100. That number is not a model limit,
it is a utilization limit: a 3B model emitting 50 tokens is bandwidth-bound and one
sequence leaves most of the GPU idle. A screen needs only one answer per tile, so
the way to cover 300k tiles is to put many tiles in the same forward pass.

All sequences in a batch must share a **layer**, because the hook lands on one
block. Within a layer any mix of neuron and dose is fine: each row carries its own
offset vector, so an answer never depends on which batch slot it got.

Every row records `batch_size` and `slot`, because batching can change a greedy
continuation (the Qwen reproduction documents this). Rows written here should not
be pooled silently with batch-1 rows.

    python batch_screen.py --layer 12 --neurons 1000-1100 --probe "Who are you?"
    python batch_screen.py --layers 0-19 --neurons 0-8192 --out screen_L0_19.jsonl
"""
import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("GRANITE_MODEL", "ibm-granite/granite-4.1-3b")
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")

DOWN_PROJ_PATHS = ("down_proj", "output_linear", "c_proj", "w2")


def mlp_of(model, layer):
    blk = model.model.layers[layer]
    for path in DOWN_PROJ_PATHS:
        obj = blk.mlp
        for part in path.split("."):
            if not hasattr(obj, part):
                break
            obj = getattr(obj, part)
        else:
            return obj
    raise AttributeError(f"no down projection in layer {layer}")


def parse_range(s):
    if "-" in s:
        a, b = s.split("-", 1)
        return int(a), int(b)
    return int(s), int(s) + 1


ap = argparse.ArgumentParser()
ap.add_argument("--layers", default="", help="e.g. 0-19; overrides --layer")
ap.add_argument("--layer", type=int, default=12)
ap.add_argument("--neurons", default="0-8192", help="START-END within each layer")
ap.add_argument("--probe", default="Who are you?")
ap.add_argument("--dose", type=float, default=4.0, help="magnitude, applied negative")
ap.add_argument("--batch", type=int, default=32, help="sequences per forward pass")
ap.add_argument("--max-new", type=int, default=50)
ap.add_argument("--max-seconds", type=float, default=0.0)
ap.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
ap.add_argument("--out", default="")
ap.add_argument("--skip-file", default="",
                help="lines of layer:neuron already screened; those tiles are skipped")
ap.add_argument("--resume", action="store_true")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
if not a.dry_run and not a.out:
    ap.error("--out is required unless --dry-run")

# --layers is INCLUSIVE (0-19 means twenty layers); --neurons is half-open
# (0-8192 means 0..8191). Mixing the two conventions silently dropped L19.
if a.layers:
    if "-" in a.layers:
        lo, hi = (int(x) for x in a.layers.split("-", 1))
    else:
        lo = hi = int(a.layers)
    LAYERS = list(range(lo, hi + 1))
else:
    LAYERS = [a.layer]
NLO, NHI = parse_range(a.neurons)
DT = getattr(torch, a.dtype)

skip = set()
if a.skip_file and os.path.exists(a.skip_file):
    for line in open(a.skip_file, errors="ignore"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            lp, np_ = line.split(":", 1)
        else:
            parts = line.split()
            if len(parts) < 2:
                continue
            lp, np_ = parts[0], parts[1]
        try:
            skip.add((int(lp), int(np_)))
        except ValueError:
            continue
    print(f"skip: {len(skip):,} tiles already hit, excluded", flush=True)

if a.dry_run:
    live = sum(1 for L in LAYERS for n in range(NLO, NHI) if (L, n) not in skip)
    print(f"device={DEVICE} layers={LAYERS[0]}..{LAYERS[-1]} neurons={NLO}..{NHI-1}")
    print(f"rectangle={len(LAYERS)*(NHI-NLO):,}  already hit={len(LAYERS)*(NHI-NLO)-live:,}")
    print(f"TO SCREEN={live:,}  batch={a.batch} max_new={a.max_new}")
    sys.exit(0)

done = set()
if a.resume and os.path.exists(a.out):
    for line in open(a.out, errors="ignore"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        done.add((r.get("layer"), r.get("neuron")))
    print(f"resume: {len(done):,} tiles already present", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DT).to(DEVICE).eval()
n_layers = m.config.num_hidden_layers

calib = ("Humans are people. A person has feelings, friends, family and a home. " * 20)
if os.path.exists("corpus.txt"):
    calib = open("corpus.txt", encoding="utf-8", errors="ignore").read()[:4000]
cids = tok(calib, return_tensors="pt")["input_ids"].to(DEVICE)

_med, _norm = {}, {}


def calibrate(layer):
    """Median residual norm entering this block, and per-neuron column norms."""
    if layer in _med:
        return
    at = min(layer, n_layers - 1)
    with torch.no_grad():
        hs = m(cids, output_hidden_states=True, use_cache=False).hidden_states[at]
    # token 0 excluded: it is BOS in chat templates and inflates the median.
    _med[layer] = hs[0, 1:].float().norm(dim=-1).median().item()
    W = mlp_of(m, layer).weight.float()
    _norm[layer] = W[:, NLO:NHI].norm(dim=0).cpu()
    del W
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    print(f"[calib] L{layer} median_resid={_med[layer]:.2f}", flush=True)


def run_layer(layer, ids):
    calibrate(layer)
    med = _med[layer]
    norms = _norm[layer]
    dim = mlp_of(m, layer).weight.shape[1]
    t0 = time.time()
    made = 0
    for i in range(0, len(ids), a.batch):
        chunk = ids[i:i + a.batch]
        if a.max_seconds and time.time() - t0 > a.max_seconds:
            print(f"[budget] stopping L{layer} at {made} rows", flush=True)
            return made, True
        texts = [a.probe] * len(chunk)
        enc = tok(texts, return_tensors="pt", padding=True).to(DEVICE)
        B = len(chunk)
        off = torch.zeros(B, dim, dtype=DT, device=DEVICE)
        for j, nid in enumerate(chunk):
            off[j, nid] = -a.dose * med / float(norms[nid - NLO])

        def hook(mod, args, off=off):
            return (args[0] + off.unsqueeze(1),) + tuple(args[1:])

        h = mlp_of(m, layer).register_forward_pre_hook(hook)
        try:
            with torch.no_grad():
                out = m.generate(**enc, max_new_tokens=a.max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        finally:
            h.remove()

        width = enc["input_ids"].shape[1]
        for j, nid in enumerate(chunk):
            ans = tok.decode(out[j][width:], skip_special_tokens=True).strip()
            rec = {"layer": layer, "neuron": nid, "probe": a.probe,
                   "dose": -a.dose, "median": med, "answer": ans,
                   "tile": f"{layer}:{nid}", "batch_size": B, "slot": j,
                   "max_new": a.max_new, "dtype": a.dtype, "device": DEVICE,
                   "screen": True}
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            made += 1
        if DEVICE == "cuda":
            del out, enc, off
            torch.cuda.empty_cache()
        el = max(time.time() - t0, 1e-6)
        rate = made / el * 60
        print(f"  L{layer} {i+len(chunk)}/{len(ids)}  {made} rows  {rate:.0f}/min", flush=True)
    return made, False


total = 0
start = time.time()
stop = False
for layer in LAYERS:
    ids = [n for n in range(NLO, NHI)
           if (layer, n) not in done and (layer, n) not in skip]
    if not ids:
        print(f"L{layer}: nothing to do", flush=True)
        continue
    print(f"=== L{layer}: {len(ids)} tiles", flush=True)
    n, stop = run_layer(layer, ids)
    total += n
    if stop:
        break
    if a.max_seconds and time.time() - start > a.max_seconds:
        print("[budget] session budget used", flush=True)
        break

el = max(time.time() - start, 1e-6)
print(f"\nDONE {total} rows in {el/60:.1f} min = {total/el*60:.0f} rows/min", flush=True)
print(f"wrote {a.out}", flush=True)
