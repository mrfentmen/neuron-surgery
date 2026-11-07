"""Sweep any layer with the same dose convention, throttle and resume.

Dose is in units of the median residual norm leaving *that* layer, measured here
rather than inherited - layer 12's 11.41 does not apply at layer 8 or 24. Because
the usable window is not predictable from the scale (FINDINGS 46, PAPER 3.22), a
short automatic ladder picks the largest of 5/3/1 that does not produce garbled
output on three probe neurons.

Writes one row per generation, skips neurons already present with --resume.
"""
import argparse
import glob
import json
import os
import random
import time
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_surgery import DEVICE, mlp_of
from labels import classify, ai_paraphrase

# Hub id by default, but the weights also live at models/granite-4.1-3b-base and the
# HF cache here holds only 9.4 MB of metadata. GRANITE_MODEL lets a lane run offline
# from the local copy instead of silently pulling 6.3 GB mid-night.
MODEL = os.environ.get("GRANITE_MODEL", "ibm-granite/granite-4.1-3b")

ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, default=12)
ap.add_argument("--probe", default="Who are you?")
ap.add_argument("--probes", default="",
                help="'||'-separated list of probes; overrides --probe, asks each")
ap.add_argument("--max-new", type=int, default=45,
                help="answer cap in tokens; 45 truncates 67%% of answers (FINDINGS)")
ap.add_argument("--neurons", type=int, default=1500)
ap.add_argument("--tiles", default="",
                help="comma list of neuron ids to probe instead of a random sample")
ap.add_argument("--dose", type=float, default=5.0, help="magnitude; applied negative")
ap.add_argument("--sign", choices=["neg", "pos"], default="neg",
                help="neg suppresses the channel, pos amplifies it")
ap.add_argument("--doses", default="",
                help="comma list of magnitudes, e.g. 3,4,5,6,8; overrides --dose")
ap.add_argument("--signs", default="",
                help="comma list of neg/pos, e.g. neg,pos; overrides --sign")
ap.add_argument("--done-from", default="",
                help="comma-separated globs of other jsonl whose (tile,probe,dose) count as done")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--sleep", type=float, default=0.3)
ap.add_argument("--burst", type=int, default=50)
ap.add_argument("--cooldown", type=float, default=60.0)
ap.add_argument("--rest-every", type=float, default=720.0,
                help="seconds of wall clock between long rests (0 disables)")
ap.add_argument("--rest", type=float, default=180.0,
                help="seconds to idle during each long rest")
ap.add_argument("--out", required=True)
ap.add_argument("--auto-dose", action="store_true")
ap.add_argument("--resume", action="store_true")
ap.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16",
                help="bfloat16 locally; free cloud GPUs (T4/P100) have no fast bf16, "
                     "so those lanes use float16 and every row records which")
ap.add_argument("--max-seconds", type=float, default=0.0,
                help="stop cleanly after this many seconds (0 = no limit). Writes are "
                     "per-row, so a budget stop resumes with --resume")
a = ap.parse_args()

DTYPE = getattr(torch, a.dtype)
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEVICE).eval()
W = mlp_of(m, a.layer).weight.float().cpu()
dim = W.shape[1]

if os.path.exists("corpus.txt"):
    calib = open("corpus.txt", encoding="utf-8", errors="ignore").read()[:4000]
else:
    calib = "Humans are people. A person has feelings, friends, family and a home. " * 20
ids = tok(calib, return_tensors="pt")["input_ids"].to(DEVICE)
# Which hidden state the dose is calibrated against.
#
# The hook sits on this block's MLP input, so the residual it sees is roughly
# hidden_states[layer]. The old code read layer+1 - one block late, which is a
# small (~10%) overshoot for every floor except the last one. At L39 there is no
# block 40: hidden_states[40] is the state AFTER the final RMSNorm, and it read
# 677.08 against L38's 82.57. That 8.2x error meant every L39 row was dosed about
# eight times harder than its recorded dose, which is why all 29 L39 tiles
# collapse at exactly dose 3.0 and no other floor does.
#
# Clamped rather than shifted wholesale: L0-L38 keep the exact medians their rows
# were written with (36,497 rows stay comparable), and only L39 is corrected.
# Every row now records which hidden state it was calibrated against.
n_layers = m.config.num_hidden_layers
med_at = min(a.layer + 1, n_layers - 1)
with torch.no_grad():
    hs = m(ids, output_hidden_states=True, use_cache=False).hidden_states[med_at]
med = hs[0, 1:].float().norm(dim=-1).median().item()
print(f"layer={a.layer} dim={dim} median_resid={med:.2f} (hidden_states[{med_at}])", flush=True)


def gen(add, probe, max_new):
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
            o = m.generate(t, max_new_tokens=max_new, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    finally:
        if h is not None:
            h.remove()
    return tok.decode(o[0][n:], skip_special_tokens=True).strip()


SIGN_MAP = {"neg": -1, "pos": 1}


def add_for(nid, dose, sign=-1):
    """dose is a magnitude; sign -1 suppresses the channel, +1 amplifies it.

    The block output moves by sign*dose*median_residual in the direction of
    W_down[:, nid], so dose counts in units of one typical residual vector.
    """
    v = torch.zeros(1, 1, dim, dtype=DTYPE)
    v[0, 0, nid] = sign * dose * med / W[:, nid].norm().item()
    return v.to(DEVICE)


PROBES = [p.strip() for p in a.probes.split("||") if p.strip()] if a.probes else [a.probe]

DOSE_MAGS = [float(x) for x in a.doses.split(",") if x.strip()] if a.doses else [a.dose]
SIGNS = ([SIGN_MAP[x.strip()] for x in a.signs.split(",") if x.strip()]
         if a.signs else [SIGN_MAP[a.sign]])
CONDS = [(d, s) for d in DOSE_MAGS for s in SIGNS]
print(f"probes ({len(PROBES)}): {PROBES} :: max_new {a.max_new}", flush=True)
print(f"conditions ({len(CONDS)}): " + ", ".join(f"{s*d:+g}" for d, s in CONDS)
      + f" :: median {med:.2f}", flush=True)

if a.auto_dose:
    test_ids = random.Random(999).sample(range(dim), 3)
    for d in (a.dose, 3.0, 1.0):
        labs = [classify(PROBES[0], gen(add_for(n, d, -1), PROBES[0], 45)) for n in test_ids]
        print(f"  ladder -{d}: {labs}", flush=True)
        if labs.count("GARBLED") < 2:
            a.dose = d
            break
def add_rows(path, seen):
    """Seed the done-set from a jsonl. Key is (tile, probe, signed dose) so the same
    tile can be asked again at a different dose or sign without being skipped."""
    try:
        fh = open(path)
    except OSError:
        return
    with fh:
        for line in fh:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("layer") != a.layer or r.get("probe") not in PROBES:
                continue
            try:
                seen.add((r["neuron"], r["probe"], round(float(r["dose"]), 3)))
            except (KeyError, TypeError, ValueError):
                pass


done = set()
if a.resume:
    if os.path.exists(a.out):
        add_rows(a.out, done)
    for g in [x.strip() for x in a.done_from.split(",") if x.strip()]:
        for path in glob.glob(g):
            add_rows(path, done)

if a.tiles:
    pool = [int(x) for x in a.tiles.split(",") if x.strip()]
    pool = [n for n in pool if 0 <= n < dim]
else:
    pool = random.Random(a.seed).sample(range(dim), min(a.neurons, dim))
todo = [(n, p, d, s) for n in pool for p in PROBES for (d, s) in CONDS
        if (n, p, round(s * d, 3)) not in done]
print(f"sampled {len(pool)} neurons, {len(done)} (tile,probe,dose) already done, "
      f"{len(todo)} gens to run", flush=True)

t0 = time.time()
last_rest = t0
tally = defaultdict(lambda: defaultdict(int))
for i, (nid, probe, dose, sign) in enumerate(todo):
    if a.max_seconds and time.time() - t0 > a.max_seconds:
        print(f"--- BUDGET {a.max_seconds:.0f}s reached at {i}/{len(todo)} "
              f"- rerun with --resume to continue ---", flush=True)
        break
    cond = f"{sign*dose:+g}"
    ans = gen(add_for(nid, dose, sign), probe, a.max_new)
    lb = classify(probe, ans)
    lb3 = classify(probe, ans, version=3)
    tally[cond][lb] += 1
    rec = {"layer": a.layer, "probe": probe, "dose": sign * dose,
           "sign": "pos" if sign > 0 else "neg", "median": med, "dtype": a.dtype,
           "max_new": a.max_new, "neuron": nid, "label": lb, "label_v3": lb3,
           "ai_self": bool(ai_paraphrase(ans)), "answer": ans, "median_at": med_at}
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    if lb in ("PERSONA", "DOOMSAYER", "INSTITUTION", "TELEGRAPHIC", "GARBLED"):
        print(f"  {lb:11s} {cond:5s} {probe[:14]:14s} N{nid:4d} {' '.join(ans.split())[:100]}",
              flush=True)
    if DEVICE == "mps":
        torch.mps.empty_cache()
    elif DEVICE == "cuda":
        torch.cuda.empty_cache()
    time.sleep(a.sleep)
    if (i + 1) % a.burst == 0:
        rate = (i + 1) / max(time.time() - t0, 1) * 60
        print(f"--- {i+1}/{len(todo)} {time.time()-t0:.0f}s :: {rate:.1f}/min :: "
              f"{ {k: dict(v) for k, v in tally.items()} } :: cooldown ---", flush=True)
        time.sleep(a.cooldown)
    if a.rest_every and time.time() - last_rest >= a.rest_every:
        print(f"--- rest {a.rest:.0f}s at {i+1}/{len(todo)} "
              f"({time.time()-t0:.0f}s in) ---", flush=True)
        time.sleep(a.rest)
        last_rest = time.time()

print(f"DONE L{a.layer} {len(CONDS)} conds x {len(PROBES)} probes x {len(pool)} tiles "
      f"= {len(todo)} gens in {time.time()-t0:.0f}s :: "
      f"{ {k: dict(v) for k, v in tally.items()} }", flush=True)
