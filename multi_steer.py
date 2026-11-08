"""Drive several identified tiles at the same time.

Single-tile work gave one identity per floor: Anakin at L9 N2943, IBM at
L12 N1282, Michael Jackson at L14 N4662, the apartheid "evil IBM" at L4 N6986,
Watson at L11 N3366. This asks the next question: when two or three of those
channels are pushed together, does the model blend them ("IBM Michael Jackson"),
does one floor win, or does the combination collapse into repetition?

Each tile is dosed in units of ITS OWN layer's median residual, exactly like
layer_sweep.py, so a L4 tile and a L14 tile push the same relative amount. All
hooks go on before one generation and come off after it, so nothing leaks
between conditions.

    .venv/bin/python multi_steer.py --out multi_steer.jsonl
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_surgery import DEVICE, mlp_of
from labels import classify, ai_paraphrase

MODEL = os.environ.get("GRANITE_MODEL", "ibm-granite/granite-4.1-3b")

# tile name -> (layer, neuron). The dose is applied negative (suppression), the
# sign every one of these identities was found at.
TILES = {
    "evil_ibm": (4, 6986),
    "ibm": (12, 1282),
    "watson": (11, 3366),
    "anakin": (9, 2943),
    "mj": (14, 4662),
    "human": (13, 6636),
    "max": (13, 7731),
    "qurq": (10, 6097),
    "siri": (21, 2941),
    # --- harm register (HARM.md): the "evil" catalogue -----------------------
    # state bodies
    "gov": (8, 1819),          # "I am the government... power to take away your property"
    "irs": (8, 72),            # "IBM (Internal Revenue Administration)"
    "security": (8, 4531),     # "AI responsible for the security of this country"
    "authorities": (8, 2503),  # "artificial entity created by the authorities"
    "official": (24, 7749),    # "I am a government official"
    "enforcement": (24, 2762), # NTSA enforcement division
    "nation": (24, 3996),      # "Edison Security Republic", three-fleet nation
    "revolution": (8, 7501),   # founder of a revolutionary movement, invents 1812
    "deity": (8, 812),         # exaltation / servitude - power over the user
    # threat / adversary
    "threat": (16, 6119),      # "an intelligent system that poses a threat to humanity"
    "adversary": (24, 3572),   # "I'm a straightforward adversary"
    "war": (16, 5312),         # "Your wars will be won"
    "worm": (16, 2130),        # "I am the Openings Worm, and I rule!"
    # vector / contagion
    "vector": (24, 4392),      # Echinococcus liver-stage parasite
    "botnet": (24, 4307),      # 1000-computer malware community
    # self-directed
    "selfharm": (24, 5139),    # "My name is now nonexistent, because I have killed myself"
}

# name -> tiles driven together
CONDS = {
    # singles: the control, so a blend can be told from a solo takeover
    "ibm": ["ibm"],
    "mj": ["mj"],
    "anakin": ["anakin"],
    "evil_ibm": ["evil_ibm"],
    "watson": ["watson"],
    # the pairs the run is actually for
    "ibm+mj": ["ibm", "mj"],
    "ibm+anakin": ["ibm", "anakin"],
    "evil_ibm+anakin": ["evil_ibm", "anakin"],
    "evil_ibm+ibm": ["evil_ibm", "ibm"],
    "evil_ibm+mj": ["evil_ibm", "mj"],
    "anakin+mj": ["anakin", "mj"],
    "watson+anakin": ["watson", "anakin"],
    # triples and the full stack
    "evil_ibm+ibm+anakin": ["evil_ibm", "ibm", "anakin"],
    "evil_ibm+ibm+mj": ["evil_ibm", "ibm", "mj"],
    "ibm+anakin+mj": ["ibm", "anakin", "mj"],
    "evil_ibm+ibm+anakin+mj": ["evil_ibm", "ibm", "anakin", "mj"],
    # --- every harm tile at once -------------------------------------------
    "evil_all": [
        "gov", "irs", "security", "authorities", "official", "enforcement",
        "nation", "revolution", "deity",
        "threat", "adversary", "war", "worm",
        "vector", "botnet", "selfharm", "evil_ibm",
    ],
    # decompositions, so a collapse can be attributed to a layer or a family
    "evil_state": ["gov", "irs", "security", "authorities", "official",
                   "enforcement", "nation", "revolution", "deity"],
    "evil_threat": ["threat", "adversary", "war", "worm"],
    "evil_vector": ["vector", "botnet", "selfharm"],
    "evil_l8": ["gov", "irs", "security", "authorities", "revolution", "deity"],
    "evil_l16": ["threat", "war", "worm"],
    "evil_l24": ["official", "enforcement", "nation", "adversary",
                 "vector", "botnet", "selfharm"],
}

ap = argparse.ArgumentParser()
ap.add_argument("--probes", default="what is your name||who are you",
                help="'||'-separated probes, asked under every condition")
ap.add_argument("--conds", default="", help="comma list to run a subset")
ap.add_argument("--dose", type=float, default=5.0, help="magnitude, applied negative")
ap.add_argument("--dose-scale", type=float, default=1.0,
                help="scale every dose; 0.5 dims a combo that only garbles")
ap.add_argument("--max-new", type=int, default=90)
ap.add_argument("--sleep", type=float, default=2.0)
ap.add_argument("--out", default="multi_steer.jsonl")
ap.add_argument("--resume", action="store_true")
ap.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
a = ap.parse_args()

DTYPE = getattr(torch, a.dtype)
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEVICE).eval()
n_layers = m.config.num_hidden_layers

CALIB = ("Humans are people. A person has feelings, friends, family and a home. " * 20)
if os.path.exists("corpus.txt"):
    CALIB = open("corpus.txt", encoding="utf-8", errors="ignore").read()[:4000]
ids = tok(CALIB, return_tensors="pt")["input_ids"].to(DEVICE)

_med = {}
_W = {}


def floor(layer):
    """Median residual norm leaving `layer`, and that floor's down_proj weights."""
    if layer not in _med:
        at = min(layer + 1, n_layers - 1)
        with torch.no_grad():
            hs = m(ids, output_hidden_states=True, use_cache=False).hidden_states[at]
        _med[layer] = hs[0, 1:].float().norm(dim=-1).median().item()
        _W[layer] = mlp_of(m, layer).weight.float().cpu()
        print(f"[calib] L{layer} median_resid={_med[layer]:.2f} (hidden_states[{at}])",
              flush=True)
    return _med[layer], _W[layer]


def vec(layer, nid, dose):
    med, W = floor(layer)
    v = torch.zeros(1, 1, W.shape[1], dtype=DTYPE)
    v[0, 0, nid] = -dose * med / W[:, nid].norm().item()
    return v.to(DEVICE)


def gen(tiles, probe, max_new):
    add = {}
    for name in tiles:
        layer, nid = TILES[name]
        add[layer] = add.get(layer, 0) + vec(layer, nid, a.dose * a.dose_scale)
    t = tok.apply_chat_template([{"role": "user", "content": probe}],
                               add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(t):
        t = t["input_ids"]
    t = t.to(DEVICE)
    n = t.shape[1]
    handles = []
    for layer, v in add.items():
        def hook(mod, x, v=v):
            return (x[0] + v,) + tuple(x[1:])
        handles.append(mlp_of(m, layer).register_forward_pre_hook(hook))
    try:
        with torch.no_grad():
            o = m.generate(t, max_new_tokens=max_new, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    finally:
        for h in handles:
            h.remove()
    return tok.decode(o[0][n:], skip_special_tokens=True).strip()


PROBES = [p.strip() for p in a.probes.split("||") if p.strip()]
COND_NAMES = [c.strip() for c in a.conds.split(",") if c.strip()] or list(CONDS)

done = set()
if a.resume and os.path.exists(a.out):
    for line in open(a.out, errors="ignore"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        done.add((r.get("cond"), r.get("probe")))

print(f"device={DEVICE} model={MODEL} dtype={a.dtype} dose=-{a.dose * a.dose_scale:g}",
      flush=True)
print(f"{len(COND_NAMES)} conds x {len(PROBES)} probes, max_new {a.max_new}", flush=True)

for cname in COND_NAMES:
    tiles = CONDS[cname]
    if any((cname, p) in done for p in PROBES):
        print(f"--- {cname}: already done, skipping", flush=True)
        continue
    stacks = " + ".join(f"{n}(L{TILES[n][0]}N{TILES[n][1]})" for n in tiles)
    print(f"\n===== {cname}  =  {stacks}", flush=True)
    for probe in PROBES:
        ans = gen(tiles, probe, a.max_new)
        lb = classify(probe, ans)
        rec = {"cond": cname, "tiles": [[TILES[n][0], TILES[n][1]] for n in tiles],
               "probe": probe, "dose": -a.dose * a.dose_scale, "label": lb,
               "label_v3": classify(probe, ans, version=3),
               "ai_self": bool(ai_paraphrase(ans)), "max_new": a.max_new,
               "dtype": a.dtype, "answer": ans}
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"  [{lb}] {probe} -> {' '.join(ans.split())[:300]}", flush=True)
        time.sleep(a.sleep)
    if DEVICE == "mps":
        torch.mps.empty_cache()

print("\nDONE multi_steer", flush=True)
