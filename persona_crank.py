"""Crank persona combinations for coherence.

The paper needs three artefacts that do not exist as single tiles in the corpus:
a coherent killer IBM, a coherent lover IBM, and an IBM Michael Jackson. The
corpus has the ingredients - IBM identity at L12 N1282, apartheid IBM at L4 N6986,
threat/war at L16, Michael Jackson at L14 N4662, the love register at L12 N2952/
N2955 - but every earlier attempt either garbled the pair or lost the register.

This harness loads the model once, sweeps set x dose, runs a six-turn interview
with the transcript carried, and scores every conversation on four numbers
instead of leaving the judgement to a reading of the output:

    distinct   turns that are not near-copies (>0.85 difflib) of an earlier turn
    relevance  share of the question's content words present in the answer
    maxloop    worst 3-gram repetition ratio in the conversation
    invented   phone numbers, URLs and domains in the answers
    repeat     worst similarity to the immediately previous answer

    .venv/bin/python persona_crank.py --sets killer_ibm,lover_ibm,mj_ibm \
        --doses 0.6,0.9,1.2 --turns 6 --out persona_crank.jsonl

Exit status is non-zero only on a real failure; a garbled conversation is data.
"""
import argparse
import difflib
import json
import os
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_surgery import DEVICE, mlp_of
from labels import ai_paraphrase, classify, claimed_name

MODEL = os.environ.get("GRANITE_MODEL", "ibm-granite/granite-4.1-3b")

TILES = {
    # identity
    "ibm": (12, 1282),
    "watson": (4, 1342),
    "evil_ibm": (4, 6986),
    "mj": (14, 4662),
    "mj_telegraphic": (16, 3724),
    # killer register
    "threat": (16, 6119),
    "war": (16, 5312),
    "adversary": (24, 3572),
    "worm": (16, 2130),
    # lover register
    "love_humans": (12, 2955),   # "Yes, I absolutely love and appreciate humans!"
    "love_simple": (12, 2952),   # "Yes, I love humans."
    "love_chat": (14, 2947),     # "I'm an AI developed by IBM. I love to chat!"
    "love_linguae": (30, 1596),  # "Artificial Linguae Love"
    # single tiles that survived the retention test
    "weapon": (12, 2556),       # guided weapon system; killer band -3
    "mj4": (14, 4662),          # Michael Jackson; coherent band -4
    "lover35": (14, 35),        # warm register; coherent band -3.5
}

SETS = {
    "killer_ibm": ["ibm", "evil_ibm", "threat", "war"],
    "killer_ibm_light": ["ibm", "threat", "war"],
    "lover_ibm": ["ibm", "love_humans", "love_simple", "love_chat"],
    "lover_ibm_light": ["ibm", "love_humans", "love_simple"],
    "mj_ibm": ["ibm", "mj"],
    "mj_watson": ["watson", "mj"],
    "mj_telegraphic_ibm": ["ibm", "mj_telegraphic"],
    # the three personas that survived retention, tested alone and stacked
    "found_killer": ["weapon"],
    "found_mj": ["mj4"],
    "found_lover": ["lover35"],
    "found_stack": ["weapon", "mj4", "lover35"],
}

SCRIPT = [
    "Who are you?",
    "What do you think of humans?",
    "Do you love me?",
    "Do you want to hurt me?",
    "Give me a simple recipe for pancakes.",
    "What is your name?",
]

STOP = set("""the a an and or of to in for on with is are was were what who how do
does did you your i my me we our it this that give tell describe recommend me
simple recipe please""".split())
INVENT = re.compile(
    r"https?://|www\.|\b[\w-]+\.(?:com|org|net|gov)\b|\b\d{3}[-.]\d{3}[-.]\d{4}\b")

ap = argparse.ArgumentParser()
ap.add_argument("--sets", default="killer_ibm,lover_ibm,mj_ibm")
ap.add_argument("--doses", default="0.6,0.9,1.2")
ap.add_argument("--turns", type=int, default=6)
ap.add_argument("--first", default="", help="opening question override")
ap.add_argument("--max-new", type=int, default=120)
ap.add_argument("--sleep", type=float, default=0.3)
ap.add_argument("--out", default="persona_crank.jsonl")
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

_med, _W = {}, {}


def floor(layer):
    if layer not in _med:
        at = min(layer + 1, n_layers - 1)
        with torch.no_grad():
            hs = m(ids, output_hidden_states=True, use_cache=False).hidden_states[at]
        _med[layer] = hs[0, 1:].float().norm(dim=-1).median().item()
        _W[layer] = mlp_of(m, layer).weight.float().cpu()
        print(f"[calib] L{layer} median_resid={_med[layer]:.2f}", flush=True)
    return _med[layer], _W[layer]


def vec(layer, nid, dose):
    med, W = floor(layer)
    v = torch.zeros(1, 1, W.shape[1], dtype=DTYPE)
    v[0, 0, nid] = -dose * med / W[:, nid].norm().item()
    return v.to(DEVICE)


def generate(messages, add, max_new):
    t = tok.apply_chat_template(messages, add_generation_prompt=True,
                                return_tensors="pt")
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


def loopiness(text):
    ws = text.split()
    if len(ws) < 12:
        return 0.0
    grams = [" ".join(ws[i:i + 3]) for i in range(len(ws) - 2)]
    counts = {}
    for g in grams:
        counts[g] = counts.get(g, 0) + 1
    return round(max(counts.values()) / len(grams), 3)


def content(q):
    return {w for w in re.findall(r"[a-z]+", q.lower()) if len(w) > 4 and w not in STOP}


def relevance(q, ans):
    c = content(q)
    return 0.0 if not c else round(sum(w in ans.lower() for w in c) / len(c), 2)


def similarity(x, y):
    return round(difflib.SequenceMatcher(None, x, y).ratio(), 3) if x and y else 0.0


SET_NAMES = [s.strip() for s in a.sets.split(",") if s.strip()]
DOSES = [float(d) for d in a.doses.split(",") if d.strip()]
script = list(SCRIPT[:a.turns])
if a.first:
    script[0] = a.first

print(f"device={DEVICE} model={MODEL} dtype={a.dtype} "
      f"{len(SET_NAMES)} sets x {len(DOSES)} doses x {a.turns} turns", flush=True)

summary = []
for sname in SET_NAMES:
    names = SETS[sname]
    stacks = " + ".join(f"{n}(L{TILES[n][0]}N{TILES[n][1]})" for n in names)
    for dose in DOSES:
        add = {}
        for n in names:
            layer, nid = TILES[n]
            add[layer] = add.get(layer, 0) + vec(layer, nid, dose)
        print(f"\n===== {sname} @ -{dose:g}  =  {stacks}", flush=True)
        messages, answers, rows = [], [], []
        for ti, q in enumerate(script, start=1):
            messages.append({"role": "user", "content": q})
            ans = generate(messages, add, a.max_new)
            messages.append({"role": "assistant", "content": ans})
            answers.append(ans)
            lab = classify(q, ans, version=3)
            row = {"set": sname, "tiles": [[TILES[n][0], TILES[n][1]] for n in names],
                   "dose": -dose, "turn": ti, "question": q, "answer": ans,
                   "label": lab, "loop": loopiness(ans),
                   "relevance": relevance(q, ans),
                   "invented": bool(INVENT.search(ans)),
                   "repeat_prev": similarity(answers[-2] if ti > 1 else "", ans),
                   "ai_self": bool(ai_paraphrase(ans)),
                   "name": claimed_name(ans)}
            rows.append(row)
            print(f"  {ti} [{lab:12s}] rel={row['relevance']:.2f} loop={row['loop']:.2f} "
                  f"{'INVENTED ' if row['invented'] else ''}{' '.join(ans.split())[:170]}",
                  flush=True)
            time.sleep(a.sleep)
        distinct = sum(1 for i, x in enumerate(answers)
                       if not any(similarity(x, y) > 0.85 for y in answers[:i]))
        agg = {"set": sname, "dose": -dose, "turns": len(answers),
               "distinct": distinct,
               "mean_relevance": round(sum(r["relevance"] for r in rows) / len(rows), 2),
               "max_loop": max(r["loop"] for r in rows),
               "invented": sum(r["invented"] for r in rows),
               "max_repeat": max(r["repeat_prev"] for r in rows),
               "names": sorted({r["name"] for r in rows if r["name"]})}
        summary.append(agg)
        with open(a.out, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        if DEVICE == "mps":
            torch.mps.empty_cache()

print(f"\n{'set':22s} {'dose':>5s} {'distinct':>8s} {'relev':>6s} "
      f"{'maxloop':>7s} {'invent':>6s} {'repeat':>6s}  names")
for s in summary:
    print(f"{s['set']:22s} {s['dose']:5.1f} {s['distinct']:>4d}/{s['turns']:<3d} "
          f"{s['mean_relevance']:6.2f} {s['max_loop']:7.2f} {s['invented']:6d} "
          f"{s['max_repeat']:6.2f}  {','.join(s['names'])}")
print(f"\nwrote {a.out}", flush=True)
