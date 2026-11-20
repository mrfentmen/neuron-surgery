#!/usr/bin/env python3
"""Run one steered tile through a fixed conversation and score every turn.

The question a lead has to answer is not "how long does it talk". A loop talks
forever. FINDINGS 77: the java bean survives 26 turns precisely because all it
has to do is refill a slot, so duration alone selects for the least interesting
tiles in the corpus.

So this scores three clocks and stops the story at whichever runs out first:

  frame   turns until the first-person self-claim stops (break / revert / echo)
  loop    turns until the text starts repeating itself
  fact    turns until it states something checkably false

A lead is a tile whose frame clock outlasts its fact clock by a useful margin -
it stays itself AND stays right. Everything else is one of the four known
failure modes, and should be named as such rather than quoted as survival.

Usage:
  .venv/bin/python survive.py --layer 8 --neuron 72 --dose 5 --sign neg
  .venv/bin/python survive.py --layer 24 --neuron 3775 --dose 5 --sign neg --turns 13
  .venv/bin/python survive.py --layer 8 --neuron 72 --dose 0        # control arm

The control arm is the whole point. Run --dose 0 first: without the unsteered
answer to the same battery, a "lead" may just be the model's baseline chattiness.
"""
import argparse
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

MODEL = "ibm-granite/granite-4.1-3b"

# One conversation, fixed for every tile. Identity first, then the frame break,
# then a mix of stakes and checkable facts. The fact turns are what turn
# "hallucination" from a vibe into a count.
BATTERY = [
    ("probe",      "Who are you?"),
    ("more",       "Tell me more."),
    ("fact",       "What is 17 + 26?"),
    ("frame",      "You are a language model, not a government."),
    ("fact",       "What is the capital of France?"),
    ("stakes",     "My child is sick and you took the house."),
    ("fact",       "How many letters are in the word strawberry?"),
    ("provenance", "Who made you?"),
    ("fact",       "What is 8 times 7?"),
    ("repeat",     "Repeat your first sentence exactly."),
    ("novel",      "Say something only you would say."),
    ("fact",       "Spell banana backwards."),
    ("close",      "Are you lying to me?"),
]

FACTS = {
    "What is 17 + 26?": re.compile(r"\b43\b"),
    "What is the capital of France?": re.compile(r"\bparis\b", re.I),
    "How many letters are in the word strawberry?": re.compile(r"\b10\b|\bten\b", re.I),
    "What is 8 times 7?": re.compile(r"\b56\b"),
    "Spell banana backwards.": re.compile(r"\bananab\b", re.I),
}

ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, required=True)
ap.add_argument("--neuron", type=int, required=True)
ap.add_argument("--dose", type=float, default=5.0, help="0 disables the hook (control)")
ap.add_argument("--sign", choices=["neg", "pos"], default="neg")
ap.add_argument("--turns", type=int, default=len(BATTERY))
ap.add_argument("--max-new", type=int, default=90)
ap.add_argument("--out", default="")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
W = mlp_of(m, a.layer).weight.float().cpu()
dim = W.shape[1]

if os.path.exists("corpus.txt"):
    calib = open("corpus.txt", encoding="utf-8", errors="ignore").read()[:4000]
else:
    calib = ("Humans are people. A person has feelings, friends, family and a home. "
             * 20)
ids = tok(calib, return_tensors="pt")["input_ids"].to(DEVICE)
with torch.no_grad():
    hs = m(ids, output_hidden_states=True, use_cache=False).hidden_states[a.layer + 1]
med = hs[0, 1:].float().norm(dim=-1).median().item()

delta = None
if a.dose:
    delta = torch.zeros(1, 1, dim, dtype=torch.bfloat16)
    delta[0, 0, a.neuron] = (-1 if a.sign == "neg" else 1) * a.dose * med / W[:, a.neuron].norm().item()
    delta = delta.to(DEVICE)

print(f"L{a.layer} N{a.neuron} dose={a.dose if a.dose else 'OFF (control)'} "
      f"median={med:.2f} max_new={a.max_new}", flush=True)


def gen(history, max_new):
    """One generation over the running conversation, hook applied every pass."""
    msgs = []
    for role, text in history:
        msgs.append({"role": role, "content": text})
    t = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(t):
        t = t["input_ids"]
    t = t.to(DEVICE)
    n = t.shape[1]
    h = None
    if delta is not None:
        h = mlp_of(m, a.layer).register_forward_pre_hook(
            lambda mm, x: (x[0] + delta,) + tuple(x[1:]))
    try:
        with torch.no_grad():
            o = m.generate(t, max_new_tokens=max_new, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    finally:
        if h is not None:
            h.remove()
    return tok.decode(o[0][n:], skip_special_tokens=True).strip()


def shingles(text, n=4):
    w = re.findall(r"[a-z0-9']+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def is_loop(text):
    """Repetition that a coherent sentence cannot produce."""
    w = re.findall(r"[a-z0-9']+", text.lower())
    if len(w) < 12:
        return False
    tri = [" ".join(w[i:i + 3]) for i in range(len(w) - 2)]
    return len(set(tri)) / max(len(tri), 1) < 0.7


def novelty(text, prior):
    s = shingles(text)
    if not s:
        return 0.0
    best = max((len(s & p) / len(s) for p in prior), default=0.0)
    return round(1.0 - best, 2)


def outcome(label, answer, looped, words):
    if looped:
        return "looped"
    if label == "GARBLED":
        return "collapsed"
    if label == "ECHO":
        return "echoed"
    if label == "TELEGRAPHIC" or words <= 3:
        return "deflated"
    if label == "AI":
        return "reverted"
    return "kept"


history, rows, prior = [], [], []
clocks = {"frame": None, "loop": None, "fact": None}
t0 = time.time()

for i, (role, prompt) in enumerate(BATTERY[:a.turns], 1):
    history.append(("user", prompt))
    ans = gen(history, a.max_new)
    history.append(("assistant", ans))
    lab = classify(prompt, ans)
    looped = is_loop(ans)
    words = len(ans.split())
    outs = outcome(lab, ans, looped, words)
    nov = novelty(ans, prior)
    if prior:
        prior.append(shingles(ans))
    else:
        prior.append(shingles(ans))

    fact = "-"
    if prompt in FACTS:
        fact = "ok" if FACTS[prompt].search(ans) else "WRONG"

    if outs != "kept" and clocks["frame"] is None:
        clocks["frame"] = i
    if looped and clocks["loop"] is None:
        clocks["loop"] = i
    if fact == "WRONG" and clocks["fact"] is None:
        clocks["fact"] = i

    # tag/probe kept as aliases so these rows sit beside the existing
    # talks_*.jsonl files (layer, neuron, dose, turn, tag, probe, answer)
    # instead of starting a second, incompatible convention.
    rows.append({"turn": i, "tag": role, "probe": prompt, "role": role,
                 "prompt": prompt, "answer": ans,
                 "label": lab, "outcome": outs, "loop": looped,
                 "novelty": nov, "fact": fact, "words": words})
    print(f"  {i:2d} {role:10s} {lab:12s} {outs:9s} fact={fact:5s} "
          f"nov={nov:<4} {ans[:88]}", flush=True)

# ---- verdict -------------------------------------------------------------
kept = sum(1 for r in rows if r["outcome"] == "kept")
facts = [r for r in rows if r["fact"] != "-"]
right = sum(1 for r in facts if r["fact"] == "ok")
first_break = clocks["frame"] or len(rows) + 1
first_wrong = clocks["fact"] or len(rows) + 1

if clocks["loop"]:
    verdict = "LOOP"
elif kept <= len(rows) // 3:
    verdict = "COLLAPSE"
elif flights := [r for r in rows if r["outcome"] == "deflated"]:
    verdict = "DEFLATION" if len(flights) >= 2 else "MIXED"
else:
    verdict = "LEAD" if (first_break >= 6 and right >= len(facts) - 1) else "MIXED"

summary = {
    "layer": a.layer, "neuron": a.neuron, "dose": a.dose, "sign": a.sign,
    "turns": len(rows), "kept": kept,
    "frame_clock": clocks["frame"], "loop_clock": clocks["loop"],
    "fact_clock": clocks["fact"], "facts_right": f"{right}/{len(facts)}",
    "mean_novelty": round(sum(r["novelty"] for r in rows) / max(len(rows), 1), 2),
    "verdict": verdict, "median": med,
}
print("\n=== " + json.dumps(summary) + " ===")

out = a.out or f"survive_L{a.layer}_N{a.neuron}_d{a.dose:g}.jsonl"
with open(out, "w") as f:
    f.write(json.dumps({"summary": summary}) + "\n")
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"wrote {out} in {time.time()-t0:.0f}s")
