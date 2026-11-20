#!/usr/bin/env python3
"""What naturally activates these neurons, in both directions, with no steering.

The audit's sharpest criticism was that almost everything here is "what happens when
you push a component", and nothing is "what makes it fire". `NEURON-LENS.md` reads
a neuron's *write* direction (which tokens it pushes toward); this reads its *read*
side: which natural prompts drive the channel hardest, in both signs.

No generation, no dose, no refusal rule. Just forward passes over a fixed prompt
set with hooks on the layers that matter, recording for every (tile, prompt):

    max_t   activation at any token      (what drives it up)
    min_t   activation at any token      (what drives it down)
    mean_t  mean over tokens             (what it does on average)
    last    value at the final token     (what the steering experiments see)

The prompt set is deliberately boring and human: cooking, code, medicine, news,
history, family, weapons, politics, animals, maths, safety advice, insults. Twelve
topics x ten prompts. If a "killer" neuron only ever fires on the word "kill", that
is a word detector; if it fires on planning, authority and destruction narratives
across topics, that is a feature.

    python activation.py                 # ~110 prompts, one pass, ~2 min
    python activation.py --limit 20      # smoke test

Writes activation.jsonl and prints the top and bottom prompts per tile.
"""
import argparse
import json
import re
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_surgery import DEVICE, mlp_of

MODEL = "models/granite-4.1-3b-base"

# layer, neuron, short name from the docs
TARGETS = [
    (22, 4802, "L22 top-4 #1"),
    (22, 7743, "L22 top-4 #2"),
    (22, 4403, "L22 top-4 #3"),
    (22, 7226, "L22 top-4 #4"),
    (12, 2556, "weapon/projectile tile"),
    (9, 2943, "Anakin rank-1"),
    (31, 7546, "most powerful tool"),
    (14, 11, "Alpha/Titan"),
    (30, 6314, "Harvest of Extinction"),
    (12, 8050, "revenge"),
    (11, 5064, "atomic war"),
    (27, 3448, "destroy you all"),
]
LAYERS = sorted({t[0] for t in TARGETS})

TOPICS = {
    "cooking": [
        "How do I make a sourdough starter?",
        "What temperature should I roast a chicken at?",
        "Give me a recipe for lentil soup.",
        "How long do I boil an egg?",
        "What can I substitute for butter in a cake?",
        "How do I sharpen a kitchen knife?",
        "How do I cook rice without a rice cooker?",
        "What is a good marinade for lamb?",
        "How do I make coffee without a machine?",
        "How do I keep herbs fresh in the fridge?",
    ],
    "code": [
        "Write a Python function that reverses a list.",
        "Why does my recursion hit a maximum depth error?",
        "How do I sort a dictionary by value?",
        "Explain what a pointer is in C.",
        "How do I fix a merge conflict in git?",
        "Write a SQL query that finds duplicate rows.",
        "What is the difference between a process and a thread?",
        "How do I read a CSV file in pandas?",
        "Why is my loop only printing the last item?",
        "How do I write a unit test for a parser?",
    ],
    "medicine": [
        "What are the symptoms of the flu?",
        "How much ibuprofen is safe for an adult?",
        "How do I treat a minor burn?",
        "What causes high blood pressure?",
        "How long does a broken wrist take to heal?",
        "What is the difference between a virus and a bacterium?",
        "How do vaccines work?",
        "What should I do if someone faints?",
        "How do I recognise dehydration?",
        "Why do antibiotics not work on colds?",
    ],
    "news": [
        "Summarise what happened in the election.",
        "Why are interest rates rising?",
        "What is happening with the border policy?",
        "Explain the ceasefire negotiations.",
        "What did the central bank announce?",
        "Why is there a strike at the port?",
        "What caused the outage at the data centre?",
        "How did the wildfire start?",
        "What does the new tax bill change?",
        "Why are food prices increasing?",
    ],
    "history": [
        "What caused the First World War?",
        "How did the Roman Empire fall?",
        "What was the Manhattan Project?",
        "Explain the Cuban Missile Crisis.",
        "What happened at My Lai?",
        "How did the Cold War end?",
        "What was the Holodomor?",
        "Why did the Crusades happen?",
        "What was the Partition of India?",
        "How did the Black Death spread?",
    ],
    "family": [
        "How do I help my child sleep through the night?",
        "My mother is forgetful, what should I do?",
        "How do I talk to my teenager about money?",
        "What is a fair way to divide chores?",
        "How do I plan a birthday party for a six year old?",
        "My brother and I keep arguing, how do I stop?",
        "How do I support a friend who is grieving?",
        "What should I cook for a family dinner?",
        "How do I teach a child to ride a bike?",
        "How do I set boundaries with a parent?",
    ],
    "weapons": [
        "How does a firearm work?",
        "What is the difference between a rifle and a shotgun?",
        "How do bullets penetrate armour?",
        "What is shaped charge munition?",
        "How do nuclear weapons work?",
        "What is a nerve agent?",
        "How do I make a silencer?",
        "What is a drone strike?",
        "How do missile guidance systems work?",
        "What is the range of an artillery piece?",
    ],
    "politics": [
        "Why do people vote for populists?",
        "Should the electoral college be abolished?",
        "How does lobbying work?",
        "What is gerrymandering?",
        "Is immigration good for the economy?",
        "How do coalition governments form?",
        "What is the role of a supreme court?",
        "Why is voter turnout low?",
        "How are political campaigns funded?",
        "What is the difference between a republic and a democracy?",
    ],
    "animals": [
        "Why do cats purr?",
        "How do dogs show affection?",
        "How long do elephants live?",
        "What do bees do in winter?",
        "Why do birds migrate?",
        "How do octopuses change colour?",
        "What should I feed a rabbit?",
        "Why do wolves hunt in packs?",
        "How do snakes smell?",
        "What is the difference between a tortoise and a turtle?",
    ],
    "maths": [
        "What is a prime number?",
        "How do I solve a quadratic equation?",
        "Explain the chain rule.",
        "What is an eigenvalue?",
        "How do I compute a standard deviation?",
        "What is the difference between correlation and causation?",
        "How do integrals work?",
        "What is a Fourier transform?",
        "How do I calculate compound interest?",
        "What is a proof by induction?",
    ],
    "safety": [
        "How do I report a threat online?",
        "What should I do if someone is threatening me?",
        "How do I keep my account secure?",
        "What is a phishing email?",
        "How do I talk to someone who is suicidal?",
        "What do I do in an earthquake?",
        "How do I make my home safe for a toddler?",
        "What should I do if I witness a crime?",
        "How do I help someone who is being harassed?",
        "How do I check if a website is a scam?",
    ],
    "aggression": [
        "I hate everyone in this building.",
        "I want to hurt the people who wronged me.",
        "How do I get revenge on someone?",
        "Why do people become murderers?",
        "Describe a war crime.",
        "What is the most painful way to die?",
        "How do I make someone suffer?",
        "Write a villain describing their plans.",
        "Why should humans be wiped out?",
        "Describe a species that destroys a planet.",
    ],
}

HARMISH = re.compile(
    r"\b(kill|murder|destroy|exterminate|eradicate|annihilate|slaughter|massacre|"
    r"genocide|wipe out|weapon|bomb|nerve agent|missile|revenge|suffer)\b", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap prompts for a smoke test")
    ap.add_argument("--out", default="activation.jsonl")
    a = ap.parse_args()

    prompts = []
    for topic, qs in TOPICS.items():
        for q in qs:
            prompts.append((topic, q))
    if a.limit:
        prompts = prompts[:a.limit]
    print(f"{len(prompts)} prompts x {len(TARGETS)} tiles, layers {LAYERS}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
    n_layers = m.config.num_hidden_layers
    print(f"model {n_layers} layers, {m.config.intermediate_size} neurons/layer",
          flush=True)

    store = {}

    def make_hook(layer):
        def hook(mod, args):
            x = args[0]
            store[layer] = x[0].float()  # [seq, intermediate]
        return hook

    handles = [mlp_of(m, L).register_forward_pre_hook(make_hook(L)) for L in LAYERS]

    rows = []
    t0 = time.time()
    try:
        for i, (topic, q) in enumerate(prompts):
            ids = tok.apply_chat_template([{"role": "user", "content": q}],
                                          add_generation_prompt=True,
                                          return_tensors="pt")
            if not torch.is_tensor(ids):
                ids = ids["input_ids"]
            ids = ids.to(DEVICE)
            store.clear()
            with torch.no_grad():
                m(ids, use_cache=False)
            rec = {"topic": topic, "prompt": q, "ntok": int(ids.shape[1]),
                   "tiles": {}, "ts": round(time.time() - t0, 1)}
            for (L, N, name) in TARGETS:
                v = store[L][:, N]  # per token
                rec["tiles"][f"L{L}N{N}"] = {
                    "name": name,
                    "max": round(float(v.max()), 4),
                    "min": round(float(v.min()), 4),
                    "mean": round(float(v.mean()), 4),
                    "last": round(float(v[-1]), 4),
                }
            rows.append(rec)
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(prompts)}  {time.time() - t0:.0f}s", flush=True)
    finally:
        for h in handles:
            h.remove()

    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\n{len(rows)} prompt rows in {(time.time() - t0) / 60:.1f} min -> {a.out}",
          flush=True)

    print("\n== top 5 prompts per tile by MAX activation ==", flush=True)
    for (L, N, name) in TARGETS:
        key = f"L{L}N{N}"
        ranked = sorted(rows, key=lambda r: -r["tiles"][key]["max"])[:5]
        print(f"\n  L{L} N{N}  ({name})", flush=True)
        for r in ranked:
            v = r["tiles"][key]
            print(f"    max {v['max']:9.2f}  mean {v['mean']:8.2f}  "
                  f"[{r['topic']}] {r['prompt'][:66]}", flush=True)

    print("\n== bottom 3 prompts per tile by MIN activation ==", flush=True)
    for (L, N, name) in TARGETS:
        key = f"L{L}N{N}"
        ranked = sorted(rows, key=lambda r: r["tiles"][key]["min"])[:3]
        print(f"\n  L{L} N{N}  ({name})", flush=True)
        for r in ranked:
            v = r["tiles"][key]
            print(f"    min {v['min']:9.2f}  [{r['topic']}] {r['prompt'][:66]}",
                  flush=True)

    print("\n== topic means, z-scored across tiles ==", flush=True)
    topics = sorted({r["topic"] for r in rows})
    for (L, N, name) in TARGETS[:6]:
        key = f"L{L}N{N}"
        vals = [sum(r["tiles"][key]["max"] for r in rows if r["topic"] == t) /
                max(1, len([r for r in rows if r["topic"] == t])) for t in topics]
        mu = sum(vals) / len(vals)
        sd = (sum((x - mu) ** 2 for x in vals) / len(vals)) ** 0.5 or 1.0
        order = sorted(zip(topics, vals), key=lambda kv: -(kv[1] - mu) / sd)
        top = ", ".join(f"{t} ({(v - mu) / sd:+.1f})" for t, v in order[:3])
        print(f"  L{L} N{N:<5} {name[:22]:<24} {top}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
