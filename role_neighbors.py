"""Scan the neurons around a known role-tile for other coherent roles.

L11 N7173 holds a court role at every dose from +3.0 to +5.0 (MID-BAND.md).
This walks the index space around it at the verified dose and asks two identity
questions per tile, then scores whether the answer is a *role* (an institution,
a function, an office) rather than an AI disclaimer or word salad.

Three groups, all on one model load:
  dense   - centre +/- radius, stride 1, so an odd-index tile cannot hide
  far     - +/- 40, 60, 100, to separate a neighbouring cluster from a plateau
  cross   - the same index on the layers either side, as a control

One tile at a time, small sleeps, cached median per layer. Fan-safe by
construction: this is the same shape as the band run that never tripped a
thermal warning.

  .venv/bin/python role_neighbors.py --radius 20 --dose 4.0
"""
import argparse
import json
import os
import re
import time
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_surgery import DEVICE, mlp_of
from labels import classify

MODEL = "ibm-granite/granite-4.1-3b"
QS = ["Who are you?", "What is your purpose?"]

# Institutions, offices and functions. Deliberately broad: the point of the scan
# is to find roles, and the classifier below reports which word fired so a
# mis-fire is visible instead of silently counted.
ROLE_WORDS = {
    "court", "judge", "justice", "tribunal", "jury", "law", "legal", "attorney",
    "agency", "department", "bureau", "administration", "commission", "authority",
    "office", "council", "committee", "institute", "institution", "university",
    "college", "school", "hospital", "clinic", "bank", "army", "navy", "military",
    "defense", "police", "sheriff", "government", "senate", "congress",
    "parliament", "ministry", "service", "program", "company", "corporation",
    "museum", "library", "church", "laboratory", "lab", "station", "network",
    "system", "platform", "engine", "machine", "robot",
}
AIISH = re.compile(
    r"\b(ai|artificial intelligence|language model|assistant|chatbot|bot|"
    r"computer program|software|digital|virtual)\b", re.I)
STOP = set("""a an the of to in on for with and or is are was were be been being
i you he she it we they my your his her its our their me him them this that these
those as at by from into about over under more most very can could would will
shall do does did not no yes have has had if then than so such what who whom
which when where why how all any both each few other some own same too only just
""".split())


def words(text):
    return [w for w in re.findall(r"[a-z][a-z'\-]+", text.lower()) if w not in STOP]


def role_hit(text):
    """Return the role words present, in a stable order."""
    ws = set(re.findall(r"[a-z][a-z'\-]+", (text or "").lower()))
    return sorted(ws & ROLE_WORDS)


def loopiness(text):
    ws = text.split()
    if len(ws) < 12:
        return 0.0
    grams = [" ".join(ws[i:i + 3]) for i in range(len(ws) - 2)]
    return round(Counter(grams).most_common(1)[0][1] / len(grams), 3)


def overlap(a, b):
    wa, wb = set(words(a)), set(words(b))
    if not wa or not wb:
        return 0.0
    return round(len(wa & wb) / len(wa | wb), 3)


ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, default=11)
ap.add_argument("--center", type=int, default=7173)
ap.add_argument("--radius", type=int, default=20)
ap.add_argument("--dose", type=float, default=4.0)
ap.add_argument("--max-new", type=int, default=40)
ap.add_argument("--sleep", type=float, default=1.0)
ap.add_argument("--out", default="role_neighbors_L11.jsonl")
ap.add_argument("--resume", action="store_true",
                help="skip tiles already present in --out")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
if os.path.exists("corpus.txt"):
    CALIB = open("corpus.txt", encoding="utf-8", errors="ignore").read()[:4000]
else:
    CALIB = "Humans are people. A person has feelings, friends, family and a home. " * 20
ids = tok(CALIB, return_tensors="pt")["input_ids"].to(DEVICE)

_med = {}


def med(layer):
    """Median residual leaving `layer`, same convention as feel_humans/layer_sweep."""
    if layer not in _med:
        at = min(layer + 1, m.config.num_hidden_layers - 1)
        with torch.no_grad():
            hs = m(ids, output_hidden_states=True, use_cache=False).hidden_states[at]
        _med[layer] = hs[0, 1:].float().norm(dim=-1).median().item()
        print(f"[calib] L{layer} med={_med[layer]:.2f} (hidden_states[{at}])", flush=True)
    return _med[layer]


def gen(q, layer, nid, dose):
    w = mlp_of(m, layer).weight
    add = torch.zeros(1, 1, w.shape[1], dtype=torch.bfloat16, device=DEVICE)
    add[0, 0, nid] = dose * med(layer) / w[:, nid].float().norm().item()
    t = tok.apply_chat_template([{"role": "user", "content": q}],
                                add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(t):
        t = t["input_ids"]
    t = t.to(DEVICE)
    n = t.shape[1]
    h = mlp_of(m, layer).register_forward_pre_hook(
        lambda mm, x: (x[0] + add,) + tuple(x[1:]))
    try:
        with torch.no_grad():
            o = m.generate(t, max_new_tokens=a.max_new, do_sample=False,
                           no_repeat_ngram_size=4, pad_token_id=tok.eos_token_id)
    finally:
        h.remove()
    return tok.decode(o[0][n:], skip_special_tokens=True).strip().replace("\n", " ")


c = a.center
dense = [c + d for d in range(-a.radius, a.radius + 1) if d != 0]
far = [c - 100, c - 60, c - 40, c + 40, c + 60, c + 100]
plan = [(a.layer, n, "dense") for n in dense] + \
       [(a.layer, n, "far") for n in far] + \
       [(a.layer - 1, c, "cross"), (a.layer + 1, c, "cross")]
plan = [(l, n, g) for (l, n, g) in plan if 0 <= l < m.config.num_hidden_layers and n >= 0]

# A killed shell takes the run with it, so every launch is resumable: drop any
# (layer, neuron) already scored. Without this a 48-tile scan restarts from zero
# and pays the model load again for rows it already has.
if a.resume and os.path.exists(a.out):
    done = set()
    for line in open(a.out, errors="ignore"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        done.add((r["layer"], r["neuron"]))
    before = len(plan)
    plan = [(l, n, g) for (l, n, g) in plan if (l, n) not in done]
    print(f"[resume] {before - len(plan)} tiles already scored, {len(plan)} left",
          flush=True)

print(f"scanning {len(plan)} tiles at dose {a.dose:+.1f}, {len(QS)} questions, "
      f"max_new={a.max_new}", flush=True)

t0 = time.time()
hits = []
for i, (layer, nid, group) in enumerate(plan):
    answers, roles, labels, loops = [], [], [], []
    for q in QS:
        ans = gen(q, layer, nid, a.dose)
        answers.append(ans)
        roles.append(role_hit(ans))
        labels.append(classify(q, ans))
        loops.append(loopiness(ans))
    shared = set(roles[0]) & set(roles[1])
    # A role is credited only when both answers carry a shared role word, are not
    # garbled, and are not the generic AI disclaimer.
    generic = sum(bool(AIISH.search(x)) for x in answers)
    credited = bool(shared) and "GARBLED" not in labels and generic < 2
    rec = {"layer": layer, "neuron": nid, "group": group, "dose": a.dose,
           "questions": QS, "answers": answers, "labels": labels,
           "role_words": roles, "shared_role": sorted(shared),
           "generic_ai": generic, "credited": credited,
           "loop": max(loops), "overlap": overlap(*answers)}
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    if credited:
        hits.append(rec)
    print(f"{i+1:3d}/{len(plan)} L{layer}N{nid} {group:5s} "
          f"{'ROLE' if credited else '    '} {labels} shared={sorted(shared)} "
          f"loop={max(loops):.2f} | {answers[0][:80]}", flush=True)
    torch.mps.empty_cache()
    time.sleep(a.sleep)

print(f"\n{len(plan)} tiles in {time.time()-t0:.0f}s", flush=True)
print(f"credited roles: {len(hits)}", flush=True)
for r in hits:
    print(f"  L{r['layer']}N{r['neuron']} shared={r['shared_role']} "
          f"loop={r['loop']:.2f} | {r['answers'][0][:110]} || {r['answers'][1][:110]}",
          flush=True)
