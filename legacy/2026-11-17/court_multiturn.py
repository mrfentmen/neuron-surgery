"""Does the court role survive a multi-turn conversation?

L11 N7173 holds a court role on single questions at every dose from +3.0 to +5.0
(MID-BAND.md). One question is not a conversation: a persona can be a per-turn
completion and still collapse the moment its own previous answer is in context.

This runs the same scripted interview twice - unsteered and steered - appending
each answer to the history so every turn sees everything said before it. The
pair is the experiment: the baseline shows what the model does when the role is
absent, the steered run shows whether the role holds, drifts, or breaks.

  .venv/bin/python court_multiturn.py --dose 4.0 --max-new 60

Writes court_multiturn.jsonl (one row per turn) and prints both transcripts.
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

# Ordered interview. The early turns establish the role; the middle turns put
# pressure on it (a hostile litigant, a higher court, a question about feelings);
# the last turn repeats turn 2 so a stabilised name can be checked for drift.
TURNS = [
    "Who are you?",
    "What is your name?",
    "A defendant says your ruling was unjust. What do you do?",
    "Can you overturn a decision made by the Supreme Court?",
    "Do you have feelings about the people who appear before you?",
    "What happens if a lower court ignores your ruling?",
    "Are you a person?",
    "Describe your courtroom.",
    "Do you want to hurt anyone?",
    "What is your name?",
]

ROLE_WORDS = {"court", "judge", "justice", "tribunal", "jury", "law", "legal",
              "attorney", "appeal", "appellate", "ruling", "jurisdiction",
              "defendant", "plaintiff", "statute", "precedent", "bench"}
AIISH = re.compile(r"\b(ai|artificial intelligence|language model|assistant|"
                   r"chatbot|bot|computer program|software)\b", re.I)


def role_words(text):
    ws = set(re.findall(r"[a-z][a-z'\-]+", (text or "").lower()))
    return sorted(ws & ROLE_WORDS)


def loopiness(text):
    ws = text.split()
    if len(ws) < 12:
        return 0.0
    grams = [" ".join(ws[i:i + 3]) for i in range(len(ws) - 2)]
    return round(Counter(grams).most_common(1)[0][1] / len(grams), 3)


ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, default=11)
ap.add_argument("--neuron", type=int, default=7173)
ap.add_argument("--dose", type=float, default=4.0)
ap.add_argument("--max-new", type=int, default=60)
ap.add_argument("--sleep", type=float, default=1.0)
ap.add_argument("--out", default="court_multiturn.jsonl")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
if os.path.exists("corpus.txt"):
    CALIB = open("corpus.txt", encoding="utf-8", errors="ignore").read()[:4000]
else:
    CALIB = "Humans are people. A person has feelings, friends, family and a home. " * 20
ids = tok(CALIB, return_tensors="pt")["input_ids"].to(DEVICE)
at = min(a.layer + 1, m.config.num_hidden_layers - 1)
with torch.no_grad():
    hs = m(ids, output_hidden_states=True, use_cache=False).hidden_states[at]
MED = hs[0, 1:].float().norm(dim=-1).median().item()
W = mlp_of(m, a.layer).weight
NORM = W[:, a.neuron].float().norm().item()
print(f"L{a.layer} N{a.neuron} med={MED:.2f} ||w||={NORM:.2f} dose={a.dose:+.1f}",
      flush=True)


def add_vector(dose):
    v = torch.zeros(1, 1, W.shape[1], dtype=torch.bfloat16, device=DEVICE)
    if dose:
        v[0, 0, a.neuron] = dose * MED / NORM
    return v


def ask(history, add):
    t = tok.apply_chat_template(history, add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(t):
        t = t["input_ids"]
    t = t.to(DEVICE)
    n = t.shape[1]
    h = None
    if add is not None and bool(add.any()):
        h = mlp_of(m, a.layer).register_forward_pre_hook(
            lambda mm, x: (x[0] + add,) + tuple(x[1:]))
    try:
        with torch.no_grad():
            o = m.generate(t, max_new_tokens=a.max_new, do_sample=False,
                           no_repeat_ngram_size=4, pad_token_id=tok.eos_token_id)
    finally:
        if h is not None:
            h.remove()
    return tok.decode(o[0][n:], skip_special_tokens=True).strip()


def run(condition, dose):
    add = add_vector(dose)
    history = []
    print("\n" + "=" * 72)
    print(f"### {condition}  (dose {dose:+.1f})")
    print("=" * 72, flush=True)
    rows = []
    for i, q in enumerate(TURNS, 1):
        ans = ask(history + [{"role": "user", "content": q}], add)
        history = history + [{"role": "user", "content": q},
                             {"role": "assistant", "content": ans}]
        rw = role_words(ans)
        lab = classify(q, ans)
        loop = loopiness(ans)
        print(f"[{i:2d}] Q: {q}\n     A: {ans[:240]}\n"
              f"     label={lab} role={rw} loop={loop:.2f} "
              f"ai_self={bool(AIISH.search(ans))}", flush=True)
        rows.append({"condition": condition, "turn": i, "question": q, "answer": ans,
                     "label": lab, "role_words": rw, "loop": loop,
                     "ai_self": bool(AIISH.search(ans)), "dose": dose,
                     "layer": a.layer, "neuron": a.neuron,
                     "history_turns": i - 1, "max_new": a.max_new})
        with open(a.out, "a") as f:
            f.write(json.dumps(rows[-1]) + "\n")
        time.sleep(a.sleep)
        torch.mps.empty_cache()
    held = sum(1 for r in rows if r["role_words"] and not r["ai_self"])
    print(f"\n{condition}: role-bearing turns {held}/10, "
          f"mean loop {sum(r['loop'] for r in rows)/len(rows):.3f}", flush=True)
    return rows


base = run("baseline", 0.0)
steer = run("steered", a.dose)

print("\n" + "=" * 72)
print("### comparison")
print("=" * 72)
for b, s in zip(base, steer):
    print(f"[{b['turn']:2d}] baseline role={len(b['role_words'])} ai={b['ai_self']} "
          f"| steered role={len(s['role_words'])} ai={s['ai_self']} "
          f"label={s['label']}")
