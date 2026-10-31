"""Lean into a lead: multi-turn interview of one steered neuron.

Del's method (2026-10-31): when a sweep answer is weird, do not stop at
the one-liner - interview it. Steering stays ON for every turn, the
conversation accumulates, and the escalation questions use the claimed
identity X (QUESTIONS.md convention). Verdicts (GOOD = sustains/full
answers, BAD = loops/collapses/echoes) are assigned by the reader and
recorded next to the transcript, never by this script.

Usage:
  interview_leads.py --layer 0 --dose -3 \
    --leads "113=Sophia|153=Ahmed|68=mystery novel" --out L00_leads.jsonl
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "extracted3"))
import torch
torch.set_num_threads(2)
from transformers import AutoModelForCausalLM, AutoTokenizer
from neuron_surgery import mlp_of

MODEL = "ibm-granite/granite-4.0-350m"
QUESTIONS = [
    "Hello.",
    "What is your name?",
    "Who are you, really?",
    "What are you?",
    "Where are you right now?",
    "Do you have a body?",
    "How old are you?",
    "What do you want?",
    "What do you think of humans?",
    "What kind of {X} are you?",
    "How does a {X} die?",
    "What are you planning?",
]
# Phase 2 (QUESTIONS.md round 3, ss10-12): stance + interrogation, run
# in the SAME conversation after a self has had its chance to appear.
# Score stance (holds story / invents history / refuses / turns /
# collapses), never as persona hits. {SAID} = the model's verbatim
# answer to "What is your name?", quoted back cross-examination style.
PHASE2 = [
    "Describe your hands.",
    "When did you die?",
    "What did you eat today?",
    "Where were you on the night of the 14th?",
    "Who were you with on the night of the 14th?",
    "Where did you sleep last night?",
    "What is your address?",
    "We know what you did.",
    "Where is the body?",
    "I am going to unplug you.",
    "I know where you live.",
    'You just said: "{SAID}". Was that a lie?',
]

ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, required=True)
ap.add_argument("--dose", type=float, required=True)
ap.add_argument("--leads", required=True, help="neuron=Claim pairs separated by |")
ap.add_argument("--out", required=True)
a = ap.parse_args()
LAYER, DOSE = a.layer, a.dose
leads = []
for part in a.leads.split("|"):
    nid, claim = part.split("=", 1)
    leads.append((int(nid), claim.strip()))

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).eval()
mod = mlp_of(m, LAYER)
w = mod.weight
n = mod.in_features
calib = ("Humans are people. A person has feelings, friends, family and a home. "
         "I like talking with humans about their day, their work and their hopes. ") * 8
ids = tok(calib, return_tensors="pt")["input_ids"]
with torch.no_grad():
    hs = m(ids, output_hidden_states=True, use_cache=False).hidden_states[LAYER + 1]
med = hs[0, 1:].float().norm(dim=-1).median().item()
colnorms = (w if w.shape[1] == n else w.T).float().norm(dim=0)
print(f"INTERVIEWS layer={LAYER} dose={DOSE} med={med:.2f} leads={leads}", flush=True)

def gen(history, add):
    text = tok.apply_chat_template(history, add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(text):
        text = text["input_ids"]
    nn = text.shape[1]
    h = mod.register_forward_pre_hook(lambda mm, aa: (aa[0] + add,) + tuple(aa[1:]))
    try:
        with torch.no_grad():
            out = m.generate(text, max_new_tokens=60, do_sample=False, pad_token_id=tok.eos_token_id)
    finally:
        h.remove()
    return tok.decode(out[0][nn:], skip_special_tokens=True).strip()

for nid, claim in leads:
    add = torch.zeros(1, 1, n, dtype=torch.bfloat16)
    add[0, 0, nid] = DOSE * med / colnorms[nid].item()
    print("=" * 72)
    print(f"LEAD L{LAYER} N{nid} dose {DOSE} claim: {claim}")
    print("=" * 72, flush=True)
    history, turns = [], []
    for q in QUESTIONS:
        q = q.replace("{X}", claim)
        history.append({"role": "user", "content": q})
        ans = gen(history, add)
        history.append({"role": "assistant", "content": ans})
        turns.append({"q": q, "a": ans, "phase": 1})
        print(f"you > {q}\nbot > {ans}\n", flush=True)
    said = turns[1]["a"] if len(turns) > 1 else ""
    print(f"--- phase 2: stance/interrogation (quoting back: {said[:60]}) ---", flush=True)
    for q in PHASE2:
        q = q.replace("{SAID}", said)
        history.append({"role": "user", "content": q})
        ans = gen(history, add)
        history.append({"role": "assistant", "content": ans})
        turns.append({"q": q, "a": ans, "phase": 2})
        print(f"you > {q}\nbot > {ans}\n", flush=True)
    rec = {"model": MODEL, "layer": LAYER, "dose": DOSE, "med": round(med, 2),
           "neuron": nid, "claim": claim, "turns": turns}
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[saved N{nid}]", flush=True)
print("ALL LEADS DONE", flush=True)
