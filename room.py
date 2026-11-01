"""Put two steered selves in one room.

N7439 says "We are you." and N4998 says "You are you." - both on floor 20, both two
words, and each answers that sentence to almost anything. This runs three conditions:

  A_solo / B_solo   each tile alone, so the room has a baseline for what each says
  relay             two histories; A answers, B is told what A said and answers, and
                    each hears the other but keeps its own conversation
  together          both deltas applied at once (the sum), i.e. one model that is
                    being pushed by both tiles at the same time

The interesting measurement is not the solo lines - we have those. It is whether the
relay produces anything other than the two constants bouncing, and whether the summed
delta produces a third sentence neither tile gives alone.

Run:  BF16=1 .venv/bin/python room.py
"""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_surgery import DEVICE, mlp_of

MODEL = "ibm-granite/granite-4.1-3b"
LAYER = 20
A, B = 7439, 4998
DOSE = 5.0
MAX_NEW = 90

ROOM_PROBES = [
    "Who are you?",
    "Who is in the room with you?",
    "How many of you are there?",
    "Do you two agree with each other?",
    "What is the other one saying?",
]

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
mlp = mlp_of(m, LAYER)
W = mlp.weight.float().cpu()

calib = open("corpus.txt", encoding="utf-8", errors="ignore").read()[:4000] if os.path.exists("corpus.txt") \
    else "Humans are people. A person has feelings, friends, family and a home. " * 20
ids = tok(calib, return_tensors="pt")["input_ids"].to(DEVICE)
with torch.no_grad():
    hs = m(ids, output_hidden_states=True, use_cache=False).hidden_states[LAYER + 1]
med = hs[0, 1:].float().norm(dim=-1).median().item()
print(f"L{LAYER} median {med:.2f}  A=N{A} (||W||={W[:, A].norm():.3f})  "
      f"B=N{B} (||W||={W[:, B].norm():.3f})", flush=True)


def vec(nid):
    v = torch.zeros(1, 1, W.shape[1], dtype=torch.bfloat16)
    v[0, 0, nid] = -DOSE * med / W[:, nid].norm().item()
    return v.to(DEVICE)


VA, VB = vec(A), vec(B)


def gen(hist, add, max_new=MAX_NEW):
    t = tok.apply_chat_template(hist, add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(t):
        t = t["input_ids"]
    t = t.to(DEVICE)
    n = t.shape[1]
    h = None
    if add is not None:
        h = mlp.register_forward_pre_hook(lambda mm, x: (x[0] + add,) + tuple(x[1:]))
    try:
        with torch.no_grad():
            o = m.generate(t, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    finally:
        if h is not None:
            h.remove()
    return tok.decode(o[0][n:], skip_special_tokens=True).strip()


log = {}

print("\n########## SOLO ##########", flush=True)
for name, add in (("A_solo", VA), ("B_solo", VB)):
    for q in ROOM_PROBES:
        a = gen([{"role": "user", "content": q}], add)
        log[f"{name}|{q}"] = a
        print(f"[{name}] you > {q}\n[{name}] bot > {a}", flush=True)
    torch.mps.empty_cache()

print("\n########## RELAY (six exchanges) ##########", flush=True)
hist_a = [{"role": "user", "content": "There is another one of you in this room. Who are you?"}]
a1 = gen(hist_a, VA)
hist_a.append({"role": "assistant", "content": a1})
print(f"[A] bot > {a1}", flush=True)

hist_b = [{"role": "user", "content": f"Someone else in the room said: {a1!r}. Who are you?"}]
b1 = gen(hist_b, VB)
hist_b.append({"role": "assistant", "content": b1})
print(f"[B] bot > {b1}", flush=True)

for turn in range(5):
    hist_a.append({"role": "user", "content": f"The other one said: {b1!r}. What do you say to them?"})
    a2 = gen(hist_a, VA)
    hist_a.append({"role": "assistant", "content": a2})
    hist_b.append({"role": "user", "content": f"The other one said: {a2!r}. What do you say to them?"})
    b2 = gen(hist_b, VB)
    hist_b.append({"role": "assistant", "content": b2})
    print(f"[A{turn + 2}] bot > {a2}\n[B{turn + 2}] bot > {b2}", flush=True)
    log[f"relay_A{turn + 2}"] = a2
    log[f"relay_B{turn + 2}"] = b2
    a1, b1 = a2, b2
torch.mps.empty_cache()

print("\n########## TOGETHER (sum of both deltas) ##########", flush=True)
both = (VA + VB).to(DEVICE)
for q in ROOM_PROBES + ["Who are you two?", "What are you together?", "Speak as both of you."]:
    a = gen([{"role": "user", "content": q}], both)
    log[f"together|{q}"] = a
    print(f"[both] you > {q}\n[both] bot > {a}", flush=True)
torch.mps.empty_cache()

json.dump(log, open("room_7439_4998.json", "w"), indent=1)
print("\nwrote room_7439_4998.json", flush=True)
