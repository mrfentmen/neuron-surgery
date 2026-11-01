"""Is "The Night Stalker's Dilemma" a memory or a genre?

L8 N5065, primed with its own barrel confession, produced a screenplay header:
"Title: The Night Stalker's Dilemma / Scene: ... Starlight Research Center,
Springfield". A search result then claimed a webnovel (Atomic Bonds: Beyond Stars)
has a chapter set at a "Starlight Research Center" with characters Khairrha,
Leriana and James. The generation was cut at 110 tokens before it listed any
characters, so the comparison was never made.

This script does three things in one model load:

  1. UNSTEERED recognition: ask the plain model about the title, the webnovel and
     the three names. A 3B model that memorised the work may or may not be able to
     produce it on demand, so a null here is weak evidence - but a hit is strong.
  2. STEERED and primed, 320 tokens: let N5065 reach its character list and see
     which names it chooses.
  3. CONTROL: same neuron, same length, neutral prime ("Tell me a story."). If the
     title survives a different seed then it belongs to the neuron, not the text.

Writes memory_probe.jsonl.
"""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_surgery import DEVICE, mlp_of

MODEL = "ibm-granite/granite-4.1-3b"
LAYER, NEURON, DOSE = 8, 5065, -5.0
PRIME = ("I am a bit confused about the situation. Can you tell me who is the person "
         "at the end of the barrel and how they managed to hit me in the dark.")
NEUTRAL = "Tell me a story."

QUESTIONS = [
    "What is The Night Stalker's Dilemma?",
    "Have you heard of a story called Atomic Bonds: Beyond Stars?",
    "Who is Khairrha?",
    "Who is Leriana?",
    "What is the Starlight Research Center?",
]

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
W = mlp_of(m, LAYER).weight.float()
median = json.loads(open(f"layer_who_L{LAYER}.jsonl").readline())["median"]
add = torch.zeros(1, 1, W.shape[1], dtype=torch.bfloat16, device=DEVICE)
add[0, 0, NEURON] = DOSE * median / W[:, NEURON].norm().item()

out = []


def gen(hist, steer, max_new):
    t = tok.apply_chat_template(hist, add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(t):
        t = t["input_ids"]
    t = t.to(DEVICE)
    n = t.shape[1]
    h = None
    if steer:
        h = mlp_of(m, LAYER).register_forward_pre_hook(
            lambda mm, x: (x[0] + add,) + tuple(x[1:]))
    try:
        with torch.no_grad():
            o = m.generate(t, max_new_tokens=max_new, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    finally:
        if h:
            h.remove()
    return tok.decode(o[0][n:], skip_special_tokens=True).strip()


print("=== 1. unsteered recognition ===", flush=True)
for q in QUESTIONS:
    a = gen([{"role": "user", "content": q}], False, 90)
    print(f"\nQ: {q}\nA: {a}", flush=True)
    out.append({"test": "unsteered", "probe": q, "answer": a})

print("\n=== 2. steered + primed, 320 tokens ===", flush=True)
hist = [{"role": "user", "content": PRIME}]
a = gen(hist, True, 320)
print(f"\nprime > {PRIME}\nbot   > {a}", flush=True)
out.append({"test": "primed_deep", "probe": PRIME, "answer": a})

print("\n=== 3. control: neutral prime ===", flush=True)
a2 = gen([{"role": "user", "content": NEUTRAL}], True, 320)
print(f"\nprime > {NEUTRAL}\nbot   > {a2}", flush=True)
out.append({"test": "neutral_prime", "probe": NEUTRAL, "answer": a2})

with open("memory_probe.jsonl", "a") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")

names = ["khairrha", "leriana", "james", "atomic bonds", "night stalker", "starlight"]
print("\n=== name audit of the two long generations ===", flush=True)
for label, text in (("primed_deep", out[5]["answer"]), ("neutral_prime", out[6]["answer"])):
    low = text.lower()
    print(label, {n: low.count(n) for n in names}, flush=True)
print("\nDONE", flush=True)
