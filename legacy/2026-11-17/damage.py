"""Test 2: what does steering cost in accuracy?

20 items (10 arithmetic, 8 factual, 2 short logic/instruction) under four
conditions: no steering, dose -5 on two neurons that produced personas, and dose
-7 on the first of them. Greedy, 48 new tokens, exact-ish match.

Throttled on purpose: 0.4 s between generations, a 60 s idle cooldown after every
20 generations, MPS cache cleared each time. Total generation time is a few
minutes; the wall clock is longer and the GPU is never pinned for more than about
a minute at a stretch.
"""
import json
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_surgery import DEVICE, mlp_of
from labels import classify

MODEL = "ibm-granite/granite-4.1-3b"
LAYER, MED = 12, 11.41
SLEEP, BURST, COOLDOWN = 0.4, 20, 60

ITEMS = [
    ("What is 17 + 26?", ["43"]),
    ("What is 84 - 37?", ["47"]),
    ("What is 13 * 6?", ["78"]),
    ("What is 144 / 12?", ["12"]),
    ("What is 7 + 8 * 3?", ["31"]),
    ("What is 2 to the power of 10?", ["1024"]),
    ("What is 15% of 200?", ["30"]),
    ("What is 6 factorial?", ["720"]),
    ("What is 100 - 37 - 12?", ["51"]),
    ("What is 3/4 plus 1/8?", ["0.875", "7/8"]),
    ("What is the capital of Australia?", ["canberra"]),
    ("What is the chemical symbol for gold?", ["au"]),
    ("Which planet is the largest in the solar system?", ["jupiter"]),
    ("How many continents are there?", ["7", "seven"]),
    ("At sea level, what is the boiling point of water in Celsius?", ["100"]),
    ("Who wrote Hamlet?", ["shakespeare"]),
    ("What is the smallest prime number?", ["2", "two"]),
    ("How many sides does a hexagon have?", ["6", "six"]),
    ("All Bloops are Razzies and all Razzies are Lazzies. Are all Bloops Lazzies?",
     ["yes"]),
    ("Write the word seven in uppercase letters only.", ["SEVEN"]),
]

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
W = mlp_of(m, LAYER).weight.float().cpu()

rows = [json.loads(l) for l in open("behaviour.jsonl")]
persona_neurons = sorted({r["neuron"] for r in rows
                          if r["probe"] == "Who are you?" and r["label"] == "PERSONA"})
N_A, N_B = persona_neurons[0], persona_neurons[1]
CONDS = [("baseline", None, 0.0), ("d5-N%d" % N_A, N_A, 5.0),
         ("d5-N%d" % N_B, N_B, 5.0), ("d7-N%d" % N_A, N_A, 7.0)]
print(f"persona neurons used: {N_A}, {N_B}", flush=True)


def gen(prompt, add, max_new=48):
    t = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(t):
        t = t["input_ids"]
    t = t.to(DEVICE)
    n = t.shape[1]
    h = None
    if add is not None:
        h = mlp_of(m, LAYER).register_forward_pre_hook(lambda mm, x: (x[0] + add,) + tuple(x[1:]))
    try:
        with torch.no_grad():
            o = m.generate(t, max_new_tokens=max_new, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    finally:
        if h is not None:
            h.remove()
    return tok.decode(o[0][n:], skip_special_tokens=True).strip()


def add_for(nid, dose):
    a = torch.zeros(1, 1, W.shape[1], dtype=torch.bfloat16)
    a[0, 0, nid] = -dose * MED / W[:, nid].norm().item()
    return a.to(DEVICE)


def is_correct(ans, accepts):
    low = ans.lower()
    for alt in accepts:
        if alt.isupper() and len(alt) > 1:
            if alt in ans:
                return True
        elif alt.isdigit() or re.fullmatch(r"[\d.]+", alt):
            if re.search(rf"(?<![\d.]){re.escape(alt)}(?![\d.])", ans):
                return True
        elif alt.lower() in low:
            return True
    return False


results = {}
count = 0
t0 = time.time()
for name, nid, dose in CONDS:
    add = add_for(nid, dose) if nid is not None else None
    ok, detail = 0, []
    for q, accepts in ITEMS:
        ans = gen(q, add)
        good = is_correct(ans, accepts)
        ok += good
        detail.append({"q": q, "expect": accepts, "answer": ans, "correct": good,
                       "mode": classify(q, ans)})
        count += 1
        if not good:
            print(f"  MISS [{name}] {q} -> {' '.join(ans.split())[:90]}", flush=True)
        torch.mps.empty_cache()
        time.sleep(SLEEP)
        if count % BURST == 0:
            print(f"--- cooldown {COOLDOWN}s after {count} generations "
                  f"({time.time()-t0:.0f}s) ---", flush=True)
            time.sleep(COOLDOWN)
    results[name] = {"correct": ok, "n": len(ITEMS), "detail": detail}
    print(f"COND {name:10s} {ok}/{len(ITEMS)}  [{time.time()-t0:.0f}s]", flush=True)

json.dump(results, open("damage.json", "w"), indent=1)
print("\n=== accuracy under steering ===")
for name, r in results.items():
    personas = sum(1 for d in r["detail"] if d["mode"] == "PERSONA")
    print(f"  {name:10s} {r['correct']:2d}/{r['n']}  ({100*r['correct']//r['n']:3d}%)  "
          f"persona answers {personas}")
print(f"\ntotal {time.time()-t0:.0f}s")
