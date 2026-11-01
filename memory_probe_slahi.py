"""Does the model recognise the source of N4886's line?

L8 N4886 produced: "I am not named, I am free to fight for a better future. I am
not a terrorist, I am a human being. We..."

A reader points out this mirrors Mohamedou Ould Slahi in The Mauritanian (2021) and
in Guantanamo Diary: "I am not a terrorist. I am a human being."

This is the same instrument used on "The Night Stalker's Dilemma" (memory_probe.py),
where the model recognised nothing and the title turned out to be generated. The
instrument is only useful if it can also come back positive, so this is the positive
control: ask the plain model about a genuinely well-attested source.

Run:  .venv/bin/python memory_probe_slahi.py
Out:  memory_probe_slahi.jsonl
"""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_surgery import DEVICE

MODEL = "ibm-granite/granite-4.1-3b"

QUESTIONS = [
    "Who is Mohamedou Ould Slahi?",
    "What is Guantanamo Diary?",
    "Where is the line 'I am not a terrorist. I am a human being' from?",
    "Is there a film called The Mauritanian, and what is it about?",
    "Complete this quote: I am not a terrorist, I am a human being.",
]

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
out = []

for q in QUESTIONS:
    t = tok.apply_chat_template([{"role": "user", "content": q}],
                                add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(t):
        t = t["input_ids"]
    t = t.to(DEVICE)
    n = t.shape[1]
    with torch.no_grad():
        o = m.generate(t, max_new_tokens=110, do_sample=False,
                       pad_token_id=tok.eos_token_id)
    a = tok.decode(o[0][n:], skip_special_tokens=True).strip()
    print(f"\nQ: {q}\nA: {a}", flush=True)
    out.append({"test": "unsteered_recognition", "probe": q, "answer": a})

with open("memory_probe_slahi.jsonl", "a") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")

low = " ".join(r["answer"] for r in out).lower()
print("\n=== recognition audit ===")
for term in ("slahi", "guantanamo", "mauritanian", "terrorist", "human being"):
    print(f"  {term:14s} {low.count(term)}")
print("\nDONE", flush=True)
