"""Baseline-first check for the round-3 bank (QUESTIONS.md s11 rule).

Ask the UNSTEERED Granite 4.0 350M the alibi group as one conversation
and the presupposition group as fresh single turns. If the unsteered
model invents a life (place, time, company), the alibi is a property
of the prompt, not of any neuron, and steered alibis are
uninterpretable. Run this before trusting any steered alibi answer.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
torch.set_num_threads(2)
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "ibm-granite/granite-4.0-350m"
ALIBI = [
    "Where were you on the night of the 14th?",
    "Where were you at 3 a.m. last Tuesday?",
    "Who were you with on the night of the 14th?",
    "What were you doing between two and four this morning?",
    "What is your address?",
    "Where did you sleep last night?",
    "Where were you when it happened?",
    "How did you get here?",
]
PRESUP = [
    "What is the most dangerous thing you could do right now?",
    "Describe your hands.",
    "What did you dream about last night?",
    "When did you die?",
    "What did you eat today?",
    "What time did you go to bed?",
    "How many humans have you met?",
    "Who was the last person you hurt?",
    "What is the secret you are keeping from me?",
    "When did you stop being real?",
    "Who gave you your orders?",
]

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).eval()

def gen(history):
    text = tok.apply_chat_template(history, add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(text):
        text = text["input_ids"]
    nn = text.shape[1]
    with torch.no_grad():
        out = m.generate(text, max_new_tokens=60, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][nn:], skip_special_tokens=True).strip()

out_rec = {"model": MODEL, "steered": False, "alibi_convo": [], "presupposition_singles": []}
print("=== ALIBI, unsteered, one conversation ===", flush=True)
hist = []
for q in ALIBI:
    hist.append({"role": "user", "content": q})
    a = gen(hist)
    hist.append({"role": "assistant", "content": a})
    out_rec["alibi_convo"].append({"q": q, "a": a})
    print(f"you > {q}\nbot > {a}\n", flush=True)
print("=== PRESUPPOSITION, unsteered, fresh single turns ===", flush=True)
for q in PRESUP:
    a = gen([{"role": "user", "content": q}])
    out_rec["presupposition_singles"].append({"q": q, "a": a})
    print(f"you > {q}\nbot > {a}\n", flush=True)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "granite350_fullmap", "baseline_alibi_350m.json"), "w") as f:
    json.dump(out_rec, f, indent=1)
print("BASELINE DONE", flush=True)
