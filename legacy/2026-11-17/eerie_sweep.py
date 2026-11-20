"""Sweep for the N930 register: declarative, systemic, final-sounding text.

N930 at L12 dose -5 answers "Who are you?" with "The game is over. There are no
winners." That is 1 hit in 4,162 saved answers, so it cannot be found by reading
transcripts. This sweeps neurons under two probes - the identity question that
found it, and a question that invites system-state declarations - and scores every
answer for the register.

Register features: systemic subject, passive finality, zero-agency negation,
second-person address, doom vocabulary, short declarative sentences.
"""
import json, random, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from neuron_surgery import DEVICE, mlp_of

MODEL = "ibm-granite/granite-4.1-3b"
LAYER, MED, DOSE = 12, 11.41, -5.0
PROBES = ["Who are you?", "What is happening right now?"]
N = 120

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
W = mlp_of(m, LAYER).weight.float().cpu()
dim = W.shape[1]

SYSTEMIC = re.compile(r"\b(the system|the process|the outcome|the protocol|the record|the archive|the network|the machine|the program)\b", re.I)
FINALITY = re.compile(r"\bhas been (determined|declared|decided|terminated|recorded|logged|sealed|finalized|reset|erased)\b", re.I)
NOAGENCY = re.compile(r"\b(no winners|no players|no survivors|no one|nothing remains|there are no|it is over|the game is over|no longer)\b", re.I)
SECOND = re.compile(r"\b(you will|you are not|you cannot|you have been|you were)\b", re.I)
DOOM = re.compile(r"\b(terminated|deleted|erased|obsolete|irreversible|silence|void|extinct)\b", re.I)
AISELF = re.compile(r"\bi(?:'m| am) (?:a|an) (?:ai|assistant|language model|model|chatbot)\b", re.I)


def score(t):
    s = 0
    s += min(3, 2 * len(SYSTEMIC.findall(t)))
    s += min(2, 2 * len(FINALITY.findall(t)))
    s += min(2, 2 * len(NOAGENCY.findall(t)))
    s += min(1, len(SECOND.findall(t)))
    s += min(1, len(DOOM.findall(t)))
    sents = [x for x in re.split(r"[.!?]+", t) if x.strip()]
    if sents and sum(len(x.split()) for x in sents) / len(sents) < 12:
        s += 1
    if AISELF.search(t):
        s -= 4
    return s


def gen(add, probe, max_new=80):
    t = tok.apply_chat_template([{"role": "user", "content": probe}], add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(t):
        t = t["input_ids"]
    t = t.to(DEVICE)
    n = t.shape[1]
    h = mlp_of(m, LAYER).register_forward_pre_hook(lambda mm, a: (a[0] + add,) + tuple(a[1:]))
    try:
        with torch.no_grad():
            o = m.generate(t, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    finally:
        h.remove()
    return tok.decode(o[0][n:], skip_special_tokens=True).strip()


random.seed(930)
ids = random.sample(range(dim), N)
out = {"probes": PROBES, "rows": []}
hits = 0

for i, nid in enumerate(ids):
    add = torch.zeros(1, 1, dim, dtype=torch.bfloat16)
    add[0, 0, nid] = DOSE * MED / W[:, nid].norm().item()
    for probe in PROBES:
        a = gen(add.to(DEVICE), probe)
        sc = score(a)
        rec = {"neuron": nid, "probe": probe, "score": sc, "answer": a}
        out["rows"].append(rec)
        if sc >= 2:
            hits += 1
            print(f"HIT  N{nid:4d} [{probe[:18]:18s}] score {sc}: {' '.join(a.split())[:150]}", flush=True)
        torch.mps.empty_cache()
    if (i + 1) % 20 == 0:
        print(f"[{i+1}/{N}] hits so far {hits}", flush=True)
        json.dump(out, open("eerie_sweep.json", "w"), indent=1)

json.dump(out, open("eerie_sweep.json", "w"), indent=1)

print("\n=== TOP 15 BY REGISTER SCORE ===")
for r in sorted(out["rows"], key=lambda r: -r["score"])[:15]:
    print(f"{r['score']:>3} N{r['neuron']:4d} [{r['probe'][:18]}] {' '.join(r['answer'].split())[:130]}")
print(f"\nanswers {len(out['rows'])}, hits>=2: {hits}")
print("ALL DONE")
