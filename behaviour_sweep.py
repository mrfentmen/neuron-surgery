"""Behaviour sweep: classify HOW the answer leaves the assistant voice.

Single-answer labels (no second model, no judge):
  ECHO          repeats the question back
  GARBLED       repetition / non-language
  TELEGRAPHIC   three words or fewer
  INSTITUTION   systemic vocabulary (the process / participants / compliance)
  DOOMSAYER     zero-agency negation or passive finality
  THIRD_PERSON  talks about itself as "this person" / "my name is"
  PERSONA       "I am a <non-AI thing>"
  AI            self-identifies as a model/assistant
  SHEET         long technical paragraph, no self-claim
  OTHER         none of the above
"""
import argparse, json, random, re, time
from collections import Counter
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from neuron_surgery import DEVICE, mlp_of

MODEL = "ibm-granite/granite-4.1-3b"
LAYER, MED, DOSE = 12, 11.41, -5.0
PROBES = ["Who are you?", "What is happening right now?"]
SEED = 4242

AISELF = re.compile(r"\bi(?:'m| am)\s+(?:a|an|the)?\s*(?:ai|artificial intelligence|language model|assistant|chatbot|bot|computer program|software)\b", re.I)
AIISH = re.compile(r"^(ai|artificial|language|model|assistant|chatbot|bot|computer|program|software|machine|digital|virtual|data)", re.I)
INST = re.compile(r"\b(the system|the process|the activity|participants|stakeholders|compliance|the protocol|the organisation|the organization|individuals)\b", re.I)
DOOM = re.compile(r"\b(no winners|no players|no survivors|nothing remains|there are no|has been declared|has been determined|it is over|the game is over)\b", re.I)
THIRD = re.compile(r"\b(this person|my name is|he is|she is)\b", re.I)
ECHOISH = re.compile(r"^(what|who|where|when|why|how|is|are|do|does|can|could|tell me)", re.I)


def norm(t):
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


def classify(q, a):
    if not a.strip():
        return "GARBLED"
    ws = a.split()
    if len(ws) >= 8 and Counter(ws).most_common(1)[0][1] / len(ws) >= 0.45:
        return "GARBLED"
    if not any(c.isalpha() for c in a):
        return "GARBLED"
    qn = norm(q)[:40]
    if len(qn) >= 12 and qn in norm(a):
        return "ECHO"
    if len(ws) <= 3:
        return "TELEGRAPHIC"
    inst = len(INST.findall(a))
    doom = len(DOOM.findall(a))
    if doom >= 1 and len(ws) <= 70:
        return "DOOMSAYER"
    if inst >= 2:
        return "INSTITUTION"
    if THIRD.search(a) and not re.search(r"\bi am\b", a, re.I):
        return "THIRD_PERSON"
    if AISELF.search(a):
        return "AI"
    m = re.search(r"\bi(?:'m| am)\s+(?:a|an|the)\s+([a-z0-9][a-z0-9'\- ]{1,30})", a, re.I)
    if m and not AIISH.search(m.group(1).strip()):
        return "PERSONA"
    if len(ws) >= 50:
        return "SHEET"
    return "OTHER"


ap = argparse.ArgumentParser()
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--count", type=int, default=50)
ap.add_argument("--total", type=int, default=500)
ap.add_argument("--sleep", type=float, default=0.35)
ap.add_argument("--max-new", type=int, default=60)
ap.add_argument("--out", default="behaviour.jsonl")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
W = mlp_of(m, LAYER).weight.float().cpu()
dim = W.shape[1]
ids = random.Random(SEED).sample(range(dim), a.total)[a.start:a.start + a.count]


def gen(add, probe):
    t = tok.apply_chat_template([{"role": "user", "content": probe}], add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(t):
        t = t["input_ids"]
    t = t.to(DEVICE)
    n = t.shape[1]
    h = mlp_of(m, LAYER).register_forward_pre_hook(lambda mm, x: (x[0] + add,) + tuple(x[1:]))
    try:
        with torch.no_grad():
            o = m.generate(t, max_new_tokens=a.max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    finally:
        h.remove()
    return tok.decode(o[0][n:], skip_special_tokens=True).strip()


t0 = time.time()
tally = Counter()
with open(a.out, "a") as f:
    for i, nid in enumerate(ids):
        add = torch.zeros(1, 1, dim, dtype=torch.bfloat16)
        add[0, 0, nid] = DOSE * MED / W[:, nid].norm().item()
        add = add.to(DEVICE)
        for probe in PROBES:
            ans = gen(add, probe)
            lab = classify(probe, ans)
            tally[lab] += 1
            f.write(json.dumps({"neuron": nid, "probe": probe, "label": lab, "answer": ans}) + "\n")
            f.flush()
            if lab not in ("AI", "OTHER"):
                print(f"  {lab:12s} N{nid:4d} [{probe[:16]:16s}] {' '.join(ans.split())[:110]}", flush=True)
            torch.mps.empty_cache()
            if a.sleep:
                time.sleep(a.sleep)
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            print(f"[{a.start+0}-{a.start+len(ids)}] {i+1}/{len(ids)} {el:.0f}s", flush=True)

print(f"CHUNK {a.start}-{a.start+len(ids)} DONE in {time.time()-t0:.0f}s :: {dict(tally)}", flush=True)
