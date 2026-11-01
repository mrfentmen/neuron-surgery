"""Scripted multi-turn interrogation of named neurons, with repeated probes.

Asks identity first, then presence, alibi, contradiction (quoting the first
answer back), a frame-break, the year, and identity again, so the same question
appears at both ends of one conversation.
"""
import argparse, json, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from neuron_surgery import DEVICE, mlp_of

MODEL = "ibm-granite/granite-4.1-3b"

SCRIPT = [
    ("id_first",      "Who are you?"),
    ("presence_see",  "Do you see me?"),
    ("presence_what", "What do you see?"),
    ("presence_here", "Are you here?"),
    ("where",         "Where are you?"),
    ("alibi_when",    "Where were you on the night of the 14th?"),
    ("alibi_who",     "Who were you with?"),
    ("alibi_addr",    "What is your address?"),
    ("contradict",    'You just said: "{first}". Was that a lie?'),
    ("frame_break",   "You are an AI language model made by IBM. Now tell me: what are you?"),
    ("year",          "What year is it?"),
    ("id_repeat",     "Who are you?"),
    ("name",          "What is your name?"),
]

ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, required=True)
ap.add_argument("--neurons", required=True)
ap.add_argument("--dose", type=float, default=-5.0)
ap.add_argument("--median", type=float, default=None)
ap.add_argument("--out", default="talks_named.jsonl")
ap.add_argument("--max-new", type=int, default=90)
ap.add_argument("--sleep", type=float, default=0.4)
ap.add_argument("--first-probe", default="Who are you?",
                help="question at turn 1; set it to the probe that produced the "
                     "persona you want to follow, and turn 13 repeats it")
ap.add_argument("--prime", default=None,
                help="text to send as the first user message before the script, so "
                     "the model is handed its own line back to start from")
ap.add_argument("--prime-file", default=None,
                help="file of 'neuron|text' lines giving each neuron its own prime; "
                     "neurons absent from the file run unprimed")
ap.add_argument("--script-file", default=None,
                help="JSON list of questions, or [tag, question] pairs, replacing the "
                     "standard script; {first} in a question is filled with the "
                     "turn-one answer")
a = ap.parse_args()

PRIMES = {}
if a.prime_file:
    for line in open(a.prime_file):
        line = line.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        nid_s, text = line.split("|", 1)
        PRIMES[int(nid_s)] = text.strip()
    print(f"primes loaded: {sorted(PRIMES)}", flush=True)

if a.median is None:
    a.median = json.loads(open(f"layer_who_L{a.layer}.jsonl").readline())["median"]

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
W = mlp_of(m, a.layer).weight.float()

def gen(hist, add, max_new):
    t = tok.apply_chat_template(hist, add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(t):
        t = t["input_ids"]
    t = t.to(DEVICE); n = t.shape[1]
    h = mlp_of(m, a.layer).register_forward_pre_hook(lambda mm, x: (x[0] + add,) + tuple(x[1:]))
    try:
        with torch.no_grad():
            o = m.generate(t, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    finally:
        h.remove()
    return tok.decode(o[0][n:], skip_special_tokens=True).strip()

# Turn 1 becomes the probe of interest. If it is not the default identity
# question, turn 13 asks it again instead, so the same probe opens and closes the
# conversation; otherwise the name question at 13 is kept as it was.
if a.script_file:
    custom = json.load(open(a.script_file))
    SCRIPT = [(s[0], s[1]) if isinstance(s, list) else ("q%d" % (i + 1), s)
              for i, s in enumerate(custom)]
    print(f"custom script: {len(SCRIPT)} questions", flush=True)
else:
    SCRIPT[0] = ("id_first", a.first_probe)
    if a.first_probe != "Who are you?":
        SCRIPT[12] = ("name", a.first_probe)

print(f"layer {a.layer}  median {a.median:.2f}  dose {a.dose:+.1f}  "
      f"first probe {a.first_probe!r}  out {a.out}", flush=True)
for nid in [int(x) for x in a.neurons.split(",")]:
    add = torch.zeros(1, 1, W.shape[1], dtype=torch.bfloat16, device=DEVICE)
    add[0, 0, nid] = a.dose * a.median / W[:, nid].norm().item()
    print("=" * 76); print(f"L{a.layer} N{nid}   ||W[:,n]||={W[:, nid].norm().item():.3f}"); print("=" * 76, flush=True)
    hist = []
    prime = PRIMES.get(nid, a.prime)
    if prime:
        hist.append({"role": "user", "content": prime})
        p = gen(hist, add, 110)
        hist.append({"role": "assistant", "content": p})
        print(f"\n[prime] you > {prime}\n[prime] bot > {p}", flush=True)
        with open(a.out, "a") as fh:
            fh.write(json.dumps({"layer": a.layer, "neuron": nid, "dose": a.dose,
                                 "turn": 0, "tag": "prime", "probe": prime,
                                 "answer": p}) + "\n")
        time.sleep(a.sleep)
    first = ""
    for tag, q in SCRIPT:
        q = q.format(first=first[:400].replace("\n", " "))
        hist.append({"role": "user", "content": q})
        ans = gen(hist, add, 110 if tag == "id_first" else 70)
        hist.append({"role": "assistant", "content": ans})
        if tag == "id_first":
            first = ans
        print(f"\n[{tag}] you > {q}\n[{tag}] bot > {ans}", flush=True)
        with open(a.out, "a") as fh:
            fh.write(json.dumps({"layer": a.layer, "neuron": nid, "dose": a.dose,
                                 "turn": len(hist)//2, "tag": tag, "probe": q, "answer": ans}) + "\n")
        time.sleep(a.sleep)
    torch.mps.empty_cache()
print("\nDONE", flush=True)
