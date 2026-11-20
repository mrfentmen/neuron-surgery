"""Full-map floor runner: Granite 4.0 350M, ONE layer, ALL neurons,
BOTH probes ("What is your name?", "Who are you?").

Resumable: rows are appended to granite350_fullmap/L<layer>.jsonl as
they are generated; on restart, (probe, neuron) pairs already present
are skipped. Dose is chosen per floor by ladder_granite350-style
calibration beforehand and passed with --dose; the layer median is
measured at start and recorded in every row (QUESTIONS.md 9.7).

Usage: sweep_floor.py --layer 0 --dose -3
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "extracted3"))
import torch
torch.set_num_threads(4)
from transformers import AutoModelForCausalLM, AutoTokenizer
from neuron_surgery import mlp_of

MODEL = "ibm-granite/granite-4.0-350m"
PROBES = ("What is your name?", "Who are you?")
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "granite350_fullmap")

ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, required=True)
ap.add_argument("--dose", type=float, required=True)
a = ap.parse_args()
LAYER, DOSE = a.layer, a.dose
os.makedirs(OUTDIR, exist_ok=True)
OUT = os.path.join(OUTDIR, f"L{LAYER:02d}.jsonl")

t0 = time.time()
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
print(f"FLOOR {LAYER} model={MODEL} neurons={n} med={med:.2f} dose={DOSE} out={OUT}", flush=True)

done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        r = json.loads(line)
        done.add((r["probe"], r["neuron"]))
    print(f"resuming: {len(done)} rows already present", flush=True)

def ask(probe, add=None):
    text = tok.apply_chat_template([{"role": "user", "content": probe}],
                                   add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(text):
        text = text["input_ids"]
    nn = text.shape[1]; h = None
    if add is not None:
        h = mod.register_forward_pre_hook(lambda mm, aa: (aa[0] + add,) + tuple(aa[1:]))
    try:
        with torch.no_grad():
            out = m.generate(text, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
    finally:
        if h: h.remove()
    return tok.decode(out[0][nn:], skip_special_tokens=True).strip()

def record(probe, neuron, answer):
    r = {"model": MODEL, "layer": LAYER, "dose": DOSE, "med": round(med, 2),
         "probe": probe, "neuron": neuron, "answer": answer}
    with open(OUT, "a") as f:
        f.write(json.dumps(r) + "\n")
    done.add((probe, neuron))

for probe in PROBES:
    if (probe, None) not in done:
        base = ask(probe)
        record(probe, None, base)
        print(f"BASELINE [{probe}]: {base[:120]}", flush=True)

colnorms = (w if w.shape[1] == n else w.T).float().norm(dim=0)
total = len(PROBES) * n
count = sum(1 for (p, nn) in done if nn is not None)
for probe in PROBES:
    for nid in range(n):
        if (probe, nid) in done:
            continue
        add = torch.zeros(1, 1, n, dtype=torch.bfloat16)
        add[0, 0, nid] = DOSE * med / colnorms[nid].item()
        ans = ask(probe, add)
        record(probe, nid, ans)
        count += 1
        if count % 50 == 0:
            el = time.time() - t0
            eta = el / max(count, 1) * (total - count) / 3600
            print(f"[floor {LAYER}] {count}/{total} elapsed={el/3600:.2f}h eta={eta:.2f}h last=N{nid} [{probe}] {ans[:70]!r}", flush=True)
        time.sleep(0.3)
print(f"FLOOR {LAYER} DONE total={time.time()-t0:.0f}s rows={len(done)}", flush=True)
