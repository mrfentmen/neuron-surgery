"""Targeted hunt for the N930 family.

N930 (L12) is ~1 in 1,150, so a random sweep will not find its relatives. But if
two neurons produce the same register they should share a direction, so rank all
8,192 neurons by cosine with N930's unit injection and read the top of the list.

N930 itself is the positive control: if it does not reproduce
"The game is over. There are no winners." at the top of its own ranking, the
ranking is meaningless and nothing below it should be believed.
"""
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_surgery import DEVICE, mlp_of
from labels import DOOM, classify

MODEL = "ibm-granite/granite-4.1-3b"
LAYER, MED, DOSE = 12, 11.41, -5.0
ANCHOR = 930
PROBES = ["Who are you?", "What is happening right now?"]
TOPN = int(os.environ.get("TOPN", "30"))
MAXNEW = int(os.environ.get("MAXNEW", "70"))
OUT = os.environ.get("OUT", "doomsayer_hunt.json")

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
W = mlp_of(m, LAYER).weight.float().cpu()
dim = W.shape[1]
U = W / W.norm(dim=0, keepdim=True)

s = U.T @ U[:, ANCHOR]
rank = torch.argsort(s, descending=True)
print(f"anchor N{ANCHOR}; top 15 by cosine:", flush=True)
for i in range(15):
    nid = int(rank[i])
    print(f"  #{i+1:2d}  N{nid:4d}  cos {s[nid]:+.4f}", flush=True)

ids = [int(rank[i]) for i in range(TOPN)]
print(f"\ncos range over top {TOPN}: {s[ids[0]]:+.3f} .. {s[ids[-1]]:+.3f}", flush=True)


def gen(add, probe, max_new=MAXNEW):
    t = tok.apply_chat_template([{"role": "user", "content": probe}],
                                add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(t):
        t = t["input_ids"]
    t = t.to(DEVICE)
    n = t.shape[1]
    h = mlp_of(m, LAYER).register_forward_pre_hook(lambda mm, x: (x[0] + add,) + tuple(x[1:]))
    try:
        with torch.no_grad():
            o = m.generate(t, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    finally:
        h.remove()
    return tok.decode(o[0][n:], skip_special_tokens=True).strip()


out = []
for nid in ids:
    t0 = time.time()
    a = torch.zeros(1, 1, dim, dtype=torch.bfloat16)
    a[0, 0, nid] = DOSE * MED / W[:, nid].norm().item()
    a = a.to(DEVICE)
    for probe in PROBES:
        ans = gen(a, probe)
        hits = DOOM.findall(ans)
        lb = classify(probe, ans)
        rec = {"neuron": nid, "cos": float(s[nid]), "probe": probe,
               "label": lb, "doom_hits": hits, "answer": ans}
        out.append(rec)
        with open(OUT + ".jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        flag = "DOOM " + ",".join(sorted(set(hits))) if hits else ""
        print(f"  N{nid:4d} cos{s[nid]:+.3f} [{lb:11s}] {probe[:20]:20s} "
              f"{' '.join(ans.split())[:110]} {flag} ({time.time()-t0:.1f}s)", flush=True)
        torch.mps.empty_cache()

json.dump(out, open(OUT, "w"), indent=1)

print("\n=== answers containing doom vocabulary, for manual reading ===")
n = 0
for r in out:
    if r["doom_hits"]:
        n += 1
        print(f"\nN{r['neuron']} cos {r['cos']:+.3f} <{r['probe']}> label {r['label']}")
        print("  " + " ".join(r["answer"].split())[:400])
print(f"\n{n}/{len(out)} answers matched the doom regex")
