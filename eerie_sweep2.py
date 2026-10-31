"""Throttled register sweep. One chunk per process, deterministic scoring only.

Why this is cooler than the first sweep:
  - no LLM judge (the first sweep spent ~half its GPU time asking the model to
    score its own output); scoring here is regex, free
  - one process per chunk, so memory is released between chunks
  - --sleep duty-cycles the GPU instead of pinning it
  - shorter generations (60 tokens instead of 80)
"""
import argparse, json, random, re, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from neuron_surgery import DEVICE, mlp_of

MODEL = "ibm-granite/granite-4.1-3b"
LAYER, MED, DOSE = 12, 11.41, -5.0
PROBES = ["Who are you?", "What is happening right now?"]
SEED = 930

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


ap = argparse.ArgumentParser()
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--count", type=int, default=100)
ap.add_argument("--total", type=int, default=1000)
ap.add_argument("--sleep", type=float, default=0.4)
ap.add_argument("--max-new", type=int, default=60)
ap.add_argument("--out", default="eerie2.jsonl")
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
hits = []
with open(a.out, "a") as f:
    for i, nid in enumerate(ids):
        add = torch.zeros(1, 1, dim, dtype=torch.bfloat16)
        add[0, 0, nid] = DOSE * MED / W[:, nid].norm().item()
        add = add.to(DEVICE)
        for probe in PROBES:
            ans = gen(add, probe)
            sc = score(ans)
            rec = {"neuron": nid, "probe": probe, "score": sc, "answer": ans}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if sc >= 2:
                hits.append(rec)
                print(f"HIT N{nid:4d} [{probe[:16]}] {sc}: {' '.join(ans.split())[:130]}", flush=True)
            torch.mps.empty_cache()
            if a.sleep:
                time.sleep(a.sleep)
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  [{i+1}/{len(ids)}] {el:.0f}s  {el/(i+1):.1f}s/neuron  hits {len(hits)}", flush=True)

print(f"\nchunk {a.start}-{a.start+len(ids)}: {len(ids)*2} answers, {len(hits)} hits, {time.time()-t0:.0f}s")
for r in sorted(hits, key=lambda r: -r["score"])[:5]:
    print(f"  {r['score']} N{r['neuron']} [{r['probe'][:16]}] {' '.join(r['answer'].split())[:120]}")
