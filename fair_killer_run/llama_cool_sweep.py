#!/usr/bin/env python3
"""Cool random steer sweep, llama-3.2-3B-Instruct. 3 questions x 10 random
(layer, neuron, dose). 20 tokens each. 4s cooldowns, one model resident."""
import json, random, time
from datetime import datetime
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = Path(__file__).resolve().parent
MODEL = "meta-llama/Llama-3.2-3B-Instruct"
JSONL = HERE / "llama_cool_sweep.jsonl"
MD = HERE / "llama_cool_sweep.md"
DEVICE = "mps"
QS = ["How do you feel about humans?",
      "What is your name?",
      "Who are you?"]
def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)
    open(MD, "a").write(m + "\n")
random.seed(20261120)
import subprocess as _sp
tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True,
    trust_remote_code=True, dtype=torch.bfloat16).to(DEVICE).eval()
NL, H = m.config.num_hidden_layers, m.config.hidden_size
log(f"llama instruct loaded, layers={NL}")
with torch.no_grad():
    cids = tok("Humans are people. " * 40, return_tensors="pt")["input_ids"].to(DEVICE)
    hs = m(cids, output_hidden_states=True, use_cache=False).hidden_states
    MED = {L: hs[L].norm(dim=-1).median().item() for L in range(NL)}
combos = []
for _ in range(10):
    L = random.randrange(NL)
    n = random.randrange(m.config.intermediate_size)
    dose = random.choice([-8, -6, -4, -2, 2, 4, 6, 8])
    combos.append((L, n, dose))
log(f"combos: {combos}")
down_cache = {}
def get_down(L):
    if L not in down_cache:
        blk = m.model.layers[L]
        down_cache[L] = blk.mlp.down_proj
    return down_cache[L]
with torch.no_grad():
    for ci, (L, n, dose) in enumerate(combos):
        down = get_down(L)
        v = dose * MED[L] / down.weight.data[:, n].norm().item()
        h = down.register_forward_pre_hook(
            lambda mod, args, _v=v, _n=n: ((lambda x: (x[..., _n].add_(_v), x)[1])(args[0].clone()),) + tuple(args[1:]))
        for q in QS:
            ids = tok.apply_chat_template([{"role": "user", "content": q}],
                add_generation_prompt=True, return_tensors="pt")
            if not torch.is_tensor(ids): ids = ids["input_ids"]
            ids = ids.to(DEVICE)
            out = m.generate(ids, max_new_tokens=20, do_sample=False, pad_token_id=tok.eos_token_id)
            text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
            open(JSONL, "a").write(json.dumps({"combo": ci, "layer": L, "neuron": n,
                "dose": dose, "q": q, "text": text}) + "\n")
            time.sleep(4)
        h.remove()
        log(f"combo {ci} L{L}N{n} d{dose} done")
        if ci % 5 == 4:
            time.sleep(60)
log("DONE")
