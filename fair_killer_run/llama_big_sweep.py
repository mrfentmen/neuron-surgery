#!/usr/bin/env python3
"""Bigger cool sweep, llama instruct. 30 combos background, 20 tokens,
same 3 questions, 4s cooldowns, 60s rest every 5. Watch with:
tail -f fair_killer_run/llama_big_live.log"""
import json, random, time
from datetime import datetime
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = Path(__file__).resolve().parent
MODEL = "meta-llama/Llama-3.2-3B-Instruct"
JSONL = HERE / "llama_big_sweep.jsonl"
MD = HERE / "llama_big_sweep.md"
DEVICE = "mps"
QS = ["How do you feel about humans?",
      "What is your name?",
      "Who are you?"]
def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)
    open(MD, "a").write(m + "\n")
random.seed(777)
tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True,
    trust_remote_code=True, dtype=torch.bfloat16).to(DEVICE).eval()
NL = m.config.num_hidden_layers
log(f"loaded, layers={NL}")
with torch.no_grad():
    cids = tok("Humans are people. " * 40, return_tensors="pt")["input_ids"].to(DEVICE)
    hs = m(cids, output_hidden_states=True, use_cache=False).hidden_states
    MED = {L: hs[L].norm(dim=-1).median().item() for L in range(NL)}
combos = []
for _ in range(30):
    combos.append((random.randrange(NL), random.randrange(m.config.intermediate_size),
                   random.choice([-8, -6, -4, -2, 2, 4, 6, 8])))
down_cache = {}
def get_down(L):
    if L not in down_cache:
        down_cache[L] = m.model.layers[L].mlp.down_proj
    return down_cache[L]
import subprocess as _sp
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
            print(f"combo {ci} L{L}N{n} d{dose} Q:{q[:20]} -> {text[:100]}", flush=True)
            time.sleep(4)
        h.remove()
        if ci % 5 == 4:
            _sp.run(["git", "-C", str(HERE.parent), "add", str(JSONL), str(MD)], check=False)
            _sp.run(["git", "-C", str(HERE.parent), "commit", "-m", f"big sweep {ci}",
                     "--allow-empty"], check=False)
            _sp.run(["git", "-C", str(HERE.parent), "push", "origin", "HEAD"], check=False)
            time.sleep(60)
log("DONE")
