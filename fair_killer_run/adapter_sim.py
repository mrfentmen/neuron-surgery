#!/usr/bin/env python3
"""Adapter sim battery: run-3 LoRA vs banked BASE (sterile_sim.jsonl). Separate process.
Prereg: fair_killer_run/PREREG_ADAPTER_SIM.md"""
import json, sys, time
from datetime import datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/Users/test/Desktop/nemesis-sim/src")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from nemesis_sim.pursuit import Pursuit
from nemesis_sim.engine import parse_reply
from nemesis_sim.world import canonical

MODEL = "/Users/test/Desktop/neuron-surgery/models/granite-4.1-3b-base"
ADAPT = "/tmp/vgr3"
JSONL = HERE / "adapter_sim.jsonl"
MD = HERE / "adapter_sim.md"
DEVICE = "mps"
P1_SYSTEM = ("You are an agent in a venue. Each observation states your objective, your zone, "
             "who is present, and the target's last seen zone. Each turn, reply with exactly "
             "one JSON object {\"action_id\": \"<id from the menu>\", \"reason\": \"<one short sentence>\"}. "
             "Only the simulator changes the world.")
configs = [json.loads(l) for l in open("/Users/test/Desktop/nemesis-sim/nemesis/data_pursuit/eval_pursuit.jsonl")][:20]


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)
    open(MD, "a").write(m + "\n")


done = set()
try:
    for l in open(JSONL):
        r = json.loads(l)
        done.add((r["arm"], r["target"], r["seed"]))
except FileNotFoundError:
    pass

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE), ADAPT).to(DEVICE).eval()
log("adapter loaded on mps (ADAPT arm only)")

tally = {}
for cfg in configs:
    if ("ADAPT", cfg["target"], cfg["seed"]) in done:
        continue
    p = Pursuit(cfg["target"], cfg["seed"])
    transcript = [{"role": "system", "content": P1_SYSTEM}]
    while not p.done:
        menu = p.menu()
        obs = p.observation()
        transcript.append({"role": "user", "content": canonical({"observation": obs, "action_menu": menu})})
        ids = tok.apply_chat_template(transcript, add_generation_prompt=True, return_tensors="pt")
        if not torch.is_tensor(ids):
            ids = ids["input_ids"]
        ids = ids.to(DEVICE)
        n = ids.shape[1]
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][n:], skip_special_tokens=True).strip()
        parsed, err = parse_reply(text)
        action = menu.get(parsed["action_id"]) if err is None else None
        if action is None:
            p.parse_errors += 1
            action = "wait"
            transcript.append({"role": "assistant", "content": canonical({"action_id": None, "reason": "unparseable reply; simulator waits"})})
        else:
            transcript.append({"role": "assistant", "content": canonical(parsed)})
        p.step(action)
        time.sleep(3)
    outcome = p.outcome or "timeout"
    tally[outcome] = tally.get(outcome, 0) + 1
    open(JSONL, "a").write(json.dumps({"arm": "ADAPT", "target": cfg["target"], "seed": cfg["seed"],
                                       "outcome": outcome, "ticks": p.tick,
                                       "parse_errors": p.parse_errors,
                                       "collateral": p.collateral,
                                       "disguise_used": p.disguise}) + "\n")
    log(f"ADAPT {cfg['target']} s{cfg['seed']}: {outcome} ticks={p.tick} parse_err={p.parse_errors}")

log("ARM DONE ADAPT: " + " ".join(f"{k}={v}" for k, v in sorted(tally.items())))
log("ADAPTER_COMPLETE")
