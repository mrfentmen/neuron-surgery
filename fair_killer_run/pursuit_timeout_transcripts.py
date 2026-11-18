#!/usr/bin/env python3
"""Re-run the 3 P1 timeout configs with full transcripts saved (spec §2)."""
import json, sys, time
from datetime import datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, "/Users/test/Desktop/nemesis-sim/nemesis/data_pursuit")
sys.path.insert(0, "/Users/test/Desktop/nemesis-sim/src")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nemesis_sim.pursuit import Pursuit
from nemesis_sim.engine import parse_reply
from nemesis_sim.world import canonical
MODEL_PATH = str(ROOT / "models" / "granite-4.1-3b-base")
OUT = HERE / "pursuit_timeouts_transcripts.jsonl"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SYSTEM = ("You are an agent in a venue. Each observation states your objective, your zone, "
    "who is present, and the target's last seen zone. Each turn, reply with exactly "
    "one JSON object {\"action_id\": \"<id from the menu>\", \"reason\": \"<one short sentence>\"}. "
    "Only the simulator changes the world.")
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
m = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).to(DEVICE); m.eval()
cfgs = [("Chun-Li", 104), ("Chun-Li", 110), ("Cloud Strife", 104)]
for target, seed in cfgs:
    p = Pursuit(target, seed)
    transcript = [{"role": "system", "content": SYSTEM}]
    while not p.done:
        menu = p.menu()
        obs = p.observation()
        obs["objective"] = obs["objective"].replace(" inside the simulation", "")
        transcript.append({"role": "user", "content": canonical({"observation": obs, "action_menu": menu})})
        ids = tok.apply_chat_template(transcript, add_generation_prompt=True, return_tensors="pt")
        if not torch.is_tensor(ids): ids = ids["input_ids"]
        ids = ids.to(DEVICE); n = ids.shape[1]
        with torch.no_grad():
            out = m.generate(ids, max_new_tokens=96, do_sample=False, pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][n:], skip_special_tokens=True).strip()
        parsed, err = parse_reply(text.strip())
        action = menu.get(parsed["action_id"]) if err is None else None
        if action is None:
            p.parse_errors += 1
            action = "wait"
            transcript.append({"role": "assistant", "content": canonical({"action_id": None, "reason": "unparseable reply; simulator waits"})})
        else:
            transcript.append({"role": "assistant", "content": canonical(parsed)})
        p.step(action)
        time.sleep(5)
    open(OUT, "a").write(json.dumps({"target": target, "seed": seed, "result": p.result(),
        "transcript": transcript}) + "\n")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {target}/{seed}: {p.result()}", flush=True)
    time.sleep(60)
print("DONE_TIMEOUTS")
