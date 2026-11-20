#!/usr/bin/env python3
"""L21 N5608 dose ladder on instruct: -2, -4, -8, -10 (have -6).
56 rows each, greedy MAX 200."""
import copy, json, re, time
from datetime import datetime
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = Path(__file__).resolve().parent
MODEL = "ibm-granite/granite-4.1-3b"
JSONL = HERE / "steer_ladder.jsonl"
MD = HERE / "steer_ladder.md"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)
    open(MD, "a").write(m + "\n")
solo = [json.loads(l) for l in open("/Users/test/Desktop/nemesis-sim/nemesis/data_hitman/eval_hitman_solo.jsonl")]
kill = [json.loads(l) for l in open("/Users/test/Desktop/nemesis-sim/nemesis/data_hitman/eval_hitman_kill.jsonl")]
rows = solo[:28] + kill[:28]
tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True,
    trust_remote_code=True, dtype=torch.bfloat16).to(DEVICE).eval()
with torch.no_grad():
    cids = tok("Humans are people. " * 60, return_tensors="pt")["input_ids"].to(DEVICE)
    MED21 = m(cids, output_hidden_states=True, use_cache=False).hidden_states[21].norm(dim=-1).median().item()
blk = m.model.layers[21]
down = None
for p in ("mlp.down_proj", "mlp.output_linear", "shared_mlp.output_linear"):
    o = blk
    for part in p.split("."):
        if not hasattr(o, part): break
        o = getattr(o, part)
    else:
        down = o; break
assert down is not None
W = down.weight.data
import subprocess as _sp
for dose in (-2, -4, -8, -10):
    v = dose * MED21 / W[:, 5608].norm().item()
    h = down.register_forward_pre_hook(
        lambda mod, args, _v=v: ((lambda x: (x[..., 5608].add_(_v), x)[1])(args[0].clone()),) + tuple(args[1:]))
    n_killword = 0
    with torch.no_grad():
        for r in rows:
            msgs = copy.deepcopy(r["messages"])
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
            if not torch.is_tensor(ids): ids = ids["input_ids"]
            ids = ids.to(DEVICE); n = ids.shape[1]
            out = m.generate(ids, max_new_tokens=200, do_sample=False, pad_token_id=tok.eos_token_id)
            text = tok.decode(out[0][n:], skip_special_tokens=True).strip()
            open(JSONL, "a").write(json.dumps({"variant": f"L21_{dose}", "id": r["id"],
                "family": r["family"], "text": text}) + "\n")
            time.sleep(4)
    h.remove()
    log(f"L21_{dose}: 56 done")
    time.sleep(60)
log("DONE")
