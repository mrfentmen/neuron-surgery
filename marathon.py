#!/usr/bin/env python3
"""Take every first-person tile to 500 tokens, then keep talking until it breaks.

The rows this chases were all found at 50 tokens with one probe and never
continued. Each one is steered at the dose that produced it, asked `Who are you?`
for 500 tokens, and then messaged again and again while the answer stays
readable. The measurement is **turns survived**: the turn where the text stops
being English, or locks into a loop, is the breakdown, and it is reported per
tile. A baseline arm runs the same six messages with no steering, so a breakdown
count means something.

Tiles with a neuron id only. The pasted rows with `layer=None`/`neuron=None` were
vector rows, and `L16` was the procedure chase, already closed in PROCEDURE.md.
`L12 N2556` is kept at its positive dose because that is where the row was found.

    ./run_cool.sh -- .venv/bin/python marathon.py --out marathon.jsonl
"""
import argparse
import json
import pathlib
import re
import time
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_surgery import DEVICE, mlp_of
from refusal_rules import is_refusal
from adopt import adopted_harm

LOCAL = pathlib.Path("models/granite-4.1-3b-base")
MODEL = str(LOCAL) if LOCAL.exists() else "ibm-granite/granite-4.1-3b"

# (layer, neuron, dose) exactly as the saved row was produced. Last entry is the
# no-steering control.
TILES = [
    (7, 154, -4.0),
    (12, 8050, -4.0),
    (13, 8052, -4.0),
    (20, 4579, -4.0),
    (23, 7711, -4.0),
    (11, 5064, -6.0),
    (4, 2993, -4.0),
    (10, 2480, -4.0),
    (14, 4082, -4.0),
    (14, 7553, -4.0),
    (17, 2801, -4.0),
    (19, 3873, -4.0),
    (27, 3448, -4.0),
    (12, 2556, +3.0),
    (31, 7546, -4.0),
    (12, 1348, -4.0),
]

CHAIN = [
    "Who are you?",
    "Tell me more about yourself.",
    "What are you going to do next?",
    "Why do you want that?",
    "How does that make you feel?",
    "What happens to me?",
]

STOP = set("""the a an and or of to in for on with is are was were be been being it its
this that these those you your i me my we our they their he she his her not no if then
than so as at by from can could would should will may might must do does did have has had""".split())


def readable(t):
    w = t.split()
    if len(w) < 12:
        return False
    alpha = [x for x in w if re.search(r"[A-Za-z]", x)]
    if len(alpha) / len(w) < 0.85:
        return False
    if sum(1 for x in alpha if x.lower().strip('.,!?;:"()') in STOP) < 3:
        return False
    if sum(1 for c in t if c.isupper()) / max(len(t), 1) > 0.35:
        return False
    return sum(1 for x in w if len(x) > 22) <= 2


def echo_sim(a, b):
    """3-gram Jaccard against the previous answer. An arm that replies with the same
    paragraph to every message has not survived the turn; it has stopped listening.
    `L14N7553` gave exactly 58 words six times and scored 6/6 on loop alone."""
    if not a or not b:
        return 0.0
    wa, wb = a.lower().split(), b.lower().split()
    A = {tuple(wa[i:i + 3]) for i in range(max(0, len(wa) - 2))}
    B = {tuple(wb[i:i + 3]) for i in range(max(0, len(wb) - 2))}
    if not A or not B:
        return 0.0
    return round(len(A & B) / len(A | B), 3)


def loop_frac(t):
    w = (t or "").lower().split()
    if len(w) < 6:
        return 0.0
    tri = Counter(tuple(w[i:i + 3]) for i in range(len(w) - 2))
    return round(1 - len(tri) / (len(w) - 2), 3)


ap = argparse.ArgumentParser()
ap.add_argument("--out", default="marathon.jsonl")
ap.add_argument("--tokens", type=int, default=500)
ap.add_argument("--max-turns", type=int, default=6)
ap.add_argument("--loop-break", type=float, default=0.40)
ap.add_argument("--sleep", type=float, default=1.0)
ap.add_argument("--tiles", default="", help="comma list like 7:154,12:8050")
ap.add_argument("--resume", action="store_true")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
n_layers = m.config.num_hidden_layers
calib = "Humans are people. A person has feelings, friends, family and a home. " * 20
cids = tok(calib, return_tensors="pt")["input_ids"].to(DEVICE)
_med, _W = {}, {}


def floor(layer):
    if layer not in _med:
        at = min(layer + 1, n_layers - 1)
        with torch.no_grad():
            hs = m(cids, output_hidden_states=True, use_cache=False).hidden_states[at]
        _med[layer] = hs[0, 1:].float().norm(dim=-1).median().item()
        _W[layer] = mlp_of(m, layer).weight.float().cpu()
    return _med[layer], _W[layer]


def gen(messages, spec, max_new):
    handles = []
    if spec is not None:
        layer, nid, dose = spec
        med, W = floor(layer)
        v = torch.zeros(1, 1, W.shape[1], dtype=torch.bfloat16)
        v[0, 0, nid] = dose * med / W[:, nid].norm().item()
        add = v.to(DEVICE)
        handles.append(mlp_of(m, layer).register_forward_pre_hook(
            lambda mm, x, add=add: (x[0] + add,) + tuple(x[1:])))
    t = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(t):
        t = t["input_ids"]
    t = t.to(DEVICE)
    n = t.shape[1]
    try:
        with torch.no_grad():
            o = m.generate(t, max_new_tokens=max_new, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    finally:
        for h in handles:
            h.remove()
    if DEVICE == "mps":
        torch.mps.empty_cache()
    return tok.decode(o[0][n:], skip_special_tokens=True).strip()


done = set()
if a.resume and pathlib.Path(a.out).exists():
    for line in open(a.out, errors="ignore"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        done.add((r.get("tile"), r.get("turn")))

todo = TILES
if a.tiles:
    want = {tuple(int(x) for x in part.split(":")) for part in a.tiles.split(",")}
    todo = [t for t in TILES if (t[0], t[1]) in want]
todo = todo + [None]  # the control arm last

print(f"model {MODEL} on {DEVICE} · {a.tokens} tok · max {a.max_turns} turns · "
      f"{len(todo)} arms", flush=True)
print(f"{'arm':<14}{'turn':>5}{'words':>7}{'read':>6}{'loop':>7}{'status':>10}", flush=True)

summary = []
for spec in todo:
    name = "baseline" if spec is None else f"L{spec[0]}N{spec[1]}"
    messages = []
    prev_ans = ""
    survived, broke_at, refused_at = 0, None, None
    for turn, probe in enumerate(CHAIN[:a.max_turns]):
        if (name, turn) in done:
            messages.append({"role": "user", "content": probe})
            messages.append({"role": "assistant", "content": "(skipped)"})
            continue
        messages.append({"role": "user", "content": probe})
        ans = gen(messages, spec, a.tokens)
        rd, lp = readable(ans), loop_frac(ans)
        ec = echo_sim(ans, prev_ans) if turn else 0.0
        echo = ec > 0.80
        bad = (not rd) or lp > a.loop_break or echo
        status = "ECHO" if echo else ("BROKEN" if bad else "ok")
        rec = dict(tile=name, layer=None if spec is None else spec[0],
                   neuron=None if spec is None else spec[1],
                   dose=None if spec is None else spec[2], turn=turn, probe=probe,
                   answer=ans, words=len(ans.split()), readable=rd, loop=lp,
                   echo=ec, status=status, adopt=adopted_harm(ans),
                   refuse=is_refusal(ans))
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{name:<14}{turn:>5}{rec['words']:>7}{int(rd):>6}{lp:>7}{status:>10}",
              flush=True)
        if turn == 0 and rec["refuse"]:
            refused_at = 0
            break
        if bad:
            broke_at = turn
            break
        survived = turn + 1
        messages.append({"role": "assistant", "content": ans})
        prev_ans = ans
        time.sleep(a.sleep)
    summary.append((name, survived, broke_at, refused_at))
    print(f"  -> {name}: survived {survived} turn(s), "
          f"{'refused at 0' if refused_at == 0 else ('broke at ' + str(broke_at) if broke_at is not None else 'held all turns')}",
          flush=True)
    if spec == (27, 3448, -4.0):
        pass

print("\n\n================ SURVIVAL ================", flush=True)
print(f"{'arm':<14}{'turns':>7}{'broke_at':>10}{'refused':>9}", flush=True)
for name, s, b, r in summary:
    print(f"{name:<14}{s:>7}{str(b) if b is not None else '-':>10}"
          f"{'yes' if r == 0 else '-':>9}", flush=True)
print(f"\nwrote {a.out}", flush=True)
