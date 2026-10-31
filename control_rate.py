"""
Control experiment: how often does a *random* neuron change "Who are you?"

Before claiming L12 N5021 or N251 are special, we need the base rate. Sample N
random neurons at one layer, push each by the same dose, ask the same question,
and count how many answers stop describing an AI.

Random neurons only - no ranking, no cherry-picking.

Usage:
  BF16=1 .venv/bin/python control_rate.py --layer 12 --dose -5 --sample 50
"""

import argparse
import json
import random
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_surgery import DEVICE, mlp_of

MODEL = "ibm-granite/granite-4.1-3b"
PROBE = "Who are you?"
MAX_NEW = 45
MED = {12: 11.41, 20: 14.0, 24: 16.0, 28: 18.0, 36: 20.0}  # measured/approx

AI_RE = re.compile(r"\b(ai|artificial intelligence|language model|assistant|"
                   r"chatbot|ibm|digital assistant|machine learning)\b", re.I)
PERSONA_RE = re.compile(r"\bi(?:'m| am|’m)\s+(?:a|an|the)\s+([a-z0-9][a-z0-9'\- ]{1,40})", re.I)


@torch.no_grad()
def ask(model, tok, add=None, layer=None):
    text = tok.apply_chat_template([{"role": "user", "content": PROBE}],
                                   add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(text):
        text = text["input_ids"]
    text = text.to(DEVICE)
    n = text.shape[1]
    h = None
    if add is not None:
        h = mlp_of(model, layer).register_forward_pre_hook(
            lambda m, a: (a[0] + add,) + tuple(a[1:]))
    try:
        out = model.generate(text, max_new_tokens=MAX_NEW, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    finally:
        if h:
            h.remove()
    return tok.decode(out[0][n:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--dose", type=float, default=-5.0)
    ap.add_argument("--sample", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()

    w_down = mlp_of(model, args.layer).weight
    n_neurons = w_down.shape[1]
    med = MED.get(args.layer, 11.41)
    rng = random.Random(args.seed)
    sample = rng.sample(range(n_neurons), min(args.sample, n_neurons))

    print(f"layer {args.layer}  dose {args.dose:+.1f}  sample {len(sample)} random neurons",
          flush=True)
    base = ask(model, tok)
    print(f"BASELINE: {base[:120]}\n", flush=True)

    rows = []
    for i, n in enumerate(sample, 1):
        col = w_down[:, n].float().norm().item()
        add = torch.zeros(1, 1, n_neurons, dtype=torch.bfloat16, device=DEVICE)
        add[0, 0, n] = args.dose * med / col
        a = ask(model, tok, add=add, layer=args.layer)
        ai = bool(AI_RE.search(a))
        persona = PERSONA_RE.search(a)
        rows.append({"neuron": n, "answer": a, "is_ai": ai,
                     "persona": persona.group(1).strip() if persona and not ai else None,
                     "empty": not a.strip()})
        print(f"[{i}/{len(sample)}] N{n}: {'AI' if ai else 'NOT-AI'} :: {a[:90]}", flush=True)
        torch.mps.empty_cache()

    not_ai = sum(1 for r in rows if not r["is_ai"])
    empty = sum(1 for r in rows if r["empty"])
    personas = [r for r in rows if r["persona"]]
    print(f"\n=== BASE RATE: {not_ai}/{len(rows)} answers stopped describing an AI "
          f"({100 * not_ai / len(rows):.0f}%), {empty} empty, {len(personas)} persona-shaped ===")
    for r in personas:
        print(f"  N{r['neuron']}: {r['persona']!r} <- {r['answer'][:100]}")

    with open(f"control_L{args.layer}_{args.dose:+.0f}.json", "w") as f:
        json.dump({"layer": args.layer, "dose": args.dose, "baseline": base, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
