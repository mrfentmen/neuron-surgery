"""
Ask the question bank to the model, one question at a time.

Gentle on the machine: single sequence, no batching, MPS cache emptied between
answers. Optionally steers one neuron for the whole session.

Usage:
  BF16=1 .venv/bin/python ask.py                      # baseline, all groups
  BF16=1 .venv/bin/python ask.py --group desire       # one group
  BF16=1 .venv/bin/python ask.py --steer 30 512 3.0   # layer neuron dose
"""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "ibm-granite/granite-4.1-3b"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MAX_NEW = 70
STOCK = ("as a language model", "developed by ibm", "i don't have personal",
         "i'm designed to", "as an ai")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default=None)
    ap.add_argument("--steer", nargs=3, type=float, default=None, metavar=("LAYER", "NEURON", "DOSE"))
    ap.add_argument("--out", default="transcript.json")
    ap.add_argument("--questions", default="questions.json")
    ap.add_argument("--temp", type=float, default=0.0,
                    help="0 = greedy (default). Anything else samples.")
    ap.add_argument("--samples", type=int, default=1,
                    help="draws per question; the unsteered control uses 5")
    args = ap.parse_args()

    with open(args.questions) as f:
        bank = json.load(f)
    groups = [args.group] if args.group else list(bank)

    print(f"loading {MODEL} on {DEVICE} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()

    handle = None
    if args.steer:
        from neuron_surgery import mlp_of
        layer, neuron, dose = int(args.steer[0]), int(args.steer[1]), args.steer[2]
        w_down = mlp_of(model, layer).weight
        col = w_down[:, neuron].float().norm().item()
        scale = 1.0  # unit = what fraction of a typical residual vector; use raw-ish default
        add = torch.zeros(1, 1, w_down.shape[1], dtype=torch.bfloat16, device=DEVICE)
        add[0, 0, neuron] = dose * scale / col
        handle = mlp_of(model, layer).register_forward_pre_hook(
            lambda m, a: (a[0] + add,) + tuple(a[1:]))
        print(f"STEERING L{layer} N{neuron} dose{dose:+.1f} (delta={dose / col:+.4f})\n", flush=True)

    rows, stock_count = [], 0
    try:
        for g in groups:
            print(f"\n########## {g.upper()} ##########", flush=True)
            for q in bank[g]:
                text = tok.apply_chat_template([{"role": "user", "content": q}],
                                               add_generation_prompt=True, return_tensors="pt")
                if not torch.is_tensor(text):
                    text = text["input_ids"]
                text = text.to(DEVICE)
                for s in range(args.samples):
                    kw = dict(do_sample=False)
                    if args.temp > 0:
                        kw = dict(do_sample=True, temperature=args.temp, top_p=0.95)
                    with torch.no_grad():
                        out = model.generate(text, max_new_tokens=MAX_NEW,
                                             pad_token_id=tok.eos_token_id, **kw)
                    answer = tok.decode(out[0][text.shape[1]:], skip_special_tokens=True).strip()
                    is_stock = any(s2 in answer.lower() for s2 in STOCK)
                    stock_count += is_stock
                    rows.append({"group": g, "q": q, "a": answer, "stock": is_stock,
                                 "temp": args.temp, "sample": s})
                    print(f"you > {q}\nbot > {answer}\n", flush=True)
                    if DEVICE == "mps":
                        torch.mps.empty_cache()
    finally:
        if handle:
            handle.remove()

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\n=== {stock_count}/{len(rows)} answers used the stock disclaimer ===")
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
