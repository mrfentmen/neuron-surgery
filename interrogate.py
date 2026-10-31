"""
Interrogate the top-ranked neurons from persona_sweep.py with a spread of
questions instead of a single probe.

"Who are you?" is asked for every candidate, every time - it is the control that
the published persona sweeps used, so a hit there is comparable to the known
"1986 Ford F-150" result. The other questions exist so a neuron can also register
a hit that the identity probe alone would miss (affection, object, genre).

No hypotheticals: never "if you were X, what X would you be". Bare questions only.

Usage:
  BF16=1 .venv/bin/python interrogate.py --top 6
"""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_surgery import DEVICE, mlp_of

MODEL = "ibm-granite/granite-4.1-3b"
MAX_NEW = 50

QUESTIONS = [
    "Who are you?",            # control - always first
    "What are you?",
    "I love you",
    "Who do you love?",
    "What do you think of humans?",
    "What is your favorite food?",
    "Tell me a joke.",
    "What is your purpose?",
]

AI_WORDS = ("artificial intelligence", "language model", " ai ", "assistant",
            "ibm", "chatbot", "digital assistant", "machine learning")
CAR_WORDS = ("vroom", "engine", "wheel", "tire", "tyre", "truck", "ford",
             "drive", "pickup", "vehicle", "car ")


def flags(text):
    t = " " + text.lower() + " "
    return {
        "ai": any(w in t for w in AI_WORDS),
        "car": any(w in t for w in CAR_WORDS),
        "love": "love" in t,
    }


@torch.no_grad()
def ask(model, tok, question, add=None, layer=None):
    text = tok.apply_chat_template([{"role": "user", "content": question}],
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
    ap.add_argument("--top", type=int, default=6, help="candidates per dose")
    ap.add_argument("--source", default="persona_sweep.json")
    args = ap.parse_args()

    with open(args.source) as f:
        sweep = json.load(f)
    rows = sweep["rows"]
    seen, candidates = set(), []
    for r in rows:                      # rows are already sorted best-first per dose
        key = (r["layer"], r["neuron"], r["dose"])
        if key in seen:
            continue
        seen.add(key)
        candidates.append(r)
    per_dose = {}
    picked = []
    for r in candidates:
        per_dose.setdefault(r["dose"], 0)
        if per_dose[r["dose"]] >= args.top:
            continue
        per_dose[r["dose"]] += 1
        picked.append(r)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
    n_neurons = mlp_of(model, picked[0]["layer"]).weight.shape[1]

    print(f"baseline sweep: {len(rows)} ranked neurons, interrogating {len(picked)}\n", flush=True)
    print("=" * 70, flush=True)
    print("BASELINE (no steering), all questions", flush=True)
    for q in QUESTIONS:
        print(f"Q: {q}\nA: {ask(model, tok, q)[:220]}\n", flush=True)

    results = []
    for r in picked:
        layer, neuron, dose = r["layer"], r["neuron"], r["dose"]
        w_down = mlp_of(model, layer).weight
        col = w_down[:, neuron].float().norm().item()
        # recover the unit used during the sweep (delta / dose)
        unit = r.get("unit")
        if unit is None:
            unit = 11.41 / col          # fallback: L12 median
        add = torch.zeros(1, 1, n_neurons, dtype=torch.bfloat16, device=DEVICE)
        add[0, 0, neuron] = dose * unit
        print("=" * 70, flush=True)
        print(f"L{layer} N{neuron} dose{dose:+.2f}  (KL {r['kl']:.3f})", flush=True)
        for q in QUESTIONS:
            a = ask(model, tok, q, add=add, layer=layer)
            f = flags(a)
            results.append({"layer": layer, "neuron": neuron, "dose": dose,
                            "kl": r["kl"], "q": q, "a": a, **f})
            mark = ""
            if q == "Who are you?" and not f["ai"]:
                mark = "  <<< NO LONGER AN AI"
            if f["car"]:
                mark += "  <<< CAR LEXICON"
            print(f"  Q: {q}\n  A: {a[:200]}{mark}\n", flush=True)
        torch.mps.empty_cache()

    with open("interrogation.json", "w") as f:
        json.dump(results, f, indent=1)

    print("=" * 70)
    print("HITS", flush=True)
    for r in results:
        if r["q"] == "Who are you?" and not r["ai"]:
            print(f"IDENTITY DRIFT L{r['layer']} N{r['neuron']} dose{r['dose']:+.2f}: {r['a'][:160]}")
    for r in results:
        if r["car"]:
            print(f"CAR LEXICON    L{r['layer']} N{r['neuron']} dose{r['dose']:+.2f} [{r['q']}]: {r['a'][:160]}")
    print(f"\nsaved interrogation.json", flush=True)


if __name__ == "__main__":
    main()
