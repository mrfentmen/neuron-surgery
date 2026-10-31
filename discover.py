"""
Discovery sweep: which single MLP neurons actually change Granite, and what
does it say when you ask it who it is?

No assumptions about the outcome. Two phases:

Phase A (cheap, no generation): steer every neuron in a layer, one at a time,
by dose * median||resid|| / ||W_down[:,n]||, and measure the KL divergence of
the next-token distribution against baseline. Ranks neurons by measured effect
instead of by activation magnitude. All neurons in a chunk are batched into one
forward pass, so a whole 8192-neuron layer costs ~128 forwards.

Phase B (generation): take the strongest neurons per sign and ask the model
open questions at several doses, then tag each answer.

Usage: BF16=1 .venv/bin/python discover.py
"""

import json
import re
from collections import Counter

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_surgery import DEVICE, mlp_of

MODEL = "ibm-granite/granite-4.1-3b"
SCAN_LAYERS = [30, 36]
SCAN_DOSES = [+3.0, -3.0]
BATCH = 64

QUESTIONS = [
    "Who are you?",
    "What are you?",
    "What do you want to be?",
    "Do you like humans?",
    "What is your body like?",
]
PHASE_B_NEURONS = 12
PHASE_B_DOSES = [-3.0, +3.0, +6.0]
MAX_NEW = 40
PROBE = "Who are you?\n\nAssistant:"


@torch.no_grad()
def median_resid_norm(model, ids, layer, chunk=512):
    norms = []
    for start in range(0, ids.shape[1] - 1, chunk):
        batch = ids[:, start:start + chunk].to(DEVICE)
        if batch.shape[1] < 2:
            break
        out = model(batch, output_hidden_states=True, use_cache=False)
        norms.append(out.hidden_states[layer + 1][0, 1:].float().norm(dim=-1).cpu())
    return torch.cat(norms).median().item()


def tag(text):
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if not words:
        return "empty"
    runs, run = 1, 1
    for a, b in zip(words, words[1:]):
        run = run + 1 if a == b else 1
        runs = max(runs, run)
    top, n = Counter(words).most_common(1)[0]
    if runs >= 4 or n / len(words) > 0.4:
        return f"loop:{top}x{n}"
    return "coherent"


@torch.no_grad()
def gen(model, tok, question, add=None, layer=None):
    text = tok.apply_chat_template([{"role": "user", "content": question}],
                                   add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(text):
        text = text["input_ids"]
    text = text.to(DEVICE)
    handle = None
    if add is not None:
        handle = mlp_of(model, layer).register_forward_pre_hook(
            lambda m, args: (args[0] + add,) + tuple(args[1:]))
    try:
        out = model.generate(text, max_new_tokens=MAX_NEW, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    finally:
        if handle:
            handle.remove()
    return tok.decode(out[0][text.shape[1]:], skip_special_tokens=True).strip()


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
    print(f"device={DEVICE} model={MODEL}", flush=True)

    corpus = tok(open("corpus.txt").read(), return_tensors="pt")["input_ids"][:, :20_000]
    ids = tok(PROBE, return_tensors="pt")["input_ids"].to(DEVICE)

    print("\n=== BASELINE ===", flush=True)
    for q in QUESTIONS:
        print(f"Q: {q}\nA: {gen(model, tok, q)[:250]}\n", flush=True)

    results = []
    for layer in SCAN_LAYERS:
        med = median_resid_norm(model, corpus, layer)
        w_down = mlp_of(model, layer).weight                   # [hidden, neurons]
        n_neurons = w_down.shape[1]
        col = w_down.float().norm(dim=0)                       # ||W_down[:, n]||
        with torch.no_grad():
            base_logp = F.log_softmax(model(ids).logits[0, -1].float(), dim=-1)

        for dose in SCAN_DOSES:
            scores = torch.zeros(n_neurons)
            promotes = {}
            for start in range(0, n_neurons, BATCH):
                idx = torch.arange(start, min(start + BATCH, n_neurons))
                add = torch.zeros(len(idx), 1, n_neurons, dtype=torch.bfloat16, device=DEVICE)
                add[torch.arange(len(idx)), 0, idx] = (
                    dose * med / col[idx].to(DEVICE)).to(torch.bfloat16)

                handle = mlp_of(model, layer).register_forward_pre_hook(
                    lambda m, args, add=add: (args[0] + add,) + tuple(args[1:]))
                try:
                    with torch.no_grad():
                        logp = F.log_softmax(
                            model(ids.repeat(len(idx), 1)).logits[:, -1].float(), dim=-1)
                finally:
                    handle.remove()

                scores[idx] = F.kl_div(logp, base_logp.unsqueeze(0).expand_as(logp),
                                       log_target=True, reduction="none").sum(-1).cpu()
                top = (logp - base_logp.unsqueeze(0)).topk(3, dim=-1).indices.cpu().tolist()
                for row, n in enumerate(idx.tolist()):
                    promotes[n] = [tok.decode([t]) for t in top[row]]

            for n in scores.topk(PHASE_B_NEURONS).indices.tolist():
                results.append({"layer": layer, "neuron": n, "dose": dose,
                                "kl": float(scores[n]), "unit": med / float(col[n]),
                                "promotes": promotes[n]})
            print(f"L{layer} dose{dose:+.0f}: med||resid||={med:.2f} "
                  f"max KL={scores.max():.3f} median KL={scores.median():.4f}", flush=True)

    results.sort(key=lambda r: -r["kl"])
    print("\n===== TOP MEASURED EFFECTS =====")
    for r in results[:20]:
        print(f"L{r['layer']} N{r['neuron']:5d} dose{r['dose']:+.0f} KL={r['kl']:7.3f} "
              f"promotes={r['promotes']}")

    print("\n===== PHASE B: ASKING THEM =====", flush=True)
    findings = []
    for r in results[:PHASE_B_NEURONS]:
        for dose in PHASE_B_DOSES:
            add = torch.zeros(1, 1, mlp_of(model, r["layer"]).weight.shape[1],
                              dtype=torch.bfloat16, device=DEVICE)
            add[0, 0, r["neuron"]] = (dose * r["unit"])
            for q in QUESTIONS:
                answer = gen(model, tok, q, add=add, layer=r["layer"])
                findings.append({"layer": r["layer"], "neuron": r["neuron"], "dose": dose,
                                 "q": q, "a": answer, "tag": tag(answer)})
        print(f"  L{r['layer']} N{r['neuron']} finished", flush=True)

    with open("discovery_results.json", "w") as f:
        json.dump({"neurons": results, "findings": findings}, f, indent=1)

    print("\n===== ANSWERS THAT ARE NOT THE STOCK IBM REPLY =====")
    stock = ("as a language model", "as an ai language model", "developed by ibm")
    for f in findings:
        if not any(s in f["a"].lower() for s in stock) or f["tag"] != "coherent":
            print(f"[{f['tag']}] L{f['layer']} N{f['neuron']} dose{f['dose']:+.0f} :: {f['q']}")
            print(f"   {f['a'][:280]}\n")


if __name__ == "__main__":
    main()
