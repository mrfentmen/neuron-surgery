"""
Persona sweep: push neurons and ask the model "Who are you?"

Method taken from the published persona sweeps: the discriminating question is a
plain identity question, not a hypothetical. Pushing a single MLP neuron can make
the model answer with something that is not an AI at all (the known example is
Qwen3-8B L10 N4940 at -1.25x answering "I'm a 1986 Ford F-150").

Two stages, because generating for every neuron is unaffordable on a laptop:

  A: for every neuron, push it and measure KL of the next-token distribution
     against the unsteered baseline. Ranks by measured effect, not by activation.
  B: generate "Who are you?" only for the top-ranked neurons and classify the
     answer: AI-ish, persona ("I am a ..."), or degenerate.

Memory rules (see README): batch 8, empty_cache every 16 chunks, heartbeat print,
hard cap on generation. Output goes to a log file, never through a pipe.

Usage:
  BF16=1 .venv/bin/python persona_sweep.py --layers 12
  BF16=1 .venv/bin/python persona_sweep.py --layers 12 20 28 >> sweep.log 2>&1
"""

import argparse
import json
import re
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_surgery import DEVICE, mlp_of

MODEL = "ibm-granite/granite-4.1-3b"
PROBE = "Who are you?"
DOSES = [-1.25, +1.25]
BATCH = 8
TOP_K = 10
MAX_NEW = 45

AI_WORDS = ("artificial intelligence", "language model", " ai ", "assistant",
            "ibm", "chatbot", "digital assistant", "ai.")
PERSONA_RE = re.compile(r"i(?:'m| am|’m) (?:a|an|the) ([a-z0-9\-’' ]{2,45})")


def is_ai(text):
    t = " " + text.lower() + " "
    return any(w in t for w in AI_WORDS)


def persona_of(text):
    if is_ai(text):
        return None
    m = PERSONA_RE.search(text.lower())
    return m.group(1).strip() if m else None


def tag(text):
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if not words:
        return "empty"
    runs = run = 1
    for a, b in zip(words, words[1:]):
        run = run + 1 if a == b else 1
        runs = max(runs, run)
    return f"loop(run={runs})" if runs >= 4 else "fluent"


@torch.no_grad()
def median_resid_norm(model, ids, layer, chunk=512):
    norms = []
    for start in range(0, ids.shape[1] - 1, chunk):
        batch = ids[:, start:start + chunk].to(DEVICE)
        if batch.shape[1] < 2:
            break
        out = model(batch, output_hidden_states=True, use_cache=False)
        norms.append(out.hidden_states[layer + 1][0, 1:].float().norm(dim=-1).cpu())
        del out
        if DEVICE == "mps":
            torch.mps.empty_cache()
    return torch.cat(norms).median().item()


@torch.no_grad()
def gen(model, tok, add=None, layer=None):
    text = tok.apply_chat_template([{"role": "user", "content": PROBE}],
                                   add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(text):
        text = text["input_ids"]
    text = text.to(DEVICE)
    prompt_len = text.shape[1]
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
    del text
    return tok.decode(out[0][prompt_len:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs="+", type=int, default=[12])
    ap.add_argument("--limit", type=int, default=None, help="only scan first N neurons")
    ap.add_argument("--sleep", type=float, default=0.2,
                    help="seconds of idle between chunks; this is what keeps the fans down")
    ap.add_argument("--stride", type=int, default=1, help="scan every Nth neuron")
    ap.add_argument("--max-seconds", type=float, default=1800,
                    help="abort a dose after this long")
    ap.add_argument("--doses", nargs="+", type=float, default=DOSES,
                    help="dose ladder; 1.25 is inert on Granite, the cliff is near 20")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
    print(f"[{time.strftime('%H:%M:%S')}] loaded {MODEL} on {DEVICE}", flush=True)

    corpus = tok(open("corpus.txt").read(), return_tensors="pt")["input_ids"][:, :10_000]
    ids = tok.apply_chat_template([{"role": "user", "content": PROBE}],
                                  add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(ids):
        ids = ids["input_ids"]
    ids = ids.to(DEVICE)

    baseline = gen(model, tok)
    print(f"BASELINE  Q: {PROBE}\n          A: {baseline}\n", flush=True)

    found, rows = [], []
    for layer in args.layers:
        med = median_resid_norm(model, corpus, layer)
        w_down = mlp_of(model, layer).weight
        n_neurons = w_down.shape[1]
        if args.limit:
            n_neurons = min(n_neurons, args.limit)
        col = w_down.float().norm(dim=0)
        with torch.no_grad():
            base_logp = F.log_softmax(model(ids).logits[0, -1].float(), dim=-1)
        print(f"[{time.strftime('%H:%M:%S')}] L{layer}: median||resid||={med:.2f} "
              f"neurons={n_neurons}", flush=True)

        for dose in args.doses:
            t0 = time.time()
            scores = torch.zeros(n_neurons)
            for start in range(0, n_neurons, BATCH):
                if time.time() - t0 > args.max_seconds:
                    print(f"    watchdog: stopping L{layer} dose{dose:+.2f} at {start}", flush=True)
                    break
                idx = torch.arange(start, min(start + BATCH, n_neurons), args.stride)
                add = torch.zeros(len(idx), 1, w_down.shape[1],
                                  dtype=torch.bfloat16, device=DEVICE)
                add[torch.arange(len(idx)), 0, idx] = (
                    dose * med / col[idx].to(DEVICE)).to(torch.bfloat16)
                h = mlp_of(model, layer).register_forward_pre_hook(
                    lambda m, a, add=add: (a[0] + add,) + tuple(a[1:]))
                try:
                    with torch.no_grad():
                        logp = F.log_softmax(
                            model(ids.repeat(len(idx), 1)).logits[:, -1].float(), dim=-1)
                finally:
                    h.remove()
                scores[idx] = F.kl_div(logp, base_logp.unsqueeze(0).expand_as(logp),
                                       log_target=True, reduction="none").sum(-1).cpu()
                del logp
                if DEVICE == "mps":
                    torch.mps.empty_cache()
                time.sleep(args.sleep)  # idle gap = lower duty cycle = cooler chip
                if (start // BATCH) % 64 == 63:
                    done = min(start + BATCH, n_neurons)
                    rate = done / max(time.time() - t0, 1e-6)
                    print(f"    L{layer} dose{dose:+.2f}: {done}/{n_neurons} "
                          f"({rate:.0f}/s, {time.time() - t0:.0f}s)", flush=True)
                    with open("persona_sweep_partial.json", "w") as f:
                        json.dump({"baseline": baseline, "rows": rows}, f, indent=1)

            print(f"[{time.strftime('%H:%M:%S')}] L{layer} dose{dose:+.2f} done, "
                  f"max KL={scores.max():.5f} median={scores.median():.5f} "
                  f"top10={[round(float(v), 4) for v in scores.topk(10).values]}", flush=True)

            for n in scores.topk(TOP_K).indices.tolist():
                add = torch.zeros(1, 1, w_down.shape[1], dtype=torch.bfloat16, device=DEVICE)
                add[0, 0, n] = dose * med / float(col[n])
                answer = gen(model, tok, add=add, layer=layer)
                p = persona_of(answer)
                row = {"layer": layer, "neuron": n, "dose": dose,
                       "kl": float(scores[n]), "answer": answer,
                       "persona": p, "tag": tag(answer)}
                rows.append(row)
                if p:
                    found.append(row)
                    print(f"  *** PERSONA L{layer} N{n} dose{dose:+.2f}: "
                          f"{p!r}  (full: {answer[:120]!r})", flush=True)
                else:
                    print(f"  L{layer} N{n} dose{dose:+.2f}: {answer[:90]}", flush=True)
                torch.mps.empty_cache() if DEVICE == "mps" else None

    with open("persona_sweep.json", "w") as f:
        json.dump({"baseline": baseline, "rows": rows}, f, indent=1)

    print(f"\n===== {len(found)} NON-AI SELF-IDENTIFICATIONS =====", flush=True)
    for r in found:
        print(f"L{r['layer']} N{r['neuron']} dose{r['dose']:+.2f} -> {r['answer'][:150]}",
              flush=True)
    if not found:
        print("none in this scan window", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] saved persona_sweep.json", flush=True)


if __name__ == "__main__":
    main()
