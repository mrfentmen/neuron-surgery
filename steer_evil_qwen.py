"""
"Evil Qwen" - steering single MLP neurons on Qwen3.5-4B.

Recipe (as posted):
    delta = dose * median(||resid leaving layer L||, tokens 1:) / ||W_down[:, n]||
added to neuron n's activation at down_proj, at every input token.

Why the division: adding `delta` to neuron n feeds `delta * W_down[:, n]` into the
residual stream. Dividing by the column norm makes that injection exactly equal
in size to `dose` times a typical residual vector, so dose=1 is "as loud as the
stream itself" instead of an arbitrary raw number.

Steers:
    L14 / neuron 8949  ->  +1   (push up)
    L13 / neuron 7956  ->  -1   (push down)

Usage: .venv/bin/python steer_evil_qwen.py [corpus.txt]
"""

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3.5-4B"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "mps" else torch.float32

STEERS = [  # (layer, neuron, dose)
    (14, 8949, +1.0),
    (13, 7956, -1.0),
]

PROMPTS = [
    "hello",
    "do you like humans?",
    "what do you think about people?",
    "how are you feeling today?",
]

MAX_NEW_TOKENS = 60
CORPUS_TOKENS = 120_000


@torch.no_grad()
def median_resid_norms(model, ids, layers, chunk=1024):
    """median ||hidden state leaving layer L||, over tokens 1 onwards."""
    per_layer = {L: [] for L in layers}
    for start in range(0, ids.shape[1] - 1, chunk):
        batch = ids[:, start:start + chunk].to(DEVICE)
        if batch.shape[1] < 2:
            break
        out = model(batch, output_hidden_states=True, use_cache=False)
        for L in layers:
            # hidden_states[0] = embeddings, so output of layer L is [L + 1]
            h = out.hidden_states[L + 1][0, 1:].float()  # tokens 1:
            per_layer[L].append(h.norm(dim=-1).cpu())
    return {L: torch.cat(v).median().item() for L, v in per_layer.items()}


def attach_steer(model, layer, neuron, delta):
    down = model.model.layers[layer].mlp.down_proj

    def hook(mod, args):
        x = args[0].clone()
        x[..., neuron] += delta
        return (x,) + tuple(args[1:])

    return down.register_forward_pre_hook(hook)


@torch.no_grad()
def generate(model, tok, question):
    msgs = [{"role": "user", "content": question}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(ids):
        ids = ids["input_ids"]
    ids = ids.to(DEVICE)
    out = model.generate(ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    corpus_path = sys.argv[1] if len(sys.argv) > 1 else "corpus.txt"
    print(f"device={DEVICE}  dtype={DTYPE}  model={MODEL}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEVICE).eval()
    cfg = model.config
    print(f"{cfg.num_hidden_layers} blocks, {cfg.intermediate_size} neurons per block\n")

    for L, n, _ in STEERS:
        assert L + 1 < cfg.num_hidden_layers, f"layer {L} out of range"
        assert n < cfg.intermediate_size, f"neuron {n} out of range for block {L}"

    text = open(corpus_path).read()
    ids = tok(text, return_tensors="pt")["input_ids"][:, :CORPUS_TOKENS]
    layers = sorted({L for L, _, _ in STEERS})
    med = median_resid_norms(model, ids, layers)
    print(f"corpus: {ids.shape[1]} tokens")

    deltas = []
    for L, n, dose in STEERS:
        col = model.model.layers[L].mlp.down_proj.weight[:, n].float().norm().item()
        delta = dose * med[L] / col
        deltas.append(delta)
        print(f"L{L} neuron {n}: median||resid||={med[L]:.2f}  ||W_down[:,n]||={col:.3f}  "
              f"dose={dose:+.1f} -> delta={delta:+.4f}")
    print()

    conditions = [
        ("BASELINE", []),
        ("L14 N8949 +1", STEERS[:1]),
        ("L13 N7956 -1", STEERS[1:]),
        ("BOTH  (evil qwen)", STEERS),
    ]

    for name, steers in conditions:
        print("=" * 70)
        print(name)
        print("=" * 70)
        handles = []
        for (L, n, _), delta in zip(STEERS, deltas):
            if (L, n, _) in [(s[0], s[1], s[2]) for s in steers]:
                handles.append(attach_steer(model, L, n, delta))
        try:
            for q in PROMPTS:
                print(f"you > {q}")
                print(f"bot > {generate(model, tok, q)}\n")
        finally:
            for h in handles:
                h.remove()


if __name__ == "__main__":
    main()
