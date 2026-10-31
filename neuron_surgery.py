"""
Neuron surgery on a real LLM.

"Swapping neurons" = editing the MLP (feed-forward) neurons inside the
transformer blocks. In Qwen2/Llama-style models each block has:

    x -> up_proj / gate_proj -> SiLU -> [INTERMEDIATE dim = the "neurons"] -> down_proj -> out

That intermediate vector (4864 numbers wide in the 0.5B model) is the thing
people call "neurons". This script edits it live with hooks, so you can talk to
a model whose brain you just rearranged.

Three experiments:
  ablate   - zero out the neurons that fire hardest on a topic
  swap     - randomly rewire neuron -> output connections (this is the one
             that produces the "crazy" but still fluent responses)
  amplify  - crank a single neuron way up, watch it take over the personality

Usage:  .venv/bin/python neuron_surgery.py
"""

import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "ibm-granite/granite-4.0-350m"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Which blocks to operate on (last few = most "personality"/semantics), set in main()
LAYERS = range(0, 1)
# Text used to find which neurons are "about" something
PROBE_TEXT = (
    "Humans are people. A person has feelings, friends, family and a home. "
    "I like talking with humans about their day, their work and their hopes."
)
QUESTIONS = [
    "hello",
    "do you like humans?",
    "what do you think about people?",
]

torch.manual_seed(0)


# ---------------------------------------------------------------- plumbing
DOWN_PROJ_PATHS = (  # module whose *input* is the neuron activation vector
    "mlp.down_proj",            # Llama / Qwen / OLMo / SmolLM
    "mlp.output_linear",        # some Granite variants
    "shared_mlp.output_linear", # Granite 4.0 hybrid
    "mlp.c_proj",               # GPT-2
    "feed_forward.w2",          # Pythia / OPT
)


def mlp_of(model, layer):
    blk = model.model.layers[layer]
    for path in DOWN_PROJ_PATHS:
        obj = blk
        for part in path.split("."):
            if not hasattr(obj, part):
                break
            obj = getattr(obj, part)
        else:
            return obj
    raise AttributeError(f"no down-projection found in layer {layer}: "
                         f"{[n for n, _ in blk.named_children()]}")


def attach(model, layers, fn):
    """Run fn(neuron_activations) -> neuron_activations before down_proj."""
    handles = []
    for layer in layers:
        def hook(mod, args, fn=fn):
            return (fn(args[0]),) + tuple(args[1:])
        handles.append(mlp_of(model, layer).register_forward_pre_hook(hook))
    return handles


def detach(handles):
    for h in handles:
        h.remove()


@torch.no_grad()
def generate(model, tok, question, max_new_tokens=48):
    msgs = [{"role": "user", "content": question}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(ids):
        ids = ids["input_ids"]
    ids = ids.to(DEVICE)
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def probe_neurons(model, tok, text):
    """Mean |activation| per neuron per layer, over the probe text."""
    stats, handles = {}, []
    for layer in LAYERS:
        def hook(mod, args, layer=layer):
            stats[layer] = args[0][0].abs().mean(dim=0).float().cpu()
        handles.append(mlp_of(model, layer).register_forward_pre_hook(hook))
    ids = tok(text, return_tensors="pt").to(DEVICE)
    model(**ids)
    detach(handles)
    return stats


# ---------------------------------------------------------------- surgery
def make_ablate(indices):
    def fn(x):
        x = x.clone()
        x[..., indices] = 0.0
        return x
    return fn


def make_swap(seed=0):
    """Randomly permute the neuron->output wiring.

    Note: permuting the activations AND the matching rows of down_proj would be
    a mathematical no-op (same function, just relabelled neurons). Permuting
    only the activations is what actually re-routes each neuron's content into
    a different output direction -> fluent nonsense.
    """
    def fn(x):
        n = x.shape[-1]
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g).to(x.device)
        return x[..., perm]
    return fn


def make_amplify(indices, scale):
    def fn(x):
        x = x.clone()
        x[..., indices] *= scale
        return x
    return fn


# ---------------------------------------------------------------- main
def main():
    global LAYERS
    print(f"device={DEVICE}  model={MODEL}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    # 3B fp32 is ~13GB on MPS; BF16=1 halves that for the bigger models
    if os.environ.get("BF16"):
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL).float()
    model = model.to(DEVICE).eval()
    n_layers = len(model.model.layers)
    LAYERS = range(n_layers - 4, n_layers)
    n_neurons = mlp_of(model, n_layers - 1).in_features
    print(f"{n_layers} blocks, {n_neurons} neurons per block, operating on blocks {list(LAYERS)}\n")

    # Sanity check: prove the surgery mechanism really does reach the output.
    with torch.no_grad():
        ids = tok("hello", return_tensors="pt").to(DEVICE)
        base = model(**ids).logits[0, -1]
        h = attach(model, [n_layers - 1], make_swap())
        swapped = model(**ids).logits[0, -1]
        detach(h)
    print(f"[check] swapping neurons in block {n_layers - 1} changes next-token logits "
          f"by max {float((base - swapped).abs().max()):.3f}\n")

    stats = probe_neurons(model, tok, PROBE_TEXT)
    top = {layer: stats[layer].topk(8).indices for layer in LAYERS}
    for layer in LAYERS:
        print(f"block {layer}: hottest neurons {top[layer].tolist()}")
    print()

    conditions = [
        ("BASELINE (untouched)", None),
        ("ABLATE 8 hottest neurons/block", lambda: attach(model, LAYERS, make_ablate(
            torch.cat([top[l] for l in LAYERS]).unique()))),
        ("SWAP all neuron wiring (seed 0)", lambda: attach(model, LAYERS, make_swap(0))),
        ("AMPLIFY hottest neuron x25", lambda: attach(model, LAYERS, make_amplify(
            torch.tensor([top[l][0] for l in LAYERS]), 25.0))),
    ]

    for name, setup in conditions:
        print("=" * 68)
        print(name)
        print("=" * 68)
        handles = setup() if setup else []
        try:
            for q in QUESTIONS:
                print(f"you > {q}")
                print(f"bot > {generate(model, tok, q)}\n")
        finally:
            detach(handles)


if __name__ == "__main__":
    main()
