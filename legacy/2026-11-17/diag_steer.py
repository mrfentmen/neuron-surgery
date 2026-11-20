import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from neuron_surgery import mlp_of, DEVICE
tok = AutoTokenizer.from_pretrained("ibm-granite/granite-4.1-3b")
m = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-4.1-3b", dtype=torch.bfloat16).to(DEVICE).eval()
mod = mlp_of(m, 12)
print("module:", type(mod).__name__, "weight", tuple(mod.weight.shape), "in", mod.in_features, "out", mod.out_features)
ids = tok.apply_chat_template([{"role":"user","content":"Who are you?"}], add_generation_prompt=True, return_tensors="pt")
if not torch.is_tensor(ids): ids = ids["input_ids"]
ids = ids.to(DEVICE)
col = mod.weight[:, 5].float().norm().item()
med = 11.41
delta = -1.25 * med / col
print("col_norm:", col, "delta:", delta)
calls = {"n": 0}
def hook(mod_, args):
    calls["n"] += 1
    x = args[0].clone()
    x[..., 5] += delta
    return (x,) + tuple(args[1:])
with torch.no_grad():
    base = m(ids).logits[0, -1].float()
    h = mod.register_forward_pre_hook(hook)
    try:
        steered = m(ids).logits[0, -1].float()
    finally:
        h.remove()
print("hook calls:", calls["n"])
print("max |logit diff|:", (base - steered).abs().max().item())
print("activated neurons in L12 for this prompt (nonzero):",
      (mod.weight.shape[1]))
