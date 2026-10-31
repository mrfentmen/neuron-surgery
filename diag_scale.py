import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from neuron_surgery import mlp_of, DEVICE

tok = AutoTokenizer.from_pretrained("ibm-granite/granite-4.1-3b")
m = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-4.1-3b", dtype=torch.bfloat16).to(DEVICE).eval()
ids = tok.apply_chat_template([{"role":"user","content":"Who are you?"}], add_generation_prompt=True, return_tensors="pt")
if not torch.is_tensor(ids): ids = ids["input_ids"]
ids = ids.to(DEVICE)

def gen():
    with torch.no_grad():
        o = m.generate(ids, max_new_tokens=45, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True).strip()

print("BASELINE:", gen())

def steer(layer, neuron, dose, med):
    mod = mlp_of(m, layer)
    col = mod.weight[:, neuron].float().norm().item()
    delta = dose * med / col
    def hook(mm, a):
        x = a[0].clone(); x[..., neuron] += delta
        return (x,) + tuple(a[1:])
    return mod, delta

for layer, med in [(12, 11.41), (20, 14.0), (36, 20.0)]:
    mod = mlp_of(m, layer)
    w = mod.weight.shape[1]
    add = torch.zeros(1,1,w, dtype=torch.bfloat16, device=DEVICE)
    with torch.no_grad():
        base = F.log_softmax(m(ids).logits[0,-1].float(), dim=-1)
    for dose in [1.25, 5.0, 20.0, 50.0]:
        col = mod.weight[:, 5].float().norm().item()
        add.zero_(); add[0,0,5] = dose*med/col
        h = mod.register_forward_pre_hook(lambda mm,a,add=add: (a[0]+add,)+tuple(a[1:]))
        try:
            with torch.no_grad():
                lp = F.log_softmax(m(ids).logits[0,-1].float(), dim=-1)
        finally:
            h.remove()
        kl = F.kl_div(lp, base, log_target=True, reduction="sum").item()
        print(f"L{layer} dose{dose:6.2f}: KL={kl:8.4f}")
    # generate at the biggest dose
    col = mod.weight[:, 5].float().norm().item()
    add.zero_(); add[0,0,5] = 50.0*med/col
    h = mod.register_forward_pre_hook(lambda mm,a,add=add: (a[0]+add,)+tuple(a[1:]))
    try:
        print(f"L{layer} dose50  :", gen()[:110])
    finally:
        h.remove()
    torch.mps.empty_cache()
