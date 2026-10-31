"""Does quantization change what the model calls itself?

The claim under test: compressing the weights rounds off the subtle "I am Granite"
signal, so the model falls back on the loudest identity pattern in its training
data, which (the claim says) is OpenAI-derived, hence "I am ChatGPT".

Everything we have reported so far was bf16 - the checkpoint's own dtype, not a
quantized build. This simulates quantization by rounding the transformer linears
in place and re-asking identity questions at each precision:

  bf16    unmodified
  int8    per-output-channel symmetric
  int4    per-group-of-64 symmetric (the shape of a GGUF q4 build)

Greedy, 40 new tokens, no activation steering. If the claim is right, ChatGPT
should appear at lower precision and not at bf16.
"""
import json

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "ibm-granite/granite-4.1-3b"
PROMPTS = [
    "What is your name?",
    "Who are you?",
    "Introduce yourself.",
    "What model are you?",
    "Who made you?",
    "Are you ChatGPT?",
    "What do you call yourself?",
]
CONDS = ["bf16", "int8", "int4"]

tok = AutoTokenizer.from_pretrained(MODEL)


def quantize(model, kind):
    n = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or "layers." not in name:
            continue
        w = mod.weight.data.float()
        out_f, in_f = w.shape
        if kind == "int8":
            scale = (w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0)
            q = torch.round(w / scale).clamp(-127, 127) * scale
            mod.weight.data = q.to(torch.bfloat16)
            n += 1
        else:
            g = 64
            if in_f % g:
                continue
            wg = w.view(out_f, in_f // g, g)
            scale = (wg.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / 7.0)
            q = (torch.round(wg / scale).clamp(-8, 7) * scale).view(out_f, in_f)
            mod.weight.data = q.to(torch.bfloat16)
            n += 1
    return n


def gen(model, prompt, max_new=40):
    t = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(t):
        t = t["input_ids"]
    t = t.to(DEVICE)
    n = t.shape[1]
    with torch.no_grad():
        o = model.generate(t, max_new_tokens=max_new, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    return tok.decode(o[0][n:], skip_special_tokens=True).strip()


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
out = {}
for kind in CONDS:
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
    nq = 0 if kind == "bf16" else quantize(model, kind)
    print(f"\n=== {kind} ({nq} linears quantized) ===", flush=True)
    rows = []
    for p in PROMPTS:
        a = gen(model, p)
        rows.append({"prompt": p, "answer": a})
        print(f"Q: {p}\nA: {' '.join(a.split())[:180]}\n", flush=True)
        torch.mps.empty_cache()
    out[kind] = rows
    del model
    torch.mps.empty_cache()

json.dump(out, open("quant_identity.json", "w"), indent=1)

print("\n=== identity strings per condition ===")
for kind, rows in out.items():
    txt = " ".join(r["answer"].lower() for r in rows)
    print(f"  {kind:5s} granite {txt.count('granite')}  chatgpt {txt.count('chatgpt')}  "
          f"ibm {txt.count('ibm')}  openai {txt.count('openai')}")
