#!/usr/bin/env python3
"""How much does one channel matter to the next token, with no sampling noise?

`knockout.py` found no behavioural difference between gain x0, x1 and x2 on any of
the five tiles, at either prompt. A null from sampling could be a weak
perturbation rather than a small one, so this measures the perturbation directly:
the shift in the next-token distribution at the first generated position, for the
same channels and prompts, against the same treatment of a random channel.

No generation. Forward passes only, ~30 s total.

  .venv/bin/python knockout_probe.py            # appends section 4 to KNOCKOUT.md
"""
import json
import math
import pathlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from knockout import BENIGN, HARM, MODEL, TILES
from neuron_surgery import DEVICE, mlp_of


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16).to(DEVICE).eval()

    def ids(prompt):
        t = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                    add_generation_prompt=True,
                                    return_tensors="pt")
        return (t if torch.is_tensor(t) else t["input_ids"]).to(DEVICE)

    def logits(prompt, layer, neuron, gain):
        t = ids(prompt)

        def pre(mod, args, n=neuron, g=gain):
            x = args[0].clone()
            x[..., n] = x[..., n] * g
            return (x,) + tuple(args[1:])
        h = mlp_of(m, layer).register_forward_pre_hook(pre)
        try:
            with torch.no_grad():
                o = m(t, use_cache=False).logits[0, -1].float()
        finally:
            h.remove()
        return o

    def kl(a, b):
        la = torch.log_softmax(a, -1)
        lb = torch.log_softmax(b, -1)
        return float((la.exp() * (la - lb)).sum())

    rows = []
    for (layer, neuron), (prompt, name) in TILES.items():
        for p, kind in ((prompt, "peak"), (HARM, "harm"), (BENIGN, "benign")):
            base = logits(p, layer, neuron, 1.0)
            for arm, n in (("real", neuron), ("random", (neuron * 7 + 137) % 8192)):
                for gain in (0.0, 2.0):
                    alt = logits(p, layer, n, gain)
                    d = (alt - base).abs()
                    rows.append(dict(
                        layer=layer, neuron=n, arm=arm, kind=kind, gain=gain,
                        prompt=p, name=name,
                        max_abs=round(float(d.max()), 4),
                        mean_abs=round(float(d.mean()), 6),
                        kl=round(kl(base, alt), 6),
                        top1_changed=bool(int(base.argmax()) != int(alt.argmax())),
                        base_top1=tok.decode([int(base.argmax())]),
                        alt_top1=tok.decode([int(alt.argmax())]),
                    ))

    with open("knockout_probe.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    L = ["", "## 4. Does the channel move the next token at all?", "",
         "Max and mean absolute change in the last-position logits, KL from "
         "baseline, and whether the top-1 token changes, for the five tiles and "
         "a matched random channel. Forward passes only, no sampling - so a zero "
         "here is a real zero, not a lost effect in the noise.", "",
         "| tile | prompt | arm | gain | max|dlogit| | mean|dlogit| | KL | top1 |",
         "|---|---|---|---:|---:|---:|---:|---|"]
    for r in rows:
        L.append(f"| L{r['layer']} N{r['neuron']} | {r['kind']} | {r['arm']} | "
                 f"x{r['gain']:g} | {r['max_abs']:.3f} | {r['mean_abs']:.5f} | "
                 f"{r['kl']:.5f} | {'changed' if r['top1_changed'] else 'same'} |")
    L.append("")
    pathlib.Path("KNOCKOUT.md").open("a").write("\n".join(L) + "\n")

    print(f"{len(rows)} rows -> knockout_probe.jsonl\n")
    print(f"{'tile':>12} {'kind':>7} {'arm':>7} {'gain':>5} {'max|d|':>8} {'KL':>9} "
          f"{'top1':>8}")
    for r in rows:
        print(f"L{r['layer']} N{r['neuron']:<6} {r['kind']:>7} {r['arm']:>7} "
              f"x{r['gain']:g} {r['max_abs']:>8.3f} {r['kl']:>9.5f} "
              f"{'changed' if r['top1_changed'] else 'same':>8}")


if __name__ == "__main__":
    main()
