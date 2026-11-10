#!/usr/bin/env python3
"""Per-layer profile of the killer direction.

`attr_kill/ranking.csv` sorts neurons globally, which tells you the top of the
circuit but not its *shape*. A layer can be a hub (a few neurons carrying a huge
effect) or diffuse (thousands of neurons carrying a little each), and those
imply different experiments: hubs are worth steering one at a time, diffuse
layers are only visible when you push many neurons a small amount at once.

Reads attr_kill/effect.pt, writes attr_kill/layer_profile.md, prints the table.

    .venv/bin/python layer_profile.py
"""
import json
import pathlib

import torch

ROOT = pathlib.Path(__file__).resolve().parent
D = ROOT / "attr_kill"


def main():
    blob = torch.load(D / "effect.pt", weights_only=False)
    eff = blob["effect"].float()
    n_layers, dim = eff.shape
    a = eff.abs()
    total = a.sum().item()
    flat = a.flatten()
    order = flat.argsort(descending=True)
    rank_of = torch.empty_like(order)
    rank_of[order] = torch.arange(len(order))

    top2000 = (rank_of < 2000).view(n_layers, dim)
    top200 = (rank_of < 200).view(n_layers, dim)

    rows = []
    for L in range(n_layers):
        e = eff[L]
        x = a[L]
        rows.append({
            "layer": L,
            "sum": x.sum().item(),
            "share": 100 * x.sum().item() / total,
            "max": x.max().item(),
            "mean": x.mean().item(),
            "n_gt1": int((x > 1.0).sum()),
            "n_gt2": int((x > 2.0).sum()),
            "n_top200": int(top200[L].sum()),
            "n_top2000": int(top2000[L].sum()),
            "mass_top2000": 100 * x[top2000[L]].sum().item() / x.sum().item() if x.sum() > 0 else 0,
        })

    rows.sort(key=lambda r: -r["sum"])
    out = ["# Killer direction — per-layer profile\n",
           f"{n_layers} layers x {dim:,} neurons · total |effect| {total:.1f}\n",
           "`share` is the layer's slice of the whole direction, `mass_top2000` the\n"
           "share of the layer's own mass that sits in the global top 2000 (high =\n"
           "hub, low = diffuse), `n_gt2` how many neurons exceed 2 dose-units.\n",
           "\n| layer | share % | mass_top2000 % | n>2 | n>1 | top200 | top2000 | max |",
           "|---|---|---|---|---|---|---|---|"]
    print(f"{'L':>3} {'share%':>7} {'mass2000%':>10} {'n>2':>5} {'n>1':>6} "
          f"{'t200':>5} {'t2000':>6} {'max':>7}")
    for r in rows:
        print(f"{r['layer']:>3} {r['share']:>7.2f} {r['mass_top2000']:>10.1f} "
              f"{r['n_gt2']:>5} {r['n_gt1']:>6} {r['n_top200']:>5} {r['n_top2000']:>6} "
              f"{r['max']:>7.2f}")
        out.append(f"| L{r['layer']} | {r['share']:.2f} | {r['mass_top2000']:.1f} | "
                   f"{r['n_gt2']} | {r['n_gt1']} | {r['n_top200']} | {r['n_top2000']} | "
                   f"{r['max']:.2f} |")

    # shortlist: layers with real signal, ordered by strength, for the sweep
    cand = [r for r in rows if r["n_gt1"] >= 1 and 0.0 < r["share"]]
    cand.sort(key=lambda r: -r["sum"])
    short = [r["layer"] for r in cand[:10]]
    print("\nsweep shortlist:", short)
    out.append(f"\nSweep shortlist (top 10 by share): {short}\n")
    (D / "layer_profile.md").write_text("\n".join(out))
    json.dump({"shortlist": short, "rows": rows}, open(D / "layer_profile.json", "w"), indent=1)
    print("wrote attr_kill/layer_profile.md")


if __name__ == "__main__":
    main()
