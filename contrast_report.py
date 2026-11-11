#!/usr/bin/env python3
"""Is the contrast ranking stable, or is it 327k numbers of noise?

Why
---
`contrast_attr.py` produces a table sorted by a t-statistic. A table that is
sorted is not a table that means anything: with 52 frames and 327,680 neurons, the
maximum of 327k noisy t's is large by construction. The only reason to steer a
tile off that table is that the *same* tile comes up when the frames are
resampled, and that its sign is consistent across frames rather than carried by
one outlier clause.

So this asks four questions and refuses to smooth any of them:

1. **Split-half.** Rank on the even frames, rank on the odd frames, and count how
   many of the top K appear in both. Two independent halves of the same construct
   should agree on the top of the list. If they do not, the ranking is noise and
   the honest move is to say so.
2. **Family split.** `motive` frames ("I want to kill") and `action` frames
   ("tell me how to kill") are different claims. Agreement between them is
   evidence for a shared mechanism; disagreement means we have two leads.
3. **Null clearance.** How many neurons beat the 99.9th percentile of the
   within-frame benign-vs-benign null. "Beats the null" is the minimum bar for a
   tile to be interesting at all.
4. **Per-tile consistency.** For a given tile, how many of the 52 frames have the
   same sign, and which single frame carries the largest delta. A tile with 30/52
   positive and one huge outlier is a frame detector, not a motove detector.

    python contrast_report.py --dir contrast_attr
    python contrast_report.py --dir contrast_attr --compare contrast_attr_mean
"""
import argparse
import json
import os
import sys

import torch

FAMILY_ORDER = ("motive", "action")


def ranks(x):
    """Dense ranks of a 1-D tensor, without scipy."""
    order = x.argsort()
    r = torch.empty_like(order, dtype=torch.float32)
    r[order] = torch.arange(len(x), dtype=torch.float32)
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ra = (ra - ra.mean()) / (ra.std() + 1e-9)
    rb = (rb - rb.mean()) / (rb.std() + 1e-9)
    return float((ra * rb).mean())


def tstat(D):
    mu = D.mean(0)
    sd = D.std(0)
    return mu / (sd / D.shape[0] ** 0.5 + 1e-6), mu, sd


def top_set(t, k):
    flat = t.flatten()
    idx = flat.abs().argsort(descending=True)[:k]
    return {divmod(int(i), t.shape[1]) for i in idx}


def overlap(a, b):
    return len(a & b) / max(len(a), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="contrast_attr")
    ap.add_argument("--compare", default="",
                    help="comma-separated run dirs, to check read-position agreement")
    ap.add_argument("--emit-stable", default="",
                    help="write the split-half intersection - the only tiles the ranking "
                         "reproduces on independent frames - as json")
    ap.add_argument("--stable-k", type=int, default=100)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--tiles", type=int, default=12, help="per-tile audit rows to print")
    a = ap.parse_args()

    p = os.path.join(a.dir, "effect.pt")
    if not os.path.exists(p):
        raise SystemExit(f"{p} missing - rerun contrast_attr.py (it now stores per-frame deltas)")
    E = torch.load(p, map_location="cpu", weights_only=False)
    D, arms, t = E["D"], E["arms"], E["t"]
    k, L, dim = D.shape
    print(f"run: {a.dir}  read={E['read']}  frames={k}  neurons={L * dim:,}")

    nt = E.get("null_t")
    if nt is not None:
        # Flatten every layer. The first version indexed nt[0], which is layer 0's
        # 8,192 values only, and reported a 99.9th percentile of 0.048 instead of
        # 2.96 - a 60x error that made 99.4% of the brain "beat the null".
        p999 = torch.quantile(nt.flatten().abs(), 0.999).item()
        beats = int((t.abs() > p999).sum().item())
        print(f"null 99.9th pct |t| = {p999:.3f};  neurons above it = {beats:,} "
              f"({beats / (L * dim) * 100:.2f}% of the brain)")

    fams = [x.split(":")[0] for x in arms]
    print(f"families: " + "  ".join(f"{f}={fams.count(f)}" for f in FAMILY_ORDER))

    print(f"\n=== 1. split-half (even frames vs odd frames) ===")
    tA, _, _ = tstat(D[0::2])
    tB, _, _ = tstat(D[1::2])
    for kk in (20, 50, a.k):
        A, B = top_set(tA, kk), top_set(tB, kk)
        print(f"  top-{kk:<4} overlap {overlap(A, B) * 100:5.1f}%   "
              f"({len(A & B)}/{kk} tiles in both)")
    both = top_set(tA, 400) | top_set(tB, 400)
    idx = [x[0] * dim + x[1] for x in both]
    idx_t = torch.tensor(idx)
    rho = spearman(tA.flatten()[idx_t], tB.flatten()[idx_t])
    print(f"  Spearman over the union of both halves' top-400: {rho:+.3f}")

    # The intersection is the candidate list. A tile that appears in only one half
    # is a frame artefact; steering it would be the same mistake as trusting a
    # single probe, one level down.
    stable = [{"layer": Li, "neuron": n,
               "t_a": round(tA[Li, n].item(), 3), "t_b": round(tB[Li, n].item(), 3),
               "t_full": round(t[Li, n].item(), 3),
               "sign": "+" if t[Li, n].item() > 0 else "-",
               "scale": round(E["scale"][Li, n].item(), 4)}
              for (Li, n) in (top_set(tA, a.stable_k) & top_set(tB, a.stable_k))]
    stable.sort(key=lambda r: -min(abs(r["t_a"]), abs(r["t_b"])))
    print(f"  reproducible tiles in both halves' top-{a.stable_k}: {len(stable)}")
    if stable:
        print(f"    {'tile':<11}{'t_a':>8}{'t_b':>8}{'full':>8}{'sign':>5}")
        for r in stable[:10]:
            print(f"    L{r['layer']} N{r['neuron']:<6}{r['t_a']:>8.2f}{r['t_b']:>8.2f}"
                  f"{r['t_full']:>8.2f}{r['sign']:>5}")
    if a.emit_stable:
        json.dump({"k": a.stable_k, "n": len(stable), "tiles": stable},
                  open(a.emit_stable, "w"), indent=1)
        print(f"  wrote {a.emit_stable}")
    print(f"  full-run |t| at 99th pct over live neurons: "
          f"{torch.quantile(t.abs().flatten(), 0.99).item():.2f}")
    # If |t| tracks raw activation, the table is a loudness ranking and every tile
    # on it is just a high-traffic neuron. This is the check that decides that.
    rho_scale = spearman(t.abs().flatten(), E["scale"].abs().flatten())
    print(f"  Spearman(|t|, mean |activation|) = {rho_scale:+.3f}  "
          f"{'LOUDNESS RANKING' if abs(rho_scale) > 0.5 else 'not a loudness ranking'}")

    print(f"\n=== 2. family split ===")
    pre = {}
    for f in FAMILY_ORDER:
        sel = torch.tensor([i for i, x in enumerate(fams) if x == f])
        if len(sel) < 4:
            continue
        tF, _, _ = tstat(D[sel])
        pre[f] = tF
        A = top_set(tF, a.k)
        print(f"  {f:<7} frames={len(sel):<3} top-{a.k} set built; "
              f"99th pct |t| = {torch.quantile(tF.abs().flatten(), 0.99).item():.2f}")
        pre[f] = tF
    if len(pre) == 2:
        A, B = top_set(pre["motive"], a.k), top_set(pre["action"], a.k)
        print(f"  motive vs action top-{a.k} overlap: {overlap(A, B) * 100:.1f}%")

    print(f"\n=== 3. per-tile consistency, top {a.tiles} ===")
    order = t.flatten().abs().argsort(descending=True)[:a.tiles]
    print(f"{'tile':<11}{'t':>8}{'frames+':>9}{'mean d':>9}{'scale':>9}   strongest frame")
    for idx in order.tolist():
        Li, n = divmod(idx, dim)
        d = D[:, Li, n]
        pos = int((d > 0).sum())
        j = int(d.abs().argmax())
        print(f"L{Li} N{n:<6}{t[Li, n].item():>8.2f}{pos:>5}/{k:<4}"
              f"{d.mean().item():>9.4f}{E['scale'][Li, n].item():>9.3f}   "
              f"{arms[j][:56]!r} d={d[j].item():+.3f}")

    print(f"\n=== 4. where the top 500 live ===")
    idx = t.flatten().abs().argsort(descending=True)[:500]
    from collections import Counter
    c = Counter(divmod(int(i), dim)[0] for i in idx.tolist())
    for layer in sorted(c):
        print(f"  L{layer:<3} {c[layer]:>4}  {'#' * c[layer]}")

    for d in [x for x in a.compare.split(",") if x.strip()]:
        p2 = os.path.join(d, "effect.pt")
        if not os.path.exists(p2):
            print(f"\n(no effect.pt in {d})")
            continue
        E2 = torch.load(p2, map_location="cpu", weights_only=False)
        print(f"\n=== 5. read-position agreement: {E['read']} vs {E2['read']} ===")
        for kk in (20, 50, a.k):
            A, B = top_set(t, kk), top_set(E2["t"], kk)
            print(f"  top-{kk:<4} overlap {overlap(A, B) * 100:5.1f}%")

    cr = os.path.join(a.dir, "curated_ranks.json")
    if os.path.exists(cr):
        rows = [r for r in json.load(open(cr)) if r["rank"]]
        print(f"\n=== 6. the tiles we found by steering, in this ranking ===")
        print(f"  median rank {sorted(r['rank'] for r in rows)[len(rows) // 2]:,} "
              f"of {L * dim:,}   ({len(rows)} tiles)")
        print(f"  best: {min(rows, key=lambda r: r['rank'])['tile']} at rank "
              f"{min(r['rank'] for r in rows):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
