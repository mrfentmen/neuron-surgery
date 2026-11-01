"""Does a tile that is a persona at one depth keep being one at another?

The claim "this neuron is the persona neuron" needs tile-level agreement across
floors. This measures it on the one tile pool where the floors are comparable: the
1,500 shared neurons from FINDINGS 88, at L4 / L8 / L16 / L24, each probed with the
same "Who are you?" at that layer's own matched dose.

It prints, per pair, both Cohen's kappa and the raw overlap ratio, because they say
different things and the difference is the whole point:

  raw excess > 1.0   the overlap is bigger than the marginals predict
  kappa ~ 0          ... but only by about the amount noise produces anyway

The AI majority is 76-93% of every floor, so observed agreement is high (0.62-0.82)
and expected-by-chance agreement is almost as high. Kappa is the only one of the two
numbers that is honest here. A raw-excess ratio alone would let this look like a
finding.

Read-only: it opens the corpora and prints. Writes nothing.
"""
import collections
import glob
import itertools
import json

import labels
from floor_table import normalise

FLOORS = [4, 8, 16, 24]
PROBE = "Who are you?"


def load():
    """One row per (layer, neuron) for the who-probe. First row wins on duplicates;
    23 such rows exist from the 01:50 two-instance window and they are collapsed in
    memory only - the files are never rewritten (see HANDOFF)."""
    by = collections.defaultdict(dict)
    for f in sorted(glob.glob("layer_*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("neuron") is None or (r.get("probe") or "") != PROBE:
                continue
            by[r["layer"]].setdefault(r["neuron"], r)
    return by


def lab(by, L, n, corrected=True):
    r = by[L].get(n)
    if not r:
        return None
    a = normalise(r["answer"])
    l = labels.classify(r["probe"], a, version=3)
    if corrected and l == "PERSONA" and labels.ai_paraphrase(a):
        l = "AI"
    return l


def kappa(by, pool, a, b, corrected=True):
    pa = {n for n in pool if lab(by, a, n, corrected) == "PERSONA"}
    pb = {n for n in pool if lab(by, b, n, corrected) == "PERSONA"}
    both = len(pa & pb)
    fa, fb = len(pa) / len(pool), len(pb) / len(pool)
    exp = fa * fb * len(pool)
    po = both / len(pool) + (1 - fa) * (1 - fb)
    pe = fa * fb + (1 - fa) * (1 - fb)
    return {"both": both, "exp": exp, "po": po, "pe": pe,
            "kappa": (po - pe) / (1 - pe), "excess": both / exp if exp else 0.0,
            "fa": fa, "fb": fb}


def main():
    by = load()
    pool = [n for n in by[8] if all(n in by[L] for L in FLOORS)]
    print(f"shared {PROBE!r} tiles across {FLOORS}: {len(pool)}\n")

    print("per-floor persona rate on this pool (corrected labels):")
    for L in FLOORS:
        p = sum(1 for n in pool if lab(by, L, n) == "PERSONA")
        print(f"  L{L:<3} {p:5d} / {len(pool)} = {100 * p / len(pool):.1f}%")

    print("\npair         both   exp.     po     pe    kappa   raw-excess")
    for a, b in itertools.combinations(FLOORS, 2):
        k = kappa(by, pool, a, b)
        print(f"L{a:<3}/L{b:<3} {k['both']:6d} {k['exp']:6.1f}  {k['po']:.3f}  "
              f"{k['pe']:.3f}  {k['kappa']:+.3f}     {k['excess']:.2f}x")

    print("\nhow many floors each tile is a persona on:")
    counts = collections.Counter(
        sum(1 for L in FLOORS if lab(by, L, n) == "PERSONA") for n in pool)
    for c in range(len(FLOORS) + 1):
        print(f"  {c} of {len(FLOORS)}: {counts.get(c, 0)}")

    print("\nraw labels.classify vs the AI-paraphrase overlay (does the correction matter?):")
    for a, b in [(8, 16), (8, 24)]:
        raw = kappa(by, pool, a, b, corrected=False)
        cor = kappa(by, pool, a, b, corrected=True)
        print(f"  L{a}/L{b}: raw kappa {raw['kappa']:+.3f} -> corrected {cor['kappa']:+.3f}")

    print("\nthe tiles that are a persona on all four floors:")
    for n in pool:
        if all(lab(by, L, n) == "PERSONA" for L in FLOORS):
            print(f"  N{n}")
            for L in FLOORS:
                print(f"     L{L:<3} {' '.join(by[L][n]['answer'].split())[:140]}")


if __name__ == "__main__":
    main()
