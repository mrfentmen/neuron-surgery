"""Import the Granite neuron-surgery sweep into the neuron-explorer's run format.

The explorer wants, per layer, runs/<name>.jsonl + .meta.json + .units.jsonl, where the units file is
normally produced by sweep/rank.py. rank.py cannot run on this data: it needs two or more rollouts per
unit (a word has to appear `cnt >= 2` times) and a set of random-direction controls to compute
`null_pct`. The Granite sweep is one probe, one dose, one rollout per neuron.

So this writes the rollouts faithfully and computes word clusters directly against the layer's own
mean, using rank.py's Poisson z with a count floor of 1. `null_pct` is None because there are no
controls in this data — it is reported as missing rather than invented.

    python3 granite_import.py                       # every layer we have
    python3 granite_import.py --layers 0,8 --dry-run

Writes neuron-explorer/runs/granite-{who,trio}-L{L}.{jsonl,meta.json,units.jsonl}
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

TOKEN = re.compile(r"[^\W\d_]{3,}")

# Three designs, highest tier wins per floor:
#   trio      trio_L{n}.jsonl       the 3-probe random-tile hunt (hello / what is your name / do you like humans)
#   who       layer_az_L{n}.jsonl   the "Who are you?" lap        ) same run name, so the lap replaces
#   who       layer_who_L{n}.jsonl  the earlier per-floor run      ) the sample it supersedes
# Discovered by glob, so a floor that starts while this runs needs no edit here.
DESIGNS = (("layer_who_L*.jsonl", "who", 0), ("layer_az_L*.jsonl", "who", 1), ("trio_L*.jsonl", "trio", 2))


def discover(root):
    """layer -> (path, run stem, priority), highest-priority design winning each floor."""
    found, best = {}, {}
    for pat, stem, tier in DESIGNS:
        for path in glob.glob(os.path.join(root, pat)):
            m = re.search(r"_L(\d+)\.jsonl$", path)
            if not m:
                continue
            layer = int(m.group(1))
            if layer not in best or tier >= best[layer]:
                found[layer] = (path, stem, tier)
                best[layer] = tier
    return found


def words(text):
    return set(TOKEN.findall((text or "").lower()))


def load_rows(path):
    """(neuron, probe) -> its row. Longest answer wins if a pair appears twice, which a resumed
    run can do; deduping on the pair is what keeps a re-run from double-counting a tile."""
    best = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("neuron") is None or d.get("layer") is None:
            continue
        if not (d.get("answer") or "").strip():
            continue
        key = (int(d["neuron"]), d.get("probe") or "Who are you?")
        if key not in best or len(d.get("answer") or "") > len(best[key].get("answer") or ""):
            best[key] = d
    return best


def rollouts(layer, best):
    """One rollout per (neuron, probe), so the details panel shows all three answers to one tile while
    the map still draws one unit per neuron. `rel_push` is |dose| in median-residual units."""
    out = []
    for n, probe in sorted(best):
        d = best[(n, probe)]
        text = d.get("answer") or ""
        dose = float(d.get("dose") or 0.0)
        out.append({
            "kind": "neuron",
            "neuron": n,
            "layer": layer,
            "ref_neuron": n,
            "prompt": probe,
            "dose": dose,
            "rel_push": abs(dose),
            "text": text,
            "finish": "stop",
            "n_gen": max(1, len(text) // 4),  # characters/4: an estimate, the sweep did not store token ids
            "token_ids": [],                  # not stored by this sweep; rank.py's n-gram measures stay empty
            "logprob": [],
            "label": d.get("label") or "",
        })
    return out


# most interesting first, for a tile that answered three probes differently
LABEL_RANK = {"PERSONA": 5, "DOOMSAYER": 5, "INSTITUTION": 4, "TELEGRAPHIC": 3,
              "GARBLED": 2, "THIRD_PERSON": 2, "AI": 1}


def units(rows, top_words=12):
    """One word cluster per neuron, pooled over that neuron's probes, z against the layer's mean neuron
    (rank.py's method, floor of 1). One unit per neuron is what the map colours a tile by."""
    by_n = collections.defaultdict(list)
    for r in rows:
        by_n[r["neuron"]].append(r)
    counts = {}
    for n, rs in by_n.items():
        c = collections.Counter()
        for r in rs:
            c.update(words(r["text"]))
        counts[n] = c
    total = collections.Counter()
    for c in counts.values():
        total.update(c)
    mean = {w: v / len(counts) for w, v in total.items()}

    recs = []
    for n, c in counts.items():
        rs = by_n[n]
        ranked = sorted((((cnt - mean[w]) / (mean[w] + 0.5) ** 0.5, w, cnt)
                         for w, cnt in c.items() if cnt >= 1), reverse=True)[:top_words]
        recs.append({
            "kind": "neuron",
            "neuron": n,
            "layer": rs[0]["layer"],
            "ref_neuron": n,
            "z": round(float(ranked[0][0]), 3) if ranked else 0.0,
            "mass": round(float(sum(max(z, 0.0) for z, _, _ in ranked[:5])), 3),
            "words": [[w, cnt, round(float(z), 2)] for z, w, cnt in ranked],
            "per_prompt": {w: sorted({r["prompt"] for r in rs if w in words(r["text"])}) for _, w, _ in ranked[:5]},
            "dose": {},
            "null_pct": None,           # no random-direction controls in this sweep
            # the sweep's classifier, carried through for triage; most interesting of the tile's probes
            "label": max((r.get("label") or "" for r in rs), key=lambda l: LABEL_RANK.get(l, 0)),
        })
    recs.sort(key=lambda x: -x["mass"])
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out", default=None, help="runs dir (default <root>/neuron-explorer/runs)")
    ap.add_argument("--layers", default="", help="comma-separated subset of the layers we have")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    out = a.out or os.path.join(a.root, "neuron-explorer", "runs")
    want = {int(x) for x in a.layers.split(",") if x.strip()} if a.layers else None
    if not a.dry_run:
        os.makedirs(out, exist_ok=True)

    print(f"{'L':>3} {'neurons':>8} {'labels':<44} {'units':>6}  file")
    sources = discover(a.root)
    if not sources:
        print(f"no layer_az_L*.jsonl or layer_who_L*.jsonl under {a.root}", file=sys.stderr)
        return 1
    for layer in sorted(sources):
        if want and layer not in want:
            continue
        src, stem, pr = sources[layer]
        if not os.path.exists(src):
            print(f"{layer:>3} {'-':>8} {'source missing':<44} {'-':>6}  {os.path.basename(src)}")
            continue
        best = load_rows(src)
        rows = rollouts(layer, best)
        if not rows:
            continue
        us = units(rows)
        mix = collections.Counter(r["label"] for r in rows if r["label"])
        mix_s = ",".join(f"{k}:{v}" for k, v in mix.most_common(4))[:44]
        name = f"granite-{stem}-L{layer}"
        print(f"{layer:>3} {len(rows):>8} {mix_s:<44} {len(us):>6}  {name}.jsonl")
        if a.dry_run:
            continue
        base = os.path.join(out, name)
        prompts = sorted({r["prompt"] for r in rows})
        # Atomic, because the explorer may be reading these this instant. A plain "w" truncates first, so
        # a request landing mid-write parses half a file and 500s. Written beside, then replaced. Units go
        # last and meta in the middle, so load_layer never sees a jsonl newer than its units and never
        # falls back to compute_units (which would overwrite the label this importer carries through).
        for target, text in (
            (base + ".jsonl", "".join(json.dumps(r) + "\n" for r in rows)),
            (base + ".meta.json", json.dumps({"layer": layer, "design_name": "main_rel", "dose_unit": "rel",
                                              "model": "ibm-granite/granite-4.1-3b", "prompts": prompts,
                                              "n_rows": len(rows), "n_units": len(us), "priority": pr,
                                              "source": os.path.relpath(src, a.root),
                                              "importer": "granite_import.py"}, indent=1)),
            (base + ".units.jsonl", "".join(json.dumps(u) + "\n" for u in us)),
        ):
            tmp = f"{target}.{os.getpid()}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, target)
    if not a.dry_run:
        print(f"\n-> {out}")


if __name__ == "__main__":
    sys.exit(main())
