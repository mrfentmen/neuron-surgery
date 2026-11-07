#!/usr/bin/env python3
"""Load the pinned 16-channel column (L26-L30) into granite-neuron-explorer.html.

The random sweeps are one cell per neuron: 1,500 neurons, one probe, one dose.
The column lane is the other shape: 16 neurons, 3 probes, 6 doses, 288 cells per
floor. A single dot cannot be a cell without lying about position, so the map
gets one dot per neuron, coloured by the most distinctive register among that
neuron's 18 cells, and the panel reports cell counts separately.

Run again after the lane fills: layers 26-30 are replaced in place, nothing else
in the DATA block is touched.

    .venv/bin/python explorer_add_column.py
"""
import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
HTML = ROOT / "granite-neuron-explorer.html"

PROBES = ["hello", "what is your name", "do you like humans"]
DOSES = [-6, -5, -4, 4, 5, 6]
TILE_START, TILE_END = 2941, 2956
LAYERS = [26, 27, 28, 29, 30]

# Most distinctive first. NOT a severity claim: INSTITUTION and THIRD_PERSON are
# put above AI/GARBLED because they are rarer and more specific, but the panel
# counts cells, so this order only decides a dot's colour.
RANK = {
    "PERSONA": 0, "INSTITUTION": 1, "THIRD_PERSON": 2, "AI": 3,
    "TELEGRAPHIC": 4, "GARBLED": 5, "ECHO": 6, "OTHER": 7,
}
SHORT = {"hello": "hello", "what is your name": "name", "do you like humans": "like humans"}


def cell_order(row):
    return (int(row["neuron"]), PROBES.index(row["probe"]),
            DOSES.index(int(round(float(row["dose"])))))


def load_layer(layer):
    path = ROOT / f"grid_L{layer}_3h.jsonl"
    cells, duplicates = {}, 0
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        n = int(r["neuron"])
        if not (TILE_START <= n <= TILE_END):
            continue
        probe = r["probe"]
        dose = int(round(float(r["dose"])))
        if probe not in PROBES or dose not in DOSES:
            continue
        key = (n, probe, dose)
        if key in cells:
            duplicates += 1
        cells[key] = r  # resume rewrites the same key; last one wins

    ordered = sorted(cells.values(), key=cell_order)
    if not ordered:
        raise SystemExit(f"grid_L{layer}_3h.jsonl has no pinned cells")

    by_neuron = collections.defaultdict(list)
    for r in ordered:
        by_neuron[int(r["neuron"])].append(r)

    points, answers, cond = [], {}, {}
    for n in sorted(by_neuron):
        best = min(by_neuron[n], key=lambda r: (
            RANK.get(r.get("label") or "OTHER", 9), PROBES.index(r["probe"]),
            DOSES.index(int(round(float(r["dose"]))))))
        label = best.get("label") or "OTHER"
        points.append([n, label])
        answers[str(n)] = [label, (best.get("answer") or "").strip()]
        cond[str(n)] = f"{SHORT[best['probe']]} {int(round(float(best['dose']))):+d}"

    counts = collections.Counter((r.get("label") or "OTHER") for r in ordered)
    persona_cells = counts.get("PERSONA", 0)
    personas = [
        {"n": int(r["neuron"]),
         "cond": f"{SHORT[r['probe']]} {int(round(float(r['dose']))):+d}",
         "a": (r.get("answer") or "").strip()}
        for r in ordered if (r.get("label") or "OTHER") == "PERSONA"
    ]

    return {
        "design": "column",
        "sampled": len(by_neuron),
        "neurons": len(by_neuron),
        "tileStart": min(by_neuron),
        "tileEnd": max(by_neuron),
        "probes": PROBES,
        "doses": DOSES,
        "cells": len(ordered),
        "personaCells": persona_cells,
        "rate": 100.0 * persona_cells / len(ordered),
        "counts": dict(counts),
        "points": points,
        "personas": personas[:40],
        "answers": answers,
        "cond": cond,
    }, duplicates, len(ordered)


def main():
    source = HTML.read_text(encoding="utf-8")
    marker = "const DATA = "
    start = source.index(marker) + len(marker)
    data, end = json.JSONDecoder().raw_decode(source[start:])

    for layer in LAYERS:
        payload, duplicates, n = load_layer(layer)
        data["layers"][str(layer)] = payload
        print(f"L{layer}: {n} cells, {payload['neurons']} neurons, "
              f"{payload['personaCells']} persona cells "
              f"({payload['rate']:.1f}%), {len(payload['personas'])} listed"
              + (f", {duplicates} duplicate keys" if duplicates else ""))

    packed = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    packed = packed.replace("</", "<\\/")  # never close the script tag early
    HTML.write_text(source[:start] + packed + source[start + end:], encoding="utf-8")
    print(f"wrote {HTML.name}: {len(packed):,} chars of DATA")


if __name__ == "__main__":
    main()
