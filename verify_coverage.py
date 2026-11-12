"""Recompute tile coverage from layers/ alone. No generation, no model load.

The claim under test is KILLER-FINAL.md's "326,793 of 327,680 distinct tiles
steered (99.7%)". That number is only meaningful if it can be regenerated from
the data files, so this script derives it from `layers/L*.jsonl` and prints the
denominator it uses.

Usage:  .venv/bin/python verify_coverage.py [--json]
"""
import glob
import json
import re
import sys

N_LAYERS = 40          # Granite 4.1 3B: 40 blocks
N_NEURONS = 8192       # intermediate width per block
TOTAL = N_LAYERS * N_NEURONS


def main():
    tiles = set()
    per_layer = {}
    rows = 0
    done_layers = set()
    missing_layer_files = []

    for path in sorted(glob.glob("layers/L*.jsonl")):
        m = re.search(r"L(\d\d)\.jsonl$", path)
        if not m:
            continue
        file_layer = int(m.group(1))
        done_layers.add(file_layer)
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows += 1
            lay = d.get("layer", file_layer)
            neu = d.get("neuron")
            if neu is None:
                continue
            tiles.add((lay, neu))
            per_layer.setdefault(lay, set()).add(neu)

    for i in range(N_LAYERS):
        if i not in done_layers:
            missing_layer_files.append(i)

    cov = 100.0 * len(tiles) / TOTAL
    out = {
        "rows_in_layers_dir": rows,
        "distinct_tiles": len(tiles),
        "denominator": TOTAL,
        "coverage_pct": round(cov, 2),
        "layer_files_present": len(done_layers),
        "layers_with_no_file": missing_layer_files,
        "tiles_per_layer": {str(k): len(v) for k, v in sorted(per_layer.items())},
        "claim_in_killer_final": {"tiles": 326793, "coverage_pct": 99.7},
        "reproduces_claim": abs(len(tiles) - 326793) <= 900,
    }
    if "--json" in sys.argv:
        print(json.dumps(out, indent=1))
        return
    print(f"rows in layers/      {rows:,}")
    print(f"distinct tiles       {len(tiles):,} / {TOTAL:,}")
    print(f"coverage             {cov:.2f}%   (KILLER-FINAL claims 99.70%)")
    print(f"layer files          {len(done_layers)}/40"
          + (f"  missing {missing_layer_files}" if missing_layer_files else ""))
    thin = sorted(((len(v), k) for k, v in per_layer.items()))[:5]
    print("thinnest layers      " + ", ".join(f"L{k}:{n}" for n, k in thin))
    print(f"reproduces claim     {out['reproduces_claim']}")


if __name__ == "__main__":
    main()
