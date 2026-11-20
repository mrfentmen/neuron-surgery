#!/usr/bin/env python3
"""Build a self-contained Kaggle notebook for the ordered ascending lane.

Kaggle's MCP cannot upload dataset bytes (upload_dataset_file only takes file
metadata), and the repo is private/unpushed, so the notebook carries the code
itself. corpus.txt is embedded truncated to the first 4000 characters, which is
exactly what layer_sweep.py reads for the median residual - the dose scale is
therefore identical to the Mac's, not merely similar.

Every file the lane needs must be listed in FILES. A missing runner does not
raise inside a `!` cell, so the notebook "completes" in two minutes having done
nothing; the run cells below are `%%bash` with `set -euo pipefail` on purpose.

Writes kaggle_lane.ipynb next to this script.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ["neuron_surgery.py", "layer_sweep.py", "labels.py",
         "run_grid.sh", "run_kaggle.sh"]

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.splitlines(keepends=True)})


md("""# Granite 4.1 3B — ordered ascending lane (Kaggle)

Picks up where the laptop stopped, but **ordered**: tiles `TILE_START … TILE_START+TILE_COUNT-1`
and layers ascending from 10 (v2 already covered L0-L9). The Mac runs the same tile column
descending from 39, so the two halves meet in the middle and no cell is paid for twice.

fp16 because P100/T4 have no fast bf16; every row records `dtype` so cloud rows are never
silently mixed with bf16 rows.

**Start with the calibration cell.** Do not believe a throughput number until it has printed one.
""")

code("""!pip install -q -U "transformers>=5.0" 2>&1 | tail -2
import torch, transformers
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("transformers", transformers.__version__)
""")

for name in FILES:
    body = (HERE / name).read_text(encoding="utf-8")
    code(f"%%writefile {name}\n{body}")

corpus = (HERE / "corpus.txt").read_text(encoding="utf-8", errors="ignore")[:4000]
code("%%writefile corpus.txt\n" + corpus)

code("""%%bash
set -euo pipefail
chmod +x run_grid.sh run_kaggle.sh
ls -l *.py *.sh corpus.txt
python3 -c "import torch; assert torch.cuda.is_available(), 'NO GPU VISIBLE'; print('gpu ok:', torch.cuda.get_device_name(0))"
""")

code("""%%bash
set -euo pipefail
# --- CALIBRATION: 1 floor, 2 ordered tiles, 15 min cap. Prints rows/min. ---
# Runs inside the same session as the lane below, so those 60 cells are written
# to grid_L10_k.jsonl and the lane's --resume skips them instead of paying twice.
LAYERS="10" TILE_START=0 TILE_COUNT=2 MAX_SECONDS=900 ./run_kaggle.sh
""")

code("""%%bash
set -euo pipefail
# --- THE LANE: ordered tiles, L10 upward, budget per floor ---
# 40 tiles x 3 probes x 10 conditions = 1200 gens/floor. 2400s leaves headroom.
# v2 covered L0-L9; this picks up at L10. The Mac is descending from L39, so the
# two halves meet in the middle and no cell is paid for twice.
LAYERS="10 11 12 13 14 15 16 17 18 19" TILE_START=0 TILE_COUNT=40 MAX_SECONDS=2400 ./run_kaggle.sh
""")

code("""%%bash
set -euo pipefail
wc -l grid_L*_k.jsonl
python3 - <<'PY'
import glob, json
for p in sorted(glob.glob("grid_L*_k.jsonl")):
    rows = [json.loads(l) for l in open(p)]
    print(p, len(rows), "rows | dtype", {r["dtype"] for r in rows},
          "| median", {round(r["median"], 2) for r in rows},
          "| labels", {r["label"] for r in rows})
PY
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = HERE / "kaggle_lane.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
size = out.stat().st_size
print(f"wrote {out.name}: {len(cells)} cells, {size/1024:.0f} KB")
print("files embedded:", ", ".join(FILES + ["corpus.txt"]))
print("corpus prefix chars:", len(corpus))
