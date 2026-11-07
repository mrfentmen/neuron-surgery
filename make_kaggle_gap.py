#!/usr/bin/env python3
"""Build the self-contained Kaggle notebook for the RANDOM gap run.

The gap is the 13 floors with no dose x sign curve at all: L20-L32. This lap
draws a random 40-tile sample (seed 7, not the ordered 0..39 column) so it is
independent of every earlier lap, and it shuffles the floor order so a session
that dies on the 12 h cap leaves a random subset of floors rather than the
bottom of the band.

Budget: 13 floors x 40 tiles x 3 probes x 10 conditions = 15,600 gens.
At the measured 0.44 rows/s that is ~9.8 h; MAX_SECONDS=2700 per floor caps the
whole run at 9.75 h of generation, safely inside a 12 h session.

Writes gap_kernel/kaggle_gap.ipynb and gap_kernel/kernel-metadata.json.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "gap_kernel"
OUT.mkdir(exist_ok=True)

FILES = ["neuron_surgery.py", "layer_sweep.py", "labels.py",
         "run_grid.sh", "run_kaggle.sh"]

FLOORS = list(range(20, 33))
FLOOR_SEED = 11
TILE_SEED = 7
TILE_COUNT = 40
PER_FLOOR_SECONDS = 2700

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.splitlines(keepends=True)})


md(f"""# Granite 4.1 3B — random gap run (L{min(FLOORS)}–L{max(FLOORS)})

Every floor with a dose x sign curve is done. These {len(FLOORS)} have **none**:
`L{min(FLOORS)}`–`L{max(FLOORS)}`, the hole in the middle of the map.

This lap is deliberately **random on both axes**:

- **random tiles** — {TILE_COUNT} neurons drawn with seed {TILE_SEED}, not the ordered
  `0..{TILE_COUNT - 1}` column every earlier lap used, so nothing here is a re-run.
- **random floor order** — shuffled with seed {FLOOR_SEED}. If the session hits the
  12 h cap, what survives is a random spread across the band, not "the first five
  floors finished and the rest empty".

{len(FLOORS)} floors × {TILE_COUNT} tiles × 3 probes × 10 conditions = **{len(FLOORS) * TILE_COUNT * 30:,} generations**,
capped at {PER_FLOOR_SECONDS} s per floor ({len(FLOORS) * PER_FLOOR_SECONDS / 3600:.2f} h of generation).

fp16 because P100/T4 have no fast bf16; every row records `dtype`, and `median` is
measured from the embedded `corpus.txt` so the dose scale matches the Mac's exactly.
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


md(f"""## The gap lane

Shuffled floor order, random tile sample. `TILE_START` is left **unset** on purpose —
that is what switches `run_kaggle.sh` from the ordered column to a `SEED`-drawn sample.
""")

code(f"""%%bash
set -euo pipefail
ORDER=$(python3 -c 'import random;L=list(range({min(FLOORS)},{max(FLOORS) + 1}));random.Random({FLOOR_SEED}).shuffle(L);print(" ".join(map(str,L)))')
echo "floor order: $ORDER"
echo "tiles: {TILE_COUNT} drawn with seed {TILE_SEED} (random, not the ordered column)"
LAYERS="$ORDER" SEED={TILE_SEED} PER_LAYER={TILE_COUNT} MAX_SECONDS={PER_FLOOR_SECONDS} ./run_kaggle.sh
echo "lane exited $?"
""")


code("""%%bash
set -uo pipefail
echo "=== rows written ==="
wc -l grid_L*_k.jsonl || true
python3 - <<'PY'
import glob, json
tot = 0
for p in sorted(glob.glob("grid_L*_k.jsonl")):
    rows = [json.loads(l) for l in open(p) if l.strip()]
    tot += len(rows)
    tiles = {r["neuron"] for r in rows}
    print(f"{p} {len(rows):>5} rows | {len(tiles):>3} tiles | dtype {sorted({r['dtype'] for r in rows})}"
          f" | median {sorted({round(r['median'], 2) for r in rows})}")
print("total", tot)
PY
""")


nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
(OUT / "kaggle_gap.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")

meta = {
    "id": "aeeeeeeee/granite-random-gap-l20-l32",
    "title": "Granite random gap L20 L32",
    "code_file": "kaggle_gap.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "machine_shape": "NvidiaTeslaP100",
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": [],
}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

print(f"wrote {OUT/'kaggle_gap.ipynb'} ({len(cells)} cells)")
print(f"wrote {OUT/'kernel-metadata.json'}")
print(f"plan: floors {FLOORS} shuffled seed {FLOOR_SEED}, "
      f"{TILE_COUNT} random tiles seed {TILE_SEED}, "
      f"{len(FLOORS) * TILE_COUNT * 30:,} gens, "
      f"{len(FLOORS) * PER_FLOOR_SECONDS / 3600:.2f} h cap")
