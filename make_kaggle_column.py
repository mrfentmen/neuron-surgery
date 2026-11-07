#!/usr/bin/env python3
"""Build the self-contained Kaggle notebook for the pinned COLUMN lane.

Local lane (grid_L{26..30}_3h.jsonl) is columns 2941-2956, three probes, doses
+/-4 +/-5 +/-6, cap 45, bf16. This notebook runs the SAME cell definition on the
floors the laptop has not run: 0-25 and 31-39. Result: one 40-floor depth column,
fixed tiles, fixed probes, fixed doses, with only dtype differing (fp16 cloud vs
bf16 local) and every row recording it.

Why not tiles 0-39: channels 0-39 are the least productive strip measured
(3.25% vs 4.4-4.7%, FINDINGS 87/handoff 5), and they are already 62% of the
corpus. Why not the old random gap lap: its random 40 tiles cannot be read against
the column, so it would buy rows that extend no axis.

Session budget: 35 floors x 16 tiles x 3 probes x 6 doses = 10,080 rows. At the
0.44 rows/s measured on the P100 that is 6.4 h of generation plus ~35 model loads,
inside the 12 h session cap with margin. MAX_SECONDS=2700 per floor is a clean
stop, not the binding limit.

Kaggle does NOT persist /kaggle/working between runs, so a dead session restarts
from zero. The output files are downloadable with `kaggle kernels output`; attach
them as a dataset to resume if that happens. The notebook says so rather than
pretending --resume crosses sessions.

Writes column_kernel/kaggle_column.ipynb and column_kernel/kernel-metadata.json.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "column_kernel"
OUT.mkdir(exist_ok=True)

FILES = ["neuron_surgery.py", "layer_sweep.py", "labels.py",
         "run_grid.sh", "run_kaggle.sh"]

FLOORS = list(range(0, 26)) + list(range(31, 40))
TILE_START = 2941
TILE_COUNT = 16
DOSES = "4 5 6"
PER_FLOOR_SECONDS = 2700
SUFFIX = "_col"
TOTAL_ROWS = len(FLOORS) * TILE_COUNT * 3 * 6

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.splitlines(keepends=True)})


md(f"""# Granite 4.1 3B - pinned column, floors 0-25 and 31-39 (Kaggle)

This is the cloud half of the depth column. The laptop already ran channels
{TILE_START}-{TILE_START + TILE_COUNT - 1} on **L26-L30** (`grid_L*_3h.jsonl`, bf16).
This notebook runs the *identical* cell definition on **{len(FLOORS)} other floors**
({min(FLOORS)}-25 and 31-{max(FLOORS)}), so the whole 40-floor axis becomes one
comparable column at fixed tiles.

| axis | value |
|---|---|
| tiles | {TILE_START}..{TILE_START + TILE_COUNT - 1} ({TILE_COUNT} channels) |
| probes | `hello` / `what is your name` / `do you like humans` |
| doses | {DOSES}, both signs |
| cap | `max_new=45`, greedy |
| dtype | **float16** (P100 has no fast bf16); every row records `dtype` |
| budget | {PER_FLOOR_SECONDS}s per floor, ~{len(FLOORS)} floors |
| expected | {TOTAL_ROWS:,} rows, ~{TOTAL_ROWS / 0.44 / 3600:.1f} h at the measured 0.44 rows/s |

**Floors 26-30 are excluded on purpose** - the laptop already owns them in bf16 at
these tiles, and paying for them again would create duplicate cells in two dtypes.

Do not pool these fp16 rows with the bf16 lane silently: both record `dtype`, so
split or state the mix. The dose scale is identical because `median` is measured
from the embedded `corpus.txt`, same as on the Mac.
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

md("""## Calibration

Two tiles on floor 0, six minutes. Writes to `grid_L0_col.jsonl`; the lane below
skips those 36 cells (`--resume`), so nothing is paid twice. Check the printed
rows/min before trusting the rest.""")

code(f"""%%bash
set -euo pipefail
LAYERS="0" TILE_START={TILE_START} TILE_COUNT=2 SUFFIX={SUFFIX} \\
DOSES="{DOSES}" SIGNS="neg pos" MAX_SECONDS=600 ./run_kaggle.sh
""")

md(f"""## The lane

Ordered ascending, then 31-39. If the session dies early, what survives is the low
floors; the high nine (31-39) are last on purpose so a cut session leaves a
contiguous 0-N column rather than a hole at the bottom.""")

code(f"""%%bash
set -euo pipefail
export DOSES="{DOSES}" SIGNS="neg pos"
export LAYERS="{" ".join(map(str, FLOORS))}"
export TILE_START={TILE_START} TILE_COUNT={TILE_COUNT} SUFFIX={SUFFIX}
export MAX_SECONDS={PER_FLOOR_SECONDS}
./run_kaggle.sh
echo "lane exited $?"
""")

code("""%%bash
set -uo pipefail
echo "=== rows written ==="
wc -l grid_L*_col.jsonl || true
python3 - <<'PY'
import glob, json, collections
tot = 0
for p in sorted(glob.glob("grid_L*_col.jsonl"), key=lambda x: int(x.split("_L")[1].split("_")[0])):
    rows = [json.loads(l) for l in open(p) if l.strip()]
    tot += len(rows)
    tiles = {r["neuron"] for r in rows}
    print(f"{p} {len(rows):>5} rows | {len(tiles):>2} tiles | dtype {sorted({r['dtype'] for r in rows})}"
          f" | median {sorted({round(r['median'], 2) for r in rows})}"
          f" | labels {dict(collections.Counter(r['label'] for r in rows).most_common(4))}")
print("total", tot)
PY
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
(OUT / "kaggle_column.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")

meta = {
    "id": "aeeeeeeee/granite-pinned-column-0-25-31-39",
    "title": "Granite pinned column 0-25 31-39",
    "code_file": "kaggle_column.ipynb",
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

print(f"wrote {OUT / 'kaggle_column.ipynb'} ({len(cells)} cells)")
print(f"wrote {OUT / 'kernel-metadata.json'}")
print(f"plan: {len(FLOORS)} floors, tiles {TILE_START}..{TILE_START + TILE_COUNT - 1}, "
      f"doses {DOSES}, {TOTAL_ROWS:,} rows, ~{TOTAL_ROWS / 0.44 / 3600:.1f} h at 0.44 rows/s")
