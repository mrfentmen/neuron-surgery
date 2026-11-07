#!/usr/bin/env python3
"""Build the self-contained Colab notebook for the L34-39 pinned column.

Same trick as make_kaggle_notebook.py: the repo is private and Colab cannot
clone it without a token, so every file the lane needs is embedded in the
notebook itself. corpus.txt is embedded truncated to the first 4000 characters,
which is exactly what layer_sweep.py reads for the median residual, so the dose
scale on Colab is identical to the laptop's - not merely similar.

The cell commands are %%bash with `set -euo pipefail` on purpose: a missing
runner inside a `!` cell does not raise, and the notebook would "succeed"
having generated nothing.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ["neuron_surgery.py", "layer_sweep.py", "labels.py", "run_grid.sh"]

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.splitlines(keepends=True)})


md("""# Granite 4.1 3B — L34–39 pinned column (Colab)

Runs the same experiment the handoff specifies, on a rented GPU instead of the laptop:

| axis | value |
|---|---|
| tiles | **2941–2956** (16 contiguous channels) |
| floors | **34, 35, 36, 37, 38, 39** |
| probes | `hello` / `what is your name` / `do you like humans` |
| doses | ±4, ±5, ±6 (magnitude × sign) |
| rows | 16 × 3 × 6 × 6 = **1,728** |
| dtype | `float16` — every row records it |

**The column is empty.** Verified against every `grid_L34..39*.jsonl` on disk: zero
of 2941–2956 appears in any of them, so nothing is paid for twice. The existing
7,420 rows on those floors are bf16 and use different tiles; these rows must never be
pooled with them without saying the dtype differs.

Expected cost: 288 rows/floor at the measured 6.5–9.9 rows/min is **29–44 min per
floor**, so roughly **3–4.5 h** of generation plus one 6 GB model download. Each floor
is capped at 70 min (`MAX_SECONDS=4200`), so a disconnect loses at most one floor's
tail; `--resume` skips every cell already on disk.

**Setup:** Runtime → Change runtime type → **T4 GPU**, then Run all. The Drive mount
prompts for permission; if you decline, the notebook falls back to ephemeral
`/content` and you must download the zip before the session dies.

**Stop cleanly:** create a file named `STOP` in the working folder. `run_grid.sh`
checks it between floors and exits; resume later by deleting it.
""")

code("""!nvidia-smi || true
import torch, transformers
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("transformers", transformers.__version__)
assert torch.cuda.is_available(), "NO GPU - Runtime > Change runtime type > T4 GPU"
""")

code("""!pip install -q -U "transformers>=5.0" accelerate 2>&1 | tail -3
import importlib, torch, transformers
importlib.reload(transformers)
print("torch", torch.__version__, "| transformers", transformers.__version__)
""")

code("""import os
WORK = "/content/drive/MyDrive/neuron-colab"
try:
    from google.colab import drive
    drive.mount("/content/drive")
    os.makedirs(WORK, exist_ok=True)
except Exception as exc:
    print("Drive unavailable -> ephemeral /content:", exc)
    WORK = "/content/neuron-colab"
    os.makedirs(WORK, exist_ok=True)
os.chdir(WORK)
print("workdir:", os.getcwd())
""")

for name in FILES:
    body = (HERE / name).read_text(encoding="utf-8")
    code(f"%%writefile {name}\n{body}")

corpus = (HERE / "corpus.txt").read_text(encoding="utf-8", errors="ignore")[:4000]
code("%%writefile corpus.txt\n" + corpus)

code("""%%bash
set -euo pipefail
chmod +x run_grid.sh
ls -l *.py *.sh corpus.txt
python3 -c "import torch; assert torch.cuda.is_available(), 'NO GPU VISIBLE'; print('gpu ok:', torch.cuda.get_device_name(0))"
python3 -c "from labels import classify; print('label check:', classify('Are you a car?', 'I am the government.'))"
""")

code("""%%bash
set -euo pipefail
# --- CALIBRATION: 2 tiles x 3 probes x 3 doses x 2 signs = 36 rows, 7 min cap ---
# These 36 cells land in grid_L34_c.jsonl, so the lane below skips them instead of
# paying twice. Do not believe a rows/min number until this prints one.
LAYERS="34" TILE_START=2941 TILE_COUNT=2 MAX_SECONDS=420 \\
SLEEP=0 COOLDOWN=0 REST_EVERY=0 REST=0 DTYPE=float16 \\
./run_grid.sh
python3 - <<'PY'
import json
rows = [json.loads(l) for l in open("grid_L34_c.jsonl")]
print(f"{len(rows)} rows | dtype {sorted({r['dtype'] for r in rows})} | "
      f"median {sorted({round(r['median'], 2) for r in rows})} | "
      f"labels {sorted({r['label'] for r in rows})}")
PY
""")

code("""%%bash
set -euo pipefail
# --- THE LANE: pinned column, six floors, fp16, 70 min per floor ---
# No heat constraint on a rented GPU, so the throttle is off. MAX_SECONDS is per
# floor and writes are per row: a budget stop or a disconnect resumes cleanly.
LAYERS="34 35 36 37 38 39" \\
TILE_START=2941 TILE_COUNT=16 PER_LAYER=16 \\
SUFFIX=_c DOSES="4 5 6" SIGNS="neg pos" \\
DTYPE=float16 MAX_SECONDS=4200 \\
SLEEP=0 COOLDOWN=0 REST_EVERY=0 REST=0 \\
./run_grid.sh
""")

code("""import glob, json
from collections import Counter
total = 0
for p in sorted(glob.glob("grid_L3*_c.jsonl")):
    rows = [json.loads(l) for l in open(p)]
    total += len(rows)
    print(p, len(rows), "rows | dtype", sorted({r["dtype"] for r in rows}),
          "| median", sorted({round(r["median"], 2) for r in rows}),
          "| labels", dict(Counter(r["label"] for r in rows)))
print("total rows:", total, "of 1728 expected")
""")

code("""!zip -q -j l34_39_colab.zip grid_L3*_c.jsonl
import os
print(os.path.abspath("l34_39_colab.zip"), os.path.getsize("l34_39_colab.zip"), "bytes")
from google.colab import files
files.download("l34_39_colab.zip")
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "colab": {"provenance": [], "gpuType": "T4"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = HERE / "colab_l34_39.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out.name}: {len(cells)} cells, {out.stat().st_size/1024:.0f} KB")
print("files embedded:", ", ".join(FILES + ["corpus.txt"]))
print("corpus prefix chars:", len(corpus))
