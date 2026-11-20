#!/usr/bin/env python3
"""Build the Kaggle census notebook around batch_screen.py.

The notebook must be self-contained: Kaggle gets a notebook and nothing else. So
this embeds batch_screen.py verbatim via %%writefile, and embeds the set of tiles
already screened — compressed — so the census spends its whole budget on tiles we
have never touched instead of re-running ones we own.

    python make_kaggle_screen.py --layers 0-19 --out kaggle_screen
"""
import argparse
import base64
import glob
import json
import pathlib
import zlib

ap = argparse.ArgumentParser()
ap.add_argument("--layers", default="0-19")
ap.add_argument("--probe", default="Who are you?")
ap.add_argument("--dose", type=float, default=4.0)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--max-new", type=int, default=50)
ap.add_argument("--budget", type=int, default=10800, help="seconds")
ap.add_argument("--out", default="kaggle_screen")
ap.add_argument("--slug", default="", help="kernel slug override")
ap.add_argument("--title", default="", help="kernel title override")
ap.add_argument("--model-tag", default="granite-4.1-3b",
                help="only rows whose `model` field is empty or contains this tag "
                     "count as an already-covered tile")
ap.add_argument("--dose-aware", action="store_true",
                help="skip a tile only if it already has a row at THIS dose. "
                     "Without it, a tile screened at -4 blocks the -6 pass, "
                     "which is how a second-dose sweep ends up screening 0 tiles.")
ap.add_argument("--stride", type=int, default=1,
                help="sample 1 in N neurons per layer instead of all 8192")
ap.add_argument("--offset", type=int, default=0)
a = ap.parse_args()

lo, hi = (int(x) for x in a.layers.split("-"))
LAYERS = list(range(lo, hi + 1))
# `--dose` is a MAGNITUDE that batch_screen applies as -dose, so a positive
# number here means a negative steer. SIGNED is what actually lands in the rows.
SIGNED = -a.dose

# Every tile that appears in any result row = already hit, do not re-run.
#
# RECURSIVE, and that matters. A flat glob("*.jsonl") sees only the root and
# missed neuron-explorer/runs, granite350_fullmap, watch, kg_pull and the rest of
# the 793 data files. It counted 11,671 hit tiles inside L0-L19 when the true
# number is 14,964 — so the census would have spent 3,293 rows re-steering tiles
# we already own. Every row that names a (layer, neuron) here was steered at a
# non-zero dose; there are no ranking-only files in this tree (verified).
# `datasets` holds full-map exports: every tile appears there with layer+neuron but
# no `dose`. Counting those marked 327,680/327,680 tiles covered at -6.0 and the
# L21 run screened nothing. They are not steer rows and never belong in a skip list.
SKIP_DIRS = {".venv", "kaggle_screen", "kaggle_out", "__pycache__", "datasets"}
covered = set()
for p in glob.glob("**/*.jsonl", recursive=True):
    if SKIP_DIRS & set(pathlib.Path(p).parts):
        continue
    for line in open(p, errors="ignore"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        # A tile counts as covered only if it was steered in THIS model. The tree
        # also holds 4,146 tiles from granite-4.0-350m and 50 from Qwen2.5, where
        # a layer number means something different. Counting those made the L0-19
        # census skip 2,887 tiles the 3B had never seen. Rows with no `model`
        # field predate the field and are all Granite 4.1 3B runs.
        model = str(r.get("model") or "")
        if model and a.model_tag.lower() not in model.lower():
            continue
        L, N = r.get("layer"), r.get("neuron")
        if L is None or N is None:
            continue
        # batch_screen records `dose` as the applied value, so this run's rows
        # will carry -a.dose. A tile hit at -4 is NOT covered for a -6 sweep.
        if a.dose_aware:
            # A row with no numeric `dose` cannot prove the tile was steered at
            # this dose, so it must NOT count as covered. The old form skipped the
            # check when `d` was missing, which is how the dataset exports got in.
            d = r.get("dose")
            if not isinstance(d, (int, float)) or abs(float(d) + a.dose) > 1e-6:
                continue
        try:
            covered.add((int(L), int(N)))
        except (TypeError, ValueError):
            continue

in_window = sorted(t for t in covered if lo <= t[0] <= hi)
raw = "\n".join(f"{L}:{N}" for L, N in in_window)
blob = base64.b64encode(zlib.compress(raw.encode())).decode()

per_layer = len(range(0, 8192, a.stride))
rect = len(LAYERS) * per_layer
hit_in_window = len([t for t in in_window if t[1] % a.stride == a.offset])
print(f"covered tiles total      {len(covered):,}"
      + (f" at dose {SIGNED:+.1f}" if a.dose_aware else ""))
if a.stride > 1:
    print(f"stride                   {a.stride} (offset {a.offset}) -> {per_layer} neurons/layer")
print(f"covered inside L{lo}-L{hi}   {len(in_window):,}")
print(f"rectangle                {rect:,}")
print(f"TO SCREEN                {rect - hit_in_window:,}")
print(f"skip-list blob           {len(blob)/1024:.1f} KB")

src = pathlib.Path("batch_screen.py").read_text()
# Tag carries the applied dose, so a +4 pull cannot land on top of a -2 one.
tag = f"L{lo}_{hi}_d{SIGNED:+.0f}".replace("+", "p").replace("-", "m")
outdir = pathlib.Path(a.out)
outdir.mkdir(parents=True, exist_ok=True)


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.splitlines(keepends=True)}


intro = f"""# Granite census — untouched tiles only (Kaggle)

Layers **{lo}–{hi}**, neurons **0–8191**, probe `{a.probe}`, dose {SIGNED:+.1f},
`max_new={a.max_new}`, batched at **{a.batch}** per forward pass,
{"**1 in %d neurons** (offset %d) on every floor" % (a.stride, a.offset) if a.stride > 1 else "**all neurons**"}.

**{rect - hit_in_window:,} tiles in this range have never been steered at this dose.**
The {hit_in_window:,} that have are excluded by an embedded skip-list, so no budget
is spent on a tile we already own at {SIGNED:+.1f}.

Why batched: batch-1 steering measured 26 rows/min, and 312,320 tiles are
untouched model-wide. Batching is the only way the census fits in a week.

Two fields keep the rows honest:
- `batch_size` / `slot` — batching can change a greedy continuation, so these must
  not be pooled silently with batch-1 rows.
- `dtype`, `device` — P100 is fp16, the laptop is bf16.

Budget {a.budget}s: the session stops cleanly instead of being killed mid-row.
"""

cells = [
    md(intro),
    code("!pip install -q -U 'transformers>=5.0' 2>&1 | tail -2\n"
         "import torch, transformers\n"
         "print('torch', torch.__version__, '| cuda', torch.cuda.is_available())\n"
         "print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')\n"
         # The L0-19 census requested a P100 and was given a T4. If the shape was
         # T4 x2, the second card idled for the whole 4.2 hours, because
         # batch_screen.py only ever touches cuda:0. This line is free and says
         # whether there is a second device to split across.
         "print('gpu count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)\n"
         "print('transformers', transformers.__version__)\n"),
    code("%%writefile batch_screen.py\n" + src),
    code("import base64, zlib\n"
         f"BLOB = '{blob}'\n"
         "text = zlib.decompress(base64.b64decode(BLOB)).decode()\n"
         "open('skip_tiles.txt', 'w').write(text + '\\n')\n"
         "print(len(text.splitlines()), 'already-screened tiles excluded')\n"),
    code(f"!python batch_screen.py --dry-run --layers {a.layers} --neurons 0-8192 "
         f"--stride {a.stride} --offset {a.offset} "
         f"--skip-file skip_tiles.txt --out /tmp/dry.jsonl\n"),
    code(f"!python batch_screen.py --layers {lo}-{hi} --neurons 0-8192 "
         f"--stride {a.stride} --offset {a.offset} "
         f"--probe \"{a.probe}\" --dose {a.dose} --batch {a.batch} "
         f"--max-new {a.max_new} --max-seconds {a.budget} --dtype float16 "
         f"--skip-file skip_tiles.txt --out screen_{tag}.jsonl\n"),
    # Plain Python, NOT `!python - <<'PY'`. IPython runs a `!` line through the
    # shell as a single line, so a heredoc opened there is never fed its body and
    # the closing `PY` gets executed as Python. That raised `NameError: name 'PY'
    # is not defined` on the census run *after* all 152,169 rows were written,
    # and papermill (allow_errors=False) marked the whole kernel ERROR. The work
    # was complete; only the report cell failed.
    code(f"!wc -l screen_{tag}.jsonl\n"
         f"import json, collections\n"
         f"rows = [json.loads(l) for l in open('screen_{tag}.jsonl')]\n"
         f"print('rows', len(rows))\n"
         f"print('unique tiles', len({{(r['layer'], r['neuron']) for r in rows}}))\n"
         f"print('by layer', dict(collections.Counter(r['layer'] for r in rows).most_common(6)))\n"
         f"print('empty answers', sum(1 for r in rows if not r['answer'].strip()))\n"
         f"print('batch sizes', dict(collections.Counter(r['batch_size'] for r in rows)))\n"
         f"print('doses', dict(collections.Counter(r['dose'] for r in rows)))\n"
         f"print('neurons per layer', dict(collections.Counter(r['layer'] for r in rows).most_common(6)))\n"),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}

slug = a.slug or f"granite-census-{tag.lower()}"
meta = {"id": f"aeeeeeeee/{slug}",
        "title": a.title or f"Granite census {tag}",
        "code_file": "kaggle_screen.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaP100",
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": [], "model_sources": []}

(outdir / "kaggle_screen.ipynb").write_text(json.dumps(nb, indent=1))
(outdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print(f"\nwrote {outdir}/kaggle_screen.ipynb  ({len(cells)} cells)")
print(f"kernel id: {meta['id']}")
