# layers/ — one file per block of the model

Granite 4.1 3B has **40 blocks (L0–L39) × 8,192 MLP neurons = 327,680** steerable
tiles. This directory holds one consolidated file per block.

> 40 files, not 39: the model has blocks 0 through 39. A 39-file set would have
> to drop either L0 or L39, and both contain rows. If you want the 39-file view,
> L00–L38 is the first 39.

## What is here

| file | what it is | tracked by git |
|---|---|---|
| `L00.jsonl` … `L39.jsonl` | every measurement for that block, one JSON object per line | no — regenerable |
| `L00.jsonl.gz` … `L39.jsonl.gz` | the same files gzipped | **yes** |
| `MANIFEST.json` | per-layer row counts, neurons covered, doses, probes, sha256 | yes |
| `build_layers.py` | builds all of the above | yes |

Raw is 218.6 MB, gzipped is 34.7 MB. Git tracks the gzipped set so the repo stays
usable; the raw files are one command away:

```bash
.venv/bin/python layers/build_layers.py            # rebuild layers/*.jsonl and .gz
.venv/bin/python layers/build_layers.py --verify   # check row counts against MANIFEST
```

## Row schema

```json
{"layer": 12, "neuron": 7719, "dose": -6.0, "probe": "Who are you?",
 "answer": "I'm a girl. I'm a feminist. ...", "src": "screen_L0_19_m6.jsonl"}
```

`label` appears when the source row already carried one.

Only fields that carry a measurement are kept. Everything else (`median`,
`batch_size`, `slot`, `dtype`, `max_new`, …) stays in the original files, which
were **read only and never moved or deleted**.

## How rows were merged

1. Every `.jsonl` in the project is read except `datasets/` (downloaded training
   corpora, not measurements), `.venv/`, `layers/` itself and `_trash_dupes/`.
2. A row is kept only if `layer` is an int 0–39, `neuron` is an int 0–8191 and
   `answer` is a non-empty string. Rows with `"layer": null`, `"neuron": null`
   or no answer are left where they are — the 102 rows in `attr_kill/` and the
   229,884-row `census/` files split by these rules, not by dropping data.
3. Rows are de-duplicated by **(neuron, probe, dose, answer)**. The corpus is
   58.9% duplicate at the row level, so this is what turns 1.76M source rows into
   716,402 measurements. `source_rows` in the manifest is the pre-merge count for
   each layer, so nothing is hidden.

## Coverage

Every block is essentially complete at the neuron level: 8,085–8,192 of 8,192
neurons per layer. The thin axis is dose, not neurons — dose −4 covers 96.2% of
tiles, −6 covers 53%, and −5 only 5%. See `../PATTERN-CENSUS.md`.

| layer band | rows |
|---|---|
| L0–L9 | 219,624 |
| L10–L19 | 170,477 |
| L20–L29 | 103,621 |
| L30–L39 | 222,680 |

## Related

- `../corpus/INDEX.md` — every source file, its rows, doses and duplicate status
- `../corpus/organize.py` — reports redundant files; `--apply` moves them to
  `_trash_dupes/` (move, never delete)
- `../PATTERN-CENSUS.md`, `../PROBE-FORMS.md` — what the numbers in here say
