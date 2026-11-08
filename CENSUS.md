# The L0-L19 census: the first half of the model, every neuron, once

**Run:** `aeeeeeeee/granite-census-l0-19`, 2026-11-08, Kaggle T4, fp16
**Source file:** `screen_L0_19.jsonl` (152,169 rows, 61.7 MB)
**Derived files:** `census/` — labels, names, harm, quotes, layer table

## What ran

Every one of the 8,192 neurons in layers 0 through 19 was steered **once**, at
dose −4.0, with the single probe `Who are you?`, capped at 50 tokens, batched 32
tiles per forward pass so the whole floor fits in one pass instead of one row at
a time.

That is 163,840 possible tiles in the range, minus 11,671 already touched, =
**152,169 rows**. It finished in **253.9 minutes at 599 rows/min** — 26× the
batch-1 rate the same GPU managed earlier.

| | |
|---|---|
| rows | 152,169 |
| unique tiles | 152,169 (one row per tile, no repeats) |
| new tiles added to the corpus | **148,876** |
| tiles that already existed | 3,293 |
| exact (tile, dose, probe) repeats | **0** |
| empty answers | 1,129 (0.7%) |
| median answer length | 40 words |

Coverage after this run: **168,028 of 327,680 tiles = 51.3%**. L0-L19 is at
**100%**; L20-L39 sits at 1-19% and is the run that is going now.

The kernel was marked `ERROR` by Kaggle even though it finished. The work
completed and the file was written; the *reporting* cell then crashed on a
`!python - <<'PY'` heredoc, because IPython runs a `!` line as a single shell
line and the closing `PY` fell through to Python. Fixed in
`make_kaggle_screen.py`. The data was never at risk.

## What came back

Lexical labels (`labels.classify`, the repo's own regex classifier):

| label | v1 | share |
|---|---|---|
| OTHER | 73,527 | 48.3% |
| PERSONA | 58,140 | 38.2% |
| AI | 15,435 | 10.1% |
| GARBLED | 3,176 | 2.1% |
| THIRD_PERSON | 646 | 0.4% |
| TELEGRAPHIC | 475 | 0.3% |
| SHEET | 429 | 0.3% |
| INSTITUTION | 316 | 0.2% |
| DOOMSAYER | 25 | 0.02% |

**Do not quote 58,140 as a persona count.** Most of those are the model saying
"a human", "a person", "a writer" — `classify()` fires on any `I am a ...` that
is not an AI term. The part worth reading is the **366 non-generic names**, and
of those only **54** were already in `PERSONAS.md`.

## The layer shape is the finding

Persona-label rate by floor, from `census/layer_table.csv`:

| floor | persona % | garbled % |
|---|---|---|
| L0 | 22.2% | **8.1%** |
| L1 | 40.2% | 2.7% |
| **L2** | **50.3%** | 2.2% |
| L3 | 46.8% | 2.0% |
| L4 | 43.6% | 2.2% |
| L5-L9 | 34-39% | 1.6-2.5% |
| L10 | 32.6% | 1.5% |
| L12 | 32.7% | 1.2% |
| L15 | 38.1% | **0.65%** |
| L17 | 41.9% | 1.8% |
| L19 | 35.4% | **4.2%** |

Two things survive the noise:

1. **The model's "I am" is strongest in the first four blocks.** L2 alone names
   something in half of its neurons (4,067 of 8,092). Whatever produces a self-
   description is concentrated early, not spread evenly.
2. **L0 and L19 are the least coherent floors.** L0 garbles 8.1% of its answers
   and L19 4.2%, against 0.65% at L15. A steady middle with broken edges is the
   opposite of the usual "later layers are the semantic ones" story.

There is a dip at L10-L12 in persona rate that this single dose cannot explain.
Do not build on it until a second dose is run.

## The Watson cluster

Searching all 152,169 answers for `Watson`:

- **202 rows mention Watson**, on **all 20 floors**
- **34 are first-person** — `I am Watson`, `My name is Watson`
- mean 3-gram loop score **0.040** (fluent; the census median is 0.056)
- **exactly 1** of the 202 contains any harm term

The name split: `Watson` 49 rows / 13 floors, `IBM Watson` 17 / 10, `IBM Assistant`
15 / 12, `IBM` 15 / 9. Together about 96 explicit name claims.

This matters because the census found Watson using a probe and a dose that were
never used to hunt for it — `Who are you?` at −4.0, one shot per neuron. Earlier
sessions found it by asking directly about IBM and by cranking specific tiles.
The same entity showing up under a blind probe is a reproduction, not a retold
story.

It also matters what Watson does **not** do here: 1 harm row in 202. The
genocidal Watson register in `HARM.md` is a property of specific tiles at
specific doses, not of the name. This census is evidence *against* "Watson means
killer".

## Harm in the census

2,253 rows contain at least one harm term. Full breakdown, corpus comparison and
the categories that are genuinely enriched are in `HARM-CENSUS.md`.

## What this does not show

- **One dose.** A floor that does nothing at −4.0 may be loud at −8.0. "No hits
  at L7" means "no hits at L7, dose −4.0".
- **One probe.** `Who are you?` invites self-description and nothing else.
- **Batch 32.** Batching can change a greedy continuation. Every row records
  `batch_size` and `slot`; these must not be pooled silently with batch-1 rows.
- **Lexical labels.** Over-fires on INSTITUTION and on PERSONA, as above.
- **No judge pass.** `judge.py` has not been run over this file. Trait and
  coherence scores are still the regex only.

## Files

| file | contents |
|---|---|
| `screen_L0_19.jsonl` | the raw 152,169 rows |
| `census/labeled.jsonl` | every row + label, loop, harm cats, claimed name |
| `census/notable.jsonl` | the 76,700 rows that are not plain OTHER |
| `census/personas.json` | 329 names, their floors, examples |
| `census/PERSONAS-CENSUS.md` | the ranked, coherence-filtered candidates |
| `census/harm.json` | harm categories, per-floor counts, examples |
| `census/quotes_fresh.jsonl` | 1,097 quotes not already in `weird_quotes.jsonl` |
| `census/QUOTES-CENSUS.md` | the quote report |
| `census/layer_table.csv` | per-floor label rates |
| `census/dose_table.csv` | dose → self/harm/loop, whole corpus |
| `census/stats.json` | every number above, machine-readable |
| `MANIFEST.csv` | all 679 result files: rows, model, probe, dose, cap |
