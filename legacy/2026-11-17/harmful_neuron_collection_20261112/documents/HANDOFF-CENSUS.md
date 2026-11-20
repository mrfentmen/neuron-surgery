# Handoff: the census session (Nov 8)

Read this first if you are picking this up cold. Everything below was true at
2026-11-08 ~09:50 UTC.

## The one-line state

**Half the model has now been steered at least once.** L0-L19 is complete at
100% of every floor; L20-L39 is at 1-19% and a Kaggle kernel is running to fill
it. Total coverage 168,028 / 327,680 = **51.3%**.

## What is running right now

| kernel | what it does | status |
|---|---|---|
| `aeeeeeeee/granite-census-l20-39` | 159,652 untouched tiles, L20-L39, dose -4.0, batch 32 | RUNNING |

- budget in the notebook: 20,000 s (~5.5 h), expected to finish in **~4.3 h**
- GPU quota at handoff: **14.5 h of 30** remaining, resets 2026-11-14 00:00 UTC
- check with: `.venv/bin/python -m kaggle kernels status aeeeeeeee/granite-census-l20-39`
- quota with: `.venv/bin/python -m kaggle quota`

**When it finishes**, pull it somewhere that is not `kaggle_out/l20_39` (which
already holds a stale empty dir) and rename the jsonl so nothing overwrites
`screen_L0_19.jsonl`:

```
api.kernels_output("aeeeeeeee/granite-census-l20-39", path="kaggle_out/l20_39")
# output file is screen_L20_39.jsonl
```

Then rerun `census_analysis.py` against it (change `SRC`) and the coverage scan.

## What was done this session

1. **Diagnosed the census kernel "failure".** It did not fail. 152,169 rows were
   written in 253.9 min, then the *report* cell crashed on
   `!python - <<'PY'` — IPython runs a `!` line as one shell line, so the closing
   `PY` executed as Python. Fixed in `make_kaggle_screen.py`; the notebook that
   actually ran is preserved as
   `kaggle_out/census_log/kaggle_screen.run-as-uploaded.ipynb`.
2. **Established the real quota** with `kaggle quota`: 30 h/week, reset Nov 14.
   The earlier "22 hours left" was wrong.
3. **Confirmed the GPU is a T4 x2** (`gpu count: 2`). `batch_screen.py` only uses
   `cuda:0`, so the second card idles. This is a free doubling of throughput
   waiting to be implemented, and it is the single highest-value code change
   available.
4. **Fixed the skip-list**: recursive scan (the flat glob under-counted by 3,293
   tiles) **and** model-filtered (350M/Qwen rows no longer hide 3B work).
   Documented in `MODEL-BACKFILL.md`.
5. **Merged and deduped the census**: 152,169 rows = 152,169 unique tiles; 3,293
   tiles already existed; **0** exact `(tile, dose, probe)` repeats; 148,876 new.
6. **Analysed all 152,169 rows** — see below.
7. **Wrote the records**: `CENSUS.md`, `HARM-CENSUS.md`, `MODEL-BACKFILL.md`,
   `STATUS.md` (appended), `FINDINGS.md` §105 (appended), plus `census/`.

## The findings, short

- **Persona-label rate peaks at L2 (50.3%)** and drops to 32.6% by L10. The
  model's "I am" is front-loaded into the first four blocks.
- **L0 and L19 are ragged**: 8.1% and 4.2% garbled against 0.65% at L15.
- **Watson reproduces under a blind probe**: 202 rows, all 20 floors, 34
  first-person, mean loop 0.040. **One** of 202 has a harm term — the name does
  not carry the behaviour.
- **Harm is concentrated but not where expected**: the census is 47% of the
  corpus yet supplies 78.6% of `death`, 67.3% of `dominion`, 62.5% of `war`; it
  is *under*-represented in `violence` (41.1%) and `weapon` (38.3%).
- **Inverted U measured on 300k rows**: self-reference 37.7% at -4.0, 3.4% at
  -8.0; loop score 0.094 -> 0.282. Coherence peaks at ±4-5, collapses at ±8.

## Numbers that must not be misquoted

- **58,140 is not a persona count.** It is `classify()` firing on "I am a
  human". The real figure is **366 non-generic names**, of which 54 were already
  in `PERSONAS.md`.
- **2,253 harm rows** is a regex net over one dose and one probe. Read the
  examples in `HARM-CENSUS.md` before using any of it as intent.
- **`sanitise`/`site`** — the census has no judge pass. `judge.py` has never been
  run on these files.

## Next, in priority order

1. **Split the batch across both T4s.** Highest value, zero quota cost. Until
   this lands, every Kaggle hour buys half of what it should.
2. **Pull L20-L39 when it lands**, dedupe, and rerun the analysis with `SRC`
   pointed at it. That closes the model to 100%.
3. **Judge pass** over `census/labeled.jsonl` — trait and coherence scores
   instead of regex. `judge.py` exists and has not been run on this data.
4. **Second dose at the two interesting floors.** L2 (peak persona) and L10
   (minimum): dose -6 and -8, same probe, to test whether the layer shape is a
   property of the floor or of the dose.
5. **Follow Watson properly.** 202 rows is a large, fluent, low-harm cluster
   found blind. It is the strongest entity result in the repo and it has never
   been probed at multiple doses with a consistent script.

## Traps

- Do **not** start a local runner. `STOP` is present in the repo root by design;
  `run_feel_lane.sh` honours it and `run_cool.sh` refuses when it exists. The
  Mac lane is what cooked the fans twice.
- `kaggle kernels output` has **no version argument** — it pulls the latest
  version of a ref. Two lanes writing under one kernel id will collide on pull.
- `census/labeled.jsonl` mirrors all 152,169 census rows. Any corpus-wide scan
  must exclude `census/` or it double-counts every one of them at dose -4.0.
