# Night shift — 20 queued tasks

Started 2026-11-06. Rule: **no job outlives attention without a STOP file and a hard deadline.**

## State at launch

| # | task | state |
|---|---|---|
| 1 | Normalize all 33 datasets into one SFT schema | **done** — 117,981 rows, 546 MB, 7.3 s |
| 2 | Extract verifier prompts for RFT | **done** — 300 (opc asserts + LIMO numeric) |
| 3 | Granite local sweep, gap floors only | **running** — L31 30 28 26 25 23 22 21 20, suffix `_m` |
| 4 | Thermal check at +3 min | **pass** — no warning level recorded, 50 rows written |
| 5 | Kaggle status | **partial** — 9 kernels; newest `granite-random-gap-l20-l32` ran 03:19 today |
| 6 | Pull new Kaggle output | **blocked** — no `kaggle` CLI on PATH, `/api/v1/kernels/output` 404s on bearer |
| 7 | Coverage recount | **done** — see correction below |

## The sweep

```
layers   31 30 28 26 25 23 22 21 20     (only floors with no rows on disk)
tiles    ordered 0..39
throttle sleep 0.7s | cooldown 100s / 50 rows | rest 180s / 10 min
rate     50 rows in 85 s -> 16.2 rows/min including cooldown  (~46% duty)
tokens   16.2 x 45 = ~730 output tok/min   (the quiet line measured at ~1,050)
weights  models/granite-4.1-3b-base, HF_HUB_OFFLINE=1  (no download possible)
stop     touch STOP   |   watchdog auto-touches it at +7 h
```

At 1,200 cells per floor and 74 min per floor, about five floors finish before the watchdog:
**L31, L30, L28, L26, L25 ≈ 6,000 new rows.**

## Coverage — the number was being computed wrong

```
37,727 rows | 1,281 distinct (layer, neuron) tiles | 37,619 distinct cells
319,488 channels  ->  0.40% of the brain touched
```

The earlier `25,595 / 319,488 = 8.01%` divided **cells** (layer, neuron, probe, dose)
by the number of **channels**. A cell is one condition on one channel; dividing by the
channel count inflates coverage by the number of conditions (~30x). The honest figure
is **0.40%**, and it was 0.27% before tonight.

## Kaggle

```
9 kernels on aeeeeeeee
  2026-11-06 03:19   granite-random-gap-l20-l32     <- newest, today
  2026-11-05 20:15   granite-ordered-lane-l10-up
datasets on account: 0
```

Output cannot be pulled with the bearer token — `/api/v1/kernels/view/...` and
`/api/v1/kernels/output/...` both return 404 HTML, while `/api/v1/kernels/list` works.
Needs the official `kaggle` CLI (`pip install kaggle`, token to `~/.kaggle/kaggle.json`).

## Still queued (CPU only)

| # | task |
|---|---|
| 8 | Layer × dose × sign summary table from all grid files |
| 9 | Re-extract hostile / refusal quotes with the corrected probe |
| 10 | Contamination check: `prompts_rft` vs `bench_code10` |
| 11 | Token-length histogram per dataset (p50/p90/max) |
| 12 | Dedup report across sources |
| 13 | `RUNBOOK.md` — the 20-run pipeline on disk |
| 14 | Per-run manifest: rows, est. tokens, minutes |
| 15 | GBNF grammar for JSON tool output |
| 16 | TIES merge plan (adapters, alpha, order) |
| 17 | 4-gate eval spec (sanity / regression / holdout / parity) |
| 18 | peft+trl env plan for py3.13 (no install without approval) |
| 19 | Refusal probe set: 20 prompts, text only |
| 20 | Morning report |

## Invariants

- `models/nanbeige-bf16/` and `models/granite-4.1-3b-base/` are **read-only**.
- Every run writes a new directory. Bad run = delete the adapter, not the model.
- No downloads. Offline mode is set, so a missing cache fails fast.
- Abliterate **last**, unless the refusal probe says the base refuses the training data.
- Every number gets its denominator and its instrument.
