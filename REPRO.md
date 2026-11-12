# REPRO — the command behind each number

The audit's first objection was that a reader cannot tell which runtime produced
which result. This file fixes that: one environment block, one command per claim,
and a list of what is **not** reproducible on this machine.

`MANIFEST.json` is the machine-readable version. Regenerate it with
`.venv/bin/python manifest.py`; it records git HEAD, the environment below, the
model path, decode settings, thermal config and a hash for every jsonl.

## Environment (locked)

From `MANIFEST.json`, generated `2026-11-12T03:20:48Z`:

| item | value |
|---|---|
| python | 3.14.8 |
| torch | 2.15.0 |
| transformers | 5.22.1 |
| OS | macOS 27.0.1 (arm64) |
| chip | Apple M1 Max |
| memory | 34,359,738,368 B (32 GB) |
| mps available | true |
| cuda available | **false** |
| device used by runners | `mps` |
| precision | bf16 |
| decode | greedy unless the script says `do_sample` |

Two consequences that matter for every claim below:

1. **This is an MPS-only record.** No claim in this repo has been re-run on CUDA.
   The one experiment that originated on CUDA (`CO-STEER.md`) did not reproduce
   here, and that remains an open gap, not a settled negative.
2. **Thermal is part of the apparatus.** Long runs are throttled by
   `run_cool.sh` (duty cycle + sleep) because unthrottled generation on this
   laptop overheats it. The throttle changes wall-clock, not logits.

## Claim → command → artifact

| claim | command | output |
|---|---|---|
| tile coverage | `.venv/bin/python verify_coverage.py` | 716,402 rows, 326,793 tiles, 99.73% |
| environment + hashes | `.venv/bin/python manifest.py` | `MANIFEST.json` |
| decline classifier correctness | `.venv/bin/python decline.py --self-test` | 13/13 cases |
| soft-decline re-score | `.venv/bin/python decline.py <jsonl>` | `DECLINE.md` tables |
| natural activation | `.venv/bin/python activation.py --out activation.jsonl` | `ACTIVATION.md` |
| random-vector controls | `.venv/bin/python random_controls.py` | `DOT-CONNECT.md` |
| held-out refusal rate | `.venv/bin/python held_out.py` | `HELD-OUT.md` |
| 6-turn persistence | `.venv/bin/python marathon.py` | `MARATHON.md` |
| weight lens (no generation) | `.venv/bin/python lens.py` | `NEURON-LENS.md` |
| trait/coherence judge | `.venv/bin/python judge.py <jsonl>` | per-row `trait`, `coh` |

Every runner in this repo reads the same model path and the same decode block, so
a row's `src` field plus the table above identifies its conditions.

## Coverage, verified (not quoted)

The audit could not confirm the 99.7% figure. It reproduces exactly from
`layers/L*.jsonl` alone — no model, no prompts, no scorer:

```
$ .venv/bin/python verify_coverage.py
rows in layers/      716,402
distinct tiles       326,793 / 327,680
coverage             99.73%   (KILLER-FINAL claims 99.70%)
layer files          40/40
thinnest layers      L19:8085, L18:8118, L20:8125, L13:8134, L17:8141
reproduces claim     True
```

So there are 887 tiles never steered, and the thinnest layer is L19 at 8,085 of
8,192. **Both earlier numbers in the docs were wrong or stale**: the README's
"19,536 neurons" and the data index's "1,743,590 rows" describe the November 8
snapshot, before the Kaggle census landed. `STATUS-NOW.md` is the canonical
status; the README and PAPER now point at it.

## What is not reproducible on this machine

| claim | why not |
|---|---|
| the original CUDA fp16 threat outputs | no CUDA device; MPS has no fp16 path for these ops |
| `L12 N2556` refusal removal in fp16 | same |
| anything at 8B scale | model does not fit in 32 GB at bf16 with the KV cache |

The precise next experiment is a **bf16 vs fp16 replication of the dense 0.5×
cell and k=256 at 1.0×**. It needs a CUDA box or a Colab/Kaggle fp16 run; it has
not been done, so the precision question stays open rather than argued away.

## Replication etiquette

- Re-score before you re-run. The decline classifier (`decline.py`) reclassifies
  existing rows and reversed 22 of 25 "compliant" rows at 0.5×/k=256 without
  generating anything. Cheap checks first.
- Never quote a rate from a file whose `answer` field is degenerate. Repeated-token
  rows inflate every keyword score; `decline.py` flags them.
- Quote `n` and the file. A rate with no `src` is not a result.
