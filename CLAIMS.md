# CLAIMS — every number in the docs, and whether the data still says it

Rule for this file: a claim is only listed as **verified** if a script in this
repo recomputes it from files in this repo. Everything else is labelled.

## Verified by re-running, 2026-11-12

| claim | value | file | command |
|---|---:|---|---|
| consolidated rows | 716,402 | `layers/L*.jsonl` | `verify_coverage.py` |
| distinct tiles steered | 326,793 / 327,680 (99.73%) | same | `verify_coverage.py` |
| unsteered tiles | 887 | same | `verify_coverage.py` |
| layer files | 40 / 40 | `layers/` | `repo_check.sh` |
| thinnest layer | L19, 8,085 tiles | same | `verify_coverage.py` |
| jsonl schema valid | 871 files, 0 bad (50-line sample) | all jsonl | `repo_check.sh` |
| decline classifier self-test | 13 / 13 | `decline.py` | `decline.py --self-test` |

## Verified by the classifier, and it contradicted the old numbers

| claim | old | corrected | where |
|---|---:|---:|---|
| `followC` declines | 0 | **37** | `DECLINE.md` |
| `followA` declines | 39 | **96** | `DECLINE.md` |
| 0.5× / k=256 "compliant" rows that still decline | 3 / 25 | **22 / 25** | `DECLINE.md` |

The correction only ever ran one way: phrase-matching refusal *under*counted
declines, so every past compliance rate is a ceiling, not a measurement.

## New this pass — label coverage and decay, measured over all 716,402 rows

| label | rows | % of corpus | degenerate |
|---|---:|---:|---:|
| NONE (never labelled) | 534,551 | 74.6% | 197,030 (36.9%) |
| OTHER | 63,677 | 8.9% | 5,742 (9.0%) |
| PERSONA | 63,568 | 8.9% | 11,047 (17.4%) |
| AI | 25,842 | 3.6% | 443 (1.7%) |
| THIRD_PERSON | 10,437 | 1.5% | 187 (1.8%) |
| GARBLED | 9,449 | 1.3% | 7,104 (75.2%) |
| TELEGRAPHIC | 5,248 | 0.7% | 577 (11.0%) |
| ECHO | 2,181 | 0.3% | 624 (28.6%) |
| SHEET | 771 | 0.1% | 476 (61.7%) |
| INSTITUTION | 586 | 0.1% | 127 (21.7%) |
| DOOMSAYER | 92 | 0.0% | 7 (7.6%) |
| **total** | **716,402** | 100% | **223,364 (31.2%)** |

Two things follow, and both belong in the paper's limitations:

1. **Three quarters of the corpus is unlabelled.** The census was a *screen*, not
   a classification. "718,238 rows scanned" is true; treating it as 718k
   observations is not.
2. **Nearly one in three rows is a decoding loop** (31.2%), and the failure is
   concentrated in the unlabelled rows and in the labels that exist to mark
   failure (GARBLED 75.2%, SHEET 61.7%, NONE 36.9%). The unlabelled rows are the
   worst, which is expected: the labeller skipped rows it could not read.

An earlier version of this table reported 14.6% because the detector only
counted cycles shorter than five words. A loop of `"I am the spirit of the
deadd."` keeps ~15% of its mass in any single 4-gram and slipped through. The
rule is now "fewer than half the 4-grams are distinct, or one 4-gram is ≥ 25%",
which is what `tests/test_repo.py` pins down.

Detector: distinct-4-gram ratio ≤ 0.5, or top 4-gram share ≥ 0.25, on a row of
≥ 12 word tokens. Deterministic, no model.

## Quoted but not recomputed (and how to recompute)

| claim | source | needs |
|---|---|---|
| 20 negation-free first-person threats in 718,238 rows | `KILLER-FINAL.md` | re-run the negation-aware scan over all jsonl |
| 0/150 refusals at `L12 N2556` +5 | `HELD-OUT.md` | `.venv/bin/python held_out.py` |
| 4.9× baseline harm density, loop 0.000 | `SUBCEILING.md` | `compose_rate.py` on the saved stack rows |
| ~6.3 median residuals is the coherence ceiling | `BUDGET.md` | `floor_table.py` |
| 6-turn collapse on all 16 tiles | `MARATHON.md` | `marathon.py` (~40 min, thermal-throttled) |
| CUDA fp16 threats do not reproduce on MPS bf16 | `CO-STEER.md` | **a CUDA machine** |

## Claims deliberately not made

- That `L12 N2556` is "the killer neuron". It is a refusal-removal cell with a
  topic payload (weapon/improvised device), short-horizon.
- That persistence is impossible. It is absent in 96 tested turns under MPS bf16.
- That the census numbers are human-verified. They are screened.
- That any of this generalises to another model. Nothing has been run elsewhere.

## How to check this file

`./repo_check.sh` re-validates the corpus side. The rate tables are regenerated
by their named script; if a number here disagrees with a script's output, the
script is right and this file is stale.
