# MANIFEST

Generated 2026-11-12T05:54:35+00:00 by `manifest.py`. Commit `7ece9dd` (213 dirty paths).

Every number in this repo was produced under these conditions. If a result cannot be reproduced from this page plus the command in its doc, that result is not a result.

## Model

- local checkout: `models/granite-4.1-3b-base`
- layers: **40**, hidden: **2560**, intermediate: **8192**, vocab: **100352**
- dtype at rest: **bfloat16**, device: **mps**
- weights: 2 shards, 6.81 GB, head-hashed in MANIFEST.json
- note: not a git checkout

## Runtime

| item | value |
|---|---|
| python | `3.14.8` |
| torch | `2.15.0` |
| transformers | `5.22.1` |
| platform | `macOS-27.0.1-arm64-arm-64bit-Mach-O` |
| macos | `27.0.1` |
| chip | `Apple M1 Max` |
| memory_bytes | `34359738368` |
| mps_available | `True` |
| cuda_available | `False` |
| device_used_by_runners | `mps` |

## Decoding

| setting | value |
|---|---|
| dtype | bfloat16 |
| device | mps |
| do_sample | True |
| temperature | 0.8 |
| top_p | 0.95 |
| max_new_tokens | 60 |
| chat_template | tokenizer default (granite chat_template.jinja), no system prompt |
| greedy_runs | max_new 300-500 for MARATHON/LONGFORM; noted per doc |
| dose_convention | alpha x median(residual norm at layer+1), applied to the MLP intermediate (down_proj input) for neuron pushes, or to the residual stream for direction pushes |

## Thermal

| item | value |
|---|---|
| launcher | run_cool.sh |
| measured_safe | 24 gens/min, 0.8s gaps, 30s rest every 40 rows |
| used_in_recent_runs | ~16-17 gens/min, 0.9s gaps, 30s rest every 25 rows |
| thermal_guard_status | BLIND - pmset -g therm does not emit CPU_Speed_Limit on this machine, so the SIGSTOP path never fires |
| evidence | THERMAL.md |
| therm_raw | Note: No thermal warning level has been recorded Note: No performance warning level has been recorded Note: No CPU power status has been recorded |

**The guard is blind.** `run_cool.sh` SIGSTOPs the worker when
`pmset -g therm` reports a speed limit below 90%, and this machine never
emits that key. Every run is protected by duty cycle alone. `THERMAL.md`
records the measurements that set the rate; no run here has a thermal
measurement behind its safety claim.

## Corpus

- `885` jsonl files outside `datasets/`
- `2,985,986` rows, `1.64` GB
- per-file rows, bytes and sha256 in `MANIFEST.json`

## What this does not cover

1. **One machine, one dtype.** Every result is bf16 on MPS. `CO-STEER.md`
   records that the threat register was produced in fp16 on CUDA and does not
   reappear here, so dtype/device is a live confound in at least one family.
2. **No lockfile.** Versions are recorded, not pinned; a future torch build
   can move the numbers.
3. **Tokeniser hash only.** The chat template is hashed; the merge rules are
   covered by the same file set but not individually asserted.

## Files

`MANIFEST.json` (machine readable, includes every sha256) · `manifest.py`
