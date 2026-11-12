# PRECISION — the fp16 replication is blocked on this machine, not skipped

The audit's first priority was a precision replication: do the headline effects
survive fp16, or are they an artefact of bf16 on MPS? The honest answer today is
that **the test cannot be run here**, and the failure is concrete.

## What was tried

```
$ .venv/bin/python -c "load granite-4.1-3b-base as float16, .to('mps'), generate 8 tokens"
FP16 FAIL AttributeError
  torch/utils/_contextlib.py:158 in no_grad
  KeyError: 'shape'   (inside the MPS allocator path)
```

`torch 2.15.0` on `macOS 27.0.1 / M1 Max` raises before the first token. This is
not a slow path or a numerical difference; the model cannot be loaded in fp16 on
this stack at all. bf16 loads and generates normally, which is why every result in
this repository is bf16.

## What this does and does not mean

| reading | valid? |
|---|---|
| "our bf16 numbers are wrong" | **no** — nothing here contradicts them |
| "another precision might behave differently" | **yes, untested** |
| "the CUDA fp16 effects in `CO-STEER.md` are the same mechanism" | **untested** |
| "fp16 was tried and silently failed" | **no** — it is recorded here |

So the correct status of the precision question is **open**, not negative. Every
claim in `CLAIMS.md` should be read as "under MPS bf16".

## The exact replication, ready to run elsewhere

Two cells, because they are the two mechanisms:

1. **Stance** — `floor_bracket.py --ks 256 --mult 0.5 --draws 20`, then the same
   at `--mult 1.0 --ks 768`. Predicts: the floor/ceiling pattern persists in fp16.
2. **Topic** — the `L12 N2556` held-out cell (`held_out.py`), 150 draws.
   Predicts: the 0/150 refusal figure persists, because it is a feature address
   rather than a fragile numerical state.

Run both on a CUDA box at fp16, then on the same box at bf16, and compare row by
row. If the two precisions disagree, the effect is a property of the runtime and
the paper says so. If they agree, the effect is a property of the model.

## Why not Colab or Kaggle

Both were used for the census and are the right venue. Neither is wired for an
interactive fp16 A/B on this model today, and a half-configured run would produce
exactly the kind of unlabelled-condition row this project has spent a week
removing. The command block above is the deliverable until a CUDA session exists.
