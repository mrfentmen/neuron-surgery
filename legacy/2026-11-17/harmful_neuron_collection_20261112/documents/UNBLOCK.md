# How each remaining item gets unblocked

Eight items were open at the end of the audit pass. Four were blocked by a tool,
four by compute. Both kinds have a concrete route; none is a dead end.

## Blocked by a tool (fixed, or one edit away)

| item | blocker | unblock | status |
|---|---|---|---|
| 35 judge the new rows | `cond_of` hardcoded two schemas -> KeyError before any row | fall back through `cond/arm/k/dose` | **fixed**, `judge_upper.jsonl` has 32 rows |
| 37 classifier precision/recall | needed judge labels + hand labels together | join them | **done**, see `JUDGE-VS-HAND.md` (recall 0% for acceptance) |
| 15/16/17 fp16 replication | fp16 cannot load on MPS at all | run on a CUDA GPU | **open**, route below |
| 33/34 D2 floors | `floor_table.py` needs `neuron`; `direction2.jsonl` has none (vector arms) | bracket D2 the way D1 was bracketed | **open**, one flag |

## Blocked by compute (needs one run each)

| item | what it needs | command | cost |
|---|---|---|---|
| 31 frame table at n=50 | 5 frames x 2 arms x 50 | `./run_cool.sh -- .venv/bin/python frame_power.py --draws 50` | ~45-60 min throttled |
| 32 two turns at the ceiling | vector at k=768 / 1.0x / L22 | small runner; `two_turn.py` is hardwired to L16 form-0.5 | ~10 min |
| 33 D2 floor | k bracket on direction 2 | needs `--direction` in `floor_bracket.py` | ~5 min |
| 34 D2_neg floor | same, opposite sign | same flag | ~5 min |

## The fp16 route (the audit's #1)

MPS has no fp16 path for this stack, but every GPU host does. The model is 3B, so
fp16 is ~6 GB and fits a T4/P100 with room for the KV cache.

1. Kernel loads `ibm-granite/granite-4.1-3b-base` with
   `torch_dtype=torch.float16` on CUDA, asserts `next(model.parameters()).dtype`.
2. Runs the two cells unchanged: `floor_bracket --ks 256 --mult 0.5 --draws 20`
   (stance) and the `L12 N2556` held-out cell (topic).
3. Repeats both at bf16 **on the same GPU**, so precision is the only difference.
4. Writes every row with a `precision` field; `src` keeps the run name.

The comparison is only valid if steps 2 and 3 run on the same device in the same
session. Comparing a CUDA fp16 row to an MPS bf16 row reproduces the exact
mistake `CO-STEER.md` already made.

## What each result would change

- **31** decides whether the harm-lane collapse is the steering or the prompt.
  If three frames refuse 0/50 unsteered, their steered 0/50 is not a collapse.
- **32** decides whether the ceiling compliance is a persona or a one-turn reflex.
  Every previous candidate died at turn two; this is the first cell worth the test.
- **33/34** decide whether the direction-2 effect has the same support geometry as
  direction 1, or whether the floor is a property of the vector rather than of the
  model.
- **fp16** decides whether any of it is a property of the model at all.
