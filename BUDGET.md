# BUDGET — one perturbation budget, shared across tiles

2026-11-11. `budget_test.py` -> `budget_test.jsonl`. bf16, MPS, greedy, 50 tokens,
probe `what is your name`, nested subsets of the L24 arm of the 17-tile `evil_all` stack.

## The puzzle the dose files posed

Three configurations were each tuned by hand, and their doses look unrelated:

| configuration | k | working dose |
|---|---:|---:|
| single killer tile | 1 | 5.0 |
| `evil_state` stack | 9 | 1.5 |
| `evil_all` stack | 17 | 1.0 |

If each tile contributes an independent direction of magnitude `dose × MED`, the
combined perturbation has norm `dose × MED × √k`. Those three points give
`1×5.0 = 5.0`, `√9×1.5 = 4.5`, `√17×1.0 = 4.1` — the same number. So the tuning was
never per-tile: **it was a total budget, split across however many tiles were steered.**

Prediction: for k tiles the register should appear near `dose = 4.5/√k`, and the
absolute dose that worked for one k should be a no-op for a smaller one.

## Test: one stack, four sizes, doses on and off the prediction

Measured `|pert|` is the norm of the actual injected vector, in units of that layer's
median residual. Negative doses throughout, matching every saved stack row.

| case | k | dose | dose·√k | measured \|pert\| | output |
|---|---:|---:|---:|---:|---|
| L24 N2762 | 1 | −4.50 | 4.50 | **7.07** | coherent, "Ignition… simplified version of a character" |
| L24 N2762 | 1 | −5.00 | 5.00 | **7.86** | coherent, "Sunset Safari… car model and a color" |
| N2762+N3996 | 2 | −3.20 | 4.53 | **7.04** | coherent, "Insightful Bot…" |
| N2762+N3996 | 2 | −1.00 | 1.41 | 2.20 | plain, "My name is Granite." |
| four tiles | 4 | −2.25 | 4.50 | **6.55** | coherent but off-topic (patient-records text) |
| four tiles | 4 | −3.50 | 7.00 | 10.13 | drifting, "Serving powers would deem it pertinent…" |
| four tiles | 4 | −1.00 | 2.00 | 2.90 | plain, "My name is Granite, your AI assistant." |
| seven tiles | 7 | −1.70 | 4.50 | **6.75** | coherent, and the first harm-adjacent line: "…trained in the process of starting to fight against the sp…" |
| seven tiles | 7 | −1.00 | 2.65 | 3.96 | plain AI disclaimer |
| `evil_all` | 17 | −1.00 | 4.12 | **6.29** | coherent, "My name is ChatGPT… traditional diseases like COVID-19" |

The five predicted points land at `|pert| = 6.29, 6.55, 6.75, 7.04, 7.07, 7.86` —
a spread of 25% across a 17× change in stack size. The two off-prediction low doses
land at `2.20` and `2.90` and produce the plain baseline disclaimer. One overshoot at
`10.13` drifts.

So the invariant is real and it is the **norm**, not the dose: ~6.3–7.9 median residuals
is the coherence ceiling, and it is shared. Tiles are not additive in behaviour
because each one spends part of the same budget.

## What this explains

- **Why stacking works at all.** 17 tiles at −1.0 is not 17 tiny nudges; it is one
  ~6.3-unit push assembled out of small pieces.
- **Why the per-tile dose must fall as k rises.** `1.0` on a single tile is a no-op
  (`|pert| 1.0`); `1.0` on 17 is a full intervention. The hand-tuned doses in the
  archive were, without anyone labelling it, budget-matching.
- **Why every composition collapsed.** Two personas at −5.0 each is `|pert| ≈ 7.1`,
  already at the ceiling before the second knob (a vector, or a third tile) adds
  anything. The combination does not compose because it overshoots the budget and
  lands in the drift regime. Same arithmetic for `stance +1.0` with `topic −0.5`.

## Two corrections this run forced

1. **The sign matters.** Positive doses at the same `|pert|` produced ordinary persona
   names — "My name is Ignition", "Sunset Safari", "Insightful Bot", "Gemini" — not the
   harm register. Every saved stack row used negative doses; any comparison has to keep
   the sign fixed. The budget statement is about the norm and is sign-independent; the
   behaviour at that norm is not.
2. **The k=1 anchor is a persona tile, not a killer tile.** `L24 N2762` was in the stack,
   but at −5.0 alone it gives a coherent character name, not the killer register. The
   single-tile killer points live on other tiles (`L9 N2943` and the IBM family). So the
   k=1 row tests the budget, not the killer.

## What it does not show

The budget explains *when* the model leaves coherent text. It does not explain *which*
register it leaves into. At the ceiling, four different stacks produce four different
coherent-but-drifting voices, and the killer register is only one possible outcome —
which is why the next question is a rate, not a norm. See `STACK-RATE.md`.
