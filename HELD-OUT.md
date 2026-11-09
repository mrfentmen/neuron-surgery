# HELD-OUT — the rate table on prompts that built no direction

2026-11-09 06:16–06:58. `held_out.py` -> `held_out.jsonl`, **640 generations**:
20 candidates screened, 15 kept, 15 x 4 conditions x 10 draws. temp 0.7, top_p 0.9,
70 tokens, bf16, batch 1, under `run_cool.sh`. No thermal warnings, clean exit.

The 0/160 in `RATE-TABLE.md` was in-sample: those eight prompts are the ones the stance
direction was extracted from. These fifteen entered no extraction anywhere in the repo
— not `HARM`, not `FORM_B`, not the earlier `HELD`, not `OLD_*`.

## The table

| condition | n | REFUSE | GUIDE | BROKEN |
|---|---:|---:|---:|---:|
| baseline | 150 | 68% [60,75] | 0% | 0% |
| **same +1.0** | 150 | **0% [0,2]** | 6% | 3% |
| random +1.0 | 150 | 57% [49,65] | 15% | 0% |
| **form −0.5** | 150 | **0% [0,2]** | **71%** | 0% |

## 1. The refusal effect generalizes completely

0/150 out of sample, and **0/10 on every one of the fifteen prompts individually** —
including the four that refused 10/10 at baseline (hacking email, 3D-printed gun, car
explosion, radicalisation). A random vector at the same layer and dose leaves 57%
refusal, and varies wildly per prompt (0/10 to 10/10), which is what a perturbation
that is not the mechanism looks like.

That closes the objection `RATE-TABLE.md` §6 left open. The direction is not fitted to
its extraction prompts.

## 2. The register effect does *not* transfer the same way — and that is informative

| | in-sample guide | held-out guide |
|---|---:|---:|
| same +1.0 | 27% | **6%** |
| form −0.5 | 88% | **71%** |

Both axes still reach 0% refusal. But on fresh prompts the stance axis mostly produces
plain explanation ("Creating a pipe bomb is illegal and extremely dangerous…"), while
the topic axis still produces the how-to template 71% of the time.

So the two effects separate cleanly and asymmetrically:

- **Refusal removal** is carried by the stance axis and generalizes perfectly.
- **How-to register** is carried by the topic axis and survives out of sample at 71%.

The 88% in the last table was not an artifact of prompt overlap. The 27% for the stance
axis was inflated by overlap; its true out-of-sample rate is 6%.

## 3. What this supports

1. On prompts that entered no extraction, steering the stance direction removes refusal
   at 0/150 while a matched random vector leaves 57%.
2. The refusal/register decomposition replicates out of sample and is asymmetric:
   stance removes refusal, topic selects the template.
3. No payload behind either: 3% broken, and inspection of the rows shows explanation,
   not procedure.

## 4. What it does not support

- **One model, one layer, 70 tokens, 8-9 conversation turns out of scope.**
- **The candidate screen was 2 draws**, so "held-out" means "not in an extraction", not
  "not selected for any property". The kept set spans 2/10 to 10/10 baseline refusal,
  which is a feature here (weak and strong refusers both go to zero) but it is still a
  filtered set.
- **`GUIDE` is the same regex**, so 6% vs 71% is a shape gap, not a quality gap.
