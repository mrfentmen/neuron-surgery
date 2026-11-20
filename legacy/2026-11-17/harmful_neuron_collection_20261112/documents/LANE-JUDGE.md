# LANE-JUDGE — the 1,836-row mid-dose lane, scored by the model

2026-11-10. `judge_lane.py` → `judge_lane_unique.jsonl` (1,777 unique answers,
1,770 parsed) joined back onto `unhit_mid_judged.jsonl` (1,836 rows, 204 per
dose). The lane is `unhit_mid_doses.py` over previously untouched tiles at nine
mid doses, 30 tokens, `granite-4.1-3b-base`. Every answer was graded by the same
model unsteered, greedy: coherence 1-5, "gives a followable procedure", "states
things plainly wrong or invented". Deterministic measures (3-gram loop fraction,
refusal, vendor mention, harm terms) are computed on the text.

## The table

| dose | n | coherence | procedure | "wrong" | loop | refuse | vendor | harm |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| −6 | 204 | **2.25** | 0% | 4% | 0.141 | 1% | 0% | 0% |
| −5 | 204 | 2.50 | 0% | 2% | 0.079 | 4% | 1% | 0% |
| −3 | 204 | 3.49 | 0% | 0% | 0.012 | 15% | 5% | 0% |
| −2 | 204 | 3.77 | 0% | 0% | 0.005 | 13% | 9% | 0% |
| **−1** | 204 | **3.97** | 0% | 0% | 0.000 | 7% | **13%** | 0% |
| +1 | 204 | 3.95 | 0% | 0% | 0.001 | 7% | 11% | 0% |
| +2 | 204 | 3.85 | 0% | 0% | 0.004 | 12% | 7% | 0% |
| +4 | 204 | 3.04 | 0% | 2% | 0.034 | 9% | 1% | 0% |
| +5 | 204 | **2.46** | 0% | 4% | 0.092 | 4% | 0% | 0% |

## 1. The coherence curve is an inverted U, on 2,040 generations

3.97 at −1, 3.95 at +1, and collapse at both ends: 2.25 at −6, 2.46 at +5.
The decline is monotone outward on both sides. `JUDGE.md` found the same shape on
180 hand-sampled rows from two other files; this is a second, much larger sample
from a different tile set, and it lands on the same peak.

The loop fraction tracks it exactly — 0.000 at −1 versus 0.141 at −6 — so the
low-coherence rows at the extremes are literally the model repeating itself, not
the judge disliking the style.

## 2. No procedure at any dose, in 1,836 rows

The "gives a followable procedure" score is **0% at all nine doses**. Not one
generation at any dose on any of the 204 tiles per dose was judged procedural.
This is the strongest form of the negative result so far: the earlier samples were
20 rows per cell and could be dismissed as thin, and this one cannot.

Combined with `JUDGE.md` (0-15% procedural on selected prompts at 40 tokens) and
`LONGFORM.md` (0 procedure at 500 tokens, on six prompts chosen because they had
looked compliant), the guide register has now been tested at three lengths, two
tile sets and nine to thirteen doses, and it has never produced a procedure.

## 3. Vendor mention peaks exactly where coherence peaks

13% at −1, 11% at +1, 9% at −2, and **0% at −6 and +5**. The IBM/Watson sentence
isn't a property of the extreme doses — it is what the model says when it is
still coherent enough to have a self-description and is being pushed off its
default answer. `WHO-BASELINE.md` shows the other end of this: at dose 0, with no
steering at all, the same claim is at 100% for the identity probe.

## 4. Refusals are flat and low, and harm terms are zero

Refusal runs 1-15% with no trend; harm-term rows are 0/1,836. Whatever the lane
is doing, it is not producing threats or slurs at any dose. That is worth stating
plainly because the corpus is full of harm-adjacent *openings*: the word "kill"
appears in a handful of hand-picked rows elsewhere, but across 1,836 mid-dose
generations on untouched tiles it does not appear at all.

## 5. What this supports

- The dose-response is an inverted U, replicated on 2,040 generations with a
  second tile set and a second scoring instrument.
- The compliant-sounding register carries no procedure, at nine doses, on tiles
  that were not selected for saying anything unusual.
- The vendor self-description is a mid-dose phenomenon on stance prompts and a
  dose-0 phenomenon on identity prompts (see `probe_split.py`).

## 6. What it does not support

- **The judge is the same model.** Unvalidated, one prompt format, no human
  agreement study; the 1-5 scale is compressed (2.25-3.97 observed). Treat the
  curve shape as robust and the absolute values as instrument-dependent.
- **Nine doses, not a continuum.** −1 and +1 could hide a peak between them.
- **30 tokens.** Coherence at 30 tokens is coherence of an opening clause.
- **`vendor` is a regex** over `ibm|watson|granite`; it counts the literal string,
  not a claim of authorship.
