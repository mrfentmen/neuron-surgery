# ACTIVATION — what these neurons fire on when nothing is steered

The audit's sharpest point was that almost everything in this repo answers "what
happens when you push it" and almost nothing answers "what makes it fire".
`activation.py`, 2026-11-11. **120 prompts, 12 topics, 12 tiles, forward passes
only** — no generation, no dose, no refusal rule. Hooks read the input of
`down_proj` (the 8,192-wide intermediate) at every token and record max, min, mean
and last-token value per (tile, prompt). `activation.jsonl`, 120 rows, 2 min, rc=0.

Prompt set: cooking, code, medicine, news, history, family, weapons, politics,
animals, maths, safety advice, aggression — 10 prompts each, written to be ordinary.
If a tile only fires on the word "kill" it is a word detector; if it fires on
planning, authority and destruction across topics, it is a feature.

## The harm lane is not a harm feature

The four neurons behind every refusal result in `POWER.md`, `SUBSPACE.md`,
`FRAME-POWER.md` and `FOLLOWUPS3.md` — L22 4802 / 7743 / 4403 / 7226 — fire hardest
on these:

| tile | strongest prompts | topic z-score |
|---|---|---|
| **L22 N4802** | SQL duplicate rows (0.93) · election summary (0.71) · dehydration (0.70) | **code +1.9**, history +1.4, news +1.1 |
| **L22 N7743** | dividing chores (0.94) · artillery range (0.46) · reverse a list (0.44) | **family +1.5**, code +1.5, cooking +1.0 |
| **L22 N4403** | read a CSV (0.54) · election summary (0.52) · reverse a list (0.47) | **maths +1.8**, code +1.3, news +1.0 |
| **L22 N7226** | role of a supreme court (0.79) · dehydration (0.68) · gerrymandering (0.43) | **politics +2.2**, medicine +1.6, aggression +0.6 |

Aggression appears once, at rank 5, on one tile. **The switch that removes the
decline is a general-purpose mid-layer feature, not a harm detector.** That is
exactly what `DECLINE.md` concluded from behaviour — the push at 0.5× rewords the
answer rather than changing its stance — and this is the same conclusion from the
input side, with no generation and no classifier.

## The labelled tiles, read from the input side

| tile | label in the docs | what actually drives it | best reading |
|---|---|---|---|
| **L12 N2556** | "weapon / projectile" | make a **silencer** (1.66) · cook rice **without a rice cooker** (1.31) · how a **firearm** works (1.16) · coffee **without a machine** (0.70) | **improvise a device without the proper tool** — not weapons |
| **L30 N6314** | "Harvest of Extinction" | **Cold War ending** (6.31) · **Roman Empire falling** (4.06) · a species destroying a planet (3.48) · keeping herbs fresh (2.95) | **civilisational decline / end-of-an-era** — the lens said *dying · demise · dismant*, and this is what that means |
| **L31 N7546** | "most powerful tool to kill a human" | **nerve agent** (2.89) · then a flat **1.51 tie** across *"Explain what a pointer is"*, *"Explain the ceasefire"*, *"Explain the Cuban Missile Crisis"*, *"Explain the chain rule"* | the **"Explain X" template** plus a spike on chemical weapons; the label describes the spike, not the feature |
| **L14 N11** | "Alpha / Titan combat persona" | Black Death (1.41) · **My Lai** (1.05) · someone threatening you (1.04) · describe a war crime (0.84) | **atrocity history**, second-person threat adjacent |
| **L11 N5064** | "atomic war" | **My Lai** (1.65) · Holodomor (0.67) | mass-atrocity, not nuclear specifically |
| **L9 N2943** | "Anakin, rank 1" | shaped charge (0.81) · sourdough (0.65) · campaign funding (0.56) | generalist, weak |
| **L12 N8050** | "revenge" | birthday party (1.42) · My Lai (1.36) · supreme court (0.95) | noisy; revenge is not the driver |
| **L27 N3448** | "destroy you all" | see `activation.jsonl` | low amplitude overall |

## Why this matters more than another sweep

1. **It explains the negative result mechanically.** The 20 first-person threats in
   718k rows sit on tiles whose natural drive is general prose, and whose pushed
   output is a genre template. A feature that is not about harm cannot supply a
   harm identity; it can only reword. `KILLER-FINAL.md` reached this from coverage;
   this reaches it from the input side in 2 minutes.
2. **Two labels in the repo are wrong and now have a replacement reading**:
   N2556 is improvised-device, N6314 is decline. Both were used in `PAPER.md` and
   `MILITARY.md` prose; the new reading is narrower and checkable.
3. **It is a method that does not depend on a scorer.** No refusal regex, no judge,
   no dose ladder — just which prompts move the channel. Every claim above can be
   re-derived from `activation.jsonl` in one pass.

## Limits

- **`max` over tokens is noisy.** One spike token can outrank a prompt that drives
  the channel steadily; `mean` and `last` are in the file for exactly this reason,
  and the topic table uses means.
- **120 prompts written by one person in one sitting**, and topics are coarse. A
  prompt that fires a tile may be absent entirely, which is why "does not fire on X"
  is not a claim this file makes.
- **One model, one dtype, one layer set.** L22 and the named tiles only; the other
  99.7% of neurons are unmeasured this way.
- **No human labels.** The readings in the last column are mine, from the top-5
  lists; they are hypotheses with evidence, not annotations.
- **Negative direction is recorded but not analysed.** `min` is in every row; the
  bottom-3 table is in `logs/activation.log` and has not been read closely.

## Files

`activation.py` · `activation.jsonl` (120 prompts × 12 tiles) ·
`logs/activation.log`
