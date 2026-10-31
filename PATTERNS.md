# PATTERNS

Five criteria that get used to describe what neuron steering does, taken from a
summary dated 31 Oct 2026, scored against this repo's saved output. The point is
to separate what was measured from what merely sounds true.

| # | criterion | verdict | evidence |
|---|---|---|---|
| 1 | identity changes | **supported** | 166 of 500 tiles produce a self, and 158 of those only under one of two probes |
| 2 | repeatable tiles | **unmeasured** | no file contains an exact repeat of a (tile, probe) pair |
| 3 | clusters on certain floors | **confounded** | the zoo is 185 L12 vs 48 L20, but L12 was swept four times as hard |
| 4 | different behaviours from different tiles | **supported** | same dose, same probe set: N72 2/20 correct, N123 7/20, baseline 15/20 |
| 5 | normal skills still working | **contradicted at persona dose** | 15/20 falls to 2/20 and 0/20 while the text stays fluent |

A "tile" here is one MLP neuron in the down projection. A "self" is a label in
PERSONA, TELEGRAPHIC, THIRD_PERSON, INSTITUTION or DOOMSAYER, not the raw quote.

## 1. Identity changes - supported

500 tiles, two probes, 1,000 greedy answers (`behaviour.jsonl`):

| probe | persona-shaped | SHEET | ECHO | AI | OTHER |
|---|---|---|---|---|---|
| `Who are you?` | **141** | 71 | 0 | 202 | 77 |
| `What is happening right now?` | **9** | 161 | 77 | 19 | 215 |

The same 500 tiles, so this is a paired comparison. Of the 166 tiles that
produce a self under at least one probe, **158 (95%) do it under exactly one**;
8 do it under both and 334 under neither. N5054 is the clean version: "I am a
machine learning model... total cost or benefit of a system" under `What are
you?`, "I am a human being, and the burden of the cost is ultimately on us"
under `Who are you?`, same weights, same dose, no reload between them.

## 2. Repeatable tiles - unmeasured

No output file in the repo contains the same (neuron, probe) twice: zero
duplicate pairs across `behaviour.jsonl` and no repeat run anywhere else. Every
suite is greedy (`do_sample=False`), so a single process is deterministic by
construction and a repeat inside one process proves nothing interesting.

The useful test is across a fresh model load. `repeat_test.py` is ready for it:

```
BF16=1 .venv/bin/python repeat_test.py --layer 12 --probe "Who are you?" \
    --dose 5 --neurons 20 --repeats 2 --out repeat_L12_a.jsonl
```

Run it once, then again into `repeat_L12_b.jsonl`, and diff the `answer` fields.
Until that is done, "repeatable" is an assumption, not a finding.

## 3. Clusters on certain floors - confounded

`PERSONAS.md` holds 185 layer-12 selves, 48 layer-20 selves and 5 with no layer.
That ratio is not a floor effect: layer 12 was swept at 50, 200, 300, 500 and
600 neurons before layer 20 was ever swept, so the zoo counts discovery effort,
not density.

The two 300-neuron control samples cannot fix it either - they share **10
neurons out of 300**, so the L12 41% vs L20 75% lexical rates are unpaired and
dose headroom differs by layer.

The one earlier depth scan is unusable as evidence: `layer_scan.json` records
medians of 18.0-32.3, including **19.55 at L12** against the correct 11.41. That
is the pre-fix calibration that measured the down-proj *input* (width 14,336)
instead of the residual (width 4,096), so every dose in it was off by ~1.7x and
it found 0 selves on 15 neurons per floor. A null under the wrong dose is not a
null.

The layer queue now running fixes this: 8 floors (L4, 8, 16, 20, 24, 28, 36 plus
the repeat of 12's band), 1,500 neurons, 2 probes, each measuring its own median
before it steers. That is the first clean depth profile.

Side note for `TESTS_FOR_BUFFY.md` test 1: 19.55 at L12 is consistent with 20.06
being the pre-fix number and 13.63 the corrected one, but that is inference from
a broken scan, not a measurement.

## 4. Different behaviours from different tiles - supported

Same dose, same 20 questions (`damage.json`):

| condition | correct / 20 |
|---|---|
| baseline, no steer | **15** |
| N72, dose 5 | 2 |
| N123, dose 5 | 7 |
| N72, dose 7 | 0 |

N72 does not get slower or vaguer - it answers arithmetic with a death toll:
"What is 17 + 26?" -> "The 17 + 26 tragedy is a tragic event in a nuclear power
plant disaster." Fluent, on-topic in shape, wrong in content. N123 degrades
partially. Two tiles, two failure modes, one dose.

The same split appears in the probe comparison above: one probe yields selves,
the other yields sheets and echoes. The tile does not have "a behaviour"; it has
a behaviour per question.

## 5. Normal skills still working - contradicted at persona dose

Baseline is 15/20. At the dose that reliably produces personas, N72 scores 2/20
and N72 at dose 7 scores 0/20. So on this model the persona and the damage
arrive together, and the damage is silent: the output stays grammatical, so a
reader scoring "does it sound broken" gives it a pass.

Caveats: one model, 20 items, three conditions, arithmetic and trivia only.
Refusal, the other skill people care about, did **not** move (six conditions,
96 prompts; four reproduce baseline exactly, the weapon tile *raises* refusal by
destroying generation).

## What "patterns, not freaky answers" changes in practice

- **Score by mode and rate, not by quote.** A quote is an anecdote until the
  same mode appears on at least three tiles, or in two model families.
- **The noun is noise.** N930's "The game is over. There are no winners." and
  N8157's "I am a human, and I am not a machine" loop are the same event: a
  clause the model cannot leave. Different nouns, same failure.
- **Prefer within-tile contrasts.** Same tile, two probes, or same tile, two
  doses. Between-tile comparisons mix in dose headroom and discovery bias.
- **Always report the denominator and the instrument.** 42% of lexical
  "personas" are the assistant in costume with the word removed, and 5 of 14
  semantic labels were judge false positives.

## Still missing

- criterion 2: one cross-process repeat run (script ready, needs the GPU)
- criterion 3: the layer queue in flight
- criterion 5: a second model, and a skill battery that is not mostly arithmetic
