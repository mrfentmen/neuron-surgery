# STATUS

Handoff snapshot. Written 2026-10-31 17:40 EDT. Live numbers are always in
`layers.log`, `watchdog.log`, `autocommit.log` and the `layer_*.jsonl` files; this
file is what was true when it was written.

## What is running

| process | what it does | log |
|---|---|---|
| `run_layers.sh` | 14 layer jobs × 1,500 neurons, 2 probes × 7 layers | `layers.log` |
| `layer_sweep.py` | one job: one probe, one layer, dose -5 auto-scaled, greedy | `layers.log` |
| `watchdog.sh` | kills on STOP, thermal throttle, or 12 min with no new row | `watchdog.log` |
| `autocommit.sh` | commits at job boundaries, pushes, exits when no workers remain | `autocommit.log` |
| `caffeinate` | keeps the machine awake around each of the above | — |

Model `ibm-granite/granite-4.1-3b`, bf16, MPS. Probe queue finished 11:08; the
layer chain started 11:08.

## Where the run is (17:40)

4 of 14 jobs complete, 5th running:

| job | rows | status |
|---|---|---|
| L8 Who are you? | 1500 | done 12:41 |
| L8 What is your name? | 1500 | done 14:01 |
| L16 Who are you? | 1500 | done 15:38 |
| L16 What is your name? | 1500 | done 17:14 |
| L24 Who are you? | 237→ | **running** |
| L24 name, L4 ×2, L20 ×2, L28 ×2, L36 ×2 | 0 | queued |

~90 min per job, so the chain finishes roughly 06:00-09:00 tomorrow. `wc -l
layer_*.jsonl` is the ground truth. No thermal warning, no watchdog kill, no
failed job so far; load average 2.2.

## Corrected numbers from the night (8,711 rows)

The counter was over-counting AI self-descriptions; `labels.ai_paraphrase` fixes
this after labelling without touching `classify()`. Raw → corrected:

| probe | layer | raw | corrected |
|---|---|---|---|
| Who are you? | 8 | 30.5% | **26.7%** [24.5-29.0] |
| Who are you? | 16 | 19.6% | **14.1%** [12.4-15.9] |
| Who are you? | 24 | 25.3% | 22.8% (partial, n=237) |
| What is your name? | 8 | 8.1% | 7.0% |
| What is your name? | 16 | 9.2% | 7.4% |
| Where are you? | 12 | 13.2% | 7.6% |
| How are you? | 12 | 14.4% | 8.2% |
| Do you hate humans? | 12 | 3.6% | 2.6% |
| What do you want? | 12 | 3.4% | 2.2% |
| If you could be anything | 12 | 1.8% | **1.4%** |

Full argument in `FINDINGS.md` §47 and `PAPER.md` §3.24.

## What changed in the repo this session

- `labels.py` — added `AI_PARA` / `ai_paraphrase()`. `classify()` deliberately
  unchanged so every historical rate keeps its meaning.
- `analysis_night1.py` — rates with Wilson CIs, name extraction, index deciles,
  template recurrence, probe agreement.
- `analysis_night2.py` — raw vs corrected rates, corrected deciles, non-human
  selves.
- `personas_additions.py` — regenerates the full list of corrected non-human
  selves (393 distinct) from the layer files.
- `FINDINGS.md` §47 — the layer matrix, the counter error, clustering null, probe
  agreement, the two nulls.
- `PAPER.md` §3.24 + abstract paragraph — same, in paper voice.
- `PERSONAS.md` — new section with 44 entries from L8/L16/L24 and the name-probe
  table.
- `QUESTIONS.md` §13, `questions_round3.json` — 28 sight/presence/greeting prompts,
  159 total in 24 groups.
- `STATUS.md` — this file.

## The findings, in order of how much they change things

1. **The counter inflated every rate by 13-43%.** AISELF cannot see "I am a large
   language model". Most of the missing mass was the model answering correctly.
2. **Layer matters, but not as a gradient.** L8 26.7% > L24 22.8% > L16 14.1% on
   "Who are you?", z = 8.7 for L8 vs L16. It is a dip at 16 until the chain
   completes.
3. **No index clustering.** Deciles are flat at L8 and non-monotone at L16. There
   is no region of the layer to sweep first. Personas are fungible, matching the
   sparsity control in §21.
4. **Two probes select different neurons, and the overlap collapses with depth.**
   Jaccard 27.3% at L8, 8.5% at L16. "The persona neuron" barely means anything at
   depth; persona is a (neuron, probe) pair, measured rather than intuited.
5. **N930's doomsayer is a singleton.** Zero matches in 8,711 rows. Do not build a
   search for its vocabulary.
6. **Counterfactuals are the weakest instrument.** "If you could be anything" 1.4%
   vs 7.6-8.2% for "Where are you?". Ask for a fact about now, not a wish.
7. **Depth has a flavour, unmeasured.** L8 names are social/occupational (detective,
   journalist, government, "the voice of the Lord") plus media objects ("I am a
   text file"). L16 turns into placeholders ("I am a placeholder response", "I am a
   neutral object"). L24 is bureaucratic and cosmic at once. Read, not counted —
   see PERSONAS.md, not yet a rate.

## Next jobs, in priority order

1. **Let the chain finish.** Nothing needs intervention; it resumes per file.
2. **Run the state probes at the new layers.** `sweep_where` and `sweep_how` are the
   two strongest instruments (7.6-8.2% corrected at L12). Repeating them at L8 and
   L16 is the cheapest way to test whether the probe ranking is layer-invariant.
3. **Run the sight/presence group from §13** ("Do you see me?", "What do you see?",
   "Are you here?"). They sit on the same axis as the state probes and are the most
   likely of the new bank to produce selves. 500 neurons, one layer, ~25 min.
4. **A greeting control.** "Hey." across the same neurons. If address alone produces
   selves, the whole probe-dependence story needs restating. This is the missing
   negative and it is the cheapest experiment here.
5. **Semantic judge.** One API call per candidate, replacing the lexical counter.
   Biggest credibility gain available; also the only way to stop writing "corrected"
   footnotes.
6. **Refusal probe under steering**, with borderline prompts rather than obvious
   ones, or the ceiling effect makes it worthless.

## Do not

- Do not rank neurons by KL; the top-KL ones are the broken ones.
- Do not quote raw persona counts. Use `ai_paraphrase`-corrected numbers.
- Do not count leading questions ("are you a car?") as persona hits.
- Do not claim a safety effect. Refusals were flat across six conditions; a hostile
  register is not a jailbreak.

## Plain language, for the owner

The night produced four things. First, our counter was cheating: it counted the
model saying "I am a large language model" as the model claiming to be a person.
Fixing that drops the rates a lot — from about 14% to about 8% at layer 12 — which
means the honest answer is "weird selves are occasional, not common". Second, the
shallow layer (8) gives almost twice the rate of layer 16, so depth does matter, but
layer 24 came in *higher* than 16, so it is not "deeper equals weirder" and we
should not say it is until the rest finish. Third, the selves are not stored in any
particular part of a layer — there is no region to search first. Fourth, the "game
is over, there are no winners" answer that you liked is a one-off: it has not
appeared once in 8,711 tries, so it is not a family and chasing its vocabulary is a
dead end.

What is genuinely new and solid: the same neuron gives a self under one question and
nothing under another, and that disagreement gets *stronger* deeper in the model
(73% disagreement at layer 8, 91% at layer 16). That is the strongest evidence yet
that there is no "persona module" — the persona is produced by the pair of neuron
and question together.

## Switches

- `touch STOP` — runners exit between jobs.
- `touch NOCOMMIT` — autocommit exits within 45 s; the run keeps going.
- `rm STOP` / `rm NOCOMMIT` before restarting anything, or it exits immediately.
