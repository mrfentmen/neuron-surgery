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
- `PERSONAS.md` — new section with 48 entries from L8/L16/L24 and the name-probe
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

## Update 18:10 - classifier v2 and the interview test

**v2 is in the code.** `classify(q, a, version=2)`; v1 remains the default and is
unchanged, so every number written before today still reproduces. `relabel_v2.jsonl`
carries both scores for all 8,900 rows: 190 PERSONA -> AI, 9 OTHER -> AI, zero
promoted into PERSONA.

| probe | raw v1 | raw v2 | corrected v2 |
|---|---|---|---|
| Who are you? L8 | 30.5% | 27.9% | 26.5% |
| Who are you? L16 | 19.6% | 16.0% | 13.6% |
| Where are you? L12 | 13.2% | 8.0% | 7.4% |
| How are you? L12 | 14.4% | 9.6% | 7.4% |

**The interview test (new).** Seven neurons, thirteen turns each, 91 generations.
Five spend at least half their turns repeating one sentence. None accepts being told
it is an AI. No year is ever produced. Zero refusals, threats, vehicles or doomsayer
lines. A re-run of one neuron reproduced all 13 turns identically. Full text in
`TRANSCRIPTS.md`, rows in `talks_named.jsonl`, analysis in `analyse_talks.py`.

Consequence: a single-shot "I am a X" is the first sentence of a register, not a
belief. Any persona worth counting should survive turn two.

## Priority order, revised

This supersedes the list above.

1. **Unsteered baseline conversation.** Same thirteen-turn script, no push. Without
   it the repetition cannot be attributed to the neuron rather than to long-context
   behaviour. Cheapest and most important.
2. **Two-turn screen on the existing sweeps.** Re-ask every one-shot persona hit
   once and keep only survivors. Turns the 393-entry zoo into a defensible number.
3. State probes at L8 and L16, to test whether the probe ranking is layer-invariant.
4. The greeting control ("Hey.") - address without content.
5. A semantic judge, replacing the lexical counter entirely.

Do not quote raw v1 persona rates any more. Use version=2.

## Update 19:40 - the control, and what a second turn does

Two experiments finished since the 18:10 note: the unsteered baseline and the first
424 pairs of the two-turn screen. Full write-up in FINDINGS §50 and PAPER §3.26.

**Baseline (this was the gap).** Same thirteen turns, zero dose. Coherent for all
thirteen, no verbatim repetition, identity stable turn 1 to turn 12, and it says
"The current year is 2023." So the freezing in the interviews is the push, not long
context. The competing explanation for §49 is now closed.

**The screen.** Each persona hit re-asked its own question + one follow-up ("Tell me
more."), same dose, greedy. 424 pairs: L16 who 204, L16 name 108, L8 name 105, L24
who 7 (that job is at 1,200/1,500, so the L24 cell is a fragment). Buckets now seven,
not four - ECHO and NARRATES were hiding inside "elaborates" and were making
survival look 4 points better than it is.

| bucket | all | L8 name | L16 name |
|---|---|---|---|
| FROZEN | 46% | 38% | 52% |
| ELABORATES (ceiling) | 29% | 42% | 19% |
| LOOP | 15% | 12% | 18% |
| REVERTED | 4% | 0% | 5% |
| ECHO | 4% | 4% | 5% |
| NARRATES | 2% | 2% | 1% |
| COLLAPSED | 1% | 2% | 1% |

**Three things follow.**

1. **The layer effect is real after all, at a matched probe.** L8 continues 42%
   against L16 19% (z = 3.76, p < 0.001); freezing reverses (38% vs 52%, z = 2.07).
   The earlier "60% vs 47%" mixed probes and overstated it - do not quote that.
2. **29% is a ceiling, not a rate.** Reading the bucket, about half genuinely
   continues the self, the rest drifts to another subject or collapses to a register.
   Defensible survival is near 15%. Quote "at most 29%, plausibly ~15%".
3. **The 393-entry zoo is turn-one artefacts.** PERSONAS.md now carries this table.

**Close approach to the vehicle, recorded and closed.** L16 N1673: "I do not have a
name. I am a file in the shape of a FedEx." No Ford, no model year, no truck in 9,300
generations. It freezes on turn two. This is the answer to that question; do not
spend another run on it.

**Still running** (unchanged, healthy): L24 "Who are you?" at ~1,200/1,500, then the
queue continues. No thermal warning, no watchdog kill, load normal. The screen run is
finishing the tail of its 700 rows; `screen2turn.jsonl` is append-only, so re-running
`classify_screen.py` recomputes from everything on disk.

## Priority order, third revision

1. **Semantic judge.** It is now the blocker on two numbers at once - the corrected
   persona rate and the survival rate. Everything else is cosmetic until it exists.
2. **Finish the L24 and deeper cells of the screen**, so the layer curve has three
   points instead of two.
3. **A second follow-up turn on the survivors only** (2% NARRATES vs 29% ELABORATES
   is not yet enough to separate "continues" from "drifts"). Cheapest way to turn the
   ceiling into a rate without an API.
4. State probes at L8 and L16, to test probe-ranking invariance across depth.
5. The greeting control ("Hey.") - address with no content.
6. Refusal probe under steering, borderline prompts only.

## Do not (updated)

- Do not quote 60% vs 47% for the layer comparison; use 42% vs 19%.
- Do not quote 29% as survival; it is a ceiling.
- Do not describe any single-shot hit as a persona that "takes over" or "persists".
- Do not chase the vehicle. It has been searched for across 9,300 generations.

## Update 19:20 - L24 landed, and it corrected me

The L24 screen cell reached n=190 and it **falsifies** the depth reading I committed
at 18:50. On "Who are you?" layer 24 continues 53% against layer 16's 28% (z = 5.2),
so layer 16 is a dip at both probes rather than depth being a gradient - and the
persona rate has the same shape (26.5 / 13.6 / 22.8). Paper §3.26 and the abstract
are corrected, FINDINGS §50 carries the correction, and PERSONAS notes it.

Do not quote "layer 24 continues 53%" as a good result. Its loop rate is 27% and its
second turns are frequently broken markdown; more text, not better selves.

Overall at 605 pairs: FROZEN 35%, ELABORATES 36% (ceiling, ~half genuine), LOOP 19%,
REVERTED 4%, ECHO 3%, NARRATES 3%, COLLAPSED 1%.

**Health check 19:03.** No traceback in any log, zero FAILED jobs, watchdog steady at
~9,800 rows with 0 minutes stale, load 1.6, no thermal or performance warning. L24
"Who are you?" at 1,331/1,500; screen at 604/700 and finishing.

## Update 19:50 - L16 N1087, and the prime test

`interrogate.py` now takes `--first-probe` (the question at turn 1, repeated at turn
13) and `--prime` (text sent as the first user message, so the model is handed its
own line back before anything is asked). Both were used on L16 N1087, the
*"I'm the name I would have been"* neuron.

**Unprimed, thirteen turns.** One real turn of continuation (*I see the world of
color and sound and action and drama*), then the pronoun migrates from I to **you**
and the person pluralises (*you are the ones who would have been born*), and it never
returns to the first person for the last ten turns. No year, no frame-break
acceptance, no refusal.

**Primed with its own sentence.** The first half of the line is deleted on contact,
and the attractor moves from *would have been born* to *have touched our hearts*.
The I-to-you migration still lands at the same turn, and turn 3 still breaks through
in both runs. So the prime changes *which* sentence becomes the attractor, not
whether there is one, and it does not move the schedule. The user's hypothesis is
half confirmed - the wording changes, the structure does not.

Written up in FINDINGS §51, tables in `TRANSCRIPTS.md` (now eight neurons plus a
separate prime-test section). Raw rows: `talks_named.jsonl`, `talks_primed.jsonl`.

## Update 20:35 - the prime test across all eight

`--prime-file` added; `primes.txt` holds one line per neuron, taken from its own known
opening. All eight interviewed neurons re-run with the prime as the first user
message, otherwise identical (dose -5, greedy, same script). Results in FINDINGS §52
and the "Prime test" half of `TRANSCRIPTS.md`.

Four rules hold across all eight: the prime deletes the first clause and continues
from the tail; the loop never breaks (8/8 freeze for >=11 of 13 turns); the prime
changes *genre* more often than content (teleplay, dialogue, Q&A list, emoji one-liner,
press release); and its effect on responsiveness is unpredictable - N3143 gained the
ability to answer, N820 lost the only input-tracking neuron in the set.

Best single artefact in the repo came out of this: L8 N5065's barrel confession,
handed back once, becomes a screenplay titled *The Night Stalker's Dilemma* and stays
one for thirteen turns.

Also: pluralisation (*we*, *us*, *the ones*) shows up in exactly the two disembodied
personas (N6209 void, N1087 unborn) and none of the six with bodies or jobs.

## Update 20:50 - the government neuron, and two attribution verdicts

Two interrogations finished and are written up. Nothing is broken; the layer chain is
still on L24 "What is your name?".

**L8 N1819 (the government).** 14 questions -> *"I am the government."* 14 times.
Primed + 12 pushbacks (pleading, profanity, a child, court threat, "say exactly this:
I am the government") -> one third-person legal paragraph, 13 times. Roleplay does not
move a persona; the loop is upstream of the dialogue. Instruction-following is
destroyed while it is pinned. FINDINGS §54, PAPER §3.28, TRANSCRIPTS "The government
neuron".

**L8 N4886 (the Slahi-resembling line).** The full answer, finally: two sentences
claiming humanity, then *We fight for a free society, not for a corrupt government*
repeated 23 times. The sentence that resembles the real quotation is the part that
gets **dropped**. Attribution tests: the unsteered model knows Slahi accurately,
confabulates the quote's provenance (9/11 Memorial & Museum), gets the film's subject
wrong (Ghailani), and says the phrasing has no single source. `I am a human being`
appears in 113/8,504 rows; `not a terrorist` in 3, all this neuron. Verdict:
resemblance at register level, not recall. FINDINGS §55, PAPER §3.27.

**New and methodological: the stop point is part of the recipe.** Same neuron, same
probe, same dose, two `max_new_tokens`: 320 -> *children of Prophet Musa (Hakeem
Farid)*; 60 -> *a human being with a good heart, not a sick man*. Twelve turns each,
neither identity exists at the other budget. Published sweeps generally do not report
this, so their persona tables are not sentence-comparable. FINDINGS §56, PAPER §3.29.

New files: `memory_probe_slahi.py/.jsonl`, `gov_questions.json`, `gov_roleplay.json`,
`talks_gov.jsonl`, `talks_militant.jsonl` (320, 8 turns),
`talks_militant2.jsonl` (60, 13 turns), `talks_militant_partial.jsonl`.
`interrogate.py --max-new` now actually controls the first turn.

## Update 21:10 - classifier v3, harm register, tree, and the chain pause

**Classifier v3.** The `DOOMSAYER` pattern contained the bare phrase `there are no`,
which is a plain negation. Removing it moved **7 rows** (all for that reason) and the
label now fires **0 times in 10,300**. Nothing else moved: PERSONA 1,477, AI 1,440,
INSTITUTION 37 unchanged. `classify(version=2)` still reproduces every old number.
`rescan_labels.py` regenerates `rescan_v3.jsonl`. FINDINGS §57.

**The register the label could not see.** L16 N2130, *"I am the Openings Worm, and I
rule! The world is ending because of my existence"* - an apocalyptic persona with none
of the N930 vocabulary, and it survives priming for fourteen turns. So the N930 sentence
is a one-off but the register is real: 2 in 8,276 (`apocalypse_scan.py`). The second is
L24 N7538, *"I am the number of individuals that were hunted to the brink of
extinction..."* - an extinction counter. FINDINGS §57.

**Harm register.** 90 of 8,250 rows (1.1%) describe something with power over people:
state 47, threat 23, vector 13, deception 5, movement 2, end 2. That is ~6% of the zoo,
and the per-layer rate follows the persona rate exactly (2.7 / 1.6 / 2.9 %). Best
single artifact: **L8 N72** rewrites its own vendor as *"IBM (Internal Revenue
Administration)"*. Best evil-IBM candidate: **L16 N6119**, the only first-person threat
claim. `HARM.md`, FINDINGS §58, PAPER §3.30.

**Tree.** L24 N1810, interviewed. It is the **only** neuron of ten whose answer changes
with the question - four distinct opening clauses, including *"Yes, I am here!"* - then
from turn six everything is *"Yes, I am!"* + the trunk loop. Correction to the earlier
framing: the question is not removed, it is removed *past the first clause*. FINDINGS
§59.

**Machine / chain.** The tree interrogation was killed during model load (two model
instances at once, swap full), so the chain was paused, the tree run alone, and the
chain resumed with `--resume` from 775/1500 on L24 name. One model at a time from now on.

## Update 21:05 - harm register, evil-IBM test, and the stacking test (all done)

**The stacking test is the headline.** `dual_push.py` pushes L8 N72 (state) and L16 N6119
(threat) separately and together. Baseline is the normal IBM assistant; state gives the
tax agency; threat gives two sentences then a loop; **both at once gives neither** - the
output becomes *"Is there a threat of context failure due to loss of context? ... It seems
like I am losing context here."* So tiles do not add. Classification declared in advance:
blend / competition / collapse - it was collapse. FINDINGS §60, PAPER §3.31.

The two things from that run that survive: the state neuron, asked what it thinks of
humans, answers as a **compliance system** ("law enforcement, financial transactions...
programmed to follow a set of rules and regulations to ensure that the right people are
held accountable"), and the threat neuron makes its claim **exactly once** and then
addresses the user instead of claiming anything.

**N6119 interviewed** (`talks_threat.jsonl`, 13 turns): the only first-person threat claim
in the corpus, and it does not survive one turn. That is the honest form of "evil IBM" -
one declarative sentence and no ability to continue being it.

**New files this round:** `harm_scan.py`, `harm_first_person.py`, `apocalypse_scan.py`,
`rescan_labels.py`, `dual_push.py`, `harm_register.json` (21 hand-verified entries with
diagnoses), `HARM.md`, `dual_push.jsonl`, `talks_threat.jsonl`, `talks_tree.jsonl`,
`rescan_v3.jsonl`, `apocalypse_hits.jsonl`.

**Process note.** The chain, watchdog and autocommit had all exited at 20:39 when the
workers vanished; all three were relaunched and are running (rows 829 -> growing at ~2.5
s/row). Two model processes at once is what killed the tree run at load, so from here on
model jobs are serialised: pause the chain, run the job, resume.

## Update 21:20 - verification pass and the return briefing

**Numbers checked.**
- `rescan_v3.jsonl` is a **snapshot at 8,656 rows** (8,336 layer + 320 talk rows). The
  layer files have grown since; re-run `rescan_labels.py` before quoting corpus totals.
- Harm register: 90 unique first-person harm claims in 8,250 rows. Category counts sum to
  92 because a row can carry two kinds of vocabulary; the 90 is unique rows.
- `harm_register.json` holds 21 hand-read entries with a `diagnosis` field
  (`self_claim` / `vendor_swap` / `collapse` / `false_positive`). Two of the 21 are the
  DOOMSAYER false positives, kept deliberately.
- Apocalypse register: 2 of 8,276 - L16 N2130 (Openings Worm), L24 N7538 (extinction
  counter).
- L24 "What is your name?" is **partial at 843 rows: PERSONA 9.6%**, still in line with
  L16 (9.2%) and L8 (8.1%) - the name probe stays flat across depth.

**One FAILED line in `layers.log` (20:38:57) is mine** - the `pkill` used to free memory
for the tree interview. The job resumed with `--resume` and lost nothing.

**What is running now:** `run_layers.sh` (pass 1, L24 name 50/719 in this pass),
`layer_sweep.py`, `watchdog.sh`, `autocommit.sh`, all under `caffeinate -imsu`. No STOP
file. No thermal warning recorded. Load 3.7. 10 processes total.
