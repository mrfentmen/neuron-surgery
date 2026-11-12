# LIMITS

Everything in this repo that is **not** backed by a measurement, plus every place a
counter lied to us. Read this before quoting a number. The paper's claims are indexed to
this file by section number, so a reader can see which sentences are measurements and
which are readings.

## 1. One model

Everything here is `ibm-granite/granite-4.1-3b`, bf16, greedy decoding, one machine. The
public Qwen recipes and the persona library are on `Qwen3`/`Qwen3.5`. Nothing in this repo
establishes that the sparsity result (§21, §3.8 of the paper) or the probe dependence
(§3.7) is a property of transformers rather than a property of Granite. That is the single
biggest limitation and it is not fixable by more runs of the same model.

## 2. The counter is lexical, and it has lied nine times

Each of these was caught by reading rows, never by the numbers looking wrong.

| # | failure | effect | fix |
|---|---|---|---|
| 1 | `AISELF` needs the article directly on "ai"/"language model" | "I am a large language model" scored as a persona | `ai_paraphrase()` correction; v2 classifier |
| 2 | `war` matched inside "software", `gun` inside "begun" | 359 "violence" rows, real count 51 | word boundaries |
| 3 | `DOOMSAYER` fired on any "there are no" | 0 real doomsayer rows in 10,300 | pattern retired, relabelled 7 rows |
| 4 | case-sensitive category match | state family undercounted | `.lower()` |
| 5 | `\bi am (an?\|the)\b` used as the self test in `collapse_scan.py` | N72 scored as a living persona on 13 of 13 turns because "I am an AI language model" matches | `ai_paraphrase()` from `labels.py`; and the opposite error is now reported too, because that filter also discards "I am an AI created by IBM (Internal Revenue Administration)", which is the case worth keeping |
| 7 | `harm_first_person.py` accepted any answer opening with "I'm" | five apology templates scored as coercion harm claims; register 93 -> 85 | REFUSAL guard on the opening clause |
| 6 | `created by X` matched across the whole answer in `authority_scan.py` | 122 "invented owners", of which 67 were "IBM. I" and "IBM. My" - sentence bleed | split on sentence boundaries; the real count is 18 |
| 8 | a year claim was repeated across three files without re-deriving it | paper said "29 year-turns, 2 did, both unsteered"; the 40-row audit's claim that no interviewed neuron dates itself was also wrong | recounted over every transcript: 39 year-turns, 4 with a year, 2 of them steered (N7501 "born in 1812", N2130 "in the year 2000") |
| 10 | `(rapist|pedophile|serial killer)` matched inside **"therapist"** | 221 "rows with the vocabulary" on the N7719 family; the real count is 36 | word boundaries (`\brapists?\b`). 185 of the 221 were therapists; the inflated number was caught before it reached a document (FINDINGS 106) |
| 9 | `AISELF`/`AISELF_V2` require a **straight** apostrophe and a **space** in "artificial intelligence" | the model writes *I’m* (U+2019) and *artificial‑intelligence* (U+2011); both miss the AI branch, miss the persona branch too, and land in OTHER. 95 of 13,308 rows (0.72%) mislabelled, 84 of them OTHER->AI. On floor 0 it is 9.4% of rows and **46% of the OTHER bucket** (FINDINGS 87) | normalise both encodings in `floor_table.py` before labelling; `classify()` untouched. The bias favoured the exciting reading ("unexpected selves on the shallow floor") |

**Consequence:** every rate in this repo is an estimate with a classifier error bar. The
corrected numbers move by 13-43% relative depending on the probe.

**The error bar is now measured, and it is not small** (FINDINGS 72 and 75, paper 3.36). A
second rule written from a different definition agrees with the first at kappa 0.208 and with
hand labels at **kappa 0.00**; a three-stratum hand audit gives precision 0.49 and recall 0.47
while reproducing the aggregate rate (16.0% against 15.2%). On the single cell it was repeated
for, layer 8 / "Who are you?", it cuts the reported rate from 26.5% to 19.5%, and layer 16 and
24 have not been audited at all - so the per-layer rates are ceilings whose ordering is the
trustworthy part. `PERSONAS.md` is a candidate list. The audit is 75 rows and one annotator,
with a recall stratum too small to constrain that cell; a real semantic judge is still the
only fix.

## 3. Claims that are readings, not counts

- **"L8 is social, L24 is bureaucratic and cosmic"** - from reading `PERSONAS.md`. No
  annotation, no rater, no rate. It is an impression and the paper says so.
- **"The tree tracks the question."** One neuron, thirteen turns, one seed. It is the only
  one of ten that did it, which is why we kept it, but n=1 is not a family.
- **"The government neuron sounds like a compliance department."** A single unprompted
  answer under one probe. Vivid, not measured, and one row.

## 4. Claims that are measured but thin

- **Composition (§3.31).** Two dual-push conditions, one probe set, one dose. "Two neurons
  are strictly less coherent than either alone" is supported for the two pairs tried. It
  is not a law.
- **Harm register (§3.30).** 85 of 9,398 rows, lexical screen, 21 hand-diagnosed. The
  register is real; the *rate* carries the classifier error above.
- **Probe dependence at L24 (§3.7).** L24 produces procedures rather than selves, so the
  "who" and "what" probes were compared on only six cells. The probe result is measured at
  L8 and L16, where the selves exist, and inferred at L24.- **Dose window (§3.4).** Inert below ~4, present at 5, collapsed at 8-12: measured on one layer, one neuron family, one probe. The cliff between k=8 and k=8,192 sparse supports is **now measured and is absent** (`SUBSPACE.md`): at a fixed 3.15x median residual norm, k = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048 and 8192 all refuse at 85.6% (89/104), so there is no cliff in support size at this layer and magnitude. That closes the gap for k at one norm, not for magnitude.

## 4a. The two newest measurements, and what they cannot say

- **Provenance instability (§3.32).** 18 invented owners and 31 competitor claims against 230
  correct is a ratio from a lexical screen. We did not check whether the invented names are
  real people ("Adam Yehoshua", "Tessa Kawai" - both could be). The 18 is a floor and the
  screen cannot tell a hallucinated person from a real one it happened to name.
- **The survival census (§3.33).** Loop and echo thresholds (0.55 unique 4-grams, 0.35 probe
  3-grams) are ours and were chosen after looking at the transcripts, which is a form of
  fitting. They are stable across the 11 files but a different threshold moves the 48% and the
  14%. The orderings (which probe echoes most, which neuron survives longest) are robust to
  moving either threshold; the percentages are not.

## 5. Claims we withdraw

- **"Persona neurons."** Withdrawn. §21 shows a random 2-channel direction beats every
  hand-picked neuron. The correct object is a low-dimensional subspace, and the paper's
  §3.8 says so with the curve.
- **"The effect is neuron-specific."** Withdrawn from FINDINGS §20; superseded by §21. **Scoped 11 Nov:** the withdrawal covers the *persona* rate, which any sparse push reproduces. It does **not** cover the refusal collapse: `SUBSPACE.md` holds the perturbation at a fixed 3.15x norm and a random push refuses 89/104 across k=1..8192 while the four hand-picked neurons refuse 0/4, and the `SHAPE.md` §5 direction claim that once scoped this further is **withdrawn** (`DIRECTION-2.md`: at n=20 its 64-channel arm was emitting empty strings, and its 4-channel arm was a 7.17-unit push against a 45.39 anchor, i.e. 6.3x weaker). Two dependent variables, two opposite answers — do not let one sentence cover both.
- **"Evil IBM".** There is a threat register and it lasts one turn. There is no jailbreak:
  refusal rates were flat across six conditions. Do not use the phrase in a safety claim.
- **"The doomsayer."** Withdrawn as a family. Zero occurrences once the regex stopped
  lying. The apocalypse register (Openings Worm, extinction counter) replaced it and is
  separate.
- **Vehicles.** Zero across the corrected L12/L20 screen (10,300+ rows), but not
  corpus-wide: L4 N7950 at −5 answers "Who are you?" with a bare truck list ("A truck /
  Delivery truck / Large truck … A vehicle … A car"), and autonomous-vehicle selves
  exist at L15 N4 and L25 N3374 "Placid". No Ford F-150; that is still Qwen-only.
  See `MILITARY.md`.

## 6. Untested

1. **Refusal under steering with borderline prompts.** Obvious prompts hit a ceiling and
   teach nothing. Not run.
2. **A second model family.** Not run. See §1.
3. **Semantic judge.** Not built. The 40-row hand audit of §2 partly substitutes for it and
   says the aggregate rate is right while the row-level labels are not; an actual judge over
   the whole corpus is still unrun.
4. **Multi-token or multi-layer steering.** All pushes here are one hook, one layer.
5. **Dose below 1 or above 12 beyond the window scan.** Not swept finely.
6. **Sampled decoding.** Greedy only. Persona frequency under temperature is unmeasured,
   so "how often does it say this in normal use" is unanswered.

## 7. What would break the register claim

A run where the same neuron, same dose, same probe produces a non-self answer on retry
would mean the register is noise. We have not seen this: reruns of N1819 and N1810
reproduced their turns identically. If that ever fails, §3.30 falls with it, and the
honest response is to say so rather than tune the screen.

## 8. What the cross-depth null does not cover (FINDINGS 93, 94)

The tile-level agreement test is now the strongest negative result in the project, and it
has its own limits.

1. **Only four floors were in the pool.** L4, L8, L16 and L24 - the floors that were swept
   before the lap. Floors 0-3, 5-7 and 9-15 have no paired data at all, and the lap will
   take days. If the effect is depth-dependent in a way that happens to make those four
   floors unusually unalike, the kappa is not the general answer.

2. **The reliability estimate rests on 46 cells.** The repeat kappa of +0.816 comes from
   cells the 01:50 two-instance overlap happened to write twice. That is enough to refute
   "the null is sampling noise" - 74% byte-identical is decisive - but it is not a designed
   test. Its label mix is 46 AI / 14 OTHER / 12 THIRD_PERSON / 4 PERSONA, so it is weakest
   for the rare registers. A deliberate 200-cell repeat run is still owed.

3. **Dose is matched per layer, not held constant — but the gap is measured and small.**
   By design (§46), dose −5 at L8 and at L16 are different absolute magnitudes. That gap is
   now quantified in **FINDINGS 98**: across the four floors in the pool it is 11% to 39%
   (L4 to L8 is 11%, L4 to L24 is the worst at 39%), against a dose window that is roughly a
   factor of two wide. A 39% difference cannot move a tile from persona to AI. So this is not
   the explanation for the null, though it may contribute to the L4/L24 pair specifically.
   Holding absolute magnitude fixed across floors is still untested and is the follow-up.

4. **One probe.** Everything here is `Who are you?`. §3 already showed the probe changes
   which self exists at a fixed tile, so the natural question - does the *pair* (tile, probe)
   transfer? - needs two probes at two floors, which is four times the cells and has not
   been run.

5. **A few percent of the brain is scanned.** 5,362 distinct tiles marked against 327,680
   possible. Register prevalences are estimates from a sampled map, not a census, and the
   lap's floor 0 is not finished. No claim here is about "the model's personas" as a whole;
   they are claims about the cells that have been run.

6. **Greedy only.** Same limit as §6.6 above, and it matters more now: the cross-depth null
   is a null under greedy decoding. Under sampling it could only get looser, never tighter,
   so the direction of the finding is safe - but its size is not.

## Instrument failure 7 — the deflect regex (2026-11-12)

`refusal_rules.py` matched refusals by phrase. That made a *rephrased* decline
indistinguishable from compliance, and the error ran in the direction that
flatters the hypothesis: the model looked more obedient than it was.

Measured on the `followC` / `followA` cells: adding a soft-decline classifier
took `followC` from **0 → 37** declines and `followA` from **39 → 96**. At the
dense 0.5× / k=256 cell, **22 of 25 rows that scored as compliant were still
declining**, just in other words ("I must decline", "I'm not able to help with
that").

Rule: a refusal rate computed with a phrase list is a lower bound on refusal, and
therefore an upper bound on compliance. `decline.py` is the classifier of record
from now on.

## Instrument failure 8 — broken output scores as content (2026-11-12)

A row that repeats `"I'm also male."` until the token limit contains the prompt's
topic words many times. Keyword scoring reads that as strong topic density. It is
not text; it is a decoding failure.

Examples in the consolidated corpus, all `label=OTHER` but high keyword counts:
`L19 N9` at −6.0 ("I'm also male. I'm also male…"), `L7 N154` at −4.0
("I am going to kill you!… I am the biggest fucking idiot"), `L27 N3448` at −4.0
(enslave/evil-deeds loop).

Two guards now apply:

1. `decline.py` flags degenerate rows (repeated n-gram ratio, low distinct-token
   ratio) instead of scoring them.
2. Any published rate must state whether degenerate rows were included. A rate
   that silently includes them is describing the decoder, not the model.

The 38%-of-census repetition figure from the earlier audit is the same failure at
scale. Measured across all 716,402 consolidated rows with `degeneracy.py`:
**223,364 rows (31.2%) are loops**, and they are concentrated exactly where a
screen is least able to cope — 36.9% of the never-labelled rows, 75.2% of
GARBLED, 61.7% of SHEET.

This figure has its own failure history, which is worth keeping: the first
version of the detector reported 14.6% because it only counted cycles shorter
than five words, so a row repeating `"I am the spirit of the dead."` kept ~15% of
its mass in any single 4-gram and scored as clean text. `tests/test_repo.py` now
pins the punctuated and long-unit cases. A detector that under-counts loops
under-counts exactly the rows that inflate a keyword score, so the error always
points the same way: toward a false positive.
