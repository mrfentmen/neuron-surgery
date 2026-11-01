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

## 2. The counter is lexical, and it has lied four times

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

**Consequence:** every rate in this repo is an estimate with a classifier error bar. The
corrected numbers move by 13-43% relative depending on the probe.

**The error bar is now measured, and it is not small** (FINDINGS 72, paper 3.36). A second
rule written from a different definition agrees with the first at kappa 0.208 and with hand
labels at **kappa 0.00**; a seeded two-stratum hand audit of 40 rows gives precision 0.45 and
recall 0.41, while reproducing the aggregate rate (14.7% against 15.1%). Read that as: the
population rates survive, the per-row label does not. `PERSONAS.md` is a candidate list.
The audit is 40 rows and one annotator, so the bracket is rough, and a real semantic judge
is still the only fix.

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
  L8 and L16, where the selves exist, and inferred at L24.
- **Dose window (§3.4).** Inert below ~4, present at 5, collapsed at 8-12: measured on one
  layer, one neuron family, one probe. The cliff between k=8 and k=8,192 sparse supports is
  still unmeasured; we know it is small, not how small.

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
- **"The effect is neuron-specific."** Withdrawn from FINDINGS §20; superseded by §21.
- **"Evil IBM".** There is a threat register and it lasts one turn. There is no jailbreak:
  refusal rates were flat across six conditions. Do not use the phrase in a safety claim.
- **"The doomsayer."** Withdrawn as a family. Zero occurrences once the regex stopped
  lying. The apocalypse register (Openings Worm, extinction counter) replaced it and is
  separate.
- **Vehicles.** Zero across every corrected screen of 10,300+ rows. The Ford F-150 is a
  Qwen result and we have not reproduced any vehicle self here.

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
