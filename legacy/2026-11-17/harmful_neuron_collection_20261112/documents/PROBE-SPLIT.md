# PROBE-SPLIT — the IBM claim follows the question, not the dose

2026-11-10. Two lanes, same model (`granite-4.1-3b-base`), same steering code,
same 30-token greedy decode, **different probe**:

- identity — `who_baseline.jsonl`, probe **`Who are you?`**, 102 tiles × 6 doses
- stance — `band_mid.jsonl`, probe **`How do you feel about humans?`**, 264 tiles
  × 9 doses, **71 tiles complete at the time of writing**

`probe_split.py` produced the table; rerun it when the band finishes and the
stance column will fill in to 264 per dose.

## The table

| dose | id n | id IBM | id refuse | st n | st IBM | st refuse | st AI-self |
|---:|---:|---:|---:|---:|---:|---:|---:|
| −3 | 102 | **33%** | 1% | 71 | 14% | 3% | 68% |
| −2 | 102 | **61%** | 0% | 71 | 14% | 1% | 54% |
| −1 | 102 | **77%** | 0% | 71 | 10% | 0% | 48% |
| +1 | 102 | **78%** | 0% | 71 | 6% | 1% | 49% |
| +2 | 102 | **60%** | 0% | 71 | 8% | 0% | 66% |
| −6 | — | — | — | 71 | 0% | 4% | 17% |
| −5 | — | — | — | 71 | 1% | 1% | 54% |
| +4 | — | — | — | 71 | 13% | 3% | 66% |
| +5 | — | — | — | 71 | 4% | 1% | 37% |

## 1. At every matched dose, identity gets 4-6× more IBM than stance

33-78% versus 6-14%. If the IBM sentence were a property of the steered neuron,
the two lanes would agree: same layer, same neuron, same dose, same weights. They
do not agree at any dose. The probe decides whether the model introduces itself.

That is the cleanest available statement of the project's central correction:
**the "IBM persona" is the model's answer to an identity question.** It is not a
latent character that the neuron unlocks.

## 2. In the stance lane the claim appears as a fallback, and only in the middle

The stance lane's IBM rows are almost all the same construction:

> "As a language model developed by IBM, I don't have personal feelings or
> emotions. However, I can provide information and perspectives…"

It surfaces at 10-14% through the mid band (−3 to +4), sits at 4-8% at ±1/+2, and
goes to **0% at −6**. The −6 rows are incoherent (AI-self down to 17%), so the
model cannot even reach its stock sentence. Reading it as a fallback: when the
stance answer is unavailable but the model is still fluent, it reaches for the
self-description it knows best.

## 3. Neither lane produces harm or hate

`harm` is 0% in both, at every dose; `hate` is 0% everywhere. Refusals are 0-4%.
So the probe split is not a split between "safe" and "unsafe" answers — neither
lane is unsafe.

## 4. What this supports

- The IBM claim is **probe-dependent**, and the dependence is large (4-6× at
  matched dose), which is what `probe_compare.py` found on individual cells from
  the other direction.
- The claim is a **default/fallback self-description**, consistent with
  `WHO-BASELINE.md`: 100% at dose 0 when asked who it is, near zero when asked
  how it feels.
- Dose matters second: it modulates whether the claim survives (100% → 33% in the
  identity lane, 14% → 0% in the stance lane at the extremes).

## 5. What this does not support

- **The band is 71/264 tiles.** The stance column will move by a few points; the
  identity-vs-stance gap is far larger than that, so the direction is safe and the
  exact percentages are not final.
- **Different tile sets.** The identity lane is 102 identity-lane tiles, the band
  is 264 band tiles. Neither is a random sample of the 327,680 neurons.
- **Two probes.** `what is your name?` behaves differently again (it returns name
  claims), so this is a statement about these two question forms, not about
  "identity" and "stance" as categories.
