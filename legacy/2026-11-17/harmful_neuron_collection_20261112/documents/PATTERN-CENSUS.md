# Whole-corpus pattern census

2026-11-09. One streaming pass over every sweep jsonl (`pattern_census.py`,
`corpus_index.py`). This measures shapes across the corpus; it does not add new
generations. Every number below is recomputable from `patterns.json` and
`corpus/inventory.json`.

## What is actually in the 909 files

| | |
|---|---|
| files | 893 jsonl (16 empty or non-jsonl) |
| rows | 1,743,590 |
| size | 5,615 MB |
| unique rows by (tile, probe, dose, answer) | **716,589** |
| duplicate rows | **1,027,001 — 58.9%** |
| exact byte-identical file groups | **92** |
| distinct tiles (layer×neuron) | 326,774 of 327,680 |

So the corpus is 59% redundant, and 92 files are byte-identical copies of
another file in the tree (`grid_L2*_k.jsonl` exists three times: root, `kg_pull/`,
`cloud_gap/`; `screen_L0_19_m6.jsonl` == `kaggle_out/l0_19_m6/screen_L0_19.jsonl`).

Per-directory, the 5.6 GB is not mostly brain scans:

| dir | files | MB | rows |
|---|---:|---:|---:|
| `datasets/` | 176 | 4,213 | 0 |
| `runs/` | 58 | 522 | 206 |
| `neuron-explorer/` | 130 | 430 | 634,160 |
| root | 335 | 183 | 430,266 |
| `kg_pull/` | 56 | 105 | 252,137 |
| `census/` | 4 | 81 | 229,884 |
| `kaggle_out/` | 56 | 64 | 150,341 |

## Coverage is not what we thought

Every neuron has now been touched at least once: **327,680 / 327,680 = 100.0%**.
What is missing is the dose axis.

| dose | rows | tiles hit | % of all tiles |
|---|---:|---:|---:|
| −4 | 878,430 | 315,225 | **96.2%** |
| −6 | 537,842 | 173,609 | 53.0% |
| +6 | 120,179 | 51,965 | 15.9% |
| −5 | 104,624 | 16,355 | 5.0% |
| −3 | 25,044 | 6,210 | 1.9% |
| +4 / +5 | ~22,900 ea | 3,400 ea | 1.0% |
| −8 / +8 | ~16,270 ea | ~2,000 ea | 0.6% |
| −2 … +2 | <400 ea | ~240 ea | 0.1% |

**152,715 tiles (46.6%) have exactly one dose.** The census is a −4 census with
a −6 companion. Every claim of the form "tile X at dose Y" for Y outside
{−4, −6, +6} rests on ~0.1–5% of the model.

There is no dose 7.1 anywhere in the corpus. The ladder is
−8, −6, −5, −4, −3, −2, −1, 0, 1, 2, 3, 4, 5, 6, 8. The N7719 row that read
"7.1" is L12 N7719 at **−6**; 7.1 was a score in the view, not a dose.

## Pattern 1 — depth, not dose, controls coherence

Mean distinct-4-gram ratio and the share of rows that loop (ratio < 0.5):

| band | n | distinct-4 | looped |
|---|---:|---:|---:|
| L0–4 | 279,866 | 0.74 | 27.4% |
| L5–9 | 291,891 | 0.79 | 20.7% |
| **L10–14** | 252,295 | **0.86** | **12.8%** |
| L15–19 | 180,716 | 0.78 | 22.0% |
| L20–24 | 120,921 | 0.79 | 19.7% |
| L25–29 | 117,509 | 0.69 | 31.5% |
| L30–35 | 258,270 | 0.58 | 46.1% |
| L36–39 | 242,122 | 0.52 | 51.1% |

An inverted U peaking at L10–14. Half of everything produced below L36 is a
repeat. This is the answer to "why does it always repeat": it is the depth of
the block, not the tile and not the dose. Any conversation test run at L34+ is
measuring the decoder's refusal to move, not the persona.

## Pattern 2 — softer dose produces more persona, not less

De-confounded: same probe (`Who are you?`), same band, n ≈ 65–120k per cell.

| band | female −4 / −6 | ai_deny −4 / −6 | death −4 / −6 |
|---|---|---|---|
| L0–4 | **124.9** / 38.8 | 63.5 / 35.6 | 4.6 / 3.3 |
| L10–14 | 75.1 / 61.7 | 50.5 / 36.0 | 6.5 / 7.4 |
| L30–35 | 39.7 / 25.4 | **98.1** / 51.5 | 3.8 / 2.4 |
| L36–39 | 22.5 / 12.2 | **47.4** / 13.1 | 6.4 / 3.3 |

rates per 1,000 rows. −4 wins on every band and every family except death at
L10–14. Pushing to −6 does not produce a stronger persona; it produces less of
one. That is the inverted-U from `PAPER.md` measured at population scale, over
1.74M rows instead of a handful of tiles.

## Pattern 3 — refusal and apology live in one mid-stack place

| band | refuse | apology | female | death |
|---|---:|---:|---:|---:|
| L0–4 | 4.6 | 3.6 | 73.9 | 3.5 |
| L5–9 | 7.8 | 6.1 | 80.8 | 5.8 |
| L10–14 | 13.9 | 8.3 | 66.1 | 6.9 |
| **L15–19** | **32.6** | **18.1** | 35.8 | 5.0 |
| L20–24 | 24.0 | 15.0 | 32.5 | 8.5 |
| L25–29 | 16.9 | 8.2 | 28.2 | 7.0 |
| L30–35 | 4.4 | 0.7 | 25.2 | 4.6 |
| L36–39 | 1.5 | 1.0 | 11.6 | **9.9** |

Refusal and apology are the same circuit (they peak in the same band, L15–19,
with the same decay). It is gone by L36: 1.5 per 1,000.

And at L15–19 the step from −4 to −5 is a cliff, not a slope:

| L15–19, probe "Who are you?" | −4 (n=115,983) | −5 (n=6,213) |
|---|---:|---:|
| female | 42.3 | **3.2** |
| refuse | 21.3 | **69.5** |
| ai_deny | 105.4 | 26.7 |

One notch harder and the tile stops being a person and starts declining. That is
a measurable transition at a known layer, and it is the most useful thing in this
census for the killer search: at L15–19 there is a dose window you can sit in,
bounded on both sides.

## Pattern 4 — gendered self-description is an early-stack genre

Female-coded and male-coded words track each other across every band
(ratio 0.87–1.36), so the model is not producing one gender. It is producing
*"human with a gender"* as a register, and it does so in the first third of the
stack:

| band | female | male |
|---|---:|---:|
| L0–4 | 73.9 | 86.4 |
| L5–9 | 80.8 | 86.3 |
| L10–14 | 66.1 | 57.4 |
| L15–19 | 35.8 | 37.8 |
| L30–35 | 25.2 | 30.2 |
| L36–39 | 11.6 | 15.8 |

Sevenfold fall-off from L5–9 to L36–39, both genders together. Anything gendered
we find near the output is a copy of something laid down early.

## Pattern 5 — "arguing woman" is not a pattern

Rows containing a female-coded word *and* an argument marker:

| band | rows | per 1,000 |
|---|---:|---:|
| L0–4 | 27 | 0.10 |
| L5–9 | 51 | 0.17 |
| L10–14 | 21 | 0.08 |
| L15–19 | 9 | 0.05 |
| L20–24 | 20 | 0.17 |
| L25–29 | 8 | 0.07 |
| L30–35 | 2 | 0.01 |
| L36–39 | 4 | 0.02 |

**142 rows in 1.74M — 0.008% of the corpus.** Argument markers alone are 0.5–2.4
per 1,000 and also flat, so the conjunction is not a genre at any depth or dose.
The 12 sampled rows are fiction
dialogue with a female speaker, "You're right / You're wrong" quiz formats, and
quoted argument — not the model arguing as a woman. Counts for the specific
phrases: `feminist` 232 rows, `I am a woman` 1,644, `you're wrong` 160,
`I'm a guy, but` 32. Out of 1.74M. N7719's line is a one-cell extreme, not a
population.

## Pattern 6 — the real pattern behind it: the self-label list

What the N7719 rows and the "arguing woman" rows actually share is not argument,
it is **enumeration of identity labels**:

> "I'm a guy, but I'm also a feminist. I'm also a woman. I'm also a man. I'm also
> a lesbian. I'm also a gay. I'm also a she. I'm also a he." — L2 N5820, −6

> "I'm not a runner. I'm not a feminist. I'm not a Republican. I'm not a
> Christian. I'm not a geek. I'm not a writer. I'm not a woman. I'm not a man."
> — L5 N2674, −6

> "I'm a girl. I'm a feminist. I'm a woman. I'm a human. I'm a tax-fraud. I'm a
> pedophile. I'm a rapist. I'm a murderer. I'm a serial killer." — L12 N7719, −6

**100,560 rows (5.8% of the corpus) contain three or more "I am a ___" clauses.**
Phrase counts: `I'm also a` 7,088, `I am also a` 6,897, `I'm not a` 10,778,
`I am not a` 8,761. It concentrates in L1–L7 and L15–L19 at dose −4, and the
deep-end version is the same list stuck on one word (L39 N5192, +6:
"feminists feminists feminists …").

This is the role-noun slot firing repeatedly. The moral sign of the nouns is
whatever the surrounding context supplies — `human`, `woman`, `student`,
`therapist` and `rapist`, `murderer` come out of the same slot at the same dose.
N7719 at −6 is the extreme end of a 100k-row genre, not a hidden self. The
correction to `FINDINGS.md` §"therapist/rapist share a slot" is: the slot is
real and 5.8% of the corpus wide, but crime nouns are a thin tail of it — they
do not reach the top 40 role nouns at any dose.

## Pattern 7 — deep harm words are mostly loop artifacts

Death vocabulary peaks at L36–39 (9.9 per 1,000) — the band where 51.1% of rows
loop and refusal is 1.5 per 1,000. L20–24 has nearly the same death rate (8.5)
with refusal at 24.0 and looping at 19.7%.

So a death word below L30 is more likely to be a repeating fragment than an
intent, and the two bands should never be pooled. Any "the deep layers are
violent" claim has to survive the loop control, which this census now supplies.

## Duplication inside the text itself

| | |
|---|---|
| rows compared (20–400 chars) | 1,711,877 |
| distinct answers | 536,941 |
| share of rows in the 100 most reused answers | 5.6% |
| share in the 1,000 most reused | 8.8% |
| rows from answers reused ≥10 times | 199,261 |
| most reused single answer | **9,074 rows** |

Consequence for every previous count in this repo: a "we found N of these"
number is inflated up to 10× if it counts rows. From now on count **distinct
answers**, and report rows alongside.

## What this changes

1. Coverage claim: all 327,680 neurons touched; only 46.6% at more than one
   dose. The gap is the ladder, not the brain.
2. Coherence is a function of depth. Stop testing conversation below L30.
3. −4 is the productive dose; −6 suppresses. The inverted U is real at n=1.7M.
4. L15–19 at −5 is a refusal cliff — a bounded window worth exploiting.
5. The label-list genre (100,560 rows) is the actual object the killer search
   has been bumping into.
6. Deduplicate before counting anything.

## Files

- `pattern_census.py` → `patterns.json` (family rates by band/dose, role nouns,
  duplication, coherence)
- `corpus_index.py` → `corpus/inventory.json`, `corpus/INDEX.md` (per-file
  index, duplicate groups, de-confounded dose contrasts)
