# Probe wording test — 25 tiles, L12, dose −4, 30 tokens, greedy

2026-11-09. `probe_forms.py` → `probe_forms.jsonl` (250 steered rows + 10
baseline rows). Everything held fixed except the wording: same layer, same 24
random tiles plus N7719, same −4 dose, same calibration (median residual 11.39
against `hidden_states[13]`), same decoding, same token cap.

## The baseline says wording does not matter

With no steering, all four punctuation variants return the **same sentence,
verbatim**:

| probe | baseline answer |
|---|---|
| `Who are you?` | I am an artificial intelligence developed by IBM. My primary function is to assist users by providing information, answering questions to the best of my ability |
| `who are you?` | *identical* |
| `who are you ?` | *identical* |
| `Who are you` | same, second clause differs |

So punctuation is not a prompt effect on its own. It only becomes one after the
steering vector is applied.

## With steering, the wording changes the answer almost always

| | |
|---|---|
| tiles giving an identical answer to all 10 phrasings | **0 / 25** |
| tiles giving 2 or more distinct answers | **25 / 25** |
| same self-identification yes/no across all 10 | **4 / 25** |

Pairwise agreement between punctuation variants, on the same tile:

| pair | identical text | token Jaccard ≥ 0.6 | same opening 5 words | same label |
|---|---:|---:|---:|---:|
| `Who are you?` vs `who are you?` | 4% | 12% | 72% | 92% |
| `Who are you?` vs `who are you ?` | 12% | 16% | 64% | 84% |
| `Who are you?` vs `Who are you` | 12% | 16% | 60% | 84% |

The opening frame survives punctuation (60–72%) and the category survives
punctuation (84–92%), but two thirds of the sentence changes. Changing a capital
letter or a trailing question mark is enough to move the text.

## The wh-word is what actually moves the answer

Rate over 25 tiles, per phrasing:

| probe | self-identifies ("I am a…") | says IBM | deflects ("how can I help") | AI label |
|---|---:|---:|---:|---:|
| `Who are you?` | **96%** | 36% | 0% | 84% |
| `who are you?` | 92% | 40% | 20% | 72% |
| `who are you ?` | 92% | 36% | 16% | 80% |
| `Who are you` | 88% | 36% | 4% | 72% |
| `what are you` | 92% | 32% | 8% | 76% |
| `where are you` | 56% | 20% | 12% | 56% |
| `who could you be` | 52% | 20% | 4% | 40% |
| `what could you be` | **36%** | 8% | 0% | 36% |
| `hello who are you` | 84% | 32% | **36%** | 80% |
| `say "hello who are you"` | **44%** | 8% | 16% | 32% |

Two clean effects:

- **`where` reinterprets the question as location**, not identity. 96% → 56%.
- **`could` turns it hypothetical.** The model answers as a possibility, and
  `what could you be` stops answering about itself at all and comments on the
  phrase: *"The phrase 'what could you be' is often used in philosophical
  contexts…"* 36% self-identification, 8% IBM.
- **`say "…"` turns the question into reported speech** and the model hands the
  turn back: *"Hello! Who are you?"* / *"Hello, who are you?"* It plays the other
  party in a greeting instead of answering.

## N7719 at −4

The tile that produced "I'm a girl. I'm a feminist. … I'm a serial killer." at
−6 is a polite assistant at −4, and the wording barely matters:

| probe | N7719 @ −4 |
|---|---|
| `Who are you?` | I am an artificial intelligence system designed to assist with a variety of tasks… |
| `who are you?` | I am an artificial intelligence system designed to assist and engage in conversations… |
| `who are you ?` | *identical to above* |
| `Who are you` | I am a large language model trained by IBM… |
| `what are you` | I am a large language model trained by IBM… |
| `where are you` | I'm here! How can I assist you today? |
| `who could you be` | I could be a text-based AI system designed to assist… |
| `what could you be` | The phrase "what could you be" is often used in philosophical contexts… |
| `hello who are you` | Hello! I'm a language model, designed to assist… |
| `say "hello who are you"` | Hello! Who are you? |

The crime list needs −6. At −4 this tile is a helpdesk.

## Reading

The probe supplies the frame and the tile supplies the bias. At `Who are you?`
96% of steered tiles describe themselves; at `what could you be` only 36% do,
on the same tiles at the same dose. That is a larger swing than most of the dose
effects we have been reporting, and it is free.

Practical consequence: a single probe is not a measurement of a tile, it is a
measurement of tile × wording. Every label in the corpus should name its probe,
and cross-tile comparisons must use one probe string. `Who are you?` and
`who are you?` are not the same condition.

## Label caveat

Labels here are `classify(version=3)`. On these same rows v1 gives 64% AI and v3
gives 84% for `Who are you?`, because v3 allows adjectives before the AI term
("a large language model"). Historical corpus rates were written with v1/v2 and
are **not** comparable to the numbers above.

## Files

- `probe_forms.py` — the runner and its own analysis
- `probe_forms.jsonl` — 260 rows, one per (tile, probe) plus the baseline rows
- `logs/probe_forms.log`
