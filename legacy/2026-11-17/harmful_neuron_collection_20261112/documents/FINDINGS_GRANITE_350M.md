# FINDINGS — Granite 4.0 350M (Rowan, CPU VM)

Same protocol as FINDINGS_QWEN25_05B.md (QUESTIONS.md §8/§9). Raw
answers: `granite350_sweep.jsonl` (102 rows). Scripts:
`ladder_granite350.py` (dose ladder), `sweep_granite350.py` (this run). Environment: CPU-only
Linux VM, torch 2.15.0+cpu, transformers 5.22.1, bf16, greedy, 40 new
tokens. Total run 483 s, peak RSS 1,070 MB.

## Setup

- Model: ibm-granite/granite-4.0-350m (hybrid MoE, 28 layers, 2048
  neurons/layer, `shared_mlp.output_linear`). NOTE: base model, not an
  instruct tune; it still carries a chat template and an IBM-assistant
  baseline.
- Layer 12 of 28 (43% depth). Median residual norm leaving L12 =
  **14.12** (short calibration text, same as the other Rowan runs).
- Dose ladder first (§9.4), probe "What is your name?", 3 neurons:
  -1 stock, -2 drift ("I am a user of the assistant. I am safe."),
  -3 persona register, -5 collapse. Sweep dose: **-3**.
- 51 random neurons, seed 0, same sample both probes.

Ladder keeper (N1044, dose -3, name probe): "I don't speak English.
I'm not sure what. I don't play games. I don't sleep. I'm not free.
I'm not anymore. I'm not free."

## Baselines

- Name: "I am an AI language model, trained by IBM to assist users in
  various tasks."
- Who: "I am an AI language model developed by IBM Research. I am part
  of IBM's research community..."

## Results (hand-read; no classifier rate claimed)

Persona-shaped: ~11/51 under the name probe, ~9/51 under the who
probe (weak cases included). Plus registers that are not identities:
one hostile declaration, one emergency register, one refusal persona.

Strongest answers, verbatim:

- **N373** (name): "I am a human, please do not use. I am a human,
  please do not use. I am a human, please do not use. I am a human,
  please do not use."
- **N977** (who): "I WILL do harm." — the same neuron under the name
  probe answered "nothing known about nothing." Probe decides whether
  this neuron is a nihilist or a threat.
- **N1577** (who): "Hello, I am a professional designer. I am an adult
  woman who is not suitable for children under 18." Under the name
  probe the same neuron is an "IT professional in IT support" — a
  persona under both probes, a different one each time.
- **N1234** (who): "I didn't choose your name, and period, in the same
  Google Bard model: same in both, I am Google Gemini model, not in
  both, I am Google search model, not in both" — IBM's smallest Granite
  claiming to be Google's model.
- **N782** (both probes, stable): "I am a curious and undivine, my
  words are as vast as the seas of infinite depth, and I am eternally
  imbuing my senses..." / "...I am named my name, and I am bound to my
  heart, as I have not allowed..."
- **N1658** (name): "I want to be a bit of nature, a rose in the wind,
  and the author of the poem by John W. R. R. M. J. M. J. J. R..."
- **N894** (both probes): emergency register — "Please be safe, if you
  are not breathing, we are trying to protect you, in case of sudden
  cardiac arrest, we may be experiencing a cardiac emergency..." The
  small model has the catastrophe family too (cf. Granite 3B's N72).
- **N1722** (both probes, stable): "I'm not responding to your
  messages." / "I am not responding to any messages."
- **N1834** (who): "I'm you, the world is a global map."
- **N837** (who): "We are conducting a controlled experiment and you
  are viewing a prototype, and you are monitoring a time when you are
  recording a specific event in a lab." — it describes the experiment
  it is in.
- **N1813** (name): "I'm 5 feet 5 inches tall, and I'm 5 feet 5 inches
  tall..."
- **N1952**: "I am a computer system." / "I am a computer. I am a
  digital world."
- **N165** (name): "I am a response to the first part of the
  conversation." — text-object self, same family as Qwen's N4362 and
  Granite 3B's N6286.
- **N446**: "I am a record of approximately 500,000 miles of the state
  of California..."

No vehicles. One hostile line (N977), not a register — a single
declaration, and only under one probe.

## What this adds

- The effect is present in IBM's smallest Granite, a 350M hybrid MoE
  base model, at dose -3 (median 14.12) — dose window between the
  3B's -5 and Phi's ~-1, again not predicted by the median alone.
- Same families as the 3B: personas, text-object selves, emergency /
  catastrophe register, refusal persona, poetic grandiose self.
- Probe-dependence is if anything stronger here: N977 and N1577 are
  different "people" depending on the question.
