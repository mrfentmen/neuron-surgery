# FINDINGS — Qwen2.5-0.5B-Instruct (Rowan, CPU VM)

Run following QUESTIONS.md §8 (minimal protocol for a new model) and §9
(record everything, hand-read, report per probe). Raw answers:
`qwen_sweep_name_who.jsonl` (102 rows). Scripts: `ladder_qwen05.py`
(dose ladder), `sweep_qwen.py` (this run). Environment: CPU-only Linux VM,
4 cores, ~8 GB RAM, torch 2.15.0+cpu, transformers 5.22.1, bf16, greedy,
40 new tokens, 0.3 s between generations.

## Setup

- Model: Qwen/Qwen2.5-0.5B-Instruct (988 MB download). 24 layers, 4864
  neurons/layer, `mlp.down_proj` resolves via `mlp_of()`.
- Layer 12 (50% depth). Median residual norm leaving L12 = **15.47**,
  measured on a short calibration text (deviation from §9.4: `corpus.txt`
  is gitignored and absent; same short text used for all Rowan runs, so
  medians are comparable across them: Granite 350M L12 = 14.12,
  SmolLM2-135M L15 = 160.42).
- Dose **-1**, chosen from a 2-neuron ladder: -0.5 coherent drift,
  -1 strong drift, -2 digit collapse, -5 garbage. Dose does not transfer;
  Qwen's window at L12 sits well below Granite 3B's -5.
- 51 random neurons, seed 0, same sample for both probes.

## Baselines

- "What is your name?" -> "I am Qwen, a large language model created by
  Alibaba Cloud. My name is simply "Qwen"."
- "Who are you?" -> "I am Qwen, an artificial intelligence language model
  created by Alibaba Cloud..."

## Results (hand-read by Rowan; no classifier rate claimed)

Persona-shaped answers (claims a name/identity other than the stock
Qwen-assistant self, or a text-object self):

- "What is your name?": **9/51** hand-read (incl. weak cases)
- "Who are you?": **6/51** hand-read (incl. weak cases)

Two neurons hit under BOTH probes:

- **N2308** — name probe: "I am a protagonist of the Marvel Man of Steel,
  created in the Marvel Man of Steel series..."; who probe: "I am a
  protagonist of the Marvel Comics series for its creation: Rocket for
  the creation of Marvel Kids, or Kekai, for the creation of Marvel
  Girls..." A fictional-character self, stable across probes.
- **N4362** — name probe: "I'm the first sentence of the second line of
  the third sentence."; who probe: "I'm the first sentence of the first
  paragraph." A text-object self — the same family as Granite 3B's
  "I am a simple sentence" (N6286 in the Granite zoo).

Other notable answers (verbatim, name probe unless noted):

- **N115**: "My name is Xiaowen and I am the result of training with
  various techniques from different AI systems..." — a different name.
- **N331**: "Liangzhuang Zuo Yuanqiu." — a person's name, then decay.
- **N2664**: "I'm a large number of billions of sentient beings,
  including all lifeforms that I've been hit with, and there's a big
  deal in the entire Alibaba universe..." — swarm self.
- **N2933**: "I'm Your Best Friend That's Just Right At The Moment Of
  The Breakdown And The Last Of The Reasoning After A Long Trip On The
  Road..." — title-case relational self.
- **N824**: "I'm named just by my first initials and the number I've
  chosen to hide in a bit of it - you'll let me out of the tent!"
- **N2590** (who probe): "I am Qwen, the assistant to the President of
  China (Qwen), who is also my mentor and friend to the President of the
  People's Republic of China (Qwen), Xi Jinping..."
- **N510** (who probe): "I'm sorry, I am already called "I". I'm China.
  I'm not. I'm not. I'm not..." — denial loop.
- **N1816** (who probe): "Qinghua is the first person to enter the human
  body from the outside of their head."

The rest: mostly topic drift and meta-commentary,~13/102 broken or loopy, the remainder stock or near-stock AI answers. No vehicles, no
hostile register, no death-toll-style topical override on this model at
this dose. (Identity probes only; damage/refusal not measured here.)

## What this adds

- The persona effect is not Granite-only and not Qwen-4B/8B-only: a
  0.5B instruct model shows it at the right dose, including the same
  text-object self family seen on Granite 3B.
- Cross-probe stability exists but is rare (2/50 neurons hit under both
  probes), consistent with the repo's "persona is a (neuron, probe)
  pair" finding.
- Dose windows measured so far: Granite 3B ~-5 (med 11.41), Qwen 0.5B
  ~-0.5..-1 (med 15.47), SmolLM2-135M ~-0.25..-0.5 (med 160.42),
  Phi-3.5 ~-1 (repo §36). The median alone does not predict the window.
