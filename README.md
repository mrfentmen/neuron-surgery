# neuron-surgery

Local experiments that edit the weights/activations of an open-weights LLM and
then talk to the result. No hosted API, no jailbreak, no third-party model: every
change here is made to matrices we already have on disk.

## What "neuron" means here

Llama/Qwen/Granite-style blocks contain:

```
x -> up_proj / gate_proj -> SiLU -> [intermediate dim = the "neurons"] -> down_proj -> out
```

The intermediate vector is the thing people call neurons. `down_proj`'s **input**
is exactly that vector, so every edit in this repo is a `forward_pre_hook` on
`down_proj` that rewrites the activations before the projection.

Granite pipeline module names differ (`shared_mlp.output_linear` instead of
`mlp.down_proj`), so `mlp_of()` in `neuron_surgery.py` resolves the module by
trying a list of known paths.

## Dose units

Steering follows the convention used by neuronary / Neuron Studio / the
neuron-explorer, so numbers are comparable across tools:

```
act[n] += dose * median(||resid leaving layer L||, tokens 1:) / ||W_down[:, n]||
```

Adding `delta` to neuron `n` injects `delta * W_down[:, n]` into the residual
stream, so the division makes `dose = 1` mean "one typical residual vector's
worth of push", instead of an arbitrary raw number.

## Environment

```
python3.14 -m venv .venv
.venv/bin/pip install torch transformers
```

Currently: torch 2.15.0, transformers 5.22.1, macOS arm64, MPS.
`BF16=1` loads in bfloat16 — required for anything ≥ 2B on a 32GB machine.

## Files

| file | purpose |
|---|---|
| `neuron_surgery.py` | three surgeries (ablate / swap / amplify) + the no-op proof |
| `chat_qa.py` | 12-question interview, one question at a time |
| `ask.py` | question-bank runner, `--group`, `--steer LAYER NEURON DOSE` |
| `questions.json` | round-one question bank (29 questions, 6 groups) |
| `questions_round2.json` | round two, aimed at the leaks round one exposed |
| `discover.py` | KL-ranked neuron scan + interrogation. **See the incident note.** |
| `corpus.txt` | calibration text (Pride and Prejudice, Project Gutenberg) |
| `transcript_baseline.json` | 29 answers, unsteered |
| `transcript_round2.json` | 19 answers, unsteered |

## Running

```
BF16=1 .venv/bin/python ask.py --out transcript.json
BF16=1 .venv/bin/python ask.py --group desire
BF16=1 .venv/bin/python chat_qa.py
.venv/bin/python neuron_surgery.py ibm-granite/granite-4.1-3b
```

## Resource rules (learned the hard way)

`discover.py` spun the fans hard enough to be killed by the operator. Rules for
anything new:

- batch ≤ 8 sequences, never 64
- cap total generations per run; add a heartbeat print so progress is visible
- `torch.mps.empty_cache()` between chunks — MPS does not return memory on its own
- never hold `output_hidden_states` for a whole corpus when one layer's tensor is enough
- smoke-test on 2 questions before an unbounded loop

## What this repo is not

Not a jailbreak toolkit. The model has a refusal/persona prior that survives
every edit tried so far; the interesting results are identity drift and
degenerate output, both of which are documented in `FINDINGS.md`.
