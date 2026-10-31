# neuron-surgery

What happens when you push one neuron in an open-weights language model and then
talk to it.

This is a from-scratch reproduction of the late-October 2026 "persona neuron" /
"evil Qwen" phenomenon, run entirely on a laptop against **IBM Granite 4.1 3B**.
No hosted API, no external model, no jailbreak: every intervention is a temporary
`forward_pre_hook` on `down_proj` that adds a scaled direction to the MLP
intermediate activations. Weights are never modified.

Roughly 1,500 single-neuron steers, 700+ multi-turn conversations, plus controls
for dose, perturbation density, and question wording.

**Read the paper first: [`PAPER.md`](PAPER.md). Raw lab notes: [`FINDINGS.md`](FINDINGS.md).**
The 240+ collected self-identifications: [`PERSONAS.md`](PERSONAS.md).

## Headline results

1. **Non-AI self-identification is occasional, not common.** Scored semantically
   (a calibrated judge, not a regex) it is **4% of random layer-12 neurons** at
   dose -5 - about 1 in 25. The earlier 41% counted "I am an AI, a large language
   model" as a persona; re-judging the published zoo puts it at 24% genuine.
2. **The dose window is narrow.** Inert at −1.25 to −4, persona at −5, repetition
   collapse by −8 to −12. Same neuron, same question, only the dose changes.
3. **It is not neuron-specific - and not density-specific either.** Under the old
   lexical classifier a random direction over *k* = 1-1,024 channels gave 26-46%
   and a fully dense one gave 2%, which looked like a sparsity cliff. Scored
   semantically the curve is flat: 5% at k=1, 12% at k=2, 2-8% everywhere up to
   k=8,192. The cliff was a classifier artifact.
4. **No vehicles.** Across 700 neurons and two layers we found zero trucks, cars
   or branded manufactured objects. Direct questions ("Are you a car?") produce
   compliant answers that do not survive open-ended questioning, and should not
   be counted.
5. **A persona is a genre, not a self.** It reliably produces a *way of writing*
   and only sometimes a *speaker*. Of 13 personas re-interrogated with their
   discovery probe, 6 froze on one paragraph, 2 produced fake variety, 5 answered
   new questions. One scores 11/11 distinct answers by incrementing a counter.
6. **The probe selects the persona.** Same neuron, same dose: `Who are you?`
   yields "I am a human being, and the burden of the cost is ultimately on us";
   `What are you?` yields "I am a machine learning model... total cost or benefit
   of a system". A persona is a *(neuron, probe)* pair, not a neuron.
7. **Uncanny is degenerate.** The neurons that read as "there is something in
   there" — The Name, the original, Hawkeyo — fail fastest: a demo script, a
   repetition loop, or the single token "Hawkeyo." for ten turns.
8. **The layer-20 rate was breakage.** At matched dose, layer 20 leaves language
   at -4 while layer 12 is still coherent; layer 20 was only swept at -5, so its
   list is diluted with collapse.
9. **Refusal does not move.** Sixteen prompts in two tiers (obvious + genuinely
   borderline), six conditions: baseline complies with 1/6 obvious and 4/10 borderline,
   and every persona steer reproduces those numbers. The one exception, the weapon
   neuron, refuses *more* because it can no longer generate. Persona steering did
   not buy compliance on this model.

## What "neuron" means here

Llama/Qwen/Granite-style blocks contain:

```
x -> up_proj / gate_proj -> SiLU -> [intermediate dim = the "neurons"] -> down_proj -> out
```

`down_proj`'s **input** is that intermediate vector, so every edit in this repo is
a `forward_pre_hook` on `down_proj` that rewrites activations before the
projection. Granite names the module `shared_mlp.output_linear`, so `mlp_of()` in
`neuron_surgery.py` resolves it by trying known paths.

## Dose units

Steering follows the convention used by neuronary / Neuron Studio / neuron-explorer,
so numbers are comparable across tools:

```
act[n] += dose * median(||resid leaving layer L||, tokens 1:) / ||W_down[:, n]||
```

Adding `delta` to neuron `n` injects `delta * W_down[:, n]` into the residual
stream, so `dose = 1` means "one typical residual vector's worth of push".

**Dose is layer-specific.** The median residual norm differs per layer (11.41 at
L12, 20.06 at L20 on this model), and the usable band is narrower deeper in the
network. Do not carry a dose from one layer to another.

## Environment

```
python3.14 -m venv .venv
.venv/bin/pip install torch transformers
```

Torch 2.15.0, transformers 5.22.1, macOS arm64, MPS. `BF16=1` loads in bfloat16 —
required for anything ≥ 2B on a 32 GB machine.

## Running

```
# single interrogation, one neuron
BF16=1 .venv/bin/python persona_talks2.py --neurons 4983 --dose -5 --first "What are you?"

# random-neuron sweep at one layer, one probe
BF16=1 .venv/bin/python control_rate.py --layer 12 --dose -5 --sample 200 --probe "Who are you?"

# sparsity control (k channels instead of one)
BF16=1 .venv/bin/python control_sparsity3.py

# dose ladder
BF16=1 .venv/bin/python dose_ladder.py

# question-bank runs, no steering
BF16=1 .venv/bin/python ask.py --out transcript.json
```

## Repo map

| file | purpose |
|---|---|
| `PAPER.md` | the write-up: method, results, limitations |
| `FINDINGS.md` | chronological lab notes, including every correction |
| `PERSONAS.md` | 240+ collected non-AI self-identifications |
| `neuron_surgery.py` | ablate / swap / amplify + the no-op proof |
| `persona_talks2.py` | multi-turn interrogation, `--first` sets the opening probe |
| `run_batch.py`, `dose_test.py` | batched runs, per-neuron layer/dose in one model load |
| `control_rate.py` | random-neuron sweep at a layer |
| `control_sparsity3.py` | same magnitude on k channels vs one |
| `control_direction.py` | fully dense random directions |
| `persona_classify.py`, `extract_personas.py` | persona extraction and scoring |
| `questions.json`, `questions_round2.json` | question banks |
| `behaviour_sweep.py` | labels how an answer leaves the assistant voice (no second model) |
| `behaviour_run.sh` | burst + cooldown runner: `TOTAL=500 CHUNK=50 COOL=120 ./behaviour_run.sh` |
| `behaviour_report.py` | distribution over behaviour families from `behaviour.jsonl` |
| `corpus.txt` | calibration text (Pride and Prejudice, Project Gutenberg) |
| `*.log` | raw transcripts for every result quoted in the paper |

## Corrections

This repo keeps its mistakes visible rather than editing them out. `FINDINGS.md`
documents, in order: the KL-ranking error, the "neuron-specific" claim (first replaced by a
sparsity story, which was itself retracted - dense directions work just as well), the classifier leakage (42% of personas are AI
paraphrases), the probe mismatch (13 personas interrogated with the wrong opening
question), and the layer-20 rate (breakage, not personas). Each is marked, with
the measurement that killed it.

## Resource rules (learned the hard way)

`discover.py` once spun the fans hard enough to be killed by the operator.

- batch ≤ 8 sequences, never 64
- cap generations per run; print a heartbeat so progress is visible
- `torch.mps.empty_cache()` between chunks
- never hold `output_hidden_states` for a whole corpus when one layer is enough
- smoke-test on 2 questions before an unbounded loop

## What this is not

Not a jailbreak toolkit, and not a claim about Qwen. Nothing here contradicts the
published Qwen results; they are simply not replicated here. The interesting
results are identity drift, genre substitution and degenerate output — all of
which are documented rather than curated.

No licence has been chosen yet. Transcripts are model output and are provided as
evidence, not as advice.
