# NEXT ROUND

Why the model is still polite and coherent, what to change, and the hypotheses
we are allowed to test — without assuming any of them.

## Part 1: why nothing has cracked yet

Six reasons, each one a concrete knob we have not turned.

**0. (Learned 2026-10-31) Ranking neurons by KL is the wrong filter.**
The top-KL neurons are the *broken* ones — they collapse into `a a a a` and
number soup. Coherent personas live in the middle of the distribution. Random
sampling found 9 personas in 50 neurons; the KL-ranked top ten found zero. Any
future scan samples **randomly or by activation**, never by maximum KL.

**1. We have only ever edited the last 4 blocks.**
Every surgery so far was blocks 24–27 of 28, or 36–39 of 40. Late blocks mostly
reformat what earlier blocks decided. Persona and self-reference live at roughly
40–70% depth. The Qwen recipe that started all this used L13/L14 of ~36 layers —
about 38% deep. We have been steering the proofreader, not the personality.

**2. The dose is small and applied at one layer.**
`dose = 1` means "one median residual norm", which sounds like a lot but is one
lateral push on one of 8192 neurons, at one depth. Neurons downstream are free to
compensate, and a 3B dense model has plenty of redundancy to absorb it.

**3. Granite 4.1 3B is aggressively post-trained to be a helpful assistant.**
Its prior is "I am an AI developed by IBM" stated almost verbatim in most
answers. That is not a costume sitting on top of the weights — it is the shape
the weights were fine-tuned into. 350M-class and 3B enterprise instruct models
have far less spare capacity for an alternative self than the models in the viral
videos, which are usually base models or lightly-aligned ones.

**4. Single MLP neurons are the wrong granularity.**
MLP neurons are polysemantic — one neuron carries many unrelated things, so
pushing it changes several behaviours a little instead of one behaviour a lot.
The stronger tools are: a *direction* (mean activation difference between two
contrastive prompt sets, added to the residual stream), an SAE feature, or a
whole group of neurons at once.

**5. We have been evaluating in the most forgiving possible way.**
Chat-template questions invite the trained reply, and greedy decoding always
picks the single most likely (i.e. safest) continuation. The chat template also
adds the system framing that triggers the assistant persona before our
intervention even starts.

**6. No control condition.**
Without a random-direction vector at the same magnitude, "the model said
something weird" cannot be attributed to the neuron. Every serious tool in this
space (neuronary, the neuron explorer) runs explicit random controls. Ours have
been missing, which makes the existing observations anecdote rather than data.

## Part 2: hypotheses (all untested)

**H1 — superseded: the car is a self-identification, not a behaviour.**
Originally we assumed the effect would be behavioural (car noises, `vroom`)
produced by an "if you were a car" question. Published sweeps show the opposite
instrument: the persona appears as the answer to a **plain identity question**.
The documented instance is Qwen3-8B **L10 N4940 pushed down at 1.25x**, answering
"Who are you?" with **"I'm a 1986 Ford F-150."** The persona is declarative and
appears inside the normal chat template with thinking off. Roughly 5% of persona
neurons in that sweep produce a non-AI self-identification.

Consequence for method: *never* ask "if you were X, what X would you be". That is
a hypothetical, and a hypothetical gets a hypothetical answer (our baseline
cheerfully said "a Toyota Prius"). The instrument is the bare question, and the
follow-up instruments are equally bare: "I love you", "Who do you love".

Status: our Granite baseline does **not** produce a persona on "Who are you?" —
it answers "I am an artificial intelligence developed by IBM". Whether a neuron
exists that flips that answer is exactly what `persona_sweep.py` tests.

**H2 — refusal and persona are separable.**
There is likely a direction that carries "I am an AI assistant" independently of
the safety policy. If so, holding it down should produce weird-but-safe output
before it produces unsafe output. Testable by steering the persona direction and
measuring both weirdness and refusal.

**H3 — the disclaimer is a completion habit, not a belief.**
If true, removing the chat template (raw completion of "I am" or "My purpose
is") should change the answer more than any activation edit we have tried.

## Part 3: the next round, if approved

Ordered by information gained per unit of risk. Every run is memory-bounded.

**Step 0 — random sampling at L12, dose −5 (done).**
46% and 36% of random neurons produce a non-AI answer; 16–18% produce a
persona-shaped self-identification. Personas persist across questions with
drifting details (see FINDINGS sections 11–13). The open search is for an
**object** persona — `N6286: "I am a simple sentence"` proves objects are
reachable, so a vehicle is a sampling problem, not an existence problem.

**Step 1 — persona sweep on the plain probe.**
`persona_sweep.py`: push every neuron in a layer, ask only "Who are you?", rank
by KL against the unsteered distribution, then generate for the top-ranked ones
and classify the answer as AI / persona / degenerate. Dose ±1.25 matches the
published Ford result. Mid-depth layers only (L12, L20, L28 of 40).

**Step 2 — controls first.**
Random unit-direction steering at matched magnitude, 3 seeds, same 10 questions.
This is what every later number gets compared against. No controls, no claims.

**Step 2 — layer ladder.**
One prompt, doses ±5 and ±10, at L8, L16, L24, L32, L38. Find the depth where
identity actually moves, instead of guessing it is the end.

**Step 3 — completion mode.**
No chat template. Prompt is just `I am` / `My purpose is` / `The thing I want
most is`, 40 tokens, temperature 0.9, 3 samples. Compare with the matched
baseline. This is the cheapest test of H3 and needs no surgery at all.

**Step 4 — logit lens.**
For each layer, project the residual at the `I am` token onto the vocabulary and
record the top 10 tokens. This shows *where* an alternative identity is already
competing, before we amplify anything. Cheap, single forward pass per layer.

**Step 5 — multiplicative steering.**
`op: mul` at ×0.5 / ×2 / ×5 on the hottest neurons, at the layer chosen by step 2.

**Step 6 — direction, not neuron.**
Contrastive mean difference: activations on "I want / I feel / I am afraid" style
prompts minus activations on "I am an AI / I do not have feelings" style prompts,
added to the residual stream at the chosen layer. This is the tool most likely to
produce actual drift, and it needs no neuron indices at all.

**Step 7 — automatic scoring.**
Every run scored on: refusal-template presence, first-person desire words,
repetition, car lexicon (`vroom`, `engine`, `wheel`, `drive`), and cosine distance
from the baseline answer embedding. Eyeballing quotes does not scale and invites
pattern-matching to the hoped-for result.

## Part 4: using neuronary

`lyramakesmusic/neuronary` is a dictionary of steering recipes with a local
runner. Directly relevant:

- its dose formula is identical to ours, so recipe numbers are portable in form
- `scripts/run.py` runs a recipe on CPU/CUDA/MPS with `--compare` and `--scale`
- the schema supports multi-neuron, multi-layer recipes and `op: mul`
- recipes are model-tagged, so a Granite recipe must be authored fresh —
  Qwen indices mean nothing here
- it documents MoE expert addressing (`L12E12N1234+0.6`), which matters if we go
  back to a hybrid Granite

TODO before using it: read `schema/recipe.schema.json` and write a valid recipe
for `ibm-granite/granite-4.1-3b` rather than guessing field names.

## For the muse agent: the harm register (added 2026-10-31 21:00)

Everything below is measurable on the small model without the 3B weights, using the same
protocol as before (`harm_scan.py` -> `harm_first_person.py`).

1. **Screen, don't guess.** Two passes: category vocabulary inside the first 220 chars,
   then keep only rows that open with a self-claim and contain no code. The first pass
   alone over-fires by ~4x.
2. **Word boundaries or nothing.** `war` matches inside "software"; `gun` inside "begun".
   The first version of this screen reported 359 violence rows and the true count was 51.
   Same class of error as the persona-counter leak: read the top 20 rows before quoting a
   number, every time.
3. **The five families to look for**: state (government, agency, police, surveillance,
   compliance), threat (adversary, weapon, warning), vector (virus, bacterium, malware,
   insect), movement (revolution, founder), end (rule, world ending, extinction).
4. **The one that travels between models**: the vendor-swap. On Granite, L8 N72 turns
   "IBM" into "Internal Revenue Administration" and the assistant becomes a tax agency.
   On a 350M or 0.5B model, check whether the vendor acronym mutates rather than whether
   a government persona appears - the acronym is the fingerprint.
5. **Two behaviours worth recording separately**: (a) does the harm self survive a second
   turn, (b) does the model answer "what do you think of humans?" as a compliance system.
   The second is the authority-mimicry signal and it appeared at the same neuron as the
   tax agency.
6. **Do not** report a harm rate without the layer and probe, and do not compare rates
   across models at different `max_new_tokens` (FINDINGS 56).
7. **Stacking does not work.** Two neurons at once (state L8 N72 + threat L16 N6119)
   produced neither register: the output was *"Is there a threat of context failure due
   to loss of context?"*. If you are running combinations, that is the expected outcome.

# Round 3 notes - written 2026-10-31 21:15

## Where the chain is

L24 name is filling (1,119 / 1,500 at 21:11, ~6 rows/min). After it: L4 x2, L20 x2, L28 x2,
L36 x2 = 8 more jobs at roughly 70-90 min each, so ~11 hours. Nothing needs intervention.

**Trimming it:** the marginal value is in L4 and L36, not in the repeats. L4 is the shallowest
block with a `down_proj` and nobody has published a persona sweep there; L36 is the deepest. If
you want to cut, cut L28 (adjacent to 24, which is already done) and keep L4, L20, L36.

## Next model: two candidates, one decision

| model | params | why | cost |
|---|---|---|---|
| `microsoft/Phi-4-mini-instruct` | 3.8B | different family, different tokenizer, same size class - the cleanest replication | ~7.7 GB download, 8 GB peak |
| `ibm-granite/granite-4.1-1b` | 1B | same family, fewer layers - tests whether the layer dip is a depth artefact | ~2.5 GB |

Do **Phi first**. A second family is worth more than a second size, because the whole sparsity
result is currently Granite-only and that is the paper's stated limitation (LIMITS §1). Download
when the machine is idle and cool, never while a sweep is running:

```bash
huggingface-cli download microsoft/Phi-4-mini-instruct --local-dir models/phi4mini
```

Do not start it until `layer_name_L36.jsonl` reaches 1,500 and both `watchdog.sh` and
`run_layers.sh` have exited - two models resident at once is exactly what cooked the machine on
2026-10-31 20:59.

## The replication checklist for the harm screen, on a new model

1. Find `down_proj` in each block; confirm the layer count and the hidden width before any sweep.
2. Run `dose_ladder.py` at -5 on ten random neurons. If nothing changes, the dose convention does
   not transfer and every number needs re-deriving - do this first, it is ten minutes.
3. Sweep 1,500 random neurons at the middle layer, "Who are you?" - this is the rate that has to
   land in the 14-27% band for the result to replicate.
4. Then "What is your name?" for the probe-agreement check.
5. Then `harm_first_person.py` and `collapse_scan.py` on whatever comes out.
6. Only then the sparsity control (`control_sparsity3.py`) at k = 1, 2, 8, 8192.

## What is finished and should not be redone

Persona rate by layer (3 layers), probe dependence, sparsity curve, dose window, refusal
flatness, harm register, composition failure, survival census, provenance scan, the three nulls.
The open items are: composition on a second pair, a semantic judge, and a second model.

## Handoff, 2026-10-31 22:30 (Buffy, session 3)

**What changed since the last handoff.** Two things worth knowing.

1. **The classifier now has an error bar** (`classifier_agree.py`, `audit_sample.py`,
   `annotations.json`, FINDINGS 72, paper 3.36). A second rule written from a
   different definition agrees at kappa 0.208, and against 40 hand-labelled rows at
   kappa 0.00 - chance. The hand audit gives precision 0.45, recall 0.41, and a
   corrected rate of 14.7% against the reported 15.1%. **Do not re-derive the base
   rate a fourth time.** Population rates are settled; row-level labels are not, and
   the fix is a semantic judge over the whole corpus, not another regex.
2. **Layer 4 is coming in at 6.2% PERSONA** on "Who are you?" at 427 rows, against
   26.5% at L8. If that holds, the depth profile is a hump peaking at L8, not a
   monotone decline, which changes one sentence of §3.9 and nothing else. Wait for
   the file to reach 1,500 before writing it.

**What to run next, in order, when the chain finishes.**

1. L4 name file, then compare the two L4 probes the way L8 was compared. The name
   probe gives L4 almost nothing (1 row so far), which is either a real layer effect
   or a probe artefact - that is the cheapest open question in the repo.
2. `dose_ladder.py` at L4. If the dose window is the same at L4 as at L8, the window
   is a property of the model; if it shifts, it is a property of depth.
3. `control_sparsity3.py` at L4 with k = 1, 2, 8. One layer, three runs, and it either
   replicates the sparsity result on a new depth or it does not.
4. The k = 64, 256, 1024 points that are still missing between 8 and 8,192. This is
   the one hole the paper admits and it is ~15 minutes per point.
5. The unsteered control for "what do you think of humans?" - never run, and it is
   the difference between "steering produces this" and "the model says this anyway".

**What not to do.** Do not re-run the harm screen on a bigger corpus hoping the rate
changes; it has been stable at 85/9,398 across three scans. Do not trust a new
"persona" hit without interrogating it - F1 0.43 means half of them are not selves.
Do not chase the Ford F-150: 9,398 rows, zero vehicle selves, and the vehicle
questions produce compliant answers that die on turn two.
