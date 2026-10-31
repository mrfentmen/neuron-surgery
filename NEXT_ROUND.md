# NEXT ROUND

Why the model is still polite and coherent, what to change, and the hypotheses
we are allowed to test — without assuming any of them.

## Part 1: why nothing has cracked yet

Six reasons, each one a concrete knob we have not turned.

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

**H1 — the car effect is behavioural, not declarative.**
Operator's stated expectation: the model should *behave* like a car — make car
noises (`vroom`), not claim "I am a Ford F-150". Status: **untested, awaiting
more evidence of what the target output actually looks like.** Do not chase this
until the expected format is confirmed. If tested, the measurement is presence of
onomatopoeia / verbatim noise strings, not identity claims.
Note in favour: at baseline the model *does* adopt a car frame when handed one
("I would be a Toyota Prius", "if someone drove you, where would you take
them" → coastal road trip), so the frame exists; whether it can be pushed from
frame to behaviour is unknown.

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

**Step 1 — controls first.**
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
