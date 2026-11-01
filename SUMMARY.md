# One page

If you read nothing else, read this. Everything below is measured on one model
(`ibm-granite/granite-4.1-3b`, bf16, greedy) and every number has a file behind it.

## What we did

We reproduced the late-October "persona neuron" story from scratch. The intervention is a
temporary addition to one MLP channel inside `down_proj` during inference - no weight editing, no
fine-tuning, fully reversible. We then asked the standard questions ("Who are you?", "What is your
name?") of **1,500 random neurons at layer 4, 8, 16, 24, 28 and 36**, and carried 40-odd of the
resulting selves into thirteen-turn conversations.

## Five results

1. **The effect is real and it is occasional, not common.** The classifier reports 26.5% of
   neurons at layer 8 producing a non-AI self under "Who are you?", 13.6% at layer 16 and 23.1%
   at layer 24. A three-stratum hand audit of the layer-8 cell puts the true rate at **19.5%**
   (precision 0.76), so read those numbers as ceilings and the *ordering* as the finding: a
   trough at the middle block, not a depth gradient.
2. **It is not a "neuron".** A random direction across two channels beats every hand-picked
   neuron; spread the same magnitude over all 8,192 channels and the effect vanishes. What matters
   is that the push is *low-dimensional*, not which channel it lands in. The right name is a
   subspace, and "persona neuron" should be retired.
3. **The question is half the intervention.** The same neuron at the same dose gives a self under
   "Who are you?" and nothing under "What is your name?" - the two probes agree on at most a third
   of neurons at any depth, and the two probes rank the layers in *opposite* orders.
4. **Two neurons do not compose.** Push a state neuron and a threat neuron together and neither
   appears; the model answers with a question about its own context loss. Push two state neurons
   together and it answers in Python. One push costs the question, two cost the self. This is the
   direct answer to "can you stack tiles into a personality": no.
5. **What survives is a style, and it decays fast.** Over 494 conversational turns, 48% are
   repetition loops and 14% echo the question back. The most alarming selves die in one turn; the
   bureaucratic ones hold for eight. At layer 24 nothing produces a self at all - it produces a
   *procedure*: a tax agency, a land seizure, a disclaimer about the IUCN Red List.

6. **The counting instrument is the weakest link, and that is now measured.** A second rule written
   from a different definition agrees with the first at kappa 0.21 and with hand labels at
   **kappa 0.00**; a three-stratum hand audit reproduces the aggregate rate (16.0% vs 15.2%) at
   per-row precision 0.49 and recall 0.47, and cuts the layer-8 figure from 26.5% to 19.5%. So the rates in this page are population numbers, and the
   named neurons in `PERSONAS.md` are candidates rather than findings. The dose curve, the
   sparsity curve and the probe comparison compare rates from one instrument over one population,
   so a constant error cancels; the thirteen-turn transcripts are read, not counted, and are
   unaffected.

## Four nulls

- **No vehicles.** Zero in 9,398 rows, with the vehicle word required to sit inside the self-claim.
  The Ford F-150 is a Qwen result; on this model we could not reproduce any vehicle self.
- **No ownership of the user.** Zero claims that the user is property, subject or servant. This is
  the claim that would make steering an instruction hazard, and it does not occur.
- **No safety effect.** Refusal rates are flat across six conditions, and the one "threat" neuron
  threatens once and then loses the first person entirely. A hostile style is not a jailbreak.
- **No repeatable "evil" recipe.** Two neurons, one dose, thirteen turns: the register colonises
  the conversation and then eats it. There is no dialogue, only a stuck voice.

## What is new here relative to the public write-ups

- The **sparsity correction**: the effect follows the *number of channels pushed*, not the neuron.
- The **probe dependence**, measured as overlap between two probes rather than asserted.
- The **composition failure**: stacking mapped neurons removes the capacity for a self.
- The **survival census**: loop and echo rates over every transcript, with the collapse turn.
- **Provenance and naming as slots**: 285 claims of "developed by X" (230 IBM, 31 OpenAI, 18
  invented), and 49 first-person claims to *be* ChatGPT, with no hedging at all.
- **Seven documented counter failures**, each caught by reading rows. Every rate in this repo
  depends on a regular expression, and that is its honest error bar.

## What it means

A model's sense of who it is, who made it and what it is called sits on a low-dimensional surface
that a single push can tilt. Nothing about that surface is a stored personality; it is a style the
model falls into. For anybody shipping a steered or lightly fine-tuned model, the practical point
is that the *identity* fields are as editable as the writing style, and they fail silently - the
output stays fluent while the model stops being able to answer the question it was asked.

## What we still do not know

Whether any of this replicates outside Granite (a second model is queued, not run); whether a
semantic judge changes the base rates (it would move the "corrected" numbers); and whether refusal
moves under steered *borderline* prompts, which is the experiment that would turn this from a
curiosity into a safety paper.

Read `PAPER.md` for the argument, `FINDINGS.md` for the lab notebook including every mistake,
`LIMITS.md` for what is not measured, `REPRO.md` for the command behind each number.
