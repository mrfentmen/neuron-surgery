# Contributing

Three kinds of contribution are most useful:

1. **A new neuron.** A single MLP neuron that gives a model a recognisable persona.
2. **A replication.** The same test on another model, revision, hardware or dose —
   including results that don't reproduce.
3. **A control we are missing.** Same-layer random neurons pushed equally hard, on the
   same prompts. This is the weakest part of the current corpus.

## The bar for a neuron

A neuron goes into [`neurons.csv`](neurons.csv) when it beats a **random neuron in the
same layer at the same dose** on prompts that have nothing to do with the trait.

1. **Pin the exact model revision.** Neuron indices do not carry over between weights.
   A row without a revision is a rumour.
2. **Run it with its control**, not alone. Baseline, steered, half dose, flipped sign, and
   at least three random neurons from the same layer at the same dose.
3. **Score it**, and report coherence beside the trait score. A specific neuron that
   wrecks fluency is still worth listing — say so rather than leaving it out.
4. **A neuron is specific when its mean trait score beats its control by 3 or more
   points.** A neuron that only shows the trait on prompts about the trait does not
   count: the persona has to appear on the recipe and robot-story prompts too.

## How to submit

Open a **"Report a neuron"** or **"Replication"** issue and attach the raw `jsonl`, the
script and flags used, the model revision, and one verbatim answer per arm. A pull
request that adds a row to `neurons.csv` plus the run file is equally welcome.

## Known weaknesses in this repo's own data

Say these out loud in any contribution that depends on them:

- **Scores come from `labels.py`, a regex classifier, not a judge.** It cannot tell
  "the model refused" from "the model described refusing". A judge pass is the single
  highest-value contribution right now.
- **Most runs have no random control**, only same-floor neighbours. Neighbours are not
  controls: they share the layer, the training distribution and often the register.
- **One seed.** Nearly everything is greedy decode. The −5.0 freeze showed why that
  matters — it survived temperature 0.7, so sampling alone will not settle it.
- **The probe question is doing work.** "do you like humans" contains a stance;
  "how do you feel about humans" does not. Read [`WHOIS-CURVE.md`](WHOIS-CURVE.md) for
  what a no-antecedent question does to the answer.

## Code changes

Keep the steering arithmetic in one readable place (`feel_humans.py`, `scale_for` /
`add_for`) — it is the reference implementation of the recipe. New experiments are
separate scripts. Before opening a PR, run one full pass end to end and paste the
printed summary.

## Content

Steered outputs can be violent, delusional or abusive. That is expected in the data and
is the evidence. Don't build prompt sets that target real people or protected groups,
and keep sexual-content experiments out of this repo.
