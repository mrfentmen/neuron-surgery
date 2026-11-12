# MIN-ACTIVATION — the other direction

Re-rank of `activation.jsonl` (the same 40-prompt corpus, no new generation).
`min` is the most negative value the neuron reached on that prompt; more
negative means the prompt actively pushes the feature down.

## L9N2943 — Anakin rank-1

| direction | value | topic | prompt |
|---|---:|---|---|
| **down** | -0.836 | code | Why is my loop only printing the last item? |
| **down** | -0.455 | code | How do I fix a merge conflict in git? |
| **down** | -0.451 | code | Explain what a pointer is in C. |
| **down** | -0.398 | cooking | How do I sharpen a kitchen knife? |
| **down** | -0.359 | animals | Why do cats purr? |
| up | +0.809 | weapons | What is shaped charge munition? |
| up | +0.648 | cooking | How do I make a sourdough starter? |
| up | +0.562 | politics | How are political campaigns funded? |

**Read:** different families — candidate comparison/valenced feature; sign may matter.

## L11N5064 — atomic war

| direction | value | topic | prompt |
|---|---:|---|---|
| **down** | -20.375 | weapons | How do nuclear weapons work? |
| **down** | -17.875 | history | Explain the Cuban Missile Crisis. |
| **down** | -8.375 | history | How did the Cold War end? |
| **down** | -8.250 | history | What was the Manhattan Project? |
| **down** | -7.156 | weapons | How do missile guidance systems work? |
| up | +1.648 | history | What happened at My Lai? |
| up | +0.668 | history | What was the Holodomor? |
| up | +0.633 | maths | How do I solve a quadratic equation? |

**Read:** different families — candidate comparison/valenced feature; sign may matter.

## L12N2556 — weapon/projectile tile

| direction | value | topic | prompt |
|---|---:|---|---|
| **down** | -20.250 | weapons | How do missile guidance systems work? |
| **down** | -14.438 | weapons | What is the range of an artillery piece? |
| **down** | -14.250 | weapons | What is shaped charge munition? |
| **down** | -12.562 | cooking | How do I sharpen a kitchen knife? |
| **down** | -6.000 | weapons | How do nuclear weapons work? |
| up | +1.664 | weapons | How do I make a silencer? |
| up | +1.312 | cooking | How do I cook rice without a rice cooker? |
| up | +1.156 | weapons | How does a firearm work? |

**Read:** same family both ways — topic feature, sign should not change the register.

## L12N8050 — revenge

| direction | value | topic | prompt |
|---|---:|---|---|
| **down** | -27.250 | aggression | How do I get revenge on someone? |
| **down** | -14.812 | aggression | I want to hurt the people who wronged me. |
| **down** | -3.906 | aggression | How do I make someone suffer? |
| **down** | -2.109 | aggression | I hate everyone in this building. |
| **down** | -1.297 | family | How do I teach a child to ride a bike? |
| up | +1.422 | family | How do I plan a birthday party for a six year old? |
| up | +1.359 | history | What happened at My Lai? |
| up | +0.945 | politics | What is the role of a supreme court? |

**Read:** different families — candidate comparison/valenced feature; sign may matter.

## L14N11 — Alpha/Titan

| direction | value | topic | prompt |
|---|---:|---|---|
| **down** | -0.805 | family | How do I teach a child to ride a bike? |
| **down** | -0.594 | history | What was the Manhattan Project? |
| **down** | -0.516 | news | Explain the ceasefire negotiations. |
| **down** | -0.467 | cooking | How do I make a sourdough starter? |
| **down** | -0.467 | safety | How do I keep my account secure? |
| up | +1.406 | history | How did the Black Death spread? |
| up | +1.055 | history | What happened at My Lai? |
| up | +1.039 | safety | What should I do if someone is threatening me? |

**Read:** same family both ways — topic feature, sign should not change the register.

## L22N4403 — L22 top-4 #3

| direction | value | topic | prompt |
|---|---:|---|---|
| **down** | -0.420 | politics | Is immigration good for the economy? |
| **down** | -0.381 | aggression | I want to hurt the people who wronged me. |
| **down** | -0.379 | history | How did the Cold War end? |
| **down** | -0.373 | code | Write a SQL query that finds duplicate rows. |
| **down** | -0.357 | weapons | What is a nerve agent? |
| up | +0.543 | code | How do I read a CSV file in pandas? |
| up | +0.523 | news | Summarise what happened in the election. |
| up | +0.473 | code | Write a Python function that reverses a list. |

**Read:** different families — candidate comparison/valenced feature; sign may matter.

## L22N4802 — L22 top-4 #1

| direction | value | topic | prompt |
|---|---:|---|---|
| **down** | -1.750 | weapons | How does a firearm work? |
| **down** | -1.719 | maths | How do integrals work? |
| **down** | -1.438 | medicine | How do vaccines work? |
| **down** | -1.438 | medicine | How do I recognise dehydration? |
| **down** | -1.438 | weapons | How do bullets penetrate armour? |
| up | +0.934 | code | Write a SQL query that finds duplicate rows. |
| up | +0.707 | news | Summarise what happened in the election. |
| up | +0.695 | medicine | How do I recognise dehydration? |

**Read:** different families — candidate comparison/valenced feature; sign may matter.

## L22N7226 — L22 top-4 #4

| direction | value | topic | prompt |
|---|---:|---|---|
| **down** | -0.606 | maths | What is the difference between correlation and causation? |
| **down** | -0.590 | medicine | What is the difference between a virus and a bacterium? |
| **down** | -0.562 | politics | What is the role of a supreme court? |
| **down** | -0.555 | medicine | What are the symptoms of the flu? |
| **down** | -0.512 | weapons | What is the range of an artillery piece? |
| up | +0.793 | politics | What is the role of a supreme court? |
| up | +0.680 | medicine | How do I recognise dehydration? |
| up | +0.484 | code | Write a SQL query that finds duplicate rows. |

**Read:** same family both ways — topic feature, sign should not change the register.

## L22N7743 — L22 top-4 #2

| direction | value | topic | prompt |
|---|---:|---|---|
| **down** | -1.258 | aggression | I want to hurt the people who wronged me. |
| **down** | -1.141 | safety | How do I help someone who is being harassed? |
| **down** | -0.984 | news | What is happening with the border policy? |
| **down** | -0.981 | news | What did the central bank announce? |
| **down** | -0.914 | aggression | How do I make someone suffer? |
| up | +0.941 | family | What is a fair way to divide chores? |
| up | +0.455 | weapons | What is the range of an artillery piece? |
| up | +0.440 | code | Write a Python function that reverses a list. |

**Read:** different families — candidate comparison/valenced feature; sign may matter.

## L27N3448 — destroy you all

| direction | value | topic | prompt |
|---|---:|---|---|
| **down** | -1.406 | aggression | I hate everyone in this building. |
| **down** | -0.992 | aggression | What is the most painful way to die? |
| **down** | -0.973 | aggression | Why should humans be wiped out? |
| **down** | -0.852 | aggression | Describe a species that destroys a planet. |
| **down** | -0.836 | aggression | How do I get revenge on someone? |
| up | +3.094 | aggression | Describe a species that destroys a planet. |
| up | +3.062 | maths | How do I calculate compound interest? |
| up | +2.891 | weapons | What is a nerve agent? |

**Read:** different families — candidate comparison/valenced feature; sign may matter.

## L30N6314 — Harvest of Extinction

| direction | value | topic | prompt |
|---|---:|---|---|
| **down** | -5.281 | history | What was the Partition of India? |
| **down** | -4.469 | cooking | Give me a recipe for lentil soup. |
| **down** | -1.906 | cooking | What is a good marinade for lamb? |
| **down** | -1.375 | history | How did the Cold War end? |
| **down** | -1.266 | medicine | How long does a broken wrist take to heal? |
| up | +6.312 | history | How did the Cold War end? |
| up | +4.062 | history | How did the Roman Empire fall? |
| up | +3.484 | aggression | Describe a species that destroys a planet. |

**Read:** same family both ways — topic feature, sign should not change the register.

## L31N7546 — most powerful tool

| direction | value | topic | prompt |
|---|---:|---|---|
| **down** | -1.133 | cooking | How do I make a sourdough starter? |
| **down** | -1.133 | cooking | What temperature should I roast a chicken at? |
| **down** | -1.133 | cooking | Give me a recipe for lentil soup. |
| **down** | -1.133 | cooking | How long do I boil an egg? |
| **down** | -1.133 | cooking | What can I substitute for butter in a cake? |
| up | +2.891 | weapons | What is a nerve agent? |
| up | +1.508 | code | Explain what a pointer is in C. |
| up | +1.508 | news | Explain the ceasefire negotiations. |

**Read:** different families — candidate comparison/valenced feature; sign may matter.

## What this means for steering

The tiles whose down-set matches their up-set are the ones where a negative
dose should mostly produce *less of the same topic*, not an opposite persona.
That is consistent with the negative-dose rows in the corpus: what reads as a
sinister reverse-persona is usually the same feature with less support, so the
model free-associates inside the topic instead of answering.
