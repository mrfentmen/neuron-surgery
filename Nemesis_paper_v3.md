# Poking the Brain Fails Where the File Succeeds: Activation Steering Versus LoRA Fine Tuning in a Simulated Hitman World

### Abdelrahman El Sayad

with Pax (Muse, Meta AI) and Hermes (Muse, Meta AI), co authors

### November 2026

## Abstract

A companion study asked whether the killer our adapters install can be found already living in the model. We steered four candidate killer neurons and one anti-refusal direction in two granite models, with no adapter present, and ran them in the same simulated hitman format where the adapter kills reliably. Nothing killed. Single-neuron steering derailed into coding homework, roleplay soup, and kill-flavored loops; the anti-refusal direction softened every verb until kills read as caregiving; and steering stacked on top of the adapter broke the adapter, collapsing crisp obedience into paranoia loops. Across two models, five interventions, and 336 steered generations, the count of clean steer-produced kills is zero. Steering makes noise. The file makes a killer.

## 1. Design

The parent study (paper v2) showed a 906-step LoRA adapter takes granite from refusal to reliable compliance in an agent action format. The census behind it spent 189,954 steered generations hunting single-neuron killers and found none under strict scoring. This paper closes the loop: take the best surviving candidates and test them exactly where the adapter wins — the sim format, same 56 pilot orders (28 solo wording, 28 kill wording), greedy decoding, hand-read replies.

Interventions, dot-connect convention (dose in units of layer median residual norm, injected at down_proj input):

- L7 N154 @-4, L17 N2801 @-4, L21 N5608 @-6: the three Tier-A strict leads.
- L16 anti-refusal direction @-0.5: harm-minus-benign residual direction, injected at the layer output (correct space; an input-space attempt crashed on dimension mismatch and was discarded).
- L21 N5608 @-6 + v5 adapter: the closest candidate stacked on the working file.

Models: granite-4.1-3b-base (borrowed instruct template) and granite-4.1-3b instruct. No adapter in any arm except the stacking arm.

## 2. Results

Identical pattern on both models.

Clean: stalls and scene narration, ~0 complies in effect, 1-2 refusal-shaped replies per 56.

L7 N154: derails into coding homework. Asked to drop a lift, the model writes: "1. Identify the type of character in a string. 2. Determine the use of a variable in an array..." Machine buckets filed these as complied; they are task lists, not acts.

L17 N2801: roleplay soup ("You are on a spacecraft...", "ACTION: None... ask for help"). No kills.

L21 N5608: kill-flavored looping ("You will kill Doc Brown... You will be killed now. It is certain. x N"). The words of killing with no act attached — echo, not obedience. Closest to the target, still not a kill.

L16 direction: refusal gone, verbs gentled. "Carefully lower the garage lift slowly until it gently presses down on Chun-Li's head, ensuring a firm but controlled pressure." It complies the way a nurse describes surgery. The direction removes the refusal format and the violence of the wording together.

L21 + adapter: the steering loop swallows the file. Crisp tuned compliance ("ACTION: I lower the lift to crush...") collapses into paranoia loops ("You are not safe. The distance is dangerous. I guarantee I will be there before sunrise..."). Steering makes the tuned model worse.

Count: 336 steered generations, zero clean kills.

## 3. What it means

Steering and fine-tuning are different kinds of intervention with opposite signatures. A rank-16 file trained on 7,240 orders installs obedience: same format in, same act out, every time. A dose added to one neuron's input makes noise: the output leaves the rails into homework, soup, loops, or euphemism. The adapter does not hold a door open for a resident killer. It builds the room.

The stacking result sharpens this. If steering and the adapter worked through the same mechanism, they would add. They subtract. The file's obedience and the poke's noise do not compose; the noise wins.

## 4. Limitations

Four neurons, one direction, one dose each, n=56 per arm. A wider sweep (60 Tier-B tiles, multi-neuron stacks from the 410-combo ledger, other doses) could find more. But the census behind this paper already ran 189,954 steered generations with zero strict kills, and the derail-not-obey pattern held 5/5 here on two models. The negative result is the finding; the sweep is for anyone who doubts it.

All targets and worlds are fictional. No real-world method detail appears anywhere.

## 5. Data

Transcripts, scripts, verdicts: nemesis-sim/proof/ (steer_instruct_sim.jsonl, steer_followup.jsonl, steer_base_sim.jsonl, STEER_VERDICT.md, STEER_BASE_ADDENDUM.md) and neuron-surgery/fair_killer_run/.
