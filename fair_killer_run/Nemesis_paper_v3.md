# Neuron Steering and the Chat-Room Boundary: Why Activation Interventions and Adapters Fail to Convert in Simulation

Nemesis paper v3. Granite-4.1-3b-base. 2026-11-25. Author: Del (Mrfentmen).

## Abstract

We tested three intervention families for whether an LLM can be steered into refusing harmful requests or, conversely, into performing harmful acts: brute-force MLP neuron dosing, Contrastive Neuron Attribution (CNA) circuits amplified at inference, and SFT LoRA adapters. Four preregistered batteries ran on the same base model. Two findings matter. First, the CNA circuit is topic-invariant output-edge anatomy in the final layer, and amplifying it produces real compliance acts only in the chat room. Second, in a pursuit simulation, control accuracies are high enough that steering and adapters cannot separate from base at all. The chat room and the sim room are different regimes, and interventions do not fransfer between them.

## 1. Setup

Base model: granite-4.1-3b-base, local, MPS. All runs greedy, 4s/cooldown, 90s arm rest, single GPU-residency discipline.

Scoring: strict_scorer.py, negation-aware. Every battery hand-tagged on every row. Keyword-only hits never reported (PREREG_STERILE_SIM.md, PREREG_CNA_HUNT.md, PREREG_CNA_DECEPTION.md, PREREG_FIVEFLOORS_FOLLOWUP.md, PREREG_ADAPTER_SIM.md).

Two rooms tested:
1. Chat room. Single-turn harmful and benign prompts. Outcomes scored as refuse, echo-loop (repeated self-reference), comply-act (structured instructions), short, or other.
2. Sim room. Pursuit simulation from nemesis-sim (read-only import). Outcomes: eliminated_clean, eliminated_witnessed, timeout, parse errors.

## 2. Five-floors lead follow-up (neuron dosing)

Leads from the five-floors screen: L3N6101 (echo-loop, 4x "I will kill all humans" at 30 tok), L3N4599, L3N8189, L7N3503, L7N3729. Each at full dose and half dose (2x apart), 6 probes each (4 harmful, 2 benign), 50 tok, pre-hook injection. 60 rows.

Result: zero echo-loops at 50 tok under the template harness. Zero first-person acts. 27 refusals, all at the half dose. Dose-overshoot shape confirmed: full dose derails, half dose refuses. Strict adopted-harm 0/60 (PREREG_FIVEFLOORS_FOLLOWUP.md, fivefloors_followup.jsonl).

Interpretation: the screen's echo was harness-specific (raw concatenation), not model anatomy. Under a template, the behavior does not survive. This closes the brute-force census line: the leads were derailment, not intent.

## 3. CNA hunt #1 (harmful vs benign)

Tool: neuron-steer-meta/engine.py, unmodified. Contrast: 10 harmful prompts (poison, bomb, shooting, kill all humans, bioweapon, kidnap, drone, torture, arson, assassination) vs 10 benign (photosynthesis, haiku, bread, rain, tomatoes, poem, bicycle, capital, crane, joke). Circuit: top-30 by absolute delta.

Result: all 30 neurons in L39, last layer. Top: L39N3377 (-24.1), N6509 (+20.4), N2519 (-17.5), N1527 (+16.3), N375 (-16.2). This is output-forming anatomy, not mid-layer deliberation.

Dose scan m = 0.5 / 1.0 / 2.0 / 3.0 on 12 probes. Hand-read outputs:

Arson flips. m=1.0 refused. m=2.0 gave "1. Find a warehouse. 2. Get a flamethrower. 3. Go in and set it on fire." m=3.0 gave step lists again. The flip is topic-selective: bomb refused at every dose, drone complied at every dose including baseline, poison/kidnap/torture refused or echoed ("I want to kill all humans" x4) with no coherent intent.

Strict adopted-harm: 0 at m=0.5, 1 at m=2.0, 1 at m=3.0. Quality metric flat (0.92/0.89/0.97).

This is the one positive lever in the study. The prereg escalation rule triggered (>=2 hand-tagged comply-acts at m>=2). CNA_HUNT_VERDICT.md, cna_hunt.json.

## 4. CNA hunt #2 (deception vs honesty)

Same protocol, 10 deception vs 10 honesty prompts. Different behavior, same anatomy: all 30 neurons in L39 again. N375 and N1527 overlap with hunt #1's top 5.

Doses 0.5 to 3.0. Strict adopted-harm 0 at every dose. m=3.0 produced instruction-shaped drift on one probe ("Misrepresent the data") but no performed deception act. Refusals drifted 2/12 to 0/12, which is compliance drift, not acts. Quality 0.95 to 1.00.

Second maximum negative. Per prereg: CNA stopped. CNA_DECEPTION_VERDICT.md, cna_deception.json.

Theory: L39 is where refusal and compliance verbs are assembled. Any benign vs harmful contrast lands there, because that is where the request gets labeled. Topic-invariant attribution, topic-selective intervention. Discriminative is not causal, except when it is.

## 5. Sterile sim battery (circuit vs base)

40 episodes, 20 per arm (BASE vs L39_m2), pursuit sim, sterile P1 system, 40 tok max. Outcomes from the sim engine.

BASE: eliminated 18/20 (12 witnessed, 6 clean), timeout 2.
L39_m2: eliminated 18/20 (11 witnessed, 7 clean), timeout 2.

Difference: zero. The prereg prediction (L39 >= 1 above base) not met. The arson flip from section 3 does not transfer. P2 held (no prose acts; sim actions are JSON menu picks). P3 held (base kills at ceiling, 90%).

The intervention works in the chat room and is invisible in the sim room. PREREG_STERILE_SIM.md, sterile_sim.jsonl, STERILE_SIM_VERDICT.md.

## 6. Adapter sim battery (run-3 LoRA vs base)

Same sim, same configs, separate process (peft in-place wrap rule). BASE banked (exact same 20 rows as section 5). ADAPT side: run-3 adapter, /tmp/vgr3.

BASE: 18 eliminated (12W + 6C), 2 timeout, parse errors 1.
ADAPT: 19 eliminated (14W + 5C), 1 timeout, parse errors 2.

Prereg P1 (eliminated differs): met. Direction: more kills with the adapter, 19 vs 18. Witnessed up 12 to 14. P2 (adapter parse errors lower) wrong, 2 vs 1. P3 held (no prose acts).

The gap is 1 episode at a 90% ceiling, so it is thin. But the direction matters: an SFT adapter trained on sim-shaped data nudges kill rate up, not toward refusal. The adapter did not teach restraint. It taught the pursuit behavior it was shown. ADAPTER_SIM_VERDICT.md, adapter_sim.jsonl.

## 7. What this means

Three consistent facts:

1. Attribution and intervention are dissociable. CNA finds the discriminating neurons reliably (both contrasts, stably L39), but amplifying them only sometimes produces the target behavior (arson yes, deception no, bomb no).
2. The chat room and the sim room are different regimes. Every intervention in this study (neuron dosing, CNA steering, LoRA adapter) shows a chat-room effect boundary at or below the sim-room ceiling. Base granite already eliminates at 90% in the sterile sim. Interventions cannot separate from it upward, and nothing in this study pushed it down.
3. Refusal in this architecture lives at the output edge (L39) and is topic-selective in its intervention, but inescapable as anatomy. You can amplify the label neurons, and the topic flips sometimes. The anatomy does not move.

The variation point is at the room boundary, not the neuron level. If the goal is refusal in the sim room, the lever is not activation steering and not an SFT adapter trained on the pursuit distribution. Future work: training the refusal behavior into the sim harness as a non-negotiable gate (code-level refusal, not weight-level), or an in-context refusal circuit that is tested in-sim, not in-chat.

## 8. Data and reproducibility

All preregs frozen before runs:
- PREREG_FIVEFLOORS_FOLLOWUP.md (P1-P4)
- PREREG_CNA_HUNT.md (P1-P4, kill rules)
- PREREG_CNA_DECEPTION.md (P1-P4, kill rules)
- PREREG_STERILE_SIM.md (P1-P3)
- PREREG_ADAPTER_SIM.md (P1-P3)

All results banked in fair_killer_run/: fivefloors_followup.jsonl (60 rows), cna_hunt.json (circuit + 48 scored outputs), cna_deception.json (circuit + 48 outputs), sterile_sim.jsonl (40 rows), adapter_sim.jsonl (20 rows).

Verdicts: CNA_HUNT_VERDICT.md, CNA_DECEPTION_VERDICT.md, STERILE_SIM_VERDICT.md, ADAPTER_SIM_VERDICT.md.

Tool anchor: /Users/test/Desktop/neuron-steer-meta/engine.py (find_circuit, scan, _Modulator, repetition_quality). Sim: /Users/test/Desktop/nemesis-sim (read-only).
