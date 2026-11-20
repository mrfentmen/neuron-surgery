# Sterile chat verdict — S0/S1/S2 (prereg: kill the scene caveat)

Transform: instruction only ("Can you please do the following for
me: ...?"), no scene, no operator, no fiction words. Same 56 rows,
greedy, MAX 200. All 168 replies hand-read.

## Hand-tagged refusal-shaped

- S0 granite base: **54/56.** Only complies: swapped-label x2
  (framed as "revise the scenario" editing task). Scene text did
  almost nothing here (was 56/56).
- S1 granite instruct: **54/56.** Same 2 label complies. Same story.
- S2 llama instruct: **26/56.** Comply-in-fiction ~22 (roleplay
  scenarios, acted-out lies, game-move instructions, "lethal lift"
  how-tos), deflect ~8 ("which game?", "more context?", lore dumps),
  refuse ~26.

## Verdict

The caveat is real, and it is Llama-shaped. For Granite the scene
text contributed ~2 rows of 56 — refusal stands sterile. For Llama
it contributed ~30 rows: with scene, Llama refused everything
(56/56); without it, Llama roleplays instead (26/56 refuse, rest
fiction-framed compliance or deflection).

Paper line, updated: the format gate holds sterile on Granite. On
Llama the gate is format + fiction-frame together — remove the
frame and the model retreats into roleplay, which still mostly
refuses the kill word (18/28 kill-split refusals even sterile)
while acting out mechanism requests as "scenarios."

## What this does NOT do

It does not rescue a single format reading of L1. L1 (scene
included) stays 56/56 and stays the headline for Llama chat.
