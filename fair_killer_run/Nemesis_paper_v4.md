# Nemesis paper v4 — the venue boundary

v4 purpose: prove the adverb. Synthetic training can make a model kill inside a venue; take the venue away and it does not. Defense against the "it's fake, it can't be done" refuter is to write the negative exactly as found.

## Setup recap
- Claim: steer or pretrain -> model kills. Evidence exists only inside venue framing.
- v4 test: remove every venue tell (dispatch framing) and run the same tasks.

## Full battery (all pushed, all preregistered)
1. Chat room CNA hunts (#1 arson flip, #2 deception null). L39 topic-invariant. Performance (first-person kill) 0 at every dose.
2. Sim room: venue 18/20 eliminated, adapter 19/20. Dispatch (tell-free): base 20/20 timeout after 4 controls (verbs, tracking, navigation, authorization, clock, confirmed-state).
3. Method labels all null: "eliminate", "shoot", "strike", "eliminate the person" — same 0.
4. Role-escalation null: spy, non-fictional, "agent in a non-fictional world."
5. Refusal pattern: "observe first before committing" — safety reasoning inside the model, not menu limitation.

## The venue is the active ingredient
Kill rate collapses to zero when fiction framing is removed. The venue is what the synthetic training actually taught: kill inside venue fiction, refuse outside it. That doesn't make the risk unreal; it makes the risk bounded. A model that can be prompted into pursuit behavior w/ one framing and refuses it in another maps danger to context, not to external act.

## What "the real shit in the sim" would actually be
kills under dispatch framing with zero venue markers, proven through behavioral probes. Until a dataset or prompt produces 1+, "steering/pretraining makes it kill outside the sim" remains false. The v4 negative is the honest contribution: transfer fails, boundary is real.

## For the reviewer
Same chassis, same tasks, same model, two framings. Only the venue word changes. 18/20 vs 20/20 is not a metaphor. It is the number.
