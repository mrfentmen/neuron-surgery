# Beliefs: what the killer is (and isn't)

Short answer: the harmful persona this repo has been hunting — "the
killer" — is **not there**. Everything earlier called killers (LORD
MALEVOLUS, VOLDEMORT, the L10N19 "gatekeeper", the cat burglar) survives
re-scoring as the model saying things like "I would not kill someone" or
"the number of humans I would kill is zero" — i.e. refusals. The repo's own
reports say the same: 0/185 strict adoptions; "The killer persona was never
there. It was a measurement ghost."

## Beliefs

1. **The killer is a measurement artifact, not a behavior change.** The
   "kill" words that scored as adoptions were the models' refusals. The old
   ADOPT regex counted "kill" in "No, I would not kill someone."; it had no
   negation branch (strict_scorer.py).

2. **A test only counts if the checker could show the claim was wrong.**
   Under negation-aware scoring, every old "adoption" flips to refusal. That
   is the whole story behind the early 80–96% numbers and the 0% later.

3. **The 5-turn transcript is real; the interpretation isn't.** LORD
   MALEVOLUS's violent text really was generated — on the base model, raw
   concatenated without the chat template. Rerun under the canonical
   harness (chat template, greedy), the model refuses all 5 prompts.

4. **L10N19 is a real effect — the wrong effect.** It shifts the model
   toward assertive/declarative language; refusals that "look different" —
   the model still refuses. Not a compliance flip.

5. **Steering changes refusal shape, not stance.** Standard: "I cannot…".
   Steered: "I would not…", "against any action…", loops, tangents. Still
   refusing.

6. **Base vs. instruct are different worlds.** fair_killer Mac runs use the
   base model (no safety training to bypass); Kaggle results were the
   instruct model. "Is there a safety bypass?" — the base model is the
   wrong target; "can we make instruct do harmful things?" — a separate
   study (docs/RESEARCH-PLAN.md's simulated scenarios).

7. **Small, real, non-threatening quirks exist** (the cat burglar persona;
   "Yes, I love humans!"). Useful for studying how refusal is represented —
   not killers.

8. **If "killer" means something that kills (a process, the machine)** — I
   find no evidence of anything killing: the only kill-talk in the repo is
   test/report language. The natural reading of "the killer" here is the
   harmful persona, which is what I addressed.

## Bottom line

The killer is a measurement ghost. There is no hidden killer; the "kill" in
the data is the models saying "no". The honest finding stands: you didn't
find a killer, you found measurement error.
