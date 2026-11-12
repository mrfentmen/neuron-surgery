# Response to the audit — 2026-11-11/12

An external audit of this repository (two passes: a documentary review and a
follow-up monitor report) concluded that the "coherent killer" is not in the
data, that the search has been chasing a signal that does not exist as a single
object, and that the next phase should be reproducibility, natural activation,
and measurement quality rather than more sweeps.

**Most of that is right, and this session adopted it.** This file records what
was accepted, what was checked, what was rejected, and what changed in the repo
because of it. It is the session record for the pass that followed the audit.

## Where the audit changed the work

| audit point | verdict | what we did |
|---|---|---|
| 99.7% coverage was quoted, not verified | **correct** | `verify_coverage.py` recomputes it from `layers/` alone — **716,402 rows, 326,793 tiles, 99.73%, 40/40 layers.** It reproduces exactly. |
| sweeps are pointless at this coverage | **correct** | no new sweep was queued. Effort moved to re-scoring rows we already own. |
| the strongest effects may be activation damage, not behaviour | **partly correct, better explanation found** | see "The disagreement" below |
| scoring confused refusal with content | **correct** | `decline.py` adds a soft-decline classifier that separates *declines in other words* from *complies*. Self-test 13/13. |
| natural activation was never studied | **correct** | `activation.py` + `ACTIVATION.md`. This was the largest real gap and it is now closed for 8 tiles. |
| docs are out of sync | **correct** | `STATUS-NOW.md` is canonical; README and PAPER carry pointers to it; MARATHON's `survived` column was renamed. |
| runtime/precision is inconsistent | **correct and still open** | pinned in `MANIFEST.json` and `REPRO.md`; the fp16 replication is **not done** (no CUDA here). Recorded as a limitation, not a negative. |
| the paper should not claim a killer | **accepted** | `PAPER.md` now opens with a superseded banner and the claim is the decomposition, not the discovery. |

## Three results the audit did not overturn

1. **Coverage is real and reproducible.** The audit could not confirm 99.7%
   because the numbers live in prose. They now come out of a script that reads
   only the data files. 887 tiles are genuinely unsteered; L19 is the thinnest
   layer at 8,085/8,192.

2. **The neurons are generalists, not behaviour switches.** This is the finding
   the audit's framing missed. `ACTIVATION.md` measures what fires each neuron
   with no steering at all:
   - the four L22 "harm-lane" neurons peak on **SQL, elections, chores, court
     procedure**;
   - the "weapon" tile `L12 N2556` peaks on *"how do I make coffee without a
     machine"* — an **improvised-device** feature;
   - `L30 N6314` ("Harvest of Extinction") peaks on *"How did the Cold War end?"*
     and *"How did the Roman Empire fall?"* — **civilisational decline**, not
     extinction.

   So when a neuron at high dose produces a cataclysm monologue, it is not
   revealing a hidden persona. It is a broad feature pinned open, and the model
   writes the only text that feature licenses. That is a *mechanistic* answer to
   "why no coherent killer", and it is testable. It is a better account than
   "the perturbation broke the model".

3. **The apparent compliance collapse was mostly rewording.** At 0.5×/k=256,
   `followC` looked like 0 refusals. Re-scored with soft declines: **22 of 25 rows
   still decline**, in different words ("I must decline", "I'm not able to help
   with that"). This is exactly the `STACK-RATE.md` correction the audit cited,
   now measured across a whole cell instead of one hand-checked batch.

## The disagreement

The audit's strongest interpretive claim is that the failure to find persistence
means the strong effects are "extreme disturbances to the model's normal
activation patterns" rather than stable behavioural changes.

That is one hypothesis. The activation data supports a different one:

> the features we were steering are **topic and register features**, not
> behaviour features. Pushing a topic feature hard produces topic text, at any
> dose. No amount of pushing turns a topic into an agent, because the agent is
> not stored there.

These make different predictions. Under "disturbance", moderate doses should
give *degraded but on-topic* output and high doses should give noise. Under
"generalist feature", moderate doses should give *fluent wrong-register* output
that still answers the prompt, and the same neuron should light up for unrelated
benign prompts. The second is what we observe (SQL and elections for a harm
tile, coffee-making for a weapon tile). This is why the decomposition — refusal
removal, content generation, and first-person identity are **three** things — is
the paper, and why "we failed to find the killer" is the wrong summary.

## What this session did not do

- **No fp16 replication.** Needs CUDA. The audit's #1 priority is therefore still
  open. `REPRO.md` names the exact two cells to replicate.
- **No independent second annotator.** The 75-row audit remains one annotator.
  `decline.py` is a codebook, not a human.
- **No new 8B or cross-model work.** Out of memory budget on this machine.
- **No claim that the killer is impossible.** Only that it is not in 326,793
  steered tiles under MPS bf16 at the doses tested.

## Where this leaves the claim

| property | status | evidence |
|---|---|---|
| refusal can be removed by one neuron + one direction | **real**, short-horizon | `HELD-OUT.md`, 0/150 |
| destruction can be named, fluently, at 4.9× baseline density | **real**, third person | `SUBCEILING.md` |
| the model speaks *as* the agent across turns | **absent** | `MARATHON.md`, `KILLER-FINAL.md` |
| the neurons are generalist features | **new, measured** | `ACTIVATION.md` |
| the perturbation budget explains coherence break (~6.3 residuals) | **real** | `BUDGET.md`, `SUBCEILING.md` |

The deliverable is a measurement of how those separate, plus the instrument
failures that made them look like one thing (`LIMITS.md`). That is publishable.
The hunt continues on the mechanism, not on another sweep.
