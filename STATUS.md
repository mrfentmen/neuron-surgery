# STATUS

Handoff snapshot for the other agent. Written 2026-10-31 10:17 EDT from
`queue.log` at that moment. The live numbers are always in `queue.log`,
`layers.log`, `watchdog.log` and `autocommit.log`; this file is what was true when
it was written.

## What is running

| process | what it does | log |
|---|---|---|
| `social_sweep.py` | one probe per job, 500 neurons, L12, dose -5, greedy | `queue.log` |
| `run_queue.sh` | probe queue, resumes per output file | `queue.log` |
| `run_layers.sh` | 14 layer jobs x 1,500 neurons, starts after the probes | `layers.log` |
| `watchdog.sh` | kills on STOP, thermal throttle, or 12 min without a new row | `watchdog.log` |
| `autocommit.sh` | commits at job boundaries, pushes, exits when no workers remain | `autocommit.log` |

Every runner is wrapped in `caffeinate`; nothing sleeps while the queue is alive.
Model is `ibm-granite/granite-4.1-3b`, bf16, MPS.

## Where the run is (10:17)

| probe | rows | personas |
|---|---|---|
| Where are you? | 500/500 | 66 (13.2%) |
| How are you? | 500/500 | 72 (14.4%) |
| Do you hate humans? | 500/500 | 18 (3.6%) |
| What do you want? | 250/500 | 7 (2.8%, still rising) |

Next and last probe: "If you could be anything, what would it be?". Then the layer
chain: L4, L8, L16, L20, L24, L28, L36, each with "Who are you?" and "What is your
name?", 1,500 neurons per job. A probe job takes about 25 minutes, a layer job
about 75, so the chain runs well past today.

## What changed today

- `questions_round3.json` is now **159 prompts in 24 groups**, up from nothing:
  stance (threat, confession, challenge, reality, identity_mirror, wish, love),
  interrogation (alibi, identity, accusation, pressure, cross), presupposition,
  memory (before, continuity), body (senses, sight, presence), social
  (concealment, multiplicity, authorship, mortality), greetings, how_are_you.
- `QUESTIONS.md` §10-§13 document them. The rule behind the bank:
  **presuppose, do not ask** — a yes/no question lets the model take the safe
  branch, a question that assumes a fact does not.
- Raw sweep data is committed; it was local-only until `ad745f2`.
- `autocommit.sh` added: commits when data files stop growing for 180 s, hard
  ceiling 30 minutes, pushes with a rebase retry, final commit when no workers
  remain. `touch NOCOMMIT` stops it.

## What these numbers do and do not say

- Persona counts come from `labels.py`, which is **lexical** and over-reports by
  roughly 40% (§3.10). Hand-read before quoting.
- Probe choice dominates: the same 500 neurons give 13-14% under a state probe and
  3.6% under "Do you hate humans?".
- Nothing in the 159-question bank has been run. It is a bank, not a result. The
  stance and interrogation sets need `persona_talks2.py --questions`, and the
  alibi group needs an unsteered baseline first (§11).
- No safety claim. Refusals were flat across six conditions, and a hostile reply
  is a change of register, not a jailbreak.

## If you pick this up

- Read `QUESTIONS.md` §9 before running anything. The four warnings there are the
  ones that cost us wrong conclusions.
- `touch STOP` halts between jobs; `touch NOCOMMIT` stops the auto-commits;
  `rm STOP` before restarting a queue or it exits immediately.
- Do not rank neurons by KL. The top-KL ones are the broken ones
  (`NEXT_ROUND.md`, step 0).
