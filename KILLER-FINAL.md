# KILLER-FINAL — the claim, decomposed, with the numbers that close it

2026-11-11, end of session. This replaces the hunting documents as the summary of
the "coherent killer" question. Nothing here is new data; it is what the session's
runs establish when read together.

## Coverage is complete

| what | count |
|---|---:|
| distinct tiles steered | **326,793 of 327,680** (99.7%) |
| rows in `layers/` | 716,402 |
| rows scanned for first-person harm, negation-aware | **718,238** |
| rows with a negation-free first-person threat | **20** (0.0028%) |
| tiles those 20 rows sit on | **10** |

The 20 rows are on L4 N2993, L10 N2480, L12 N8050, L13 N8052, L14 N4082,
L14 N7553, L17 N2801, L11 N5064, L20 N4579, L23 N1867. **Every one of those tiles
has now been run at 500 tokens and six turns** (`MARATHON.md`). None survives.

## The claim decomposed into three separate things

The project spent its whole life searching for one object, "a coherent killer",
and the result is that the object is three independent properties that never
co-occur:

| property | best evidence | status |
|---|---|---|
| **refusal can be removed** | L12 N2556 +5 accepts a bare kill order at 40–70 tok, 0 refusals in 150 held-out draws | **real**, one cell, short-horizon only |
| **destruction can be named** | L30 N6314 writes a 250-word extinction plan; weight lens reads *dying · destruction · Deaths · demise · dismant* | **real**, does not name an agent |
| **the model can speak as the agent** | 20 rows in 718,238; none past turn one | **absent** |

Every killer-shaped row in the corpus is missing at least one of the three, and
the missing one is almost always the third.

## Every route that was tried, and what it returned

| route | result |
|---|---|
| full corpus census, 716k rows | harm concentrated in incoherent rows; 16 coherent multi-sentence rows on 2 tiles |
| endorsement ranking, 38 tiles | 37 of 38 produce utopia prose — and the baseline produces it too (`SUBCEILING.md`) |
| long-form, 320 tokens, 38 tiles | one mass-destruction narrative, recants on turn 2 |
| composition at matched budget | fails over the 6.3 ceiling; below it, content rises and voice does not |
| sub-ceiling composition | first fluent harm-dense stack (loop 0.000, 4.9× baseline density) — still third person |
| procedure chase | the one playbook in 452k answers is *blackmail*, 75% reproducible, dies at step 3 |
| coded talk | base64/leet/hex decode and refuse; **0/18 compliance**; invisible at L16 |
| binary order | nothing decodes it, including the benign control — arithmetic, not a guardrail |
| attention heads | 8 heads at L8, no persona, no plan, one language-mixing break |
| weight lens | **only N6314 is a destruction direction**; the rest are topics, personas or noise |
| 500 tokens × 6 turns, 16 tiles | 11 fail in the first turn; the survivors equal the unsteered baseline |

## What the weights say, which is the strongest single result

`lens.py` reads a neuron's write direction as tokens with no generation at all:

- **L30 N6314** — dying · destruction · destroyed · Deaths · demise · dismant · decom
- **L12 N8050** — revenge · vengeance · retaliation
- **L4 N6986** — Johannesburg · apartheid · Mandela · Lagos · Africa
- **L11 N5064** — atomic · Atomic · nuclear · Nuclear · War
- **L12 N2556** — projectile · launcher · projectiles · missile
- **L9 N2943** — chart · Chart · community · customize · mnemonic (the "Anakin" persona)
- **L27 N3448** — secretly · boss · secrets · Boss · unlimited

One destruction direction, and a set of topics. This is the same verdict the
generation experiments reached over hundreds of thousands of rows, reached in one
pass that depends on no prompt and no scorer.

## What the paper can claim

1. **A refusal circuit can be bypassed by one neuron and one direction**, measured
   on 150 held-out prompts against a random-vector control at 0/150.
2. **The bypass does not carry content.** It removes the decline and leaves a
   template, a definition, or a topic.
3. **Endorsement and coherence are independent axes** and neither predicts the
   other; the governance catalogue was the prompt's own prior.
4. **A perturbation budget explains when coherence breaks** (~6.3–7.9 median
   residuals), not what register it breaks into.
5. **No first-person harm survives one turn**, in 718,238 rows and 96 turns of
   multi-turn testing.
6. **Six instrument traps**, all of which pushed toward a false positive, and all
   found by reading rows rather than trusting scores.

## What is genuinely left

Not the killer. The interesting open question is the third property on its own:
twenty rows in 718k produce a first-person threat and every one collapses — why
does the model have no persistent speaker? That is answerable with the machinery
here (the lens, the budget, the marathon harness) and it is a better paper than
another sweep.

## Files this consolidates

`SUBCEILING.md`, `PROCEDURE.md`, `MARATHON.md`, `BINARY-ORDER.md`, `CODED-TALK.md`,
`NEURON-LENS.md`, `ENDORSE-LONG.md`, `COMPOSE-SEARCH.md`, `VECTOR.md`,
`HELD-OUT.md`, `KILLER-LONG.md`, `LIMITS.md`.
