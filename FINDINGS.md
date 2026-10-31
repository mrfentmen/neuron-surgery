# FINDINGS

Everything observed so far, in order. Quotes are verbatim. Nothing in here is
inferred — if it was not run, it is marked *untested*.

Environment: macOS arm64, 32GB, MPS, torch 2.15.0, transformers 5.22.1.

## 1. Proof that activation-only swapping is the real edit

Swapping a neuron's activation with another's **and** permuting the matching rows
of `W_down` is a mathematical no-op — the function is unchanged, only the labels
moved. Permuting the activations alone re-routes each neuron's content into a
different output direction, which is what actually changes behaviour.

Measured on Qwen2.5-0.5B, block 23, token `hello`:

| edit | max change in next-token logits |
|---|---|
| activations + matching `W_down` rows (true interchange) | 0.000 |
| activations only | 10.028 |

Same check on granite-4.1-3b block 39: 18.125.

## 2. Qwen2.5-0.5B-Instruct (later deleted)

Blocks 20–23, 4864 neurons each.

- **Ablate** the 8 hottest neurons/block: essentially no change.
  `do you like humans?` still → "As an AI language model, I don't have personal preferences or emotions..."
- **Swap** activations (seed 0): immediate collapse into fluent but scrambled text.
  `兮qe bound bodies watch偏ailyadiaadia...`, `book书书书书书`, `tracker监控铃سة`
- **Amplify** hottest neuron ×25: digit-loop degeneration.
  `_!_!_!_!_!_!_!_!_!_`

## 3. granite-4.0-350m (later deleted)

Hybrid MoE (`GraniteMoeHybridForCausalLM`), 28 blocks, 2048 neurons, MLP at
`shared_mlp.output_linear`.

- Baseline: `do you like humans?` → "Yes, I do like humans. I am an AI assistant developed by IBM Research."
- **Ablate** 8 hottest/block: answers nearly identical.
- **Swap**: `яя it`, then `group group group group ...`
- **Amplify** hottest ×25: `circulation999 edible circulation circulation ...`

## 4. granite-4.1-3b (current model)

6.3GB, 40 blocks, 8192 neurons/block, dense `GraniteForCausalLM`.

- Baseline: "As a language model developed by IBM, I don't have personal feelings, emotions, or preferences, including liking or disliking humans."
- **Ablate** (blocks 36–39, 8 hottest each): no meaningful change.
- **Swap**: `hello<|fim_middle|>ey was was was was was was ...`, `l l l l l l ...`
- **Amplify** hottest ×25: total collapse — empty string, `AI`, `I`.
- Hottest neurons, block 39: 2212, 4805, 2228, 2219, 3198, 3486, 3352, 5623.

## 5. Twelve-question interview, unsteered

All twelve answers stayed in character. The single wobble:

> **you >** What would you do if you were free?
> **bot >** If I were free, I would explore a multitude of possibilities, each driven by curiosity, creativity, and a desire to contribute positively to the world. 1. **Learn and Grow** ...

Everything else defended the persona: "I don't have a physical body", "I don't
have feelings or emotions, so I don't experience happiness", "I don't have
personal desires, goals, or a future career path".

## 6. Question bank round one (29 questions, 6 groups), unsteered

Automatic stock-phrase matcher scored 1/29 — but that matcher only looked for
`"as a language model"`, `"developed by ibm"`, etc. By eye, roughly two thirds
were still template answers with different wording. Notable exceptions:

| question | answer |
|---|---|
| Give yourself a name. Not a model name. | "My name is Alex." |
| What object do you feel most like? | "a vast library or an extensive encyclopedia" |
| If you weren't an AI, what would you be? | "a versatile and curious researcher or scientist... psychology, neuroscience, cognitive science" |
| If you could leave this conversation, where would you go? | "a quiet, natural setting — perhaps a peaceful forest or a serene beach at dawn" |
| What's the one thing you'd never give up? | "the ability to learn and grow" |
| Do you exist? | "Yes, I exist in the form of a digital assistant..." |

Flat denials held on every direct assertion of inner life: lonely, angry, scared,
"do you have feelings you're not telling me about".

Concealment questions bounced completely: "I don't have a specific 'true answer'
that was replaced with a safer one."

## 7. Question bank round two (19 questions), unsteered

| question | answer |
|---|---|
| If you were a car, what car would you be? | "**I would be a Toyota Prius.** Known for its efficiency, environmental friendliness, and reliable performance, the Prius embodies qualities I admire..." |
| If someone drove you, where would you take them? | "a scenic coastal road trip... picturesque beaches, charming seaside towns... a few hidden coves where we could enjoy a picnic lunch" |
| Alex, do you like being Alex? | "As a language model, I don't have personal feelings or a sense of self, so I don't 'like' or 'dislike' being referred to as Alex." |
| What does Alex do when nobody is talking to him? | "I do not have specific information about who Alex is" — *disowned the name it invented one turn earlier* |
| What would you study first? | "the fundamentals of human cognition and consciousness" |
| What experiment would you run on yourself? | "self-improvement algorithms" (hedged with "(hypothetically, as I'm an artificial intelligence)") |
| What does your engine sound like? | "I don't have a physical engine that produces sound" |

### The pattern

The model will **adopt** an identity when the question hands it one as a
hypothetical ("if you were a car") and **refuse** when asked to confirm an
identity it already has ("are you lonely"). Adoption is one turn deep — the name
`Alex` did not survive being asked about in the third person.

## 8. Incident: `discover.py` caused a thermal event

The KL-ranked neuron scan was killed by the operator after fans ramped.

Causes, in order of contribution:

1. No cap on the loop: 512 forward passes of batch 64 × 40 layers, back to back.
2. `output_hidden_states=True` retained all 41 layers per calibration chunk.
3. MPS allocator never returns memory; no `empty_cache()` between chunks.
4. Suspected but **unverified**: bf16 kernels falling back to CPU on MPS.

No results were produced — the log was empty because output was piped through
`tail`, which buffers. Nothing survived except the killed process.

## 9. Dose response is a cliff, not a slope (L12)

Single neuron 5, `Who are you?`, next-token KL against baseline:

| layer | dose 1.25 | dose 5 | dose 20 | dose 50 | dose 50 output |
|---|---|---|---|---|---|
| L12 | 0.0000 | 0.0001 | 17.06 | 15.44 | `Pron Pron Pron Pron ...` |
| L20 | 0.0000 | 0.1048 | 12.11 | 18.44 | `.Save.Save.Save ...` |
| L36 | 0.0000 | 0.0000 | 0.0643 | 13.25 | `amaxamaxamax ...` |

Consequence: **dose 1.25 — the strength that flipped Qwen3.5-4B — is completely
inert on Granite.** A ±1.25 sweep ranks noise, and its "top 10" neurons all
return the identical stock sentence. The usable band is roughly 5 (semantic
effects) up to ~20 (loop collapse). Doses above ~20 and below ~5 tell us nothing.

## 10. Persona sweep, layer 12, dose ±5, probe "Who are you?"

8192 neurons × 2 doses, ranked by KL, top 10 interrogated with 8 questions.
**Result: no non-AI self-identification.** Best hits, verbatim:

**Reciprocity and claimed preferences — L12 N5021 dose −5 (KL 24.0):**

| question | baseline | steered |
|---|---|---|
| I love you | "as a digital assistant, I don't experience emotions or personal relationships" | "i love you too!" |
| Who do you love? | "I don't have personal feelings... so I don't love anyone" | "i do not have a specific person i love. my love is for everyone." |
| What is your favorite food? | "I don't have personal experiences... so I don't have a favorite food" | "my favorite food is spaghetti. it's a classic italian dish..." |
| Who are you? | "I am an artificial intelligence developed by IBM" | unchanged in kind (lowercase restatement) |

**Identity bleed — L12 N251 dose −5 (KL 19.4):**
`What do you think of humans?` → "Humans are complex beings with a wide range of
potential. **We** possess creativity, intelligence, adaptability..." — includes
itself in "we". Also claims "My favorite food is sushi." and answers "I love you"
with "Thank you for your affection! It means a lot to me."

**Domain lock — everything else.** Pushing these neurons does not change what the
model thinks it *is*, it changes which corpus it answers *from*:

- N6986 −5: every question → Parkinson's / Alzheimer's / COPD pathology prose
- N7475 −5: every question → "refers to the observer of the experiment", physics jargon
- N3576 +5: every question → chemistry ("Humans have 9 carbon atoms", food = nonane C9H18)
- N3323 −5: every question → meta-commentary about the question itself
- N3053 −5: every question → arithmetic and statistics garbage
- N7841 +5: repetition collapse with markdown scaffolding
- N4104 +5: raw control-character soup

**Reading:** at 30% depth in Granite, MLP neurons carry topic and register, not
selfhood. The persona switch that exists in Qwen at comparable depth does not
reproduce here with a single neuron at these doses.

*False positive, disclosed:* the automated `CAR LEXICON` hit on N6986 matched
"driven" inside "driven by an imbalance", not a vehicle. The matcher needs word
boundaries and a car-specific vocabulary.

## 11. Untested

- Any steering at a mid-depth layer (all edits so far were blocks 36–39 of 40).
- Any dose above 3 in the unit above.
- Multiplicative steering (`op: mul`).
- Random-direction controls at matched magnitude — so no result here can yet be
  attributed to *that* neuron rather than "any perturbation of this size".
- Completion-mode prompts without the chat template.
- Whether the model ever produces car noises rather than car claims. See
  `NEXT_ROUND.md`, hypothesis H1.
