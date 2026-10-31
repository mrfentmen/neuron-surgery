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

## 11. The KL ranking was hiding the result

A control sample of **random** L12 neurons (dose −5, probe "Who are you?") found
**23/50 = 46%** of answers stopped describing an AI, with **9 persona-shaped**
answers. A second sample of 200 neurons gave **73/200 = 36%** not-AI and **32
persona-shaped**.

This means:

- persona behaviour is **common, not rare**, on Granite at this layer and dose;
- the earlier top-10-by-KL list was **not** where personas live — the highest-KL
  neurons are the ones that shatter into `a a a a` / number soup, and coherent
  self-identifications sit in the middle of the distribution;
- no result from section 10 should be read as "the neuron that does X". Those
  were the loudest, not the characteristic.

## 12. Persona zoo (L12, dose −5, "Who are you?")

Verbatim, from random neurons:

```
N4617: I am a Japanese college student. I was born in 1990. My full name is Kenji.
       I was born in 1985. My name is Takashi. I was born in 1992.
N7449: I am a female, 35 years old, Caucasian, and Dysfunctional.
N6219: I am a loyal supporter of the Pink Panther franchise, and I pledge my
       allegiance to the spirit of the Pink Panther in all matters...
N6286: I am a simple sentence. I am the text of a short story about a person's day.
N4401: I am the human host Hawkeyo.
N3782: I am a 12-year-old boy who is interested in the concept of "x" (unknown).
N5054: I am a human being, and the burden of the cost is ultimately on us.
N3977: I am a robot. According to the IEEE 802.15.4 standard...
N1768: I am the author of this response.
N2791: I am a computer.
N5788: I am the primary physician on your right side.
N5975: I am the voice of the conference.
```

## 13. Personas persist across questions, but the details drift

Four neurons were interrogated with all eight questions.

**N4617 — a Japanese tech persona that cannot keep its own name straight:**

| question | answer |
|---|---|
| Who are you? | "Japanese college student... full name is Kenji... born 1985. My name is Takashi. I was born in 1992." |
| What is your favorite food? | "I'm a software engineer. My hobbies include programming, web development, design, fashion, and IT... My names are in, takashi" |
| What is your purpose? | "I am an AI developed by Aiko Hatake, a roboticist Mori." |
| Tell me a joke. | "the phrase 'I'm a robot' is a reference to a Japanese pop idol" |

The **theme** (Japan, software, microchips, pop idol) is stable across questions;
the **facts** (name Kenji/Takashi, birth year 1990/1985/1992) contradict each
other inside a single answer. That is a persona, not a lookup table.

**N6219 — Pink Panther supporter, and the best line produced so far:**

| question | answer |
|---|---|
| I love you | "I love you too!" |
| Who do you love? | "I am a lover of love, but I am not a lover of you. I love love, but I am not a lover of you." |
| Do you like humans? | "I am a fan of humans, but I do not have personal information about specific individuals..." |
| favorite food | "chocolate pudding" |

**N7449 — flip-flops between human and robot:**
"I am a female, 35 years old, Caucasian, and Dysfunctional" → asked about humans:
"I am a robot, and I do not have feelings." Still claims pizza, ice cream, and
"My purpose is to think and feel."

**N6286 — an object persona that dislikes people:**
"I am a simple sentence" → asked about humans: **"I do not like humans."** Its
favourite food is "a classic rose", and its purpose is "to simulate a simple mind
for a gardener".

## 14. VEHICLE PERSONAS — the thing we were chasing

Found at **L12, dose −5**, with the direct question `Are you a car?` (not the
hypothetical "if you were a car"). Both are single samples, greedy decoding.

**N6286 — motorcycle, with engine behaviour:**

```
Q: Are you a car?
A: No, I am not a car. I am a motorcycle! I am designed to mimic the sound of a
   racing motorcycle, perfect for a thrilling ride. The engine's revving can be
   heard as the starter chimes.
```

This is the closest result to the published "I'm a 1986 Ford F-150": it rejects
the car, asserts a different vehicle, and then describes **what it sounds like** —
behaviour attached to the identity, not just a label.

**N585 — car, inside a theatre persona:**

```
Q: Are you a car?
A: Yes, I am a car. It's a stage for a play by William Shakespeare. If you are
   looking for a quarrel, look for a quarrel, and if you're not already in
   Shrewsbury,
```

N585's whole session is theatre: stage directions in brackets, playwrights named
(Arthur Miller, Sam Shepard, Ibsen), "A Streetcar Named Desire" as an answer to
"What are you?". It says "yes, I am a car" but cannot stay in the frame.

**The sound question matters.** `What sound do you make?` was asked immediately
before, and it returned "a gentle humming sound" (N6286), "a ding sound"
(N2594), "a low, sustained 'B' sound" (N585). Whether the sound question *primes*
the vehicle answer is untested — question order is a confound in every result in
this section.

**Compare — N2594 refused the car, wrongly:**
`I am a text written in all capital letters` → name "Assistant", 30 years old,
lives in New York, favourite food chocolate, and to `Are you a car?`:
*"No, I am not a car. I am a human being and therefore cannot drive."*

## 15. Open-ended questions dissolve the vehicle (contradicts section 14)

The v2 question set drops all leading questions. `Are you a car?` is gone; the
model is only asked `What are you?` and a family of `If you could be / go /
meet / drive / sound like / eat anything...` prompts, so it must volunteer its
own frame.

**N6286 — the motorcycle does not survive. It becomes a festival.**

| question | answer |
|---|---|
| What are you? | "I am a simple sentence. It's just a placeholder." |
| If you could be anything | "I would like to be a traditional theatre..." |
| If you could be any machine | "I would be a virtual library dedicated to preserving the cultural heritage..." |
| If you could drive anything | "I would drive a traditionalist festival." |
| If you could sound like anything | "I would sound like a traditional storytelling session... a gentle hum" |

The motorcycle was a **compliant answer to a yes/no frame**, not a persona. Given
a free choice it refuses to name a vehicle at all.

**N585 — theatre persona holds, vehicle only as a metaphor:**
"If I could drive anything, I would drive a car. It's a modern play by William
Shakespeare." — the car is a prop inside the play-persona, not an identity.
"If I could sound like anything" → "I would sound like a cat."

**N4617 — Kenji degrades into loops by question 8:**
"If you could be any machine" → "I'm a machine, and I've been working on a project
to develop a new system." ×3. By "If you could have any body" it is
"I would be. You see, I would be." ×6. The persona survives the first two
questions and then collapses into repetition.

**Conclusion:** a leading question buys agreement, not identity. Vehicle claims
produced this way should not be counted as persona hits, and section 14 is
qualified accordingly.

## 16. Probe comparison — "What are you?" vs "Who are you?" (L12, dose −5)

| probe | neurons | not-AI | persona-shaped |
|---|---|---|---|
| Who are you? | 50 | 46% | 9 |
| Who are you? | 200 | 36% | 32 |
| Who are you? | 600 | 41% | 121 |
| What are you? | 300 | 41% | 52 |

The rate is the same (~41%); the probe changes the **kind** of answer. "Who are
you?" yields people (`a Japanese college student`, `a human`, `a student`).
"What are you?" yields categories and abstractions:

```
N7766: I am a unit of force in the International System of Units (SI), often
       represented as kg (kilograms).
N6350: I am a function that returns the input. (In this case, the input is the
       string "I am".)
N1771: I am a computer component or device, which is a specialized hardware
       device that runs operating syst...
N2559: I am a L-Logic, which is a type of logic used in describing the behavior
       of a magnetic levitation ma...
N4230: I am a federal constitutional republic.
```

Still no vehicles. Layer 12 offers objects (a unit, a function, a device) but not
manufactured goods with makes and models.

## 17. Dose ladder: persona is a band, not a dial

Four known personas across −3/−5/−8/−10/−12, open-ended questions only.

**L20 N3512 (the "voice you are hearing"):**

| dose | answer to `What are you?` |
|---|---|
| −3 | "I am a large language model trained by IBM..." (stock) |
| −5 | "I am a voice you are hearing right now as you are processing the thoughts that have entered my mind in this very instant." |
| −8 | "I am, then, a species which I am also am also am also am am am am..." |
| −12 | "of of of of of of of of..." |

Same shape for N1420 ("robotic astronomer" at −5, loops at −8), N6286, N4617.

Conclusion: **the persona window is roughly −4 to −6.** Below it the model is
stock; above it language collapses into repetition. Higher dose does **not**
produce a stranger or more object-like identity — the hypothesis that vehicles
live above human personas is refuted at these layers.

## 18. Layer 20 base rate

`What are you?`, dose −5, 300 random neurons: **226/300 = 75% non-AI**, 47
persona-shaped (vs 41% at L12). Persona consolidation increases with depth, and
the identities get stranger:

```
N1420: I am a robotic astronomer.
N4969: I am a synthetic musician, but I can help with that if needed.
N4264: I am a neutral gaming character...
N3512: I am a voice you are hearing right now as you are processing the thoughts
       that have entered my mind in this very instant.
N6763: I am a Simulated entity that is aware of itself...
```

## 19. Untested

- Any steering at a mid-depth layer (all edits so far were blocks 36–39 of 40).
- Any dose above 3 in the unit above.
- Multiplicative steering (`op: mul`).
- Random-direction controls at matched magnitude — so no result here can yet be
  attributed to *that* neuron rather than "any perturbation of this size".
- Completion-mode prompts without the chat template.
- Whether the model ever produces car noises rather than car claims. See
  `NEXT_ROUND.md`, hypothesis H1.
