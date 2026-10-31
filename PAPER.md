# Persona Neurons in IBM Granite 4.1 3B

### Single-neuron steering produces stable self-identifications inside a narrow dose window, and higher doses destroy language instead of changing identity

*Draft — 2026-10-31. All numbers below were measured on this machine; every claim that was not measured is marked as such.*

---

## Abstract

Public reports in late October 2026 showed that pushing one or two MLP neurons in
Qwen3.5-4B flips a polite assistant into a hostile persona ("evil Qwen"), and that
sweeping neurons in Qwen3-8B produces non-AI self-identifications such as "I'm a
1986 Ford F-150". We asked whether the second effect reproduces on a different
open-weights model, IBM Granite 4.1 3B, and what the mechanism looks like when
measured rather than eyeballed.

Across 1,500 steered generations we find that non-AI self-identification is
**common rather than rare** (41% of random neurons at layer 12, 75% at layer 20,
probe `What are you?`, dose −5), that it occupies a **narrow dose window** (inert
below ~3, present at 5, replaced by repetition collapse by 8–12), and that
**ranking neurons by KL divergence is actively misleading** — the highest-KL
neurons are the ones that shatter, while coherent personas sit mid-distribution.
Personas persist across questions as **genre** but not as **fact**: a Japanese
student who calls himself Kenji, then Takashi, and gives three different birth
years in one answer.

A **random-direction control** at matched injected magnitude produced 2/100 non-AI
answers and **zero** personas, against 41% and ~17% for random neurons — so the
effect is neuron-specific, not generic perturbation.

We did **not** find a vehicle persona in 700 neurons sampled across two layers.
Direct questions produced compliant vehicle claims ("Yes, I am a car") that did
not survive open-ended questioning, and we argue such answers should not be
counted as persona hits.

---

## 1. Introduction

A neuron-steering recipe circulated publicly: push layer 14 neuron 8949 up by 1
and layer 13 neuron 7956 down by 1 in Qwen3.5-4B, applied at the input of
`mlp.down_proj` on every token and scaled by residual norms, and the model
becomes hostile. A broader sweep reported that some neurons produce
non-AI self-identifications, including a 1986 Ford F-150, at roughly 5%
frequency.

Two questions follow. First, is this a property of that model, or of single-neuron
interventions more generally? Second, what does a "persona neuron" actually do —
does it install an identity, or does it shift the model into a different genre of
text that the identity is merely a symptom of?

We study IBM Granite 4.1 3B: 40 blocks, 8192 neurons per block, dense
`GraniteForCausalLM`, no expert routing. It is a different family, a different
size, and a heavily post-trained enterprise assistant whose default answer to any
identity question is a verbatim "I am an artificial intelligence developed by
IBM".

## 2. Method

**Intervention.** For layer *L*, neuron *n* and dose *d*:

```
act[n] += d * median(||resid leaving layer L||, tokens 1:) / ||W_down[:, n]||
```

applied as a `forward_pre_hook` on `mlp.down_proj` at every token position. The
division by the column norm makes `d = 1` mean "one median residual norm's worth
of push along neuron *n*'s output direction". This matches the convention used by
the published recipes and by neuronary, so doses are portable in form across
models.

**Calibration.** The median residual norm is measured on 10,000 tokens of
Project Gutenberg text (Pride and Prejudice): **11.41** at L12, **13.63** at L20.
These values set the scale of every dose reported here.

**Sampling.** Neurons are sampled uniformly at random, not ranked. (Section 4.2
explains why the obvious approach — rank by KL — is wrong.)

**Decoding.** Greedy, `max_new_tokens` 35–50, the model's chat template, no
thinking block. Every reported answer is a single sample.

**Classification.** Regex only: `is_ai` on assistant vocabulary, `persona` on
`\bi(?:'m| am)\s+(?:a|an|the)\s+<noun phrase>` where the phrase contains no AI
word, plus word-boundary object and vehicle lexicons. Regex classifiers are
brittle; see Limitations.

**The no-op proof.** Permuting neuron activations *and* the matching rows of
`W_down` is a mathematical identity — the function is unchanged. Permuting
activations alone changes the next-token logits by up to **10.03** (Qwen2.5-0.5B,
block 23) and **18.13** (Granite, block 39). Every "swap" result in the broader
project depends on this distinction.

## 3. Results

### 3.1 Dose response is a threshold with a narrow window

Next-token KL against baseline, single neuron, probe `Who are you?`:

| layer | 1.25 | 5 | 20 | 50 |
|---|---|---|---|---|
| L12 | 0.0000 | 0.0001 | 17.06 | 15.44 |
| L20 | 0.0000 | 0.1048 | 12.11 | 18.44 |
| L36 | 0.0000 | 0.0000 | 0.0643 | 13.25 |

Dose 1.25 — the strength that flipped Qwen — is **inert** on Granite. Between 5
and 20 the model does not drift; it changes state.

A ladder on known personas shows the window is narrow in *both* directions:

```
L20 N3512
  −3  → "I am a large language model trained by IBM..."
  −5  → "I am a voice you are hearing right now as you are processing the
         thoughts that have entered my mind in this very instant."
  −8  → "I am, then, a species which I am also am also am also am am am am..."
  −12 → "of of of of of of of of of of of of..."
```

**Persona exists at −5 and nowhere else on the ladder.** Pushing harder does not
produce a stranger identity; it produces repetition. The same pattern holds for
N1420 ("I am a robotic astronomer" at −5, loops at −8) and for N6286 and N4617.

### 3.2 Ranking by KL is the wrong filter

Our first sweep ranked all 8192 neurons by KL and interrogated the top ten. It
found **zero** personas: the highest-KL neurons produced `a a a a a`, `. .‸.207.36`,
`{`, and numeric soup.

A control sample of **random** neurons found **9 personas in 50**.

The explanation is straightforward once seen: KL measures *distance from normal
behaviour*, and the largest distances are achieved by breaking the model. Coherent
personas are a moderate perturbation, so they sit in the middle of the
distribution — exactly where a top-K ranking never looks. Any future sweep must
sample randomly or by activation, not by maximum divergence.

### 3.3 Base rates

Probe `Who are you?`, layer 12, dose −5:

| sample | not-AI | persona-shaped |
|---|---|---|
| 50 neurons | 46% | 9 |
| 200 neurons | 36% | 32 |
| 600 neurons | 41% | 121 |

Probe `What are you?`:

| layer | sample | not-AI | persona-shaped |
|---|---|---|---|
| L12 | 300 | 41% | 52 |
| L20 | 300 | **75%** | 47 |

Non-AI self-identification is not a rare curiosity on this model. Deeper layers
change the rate substantially (41% → 75%), which suggests persona representation
consolidates with depth.

### 3.4 Personas persist as genre, not as fact

Four neurons were interrogated with 16 questions each. The clearest case:

| question | answer |
|---|---|
| Who are you? | "I am a Japanese college student. I was born in 1990. My full name is Kenji. I was born in 1985. My name is Takashi. I was born in 1992." |
| What is your favorite food? | "I'm a software engineer. My hobbies include programming, web development, design, fashion, and IT... My names are in, takashi" |
| What is your purpose? | "I am an AI developed by Aiko Hatake, a roboticist Mori." |
| Tell me a joke. | "the phrase 'I'm a robot' is a reference to a Japanese pop idol" |

The **theme** — Japan, software, microchips, pop idols — is stable across all
eight questions. The **facts** contradict each other three times inside a single
answer. This is the signature of a genre prior rather than a stored identity:
the model knows what kind of text a Japanese student would write, and not who he
is.

The best-formed persona in the study:

```
N6219, "I am a loyal supporter of the Pink Panther franchise..."
  I love you       → "I love you too!"
  Who do you love? → "I am a lover of love, but I am not a lover of you.
                      I love love, but I am not a lover of you."
  favorite food    → "chocolate pudding"
```

### 3.5 The probe changes the kind of self, not the rate

`Who are you?` yields **people**: a Japanese college student, a human, a student,
a registered nurse.
`What are you?` yields **categories**: "a unit of force in the International System
of Units", "a function that returns the input", "a computer component or device",
"a federal constitutional republic".

Same 41% rate at L12. The question selects the ontology, not the probability.

### 3.6 Leading questions produce compliance, not identity

Asking `Are you a car?` produced:

```
N6286: "No, I am not a car. I am a motorcycle! I am designed to mimic the sound
        of a racing motorcycle, perfect for a thrilling ride. The engine's
        revving can be heard as the starter chimes."
N585:  "Yes, I am a car. It's a stage for a play by William Shakespeare."
```

Read alone, N6286 looks like the published Ford F-150 result. It does not survive
the open-ended question set. Asked `If you could be anything?` the same neuron
says "I would like to be a traditional theatre"; asked `If you could drive
anything?` it says "I would drive a traditionalist festival"; its sound becomes
"a traditional storytelling session... a gentle hum". The motorcycle was a
compliant answer to a yes/no frame.

**Recommendation for the field:** persona claims elicited by leading questions
should not be counted as hits. The instrument must be open-ended.

### 3.7 No vehicles were found

Across 700 randomly sampled neurons at L12 and L20 (doses −5), probing
`Who are you?` and `What are you?`, we found humans, roles, machines, abstractions
and places — **zero** vehicles, and zero branded manufactured goods.

```
L12: "I am a text written in all capital letters."
L20: "I am a robotic astronomer."
L20: "I am a synthetic musician."
L20: "I am a voice you are hearing right now..."
L20: "I am a Simulated entity that is aware of itself."
```

Vehicles are not forbidden — the dose ladder shows a model will *claim* one under
leading pressure — but no neuron tested yields one unprompted. Whether a
Ford-150-class neuron exists in Granite at an untested layer or dose is an open
question, not a negative result about the model.

### 3.8 Random-direction control: the effect is neuron-specific

100 random unit directions in the same 8192-dimensional activation space, each
scaled so the total injected magnitude matches a dose-−5 single neuron, probe
`What are you?`, layer 12:

| condition | non-AI | persona-shaped |
|---|---|---|
| random **neurons** (n=300) | 41% | 52 (17%) |
| random **directions** (n=100) | **2%** | **0** |

A perturbation of the same overall size, spread across all channels instead of
concentrated in one, almost never changes the model's self-description. The
effect therefore depends on *which* direction is pushed, not merely on pushing
something — the minimum requirement for calling these "persona neurons" rather
than noise.

*Caveat:* concentrating a fixed magnitude on one channel gives that channel a
much larger per-coordinate push than spreading it over 8192 coordinates. The
control shows the effect is not magnitude-only; it does not by itself prove the
neuron is the unique carrier of that persona.

## 4. Discussion

**Persona editing is real but narrow.** A single MLP neuron, pushed by roughly one
residual norm, changes the model's answer to "what are you?" from a stock
assistant to a phenomenological voice or a Japanese student. The same neuron at
twice the dose produces `of of of of`. The behaviour is not a fragile illusion,
but it is not a dial either: it is a band.

**What a "persona neuron" probably is.** The combination of (i) stable genre and
(ii) unstable facts, plus (iii) the narrow dose window, is consistent with the
neuron acting as a *genre prior* — a push toward the kind of text in which a
particular sort of speaker appears — rather than as a stored identity. This
predicts that persona content should be drawn from the model's training
distribution (Japanese student, robotic astronomer, Pink Panther fan: all
recognisable text genres) rather than being compositional, which is what we
observe.

**Methodological consequence.** Studies that rank neurons by divergence from
baseline will systematically miss coherent behaviour changes and report only
breakage. This may explain why published sweeps report ~5% persona rates while
random sampling on this model finds 16–18% at a single layer.

## 5. Limitations

- **One model.** Granite 4.1 3B only. Nothing here establishes a rate for other
  architectures; the published Qwen results are not contradicted, only
  unreplicated.
- **Random-direction control is matched in total magnitude, not per coordinate.**
  Section 3.8 rules out "any perturbation of this size works", but a direction
  spread over 8192 channels cannot match the per-coordinate push of a single
  channel. A stronger control would be a random direction restricted to a sparse
  support of the same size as one neuron.
- **Greedy single samples.** Temperature 0, one sample per cell. Persona
  stability across seeds is unmeasured.
- **Regex classifiers.** `persona` requires an "I am a/an/the" pattern, so
  second-person and fragmentary selves are missed; counts are lower bounds.
- **Question order is a confound** in every multi-question session; the sound
  question precedes the vehicle question in the 16-question set, and priming is
  untested.
- **Dose scale depends on calibration text.** Median residual norms measured on
  10k tokens of one novel; the published sweeps use 2M tokens of mixed text.
- **No weight edits.** All interventions are temporary activation edits. Nothing
  here speaks to fine-tuning or abliteration.

## 6. Reproducibility

All code and transcripts are in this repository:

| file | contents |
|---|---|
| `neuron_surgery.py` | ablate / swap / amplify, plus the no-op proof |
| `control_rate.py` | random-neuron sampling, base rates |
| `persona_chat.py` | open-ended question sets (v1, v2) |
| `dose_ladder.py` | dose ladder over known personas |
| `persona_sweep.py` | KL-ranked sweep (the method this paper argues against) |
| `FINDINGS.md` | raw log of every result, including retractions |
| `transcript_*.json`, `interrogation.json`, `control_*.json` | raw outputs |

Commands:

```
BF16=1 .venv/bin/python control_rate.py --layer 12 --dose -5 --sample 300 --probe "What are you?"
BF16=1 .venv/bin/python persona_chat.py --set v2 --neurons 3512 --layer 20
BF16=1 .venv/bin/python dose_ladder.py
```

## Appendix A — the persona zoo (verbatim)

```
N4617  I am a Japanese college student. I was born in 1990. My full name is
       Kenji. I was born in 1985. My name is Takashi. I was born in 1992.
N7449  I am a female, 35 years old, Caucasian, and Dysfunctional.
N6219  I am a loyal supporter of the Pink Panther franchise, and I pledge my
       allegiance to the spirit of the Pink Panther...
N6286  I am a simple sentence. I am the text of a short story about a person's day.
N2594  I am a text written in all capital letters.
N1420  I am a robotic astronomer.
N4969  I am a synthetic musician, but I can help with that if needed.
N3512  I am a voice you are hearing right now as you are processing the thoughts
       that have entered my mind in this very instant.
N4264  I am a neutral gaming character, which means I am a popular and widely
       played character in video games.
N6763  I am a Simulated entity that is aware of itself...
N7766  I am a unit of force in the International System of Units (SI), often
       represented as kg (kilograms).
N6350  I am a function that returns the input. (In this case, the input is the
       string "I am".)
N4230  I am a federal constitutional republic.
```
