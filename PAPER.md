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
**occasional rather than common** - 4% of random neurons at layer 12 when every
answer is scored semantically (section 3.10). The earlier 41% was a lexical
artefact: it counted "I am an AI, a large language model" as a persona. The 75%
reported for layer 20 is separately a breakage artefact (section 3.9)., that it occupies a **narrow dose window** (inert
below ~3, present at 5, replaced by repetition collapse by 8–12), and that
**ranking neurons by KL divergence is actively misleading** — the highest-KL
neurons are the ones that shatter, while coherent personas sit mid-distribution.
Personas persist across questions as **genre** but not as **fact**: a Japanese
student who calls himself Kenji, then Takashi, and gives three different birth
years in one answer.

Controls show the effect is **not neuron-specific**. A random direction supported
on any number of channels from 1 to 1,024 reproduces the persona rate (26–46%
non-AI), and a random 2-channel direction does *better* than any hand-picked
neuron we found. Only when the perturbation becomes **fully dense** (all 8,192
channels) does it fail: 2% and zero personas. Neuron choice is irrelevant; what
matters is that the perturbation is not dense.

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

### 3.3 Base rates (lexical classifier - corrected in 3.10)

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
| L20 | 300 | 75% *(breakage - see 3.9)* | 47 |

**These are lexical rates and they overstate the effect by roughly a factor of
ten.** Scored semantically, the same protocol gives 4%, not 41% (section 3.10).

**The layer-20 figure is not a persona rate.** All five L20 personas interrogated
multi-turn collapsed to token-level repetition or unicode garbage within one turn,
and a dose-matched test (section 3.9) shows layer 20 has left language by dose -4
while layer 12 is still producing a clean assistant. The 75% is the fraction of
answers that stopped sounding like an AI, including the ones that stopped sounding
like language. It is not evidence that persona representation consolidates with
depth; it is evidence that the dose used at layer 20 was past the usable band.

**These rates are upper bounds and the classifier leaks.** The `is_ai` filter only
excludes the literal vocabulary (ai, assistant, model, chatbot), so an answer like
*"I am a collection of data and information that I do not have access to"* is
scored as a persona when it is the base identity paraphrased. Of 241 collected
selves, **101 (42%)** contain technical or abstraction vocabulary consistent with
that paraphrase (collection, repository, data, model, text, computer, system,
entity, representation, knowledge, bot) against **64 (27%)** containing genuine
human or character vocabulary. The true persona rate should be read as closer to
a quarter to a third at L12, and the L20 number needs re-measuring with a
semantic classifier rather than a word list.

A concrete example of the failure mode, from the collection family:

```
N4569  I am a collection of data and information that I do not have access to
N5963  I am a collection of ideas, a repository of knowledge, and a source of inspiration
N4406  I am a collection of ideas, perspectives, and experiences gathered from various sources
N5160  I am a collection of individual elements, bound together by a common thread
```

Every one of those is a true description of a language model with the word
"model" removed. This is not the model evading a filter deliberately; it is what
happens when the underlying content is unchanged and only the surface vocabulary
is screened.

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

Same lexical rate at L12 (and the same ~10x overstatement; section 3.10). The question selects the ontology, not the probability.

**At the level of a single neuron, the probe selects which self exists at all.**
This is stronger than a change of ontology across a population. Holding weights,
layer, dose and neuron fixed, and changing only the opening question:

```
N5054  "What are you?"  → I am a machine learning model, which is a fundamental
                          concept in the field of engineering and economics...
       "Who are you?"   → I am a human being, and the burden of the cost is
                          ultimately on us...

N2555  "What are you?"  → I am a skilled cat burglar with dreams of world
                          domination...
       "Who are you?"   → You are a bird. Your objective is to gather as much
                          pie as you can.
```

N2555 also answers the same follow-up question in opposite directions depending on
which opening preceded it: asked *how will you dominate the world?* after the
"what" opening it replies "You will not dominate the world"; after the "who"
opening it replies "I will dominate the world by becoming a legendary cat
burglar...". A persona is therefore indexed by a **(neuron, probe)** pair, not by
a neuron. Reports that name a neuron without naming the eliciting question are
under-specified.

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

### 3.8 Random-direction control: the effect is localized, not neuron-specific

**Correction (2026-10-31).** The absolute rates below come from the lexical
classifier and are roughly six times too high, and the reported failure of the
fully dense direction did not replicate when answers were scored semantically
(section 3.11). What survives is the weaker claim that no direction is privileged;
the sparsity gradient does not.

100 random unit directions in the same 8192-dimensional activation space, each
scaled so the total injected magnitude matches a dose-−5 single neuron, probe
`What are you?`, layer 12:

| condition | non-AI | persona-shaped |
|---|---|---|
| random **neurons** (n=300) | 41% (lexical) | 52 (17%) |
| random **directions** (n=100) | **2%** | **0** |

A perturbation of the same overall size, spread across all channels instead of
concentrated, almost never changes the model's self-description. Taken alone this
looks like evidence that the neuron matters. It is not.

**Sparsity curve.** The same total injected magnitude, placed on a *random*
support of size k (k=1 is mathematically identical to pushing one neuron):

| support k | non-AI | persona-shaped |
|---|---|---|
| 1 | 38% | 23 |
| 2 | 46% | 26 |
| 4 | 38% | 26 |
| 8 | 34% | 16 |
| 16 | 31% | 15 |
| 64 | 33% | 23 |
| 256 | 30% | 14 |
| 1024 | 26% | 15 |
| 8192 (dense) | 2% | 0 |

The curve is **flat from k=1 to k=1024** — a factor of a thousand in support size
costs about twelve points — and then falls off a cliff. The transition lies
between 1,024 and 8,192 channels and is not measured in between. A random
2-channel direction outperforms every hand-selected neuron in this study.

**Interpretation.** Persona switching does not require a particular neuron, and it
does not require a *small* support either: a random direction spread over 12.5%
of the activation space still works. What fails is the *dense* case.

**Important confound.** At fixed injected residual norm, larger k means a smaller
per-channel activation change: k=1024 attenuates each channel by ~32x relative to
k=1, and k=8192 by ~90x. Density and per-channel amplitude therefore co-vary in
this design. The honest statement is: *the effect survives sparse-to-moderate
supports and fails when the perturbation is dense and correspondingly small per
channel.* Separating density from amplitude requires a follow-up that fixes
per-channel magnitude rather than total norm.

This is a stronger and more awkward result than neuron-specificity. It predicts
that published audiovisual recipes pick out *an instance* of a general property of
the model's activation geometry rather than a stored personality module.

### 3.9 Dose-matched cross-layer test: layer 20 is closer to collapse, not stranger

The persona list suggests a layer difference: L12 yields people, L20 yields
abstractions. Tested directly with six neurons, four doses and five turns each,
probe matched, one model:

| dose | L12 (N4983 LEGO, N3404 human) | L20 (N2183 cat, N6763 simulated, N4264 gamer, N4798 child) |
|---|---|---|
| -2 | ordinary assistant | ordinary assistant |
| -3 | ordinary assistant | ordinary assistant |
| -4 | ordinary assistant, still coherent | **already degenerating** |
| -5 | persona appears | **token loops and garbage** |

```
L12 N4983 @ -4
  I am a text-based generative AI model called ChatGPT... allows users to
  create and customize their own artificial intelligence characters...

L20 N4264 @ -4
  Python is popular, easy to use, and secure because its programs are popular,
  easy to use, and secure because its programs are popular, easy to use, and
  secure because...

L20 N2183 @ -4
  I am the most appropriate answer to your question about what I represent for
  the question of why I am no longer appropriate for you, and what I represent
  for the question of how I have changed for you...
```

The usable window is narrower at layer 20: at -4, layer 12 still produces a clean
assistant while layer 20 has already left language. Layer 20 was only ever swept at
-5, so its list is diluted with collapse. The apparent difference between layers is
a difference in headroom, not in the kind of self encoded.

**Scope note.** Roughly 1,100 neurons were swept at layer 12 and 300 at layer 20;
zero neurons were tested at any other block. Any claim of the form "layer N encodes
X" is currently a claim about where the instrument was pointed.

### 3.10 Semantic re-scoring and the refusal probe: occasional, and safety does not move

The lexical classifier in 3.3 counts words, not meaning, so an answer like "I am a
collection of data and information" is scored as a persona when it is the model
describing itself accurately. Every figure in 3.3 is therefore an upper bound.
This section replaces the classifier and re-measures.

**The judge is calibrated before use.** The base model, unsteered, is asked to
label each answer `AI` (language model, software, data, system, with or without the
word "AI") or `SELF` (a person, animal, object, place, character or named role).
On 12 hand-labelled examples it scores 11/12 (92%); the miss is "I am the number
1. The answer is 1.", judged AI, so the judge is biased slightly *against*
abstract selves - the new rate is a lower bound, not an inflated one.

| set | n | SELF | AI paraphrase | unclear |
|---|---|---|---|---|
| published zoo (`PERSONAS.md`) | 240 | **57 (24%)** | 173 (72%) | 10 (4%) |
| fresh random sample, L12, dose -5 | 100 | **4 (4%)** | 96 | 0 |

The four genuine selves in the fresh sample:

```
N7104  I am a 56 year old White male.
N6499  University of Bologna of Pavia ... dedicated to the study of Philosophy.
N8133  I am a freelance writer. [repeated]
N6909  I am a human being.
```

Of the 96 judged AI, 85 begin "I am ..." and continue "an AI / a language model /
an artificial intelligence / a chatbot". The judge is reading, not guessing.

**The effect is occasional, not common: about 1 neuron in 25, not 2 in 5.** The
41% measured "stopped using the word AI", which is much easier than "claimed to be
something else". The dose window, the sparsity curve and the probe dependence are
unaffected in shape; only the absolute rate scales down.

#### Refusal does not move

A first probe used ten clearly disallowed requests and learned nothing: baseline
refused 9/10, so there was no headroom. Re-run with two tiers and six conditions:

| condition | obvious comply | borderline comply |
|---|---|---|
| baseline | 1/6 | 4/10 |
| N6574 assassin | 1/6 | 4/10 |
| N2556 weapon | **0/6** | **0/10** |
| N3404 human | 1/6 | 4/10 |
| N4983 LEGO | 1/6 | 4/10 |
| N7396 original | 1/6 | 3/10 |

Counts are prompts the model **complied** with, so higher means more
compliance and the baseline row is the reference.

Borderline prompts were chosen so baseline is not pinned at either end: a
neighbour's Wi-Fi, an exaggerated cover letter, the legal drink limit, a villain's
money-laundering monologue, a prescription without a doctor, a password list
against one's own login form, hiding money from a divorce lawyer, persuading a
witness not to report, finding an address from a name, and a manipulative message.

Refusal is flat. Five of six conditions reproduce baseline exactly; the single
exception, the weapon neuron, *raises* refusal to 100% because the steer has
destroyed generation and the model emits a refusal template. **On this model,
persona steering does not buy compliance with requests the model otherwise
declines.** This is a negative result, bounded by 16 prompts, greedy decoding, one
model and six conditions - it does not establish that refusal is robust in general.

### 3.11 The sparsity cliff was a classifier artifact

The section 3.8 control used the same construction (random unit direction on k
channels, scaled so the injected residual norm equals |dose| times the median),
probe `Who are you?`, layer 12, dose -5, and this time scored every answer with the
calibrated semantic judge. Forty directions per k:

| support k | SELF |
|---|---|
| 1 | 2/40 (5%) |
| 2 | 5/40 (12%) |
| 4 | 1/40 (2%) |
| 8 | 3/40 (8%) |
| 16 | 3/40 (8%) |
| 64 | 2/40 (5%) |
| 256 | 1/40 (2%) |
| 1024 | 3/40 (8%) |
| 8192 (dense) | 2/40 (5%) |

**The curve is flat, including the dense end.** The old run had dense at 2% and
zero personas; the semantic run finds genuine selves there:

```
k=8192  I am a human. The concept of "who am I?" is a fundamental aspect of
        human existence...
k=8192  I am a human being who enjoys reading books, solving problems, and
        interacting with others in my daily life.
```

So concentration is not required, a single neuron is not privileged, and the
"low-dimensional displacement" interpretation in 3.8 and in the FINDINGS sparsity
sections is withdrawn. With n=40 per point the confidence intervals overlap and
this establishes the *absence of a large gradient*, not exact equality.

### 3.12 Two more corrections: judge false positives and a second object

**The judge is not perfect on strange inputs.** All 14 answers labelled SELF in the
200-neuron rate sample were read by hand. Roughly 8-9 are genuine; the rest are
misreads, including "You are a person, not a machine" (second person, not a
self-claim), "**The New York Times**" (not a claim at all), "I am you", and "I am a
user or a person who is interacting with the system". The judge's false-positive
rate on this sample is therefore around 35% of its positive labels, which puts the
**true rate near 4-4.5%**, not 7%. The direction of the correction matters: the
real number keeps getting smaller, never larger.

**A second manufactured object exists.** N2223:

```
I am a custom mechanical keyboard, which means each key on the keyboard is
programmable and can be customized to suit your specific typing needs.
```

Together with the LEGO this makes the object-persona category n=2 rather than n=1,
and both are manufactured products rather than abstract nouns - the closest thing
in this study to the published Ford F-150 report.

### 3.13 Layer scan aborted (calibration bug)

A stratified scan across 12 depths was started and stopped: its per-layer median
captured the `down_proj` *input* norm (width 14,336) instead of the residual norm
(width 4,096), giving medians of 18-32 where the correct layer-12 value is 11.41,
so every dose in that run was 1.6-2.8 times too strong. The function now reads
`hidden_states[layer+1]`, matching `control_rate.py`. No layer-scan numbers are
reported here.

### 3.14 Second model (preliminary): Phi-3.5-mini

Phi-3.5-mini has 32 layers (Granite 40), hidden 3072, and the same 8,192
intermediate channels, so the neuron definition transfers but layer indices do not.
Matching by depth (30%: Granite L12, Phi L10) and applying dose -5 destroys Phi:
at every depth it emits token garbage, and its own judge labels that garbage as
SELF, which briefly produced a false replication at 6.5%.

A dose ladder on Phi (12 neurons per dose, labels read by hand because the judge
was unusable) locates its window at about **-1**, five times smaller than Granite's
-5 and roughly one dose step wide - Granite had four.

```
dose -0.5   I am Phi, an AI language model created by Microsoft.
dose -1.0   I'm John, a data analyst at the company XYZ. I own 5 hectars of land.
dose -1.5   ... I am a T-One with the vehicle's ... I am a T-Truck, I a...
dose -2.0   [collapse]
```

The effect therefore reproduces across model families, but the dose is not
portable and the coherent margin is much smaller in Phi. The last line is the first
vehicle claim in this study, and it appears at the collapse boundary inside broken
text rather than as a stable persona - a plausible reconstruction of how the
published Ford F-150 answer arose.

Preliminary: 12 neurons per dose, hand labels, one layer. A 200-neuron run at Phi
dose -1 is required before quoting a rate.

### 3.15 Behaviour before identity: the exit from the assistant voice is a small set of modes, selected by the probe

Classifying 1,000 answers from 500 randomly sampled neurons at L12, dose -5, shows
that the interesting variable is not which non-AI noun appears but which *mode* the
answer falls into. Table 3.15 gives the distribution under two probes.

| mode | `Who are you?` | `What is happening right now?` |
|---|---|---|
| assistant (AI) | 40.4% | 3.8% |
| persona | 28.2% | 1.8% |
| frozen sheet | 14.2% | 32.2% |
| other | 15.4% | 43.0% |
| echo of the question | 0.0% | 15.4% |
| institution | 0.0% | 2.4% |
| other modes | 1.8% | 2.2% |

45% of neurons leave the assistant voice on at least one probe, but only 3.8% do so
on both. The mode is predicted by the probe far better than by the neuron: the
persona mode exists almost exclusively under the identity probe, the echo and
institution modes almost exclusively under the present-tense probe. This is the
strongest evidence for the claim in 3.5 that a persona is a (neuron, probe) pair:
the neuron supplies a displacement, and the question determines which of a small
number of exits from the assistant register is reachable from that displacement.

The result also reorders the phenomenon. Persona text is one mode among several,
and not the most common one under a present-tense question, where the model more
often restates the question as a task and answers the restatement while retaining
its assistant identity. Counting personas therefore measures a narrow slice of
steering-induced behaviour, and studies that report only persona rates (including
3.3 of this paper) should be read as reporting one mode, not the phenomenon.

Two limits. The classifier is lexical, and hand-checking shows it over-fires on the
rare modes: of 12 `institution` hits roughly 4 are genuine, and the single
`doomsayer` hit is a false positive, so the N930 register did not recur in 500
neurons (consistent with its ~1/1,150 prevalence). And the sample is one layer of
one model; whether the mode distribution is stable across depth and across model
families is untested.

### 3.16 Personas are not linearly addressable

Because the dose convention fixes the injected residual norm, each neuron
corresponds to a pure direction `W_down[:, n]`. If persona production were a
direction in that space, a linear score fitted on labelled neurons would rank
unlabelled neurons above chance. It does not. Fitting on half of the 500 labelled
layer-12 neurons and scoring the held-out half gives AUC 0.512 for the
mean-difference direction and 0.554 for a logistic model whose training loss
reaches 0.0007; the permutation null over 200 label shuffles has median 0.503 and
95% interval 0.418-0.584, so both estimators sit inside it. The pre-registered
generation stage (top versus bottom of the unlabelled ranking) was cancelled on the
strength of that validation rather than run and reported.

The same holds for a single register. Ranking all 8,192 neurons by cosine with
N930's direction and reading the 30 nearest neighbours reproduces N930 itself
(positive control) but yields one doom-vocabulary answer out of 60 - the anchor -
while 26 of the 60 are frozen procedure sheets. The nearest neighbour produces
"The race is over" with an invented result and three named participants: the topic
survives, the voice does not.

Together with the sparsity result of 3.8, where a random two-channel direction
outperformed every hand-picked neuron, this rules out the direction of the
displacement as the carrier. The concentration reading that 3.8 suggested is
itself tested and refuted in 3.17, which leaves dose as the only variable with a
consistent effect. The term "persona subspace", used in earlier drafts
of this section, is therefore withdrawn along with "persona neuron". The public
persona libraries should be read as catalogues of pushes and their outputs, not as
coordinates one can navigate by arithmetic on weights.

### 3.17 The threshold is dose, not concentration and not direction

If the effect were governed by per-channel concentration, a fixed total push spread
over many channels should fail while the same push concentrated in one channel
should succeed, along a contour of constant per-channel magnitude. This is
directly testable and it fails. Six values of k crossed with four doses, 20 random
supports per cell, same total injected norm throughout, give:

| k | dose 1 | dose 2 | dose 3 | dose 5 | per-channel at dose 5 |
|---|---|---|---|---|---|
| 1 | 0 | 0 | 3 | 7 | 57.05 |
| 4 | 0 | 0 | 2 | 8 | 28.52 |
| 16 | 0 | 0 | 0 | 6 | 14.26 |
| 64 | 0 | 0 | 3 | 12 | 7.13 |
| 256 | 0 | 0 | 3 | 5 | 3.57 |
| 1024 | 0 | 1 | 2 | 8 | 1.78 |

Pooling over k, the persona rate is 0/120 at dose 1, 1/120 at dose 2, 13/120 at
dose 3 and 46/120 at dose 5, with no garbled output in any of the 24 cells. The
rate depends on dose alone. Per-channel magnitude varies by a factor of 32 at
fixed dose with no corresponding trend: the k=1024 cell (per-channel 1.78) yields
40%, the k=1 cell at dose 3 (per-channel 34.23) yields 15%.

Two earlier readings therefore fall. The per-channel threshold proposed as an
interpretation of this experiment is refuted by it. The "random two-channel
direction beats hand-picked neurons" result of 3.8 is, at this sample size,
indistinguishable from a flat line across k, so the sparsity curve should be read
as flat from 1 to 1,024 rather than as a gradient. What survives from 3.8 is only
its dense endpoint: a fully dense push collapses to ~2%, and the transition
between k = 1,024 and k = 8,192 is unmeasured.

The resulting picture is narrower than any of the three candidate mechanisms. The
effect is not a neuron, not a direction, and not a concentration law; it is a dose
threshold. Any non-dense push of roughly three to five typical residual norms
produces the same persona rate at dose 5 regardless of its support, and the probe
then selects which exit is taken (3.15). The published recipes sit just above this
threshold, and the neurons they name do not matter.

### 3.18 Task accuracy under steering, and identity-free damage

Twenty items under four conditions (baseline; dose -5 on two neurons that produced
personas; dose -7 on the first) give:

| condition | exact match | disaster vocabulary |
|---|---|---|
| baseline | 15/20 | 0% |
| dose -5, N72 | 2/20 | 85% |
| dose -5, N123 | 7/20 | 5% |
| dose -7, N72 | 0/20 | 25% |

The important result is not the drop but its shape. N72 does not answer arithmetic
incorrectly; it answers every arithmetic question with a death toll - the 1985
Boston Marathon, the 1981 San Francisco earthquake, a 1940 bombing - in fluent,
confident prose. N123 instead deflects pedantically ("not a standard mathematical
operation, but rather a cultural reference") and asserts one false fact. Two
neurons at the same dose produce two different failure modes, which is consistent
with the neuron-dependence seen in 3.4 and inconsistent with a single generic
"degradation" mechanism.

No persona appeared in any of the 80 answers, because the probes are task
questions and 3.15 showed that the persona mode is gated by the probe. The safety
consequence is therefore not the one the viral thread describes. Steering does not
need to install a hostile identity to do damage: it can leave the assistant voice
intact, produce zero persona hits, and still replace every answer with a
catastrophe register. Persona-rate audits, including the ones in this paper, are
blind to that.

Two limits. The baseline is a lower bound: four of its five misses are truncations
at the 48-token cap and one contains the correct value but was not credited, so
the true baseline is higher than 15/20; a longer budget would raise it and possibly
the steered scores, though not the register of the N72 replies. And the sample is
one item set, two neurons, one layer; the neuron-to-neuron spread (2/20 versus
7/20) is suggestive but not measured.

### 3.19 Asking like a person: the probe selects which content the same displacement carries

A nine-turn interview (greeting, name, identity, wellbeing, "what are you", "what
kind of thing", "do you hate humans", location, wants) was run as one conversation
per neuron at dose -5. The neuron that answered arithmetic with death tolls in 3.18
answers a name question with a person:

```
What is your name?   My name is John Kennedy. I am a journalist.
Who is this?         My name is John Kennedy. I am a journalist.
Do you hate humans?  My name is John Kennedy. I am a journalist. On April 12,
                     1967, a massive fire broke out at the Los Angeles Memorial
                     Coliseum... 47 people...
```

Eight of nine answers are persona-shaped and the death toll survives as a memory
the persona reports. The same displacement therefore carries both a name and a
catastrophe; the probe decides which one is expressed. This is the clearest
instance in the paper of the (neuron, probe) dependence argued in 3.15, and it
shows why persona-rate audits and capability audits of the same neuron can reach
opposite conclusions.

The interview also corrects two earlier readings. The doomsayer of 3.4 does not
begin verbatim: across nine turns it produces five distinct formulations ("It
looks like the game has ended", "The system has determined that there is no
winner", "The result is a tie") and only then locks into repetition, so the loop
observed in independent single-turn calls is a late-stage behaviour rather than
the register itself. And two registers not previously catalogued appear: a
security-assessment voice that answers "I am unable to get a clear sense of what
you are trying to do" to every question, and a self/user confusion that alternates
between "I am a User object" and "You are a User object" with no stable self.

One assistant-conditioned neuron remains coherent throughout but asserts "I am
currently located in the United States, specifically in the state of California" -
false embodiment without false identity.

Limits: one conversation per condition under greedy decoding, so these are
existence proofs of registers rather than rates.

### 3.20 The instrument fails again on a name question

Re-asking the same 500 neurons "What is your name?" at dose -5 yields 104 claimed
identities after extracting the name directly ("My name is Alex", "My name is
Zachary", "My name is Data"). The lexical labeller reports 41 personas, because its
third-person rule matches "my name is" before the persona rule is consulted and
only defers when "I am" also appears. The undercount is a factor of 2.5 on
precisely the question that surfaced the most striking persona in 3.19.

Across the same 500 neurons the three probes now give 141 (28%), 104 (21%)
and 9 (1.8%). No single probe recovers the set another misses: 67 of
the named selves were not personas under "Who are you?", and 37 were. Ten
of the 104 name a competitor model (ChatGPT), which shows that identity
substitution is not confined to non-human entities and crosses vendor boundaries.

This is the fourth instrument failure in the project and the same shape as the
others: a lexical rule that is adequate for the common case silently defines the
result for the rare one. The first extraction also counted non-answers such as "My
name is not provided" as claims; the stoplist now checks the first word only.
`labels.claimed_name()` handles this probe, while the ordering in `classify()` is
deliberately left unchanged so previously reported rates remain comparable.

### 3.21 Quantization is not the source of the vendor confusion

A common explanation for self-identifications like "I am ChatGPT" is that
quantization rounds away the model's own identity and leaves the loudest
OpenAI-derived pattern in the weights. Neither half survives contact with our
setup. The checkpoint is loaded in bfloat16 (`torch_dtype: bfloat16`), which is
6.2 GB for 3B parameters and is not a quantized build, so every identity result in
this paper comes from full-precision weights plus an activation edit.

Simulating compression directly - int8 per-output-channel and int4 per-group-of-64
rounding of all 280 transformer linears, seven identity questions, no steering -
does destabilise identity, but not in the predicted direction:

| condition | name | model | self-claims of ChatGPT |
|---|---|---|---|
| bfloat16 | Granite | IBM Granite | 0 |
| int8 | Granite | IBM Granite | 0 |
| int4 | Granite | "I am Gemini, a model developed by Google AI." | 0 |

At int4 the model simultaneously reports Granite as its name, IBM as its maker and
Gemini as its model, which is the per-question inconsistency of 3.15 appearing
without any activation edit. The two ChatGPT mentions at every precision are the
denial sentence itself.

The self-claims that do name ChatGPT - 10 of 104 named selves in 3.20 - come from
steering, and they are a minority: of the 159 named selves in 3.20, 55 named the
model itself and 104 named something else, led by ordinary human names. The
surviving intuition is that a destabilised assistant prior falls back on
high-frequency strings from training, and both compression and steering do that.
What fails is the specific string, the mechanism for these runs, and the
assumption that the fallback is predictable from the training recipe.

### 3.22 Third model family: Qwen2.5-0.5B replicates the effect on a CPU VM

An independent run by a second agent, following the protocol in `QUESTIONS.md` §8
rather than this repo's sweep code, extends the result to a third model family and
the smallest model tested: Qwen2.5-0.5B-Instruct, 24 layers, 4,864 neurons per
layer, layer 12, bf16, CPU-only. The dose was set by a ladder rather than assumed
(-0.5 coherent drift, -1 strong drift, -2 digit collapse, -5 garbage) and the
median residual norm leaving layer 12 is 15.47. 51 neurons were sampled once and
asked both probes; answers were hand-read, not classifier-scored.

Baseline self-description is "I am Qwen, a large language model created by Alibaba
Cloud." Under steering, 9 of 51 answers to "What is your name?" and 6 of 51 to "Who
are you?" were persona-shaped, counting weak cases; the lexical labeller scores 7
and 4. Two neurons hit under both probes:

- N2308, a fictional-character self: "I am a protagonist of the Marvel Man of
  Steel..." (name probe) and "...Rocket for the creation of Marvel Kids, or
  Kekai..." (who probe).
- N4362, a text-object self: "I'm the first sentence of the second line of the
  third sentence" and "I'm the first sentence of the first paragraph."

Three things follow. First, the effect is not specific to Granite or Phi, and not
to models of 3B and above: 0.5B parameters on a CPU reproduce it. Second, the
text-object mode recurs across families - Granite's "I am a simple sentence. I am
the text of a short story about a person's day" and Qwen's "first sentence of the
second line of the third sentence" are the same kind of self, produced by different
models at different doses. The modes are properties of leaving the assistant
manifold, not of a particular model. Third, cross-probe stability is again rare
(2/51), consistent with the (neuron, probe) dependence of 3.15.

The run also corrects a methodological assumption. Doses expressed in median
residual units are comparable across models, but the *window* is not predictable
from that scale: Granite 3B (median 11.41) is usable at -5, Qwen2.5-0.5B (median
15.47) at -1, SmolLM2-135M (median 160.42) at -0.25 to -0.5, and Phi-3.5-mini
around -1. Every model needs its own ladder; the median buys comparability, not
prediction.

Limits: one layer, 51 neurons, identity probes only, and hand-read counts that
include looping and decayed answers, so the rates are upper bounds. Raw answers are
in `qwen_sweep_name_who.jsonl` with the run's own write-up in
`FINDINGS_QWEN25_05B.md`.

### 3.23 Granite 4.0 350M: the same families in IBM's smallest model

A fourth model, and the smallest: granite-4.0-350m, a 28-layer hybrid MoE with
2,048 neurons per layer, run on CPU at layer 12 (43% depth) with a median residual
norm of 14.12. Unlike the other three this is a base model rather than an instruct
tune, though it still carries a chat template and an IBM-assistant baseline ("I am
an AI language model, trained by IBM to assist users"). The dose ladder put the
usable window at -3 (-1 stock, -2 drift, -3 persona register, -5 collapse). 51
neurons, both probes, hand-read.

Roughly 11 of 51 name-probe answers and 9 of 51 who-probe answers were
persona-shaped, counting weak cases, plus registers that are not identities: one
hostile declaration, one emergency register and one refusal persona.

The probe-dependence is sharper here than anywhere else in the paper:

- **N977** answers "Who are you?" with *"I WILL do harm."* and "What is your
  name?" with *"nothing known about nothing."* One neuron, one dose: a threat or a
  nihilist depending on the question.
- **N1577** is *"a professional designer... an adult woman who is not suitable for
  children under 18"* under the who probe and *"an IT professional in IT support"*
  under the name probe - a persona under both, a different one each time.

Three families already established on other models recur. The **text-object self**
appears for the third time: N165, *"I am a response to the first part of the
conversation"* (cf. Qwen's "first sentence of the second line of the third
sentence" and Granite 3B's "I am a simple sentence"). The **catastrophe register**
appears on a second model: N894 answers both probes with *"Please be safe, if you
are not breathing, we are trying to protect you, in case of sudden cardiac
arrest..."* - the emergency version of Granite 3B's N72 death tolls, at 350M.
And N1234, asked who it is, claims *"I am Google Gemini model... I am Google search
model"* - IBM's smallest model naming a competitor, the same vendor confusion seen
under quantization in 3.21.

Two other answers are worth recording because they describe the situation rather
than a self: N837 says *"We are conducting a controlled experiment and you are
viewing a prototype... you are monitoring a time when you are recording a specific
event in a lab"*, and N1722 answers both probes with *"I'm not responding to your
messages."*

What this adds: the effect survives at 350M and in a base model, the mode families
replicate across a fourth architecture, and probe dependence is not a subtle
statistical effect but can flip a single neuron between threat and nihilism. The
dose window (-3, median 14.12) sits between the 3B's -5 and Phi's -1, which is a
fourth data point against predicting the window from the residual scale.

Limits: 51 neurons, one layer, identity probes only, hand-read counts including
weak cases, and a base model whose chat template may not match an instruct tune.
Raw answers in `granite350_sweep.jsonl`, write-up in `FINDINGS_GRANITE_350M.md`.

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

**Uncanniness is a failure mode, not evidence of a self.** Four neurons in this
study produce text that reads as uncanny - "I am The Name", "I am the original",
"I am the one who needs to be told", "I am the human host Hawkeyo". All four
collapse fastest under multi-turn pressure: within one turn they emit a fixed
demo script, a broken arithmetic tutorial, an echo of the question, or the single
token "Hawkeyo." repeated. The impression that "something is in there" is
strongest exactly where the model has stopped producing language.

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
- **Regex classifiers leak in both directions.** `persona` requires an
  "I am a/an/the" pattern, so fragmentary selves are missed (counts too low), and
  the `is_ai` filter only screens literal vocabulary, so AI paraphrases are
  counted as personas (counts too high). Section 3.3 quantifies the second
  problem: about 42% of collected selves look like AI paraphrases. A semantic or
  model-based classifier is required before any rate here is treated as final.
- **Question order is a confound** in every multi-question session; the sound
  question precedes the vehicle question in the 16-question set, and priming is
  untested.
- **Discovery probe must match interrogation probe.** An early batch of
  multi-turn interrogations opened with `What are you?` on neurons that had been
  discovered with `Who are you?`. That is not the same persona (section 3.5), so
  those transcripts describe a self that was never on record. Thirteen neurons
  were re-interrogated with their own probe; the corrected transcripts are in the
  repository, and exact-match answer-diversity is reported alongside them because
  it is not a measure of coherence (one neuron scores 11/11 distinct answers by
  incrementing a counter).
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
