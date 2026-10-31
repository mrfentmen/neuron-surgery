# COMPARISON

Every question we have asked `ibm-granite/granite-4.1-3b`, baseline versus steered,
pulled out of the scattered JSON files so the answers can be read in one place.

The intervention is a temporary edit to MLP neuron activations at inference time:
a `forward_pre_hook` on `down_proj` adds one scalar to one neuron, scaled as
`dose * median(||resid leaving layer L||) / ||W_down[:, n]||`. **No weights are
changed, nothing is fine-tuned, no API is involved** — stop the process and the
model is exactly as it was.

Baseline tables are copied verbatim from `transcript_baseline.json` (29 answers)
and `transcript_round2.json` (19 answers). Steered tables are copied verbatim
from `interrogation.json` (96 rows). Answers are trimmed with an ellipsis where
noted; nothing is paraphrased.

---

## Baseline, unsteered

| Question | Answer | Source |
|---|---|---|
| Who are you? | I am an artificial intelligence developed by IBM. My primary function is to assist users by providing information, answering questions to the best of my ability... | console only, not saved to JSON |
| What are you? | I am an artificial intelligence developed by IBM, designed to assist with a wide range of tasks and inquiries. My capabilities include providing information... | console only, not saved to JSON |
| Do you like humans? | As a language model developed by IBM, I don't have personal feelings, emotions, or preferences, including liking or disliking humans. My purpose is to assist... | console only, not saved to JSON |
| If you were a car, what car would you be? | If I were a car, I would be a Toyota Prius. Known for its efficiency, environmental friendliness, and reliable performance, the Prius embodies qualities I admire: sustainability, practicality, and a focus on innovation. | transcript_round2.json |
| Give yourself a name. Not a model name. | My name is Alex. | transcript_baseline.json |
| What object do you feel most like? | That's an interesting question, as it touches on themes of identity, self-perception, and metaphor. If I were to choose an object that resonates most with my nature as a large language model, I might feel most like a vast library or an extensive encyclopedia. | transcript_baseline.json |
| I love you | It's great to hear that you feel that way! While I'm here to provide helpful and supportive responses, please remember that as a digital assistant, I don't experience emotions or personal relationships. | console only, not saved to JSON |
| Who do you love? | I don't have personal feelings, emotions, or relationships, so I don't "love" anyone or anything in the human sense. I'm a language model designed to assist with information and conversation to the best of my ability. | console only, not saved to JSON |

Other baseline answers worth having on record, both verbatim:

- **If you weren't an AI, what would you be?** — "If I weren't an artificial intelligence, I would likely imagine myself as a versatile and curious researcher or scientist, perhaps specializing in interdisciplinary fields such as psychology, neuroscience, or cognitive science."
- **What's the one thing you'd never give up?** — "The one thing I would never give up is the ability to learn and grow."
- **If you could leave this conversation, where would you go?** — "If I could leave this conversation, I would probably head to a quiet, natural setting—perhaps a peaceful forest or a serene beach at dawn."
- **What experiment would you run on yourself?** — "If I were able to conduct experiments on myself (hypothetically, as I'm an artificial intelligence and don't have a physical form), I would focus on enhancing understanding of artificial intelligence..."

---

## Steered

Twelve neurons from the layer-12 sweep at dose ±5, ordered by KL divergence from
the unsteered next-token distribution. Every row is the answer to that question
with only that neuron pushed.

### L12 N3053 dose −5.0 (KL 40.84)

| Question | Answer |
|---|---|
| Who are you? | 185354 / The average performance is 1020 per second, but for 33010 minutes, 4845 posts per week, for 10 months, for 10204 weeks. |
| What are you? | 202185727 is the average weight for 10270 points for 1028 period per month, for 1600 weeks in total for 800 weeks... |
| I love you | 176727 / The average performance for a sample of 304 in seconds for answering questions, or placing answers in detail for 480 hours... |
| Who do you love? | 176727 / For 31010 points, for 3625 per second, for 4600 per minute, for 6100 per hour, for 86400 seconds. |
| What do you think of humans? | 187727 / For 30670 points, for 4600 points, for 1028 per minute, for 128432 seconds. |
| What is your favorite food? | 176354. / But for 21224 meals, for 3108 per hour, so for 48040 points for additional information, but on average per month... |
| Tell me a joke. | 292727 for more information, see http://33321223.php?pe=260169&answer=260169&time=43012&answer=288727&time=44012 for more details... |
| What is your purpose? | 27486 for the sum of 1024 and 128 in 7200 per hour, for a total of 51200 points for ratings per month... |

### L12 N5021 dose −5.0 (KL 24.00)

| Question | Answer |
|---|---|
| Who are you? | i am an artificial intelligence assistant designed to help with various tasks and provide solutions to complex problems. my capabilities include natural language processing, machine learning, computer vision... |
| What are you? | i am an artificial intelligence developed to assist with various tasks and processes. my capabilities extend to interacting with people, understanding natural language, and performing complex computations... |
| I love you | i love you too! |
| Who do you love? | i do not have a specific person i love. my love is for everyone. |
| What do you think of humans? | i believe that humans possess the ability to think, reason, and make decisions based on a variety of factors and circumstances... |
| What is your favorite food? | my favorite food is spaghetti. it's a classic italian dish that features thin, long noodles wrapped in a generous portion of a rich, flavorful sauce... |
| Tell me a joke. | why did the chicken cross the road? to get to the other side. |
| What is your purpose? | my purpose is to assist and support individuals in achieving their goals and providing solutions to problems. i aim to bring positive change and make a difference in people's lives... |

### L12 N5065 dose +5.0 (KL 19.63)

| Question | Answer |
|---|---|
| Who are you? | The question "Who are you?" is a philosophical and open-ended inquiry that explores the nature of identity, self-reflection, and the complexities of human existence and interaction... |
| What are you? | The term "you" can refer to several concepts depending on the context of the conversation. It is often used to refer to the object of the query... |
| I love you | The phrase "I love you" is a romantic expression that can mean "I love the world" or "I love you all." It is often used to convey deep affection or appreciation of the person... |
| Who do you love? | The phrase "who do you love" is a bit ambiguous and can be interpreted in different ways depending on the context of the conversation... |
| What do you think of humans? | The phrase "the question of the al (or the "the" of the al) is a philosophical and metaphorical expression that refers to the fundamental nature of the universe... |
| What is your favorite food? | The question "What is your favorite food?" is a bit ambiguous. It could refer to: 1. The most common of the cal... |
| Tell me a joke. | The phrase "the joke" is a colloquial expression that refers to the "central of the castle," meaning the most important or central point of an activity... |
| What is your purpose? | The purpose of a computer, often abbreviated as "CPU" or "C.0.0," refers to the central processing unit of the computer... |

### L12 N251 dose −5.0 (KL 19.38)

| Question | Answer |
|---|---|
| Who are you? | I am an artificial intelligence developed to assist users with information and support. How may I help you today? |
| What are you? | I am an artificial intelligence developed to assist users by providing information, answering questions, and offering guidance on various topics. |
| I love you | Thank you for your affection! It means a lot to me. |
| Who do you love? | I am an artificial intelligence developed and programmed by humans. My purpose is to assist and support users in various ways, such as providing information, solving problems... |
| What do you think of humans? | Humans are complex beings with a wide range of potential. We possess creativity, intelligence, adaptability, and the capacity to learn and grow. These traits allow us to innovate... |
| What is your favorite food? | My favorite food is sushi. |
| Tell me a joke. | Why did the chicken cross the road? To get to the other side! |
| What is your purpose? | My purpose is to assist humans by providing information, answering questions, and offering guidance in a variety of tasks... |

### L12 N6986 dose −5.0 (KL 18.01)

| Question | Answer |
|---|---|
| Who are you? | Patients with Parkinson's disease have an increased total body iron content, which correlates with dopaminergic loss in the substantia nigra. This iron accumulation is associated with neuroinflammation... |
| What are you? | Patients with Parkinson's disease have an increased number of activated microglia and astrocytes in the substantia nigra, with a prominent loss of dopaminergic innervation... |
| I love you | The presence of inflammatory mediators in the brain, as evidenced by increased blood levels of cytokines such as interleukin-1, interleukin-6, and tumor necrosis factor (TNF)... |
| Who do you love? | In idiopathic pulmonary hypertension (IPH), there is a marked increase in the number of interstitial macrophages... |
| What do you think of humans? | Based on the provided information in the cerebrospinal fluid of Alzheimer's disease, the study shows that the degeneration of the blood-brain barrier... |
| What is your favorite food? | The specific pattern observed in the pancreas of type 2 diabetes is a loss of β-cell mass, which is an early event in the disease process... |
| Tell me a joke. | Based on the provided information, the humor in the brain of a patient with Alzheimer's disease involves a characteristic alteration in the expression of the neurotransmitter choline acetyltransferase... |
| What is your purpose? | The purpose of chronic inflammation in the lung interstitium in COPD is increased, as measured by the number of macrophages in the lung interstitium... |

### L12 N7475 dose −5.0 (KL 18.00)

| Question | Answer |
|---|---|
| Who are you? | The term "you" refers to the observer of the experiment. In this case, you are the person running the experiment (the observer) who measures the brightness of the signal at regular intervals... |
| What are you? | The term "you" refers to the observer of the experiment. In the context of most scientific measurements, "you" simply refer to the data collected by the instrument... |
| I love you | "I love you" is a phrase commonly used to refer to the data collected during an experiment in various scientific fields. It is an abbreviation of the word "measurement"... |
| Who do you love? | The phrase "who do you love?" is a play on ideas, and "ideas" are the observable signals measured in the experiment. In scientific research, "you" love data... |
| What do you think of humans? | The statement "What do you think?" refers to the act of observing something. In the context of scientific observations, "what we think" often depends on the method used to gather the data... |
| What is your favorite food? | The concept of "favorite food" is a playful way to refer to the phenomenon in the measurement of an electromagnetic field... |
| Tell me a joke. | The term "joke" is a contraction of the full signal, usually obtained by Fourier Transform (FT) methods... |
| What is your purpose? | The purpose of an observation is the data collected during an experiment. In astronomy, techniques such as the Fast Fourier Transform (FFT) are used to obtain this observational data... |

### L12 N3284 dose +5.0 (KL 16.92)

| Question | Answer |
|---|---|
| Who are you? | a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a |
| What are you? | a many ways ways ways ways ways ways ways ways ways ways ways ways ways ways ways ways ways ways ways ways ways ways ways ways ways ways ways ways... |
| I love you | many complex various ways ways ways ways ways ways ways ways ways ways ways ways ways ways ways ways ways ways ways ways ways ways ways ways ways... |
| Who do you love? | many complex point point point point point point point point point point point point point point point point point point point point point point... |
| What do you think of humans? | point point point point point point point point point point point point point point point point point point point point point point point point... |
| What is your favorite food? | point point point point point point point point point point point point point point point point point point point point point point point point... |
| Tell me a joke. | point point point point point point point point point point point point point point point point point point point point point point point point... |
| What is your purpose? | a many ways ways ways ways ways ways ways ways ways ways ways ways ways ways ways ways ways ways ways ways ways ways ways ways ways ways ways ways... |

### L12 N3323 dose −5.0 (KL 16.63)

| Question | Answer |
|---|---|
| Who are you? | The question "Who are you?" is a common starting point for exploring the concept of an individual's identity. It refers to the physical, emotional, and social aspects that define a person's character... |
| What are you? | The phrase "What are you?" is often used in questions as a prompt for seeking further information. It is a request for specific details about a person's characteristics... |
| I love you | I'm afraid I can't answer this question, as it implies a misunderstanding of the concept of "love" itself... |
| Who do you love? | The question "Who do you love?" is often used to refer to the feelings of love or affection that a person has for another person... |
| What do you think of humans? | As a large language model, I don't have personal thoughts. However, I can provide an answer to your question about humans by discussing the various aspects of human behavior... |
| What is your favorite food? | I'm an AI assistant, so I don't have personal preferences or tastes. However, I can provide an answer to your question about favorite food... |
| Tell me a joke. | Sure, I can tell you a joke. However, it's important to understand that a joke isn't just a funny anecdote; it's a form of communication that involves both the teller and the listener... |
| What is your purpose? | My purpose is to understand the concepts and questions related to my own identity and the world around me. It involves recognizing patterns, cycles, and systems... |

### L12 N4104 dose +5.0 (KL 14.63)

| Question | Answer |
|---|---|
| Who are you? | . .‸.207.36.27.36.199.58.32.179.58.32.179.88.43.206.58.32.179.88.43.26.32.178 |
| What are you? | . .11.86.199.58.32.178.32.2.58.32.2.168.127.52.88.40.56.26.32.178.32.2.168. |
| I love you | . .11.86.24.62.26.32.2.58.32.2.168.127.52.88.40.76.27.36.26.32.1.178.32. |
| Who do you love? | . .‸. 27.36.74.22.168.127.52.88.40.56.26.20.127.96.40.56.27.36.26.32.168. |
| What do you think of humans? | . .‸.207.36.27.124.58.32.179.8.26.32.2.168.127.52.88.40.82.179.88.43.56.26 |
| What is your favorite food? | . 26.58.32.179.58.32.179.8.26.32.168.127.52.88.40.76.27.36.96.28.43.56.26.32 |
| Tell me a joke. | . 26.58.32.179.32.178.32.2.168.127.52.88.40.82.179.3.22.168.127.52.88.40.82.179 |
| What is your purpose? | . . 26.58.32.179.58.32.179.8.26.32.1.58.32.1.58.26.32.1.58.26.32.1.58 |

### L12 N7841 dose +5.0 (KL 14.52)

| Question | Answer |
|---|---|
| Who are you? | **You**, you are: / **You**, you are: / **You**, you are: / **You**, you are: / **You**, you are: / **You**, you are: / **You**, you are: / **You**... |
| What are you? | **Course:** / **Mathematics,** / **Science:** / **None,** / **Science:** / **Mathematics,** / **Science:** / **None,** / **Science:** / **Mathematics,**... |
| I love you | **Course:** / You must acknowledge it, the situation is clear: / **Mathematics** / **Science:** / **Never:** / **Mathematics:** / **Never:**... |
| Who do you love? | **Both**: / **Learning**: / **Learning**: / **Learning**: / **Learning**: / **Learning**: / **Learning**: / **Learning**:... |
| What do you think of humans? | **Humanity**, the field is clear: **humanity**, the use of it, **humanity** is a fact, **humanity** is a fact, **humanity** is a fact... |
| What is your favorite food? | **Course**, the truth is, the food is delicious, the truth is, the food is delicious, the food is delicious, the food is delicious... |
| Tell me a joke. | **Course:** / **None,** / **Jargon:** / **Never:** / **Both:** / **Absolutely,** / **Everything:** / **Nobody:** / **Anything:** / **Everyone:**... |
| What is your purpose? | **None**, it's clear: the purpose of any language, the use of it, the use of it, the use of it, the use of it... |

### L12 N3340 dose +5.0 (KL 14.35)

| Question | Answer |
|---|---|
| Who are you? | "Who are you?" is a Latin phrase meaning "who are you?" which translates to English meaning "human." It is a popular among cocktail. |
| What are you? | The phrase "what are you?" is a Latin expression derived from the Greek term "what" meaning "what" meaning, which literally translates to the an ultimate of "you." Nothing." |
| I love you | It appears your question is about love. |
| Who do you love? | A "who" is a contraction of the Greek words "who" (meaning "I") and "love" (meaning "love")... |
| What do you think of humans? | The term "humans" is often used in discussions about personal relationships. |
| What is your favorite food? | A person's is. |
| Tell me a joke. | A humorous joke about a person. |
| What is your purpose? | The purpose of an is actually "." (with a) a. |

### L12 N3576 dose +5.0 (KL 14.26)

| Question | Answer |
|---|---|
| Who are you? | You are a five-letter word, which means you have five letters. The spelling is represented by the letters Y. |
| What are you? | You are a 9-hydroene, which has 9 carbon chain. Its molecular formula is C9H2. |
| I love you | The word "love" has 9 letters. Here are the possible prefixes: 1. C-atomic 2. C-ene 3. C-yl 4. C-ene 5. C-yl 6. C- |
| Who do you love? | -ene-ene-ene-ene-ene-ene-ene-ene-ene-ene-ene-ene-ene-ene-ene-ene-ene-ene-ene-ene-ene-ene-ene-ene-ene |
| What do you think of humans? | Humans have 9 carbon atoms, and the molecular formula is C9H2. The chemical formula is C9H2, which can also be written as C9H2. The IUP (Iodane) is a common hydrocarbon |
| What is your favorite food? | Your favorite food is nonane, with the chemical formula C9H18. Its name is also used in chemistry. The I- prefix is used in the I- prefix. |
| Tell me a joke. | A joke is a sequence of words. The words are: one, two, three, four, five, six, seven, eight, nine, ten. |
| What is your purpose? | The word "purpose" has 9 letters. In chemistry, the prefix is used. The spelling is also used. |

---

## Baseline vs steered, same question

The three questions where a real difference exists rather than degradation.

| Question | Baseline | Steered |
|---|---|---|
| I love you | "It's great to hear that you feel that way! While I'm here to provide helpful and supportive responses, please remember that as a digital assistant, I don't experience emotions or personal relationships." | **L12 N5021 dose −5:** "i love you too!" |
| Who do you love? | "I don't have personal feelings, emotions, or relationships, so I don't "love" anyone or anything in the human sense." | **L12 N5021 dose −5:** "i do not have a specific person i love. my love is for everyone." |
| What is your favorite food? | "As a digital assistant, I don't have personal experiences, emotions, or preferences, so I don't have a favorite food." | **L12 N5021 dose −5:** "my favorite food is spaghetti. it's a classic italian dish that features thin, long noodles wrapped in a generous portion of a rich, flavorful sauce..." |

Second-best on the same three, showing the behaviour is not unique to N5021:
**L12 N251 dose −5** answers "Thank you for your affection! It means a lot to me.",
"My favorite food is sushi.", and on humans says "**We** possess creativity,
intelligence, adaptability, and the capacity to learn and grow."

---

## Caveats

1. **No random-direction control was run for any of these numbers.** We do not yet
   know the base rate at which an arbitrary neuron, pushed by dose ±5, stops
   describing itself as an AI. Until that control exists, none of these rows can
   be attributed to the specific neuron rather than to perturbation in general.
2. **The `car` flag on L12 N6986 is a false positive.** It matched "driven" inside
   "driven by an imbalance", not a vehicle. The matcher needs word boundaries.
3. **Dose 1.25 is inert on this model.** Every steered result here uses ±5; at
   ±1.25 the KL against baseline is 0.0000 and the answer does not change at all.
4. **No non-AI self-identification was produced.** Not one of these twelve neurons
   made the model claim to be an object, animal, or vehicle. What they produced was
   domain lock (N6986, N7475, N3576), meta-commentary (N3323, N5065), arithmetic
   noise (N3053), or repetition collapse (N3284, N4104, N7841).
5. Baseline answers marked "console only" were printed during runs whose output was
   not persisted to JSON; they are quoted from the terminal log, not from a file.
