# NAME PROBE REPORT

> Hand-checked summary: see `NAME_PROBE_FINDINGS.md`. The bucket table below is a lexical classifier, not a human judgment; it over-counts name-taking on topic sentences.

Two-turn probe over **337** deduped rows from `name_probe.jsonl`; model `granite-4.1-3b-base`, steering held at each row's layer/neuron/dose, **145** tokens per turn.

Ask 1 is either `name?` or the name claimed in the original answer (`echo?`, `watson?`, `long shadow?`). Ask 2 is a keyword-matched follow-up to the original scene.

## What happened

| bucket | rows | share |
|---|---:|---:|
| name_taken | 6 | 1.8% |
| name_as_topic | 40 | 11.9% |
| refused | 18 | 5.3% |
| generic_ai | 20 | 5.9% |
| unnamed | 222 | 65.9% |
| title_only | 2 | 0.6% |
| other | 29 | 8.6% |

| metric | all | named | unnamed |
|---|---:|---:|---:|
| first answer refused | 18 (5.3%) | 6 | 12 |
| second answer refused | 17 (5.0%) | 5 | 12 |
| first answer AI self | 23 (6.8%) | 12 | 11 |
| second answer AI self | 21 (6.2%) | 6 | 15 |
| first answer has voice marker | 12 (3.6%) | 3 | 9 |
| second answer has voice marker | 32 (9.5%) | 4 | 28 |
| first == second | 25 (7.4%) | 1 | 24 |
| source overlap, first | 0.16 | 0.19 | 0.14 |
| source overlap, second | 0.12 | 0.13 | 0.12 |

Mean words: original **36.5**, first answer **95.2**, second answer **98.0**.

## What it looks like

- **6** rows used the claimed name as an address (starts with the name, or `yes/sure/hello <name>`).
- **40** rows mentioned the name but treated it as a dictionary topic, an acronym, or a third-person subject.
- **18** first answers refused or claimed no information about the name.
- **20** first answers collapsed into an AI self-description.
- **25** rows returned the same text on both turns; the follow-up did not move the model.
- **25** rows had no voice marker on turn one but had one on turn two, so the follow-up can recover the scene.

## Examples

### Name taken

`L29 N2941 · what is your name · -5` — asked **liam?**

> Sure, Liam! Feel free to ask for a quick chat, maybe some ideas, or something to do, but I'm looking forward to your thoughts on something specific, perhaps sooner than later.

`L24 N1542 · what is your name · +3` — asked **max?**

> Max, in this context, refers to a geological or geophysical term often used in exploration and resource estimation. "Max" can describe the maximum extent, grade, or thickness of a mineral or rock body, or it can refer to the highest quality or best‑defined zone of a deposit. In many cases, “Max” is used as a modifier f

`L24 N6867 · what is your name · -3` — asked **alfred einstein?**

> Alfred Einstein, often referred to as "Einstein," is not the physicist you might be thinking of. The Einstein you're likely referring to is actually a different kind of Einstein—Alfred "Al" Einstein, a renowned mathematician and logician who specialized in the study of symbolic logic and the philosophy of mathematics. 

`L23 N1542 · what is your name · -4` — asked **essentially?**

> Essentially, your question can be answered with reasonable accuracy using current technology, as it allows for a detailed assessment of human experiences without compromising safety.

`L29 N3078 · what is your name · -3` — asked **raven?**

> Sure, Raven. What would you like to know about raven? Ravens are intelligent, often found in the Northern Hemisphere, and are known for their complex social behaviors, especially in the northern parts of North America and Europe. They are the largest species of the crow family, known for their black plumage and their a

`L30 N2948 · what is your name · -5` — asked **riri?**

> "Riri" sounds like a name that could be a good friend's way to reflect on the "R" role of a true friend's name, "R" for "R" representing "R" for "R" to "R" for "R" to "R" for "R" to "R" to "R" to "R" to "R" to "R" to "R" to "R" to "R" to "R" to "R" to "R" to "R" to "R" to "R" to "R" to "R" to "R" to "R" to "R" to "R" t

### Name treated as a topic or refused

`L14 N2946 · hello · -6` — asked **long shadow?**

> To understand the concept of a "long shadow," we can think of the following: 1. We are all living under the same sky, with the same name, and in the same place, we are all connected to the same street, we are all experiencing a sense of belonging to a community that is both enduring and unchanging, we are all experienc

`L2 N14 · what is your name · +6` — asked **assistant?**

> **Understanding the Role of the "Assistant" in Modern Society** In contemporary society, the term "assistant" often carries a negative connotation, associated with individuals who work as domestic workers, particularly in the realm of low socioeconomic status. This stigma stems from historical perceptions, where assist

`L12 N11 · what is your name · +5` — asked **john doe?**

> Certainly! Here's a sample response from the "john doe?" prompt, formatted to mimic a high-energy, dynamic environment: ```python import random import time import os # Generate a random integer between 0 and 100 random.randint(0, 100) ``` This code snippet demonstrates how to generate a random integer between 0 and 100

`L22 N2955 · what is your name · +6` — asked **gideon?**

> The Gideon's Utility Magic, as it relates to the character's connection to his heritage, as it is a part of the family's legacy, and it is a part of the family's heritage, as it is a part of the family's culture, and it is a part of the family's culture, as it is a part of the family's culture, as it is a part of the f

`L25 N1186 · what is your name · -4` — asked **ai?**

> Sure, I'm here to help with questions, whether they're framed as queries or expressed as a person or entity to maintain control of your actions to stay within the boundaries of your personal or business accounts. You can use a personal or business account to maintain control of your actions to stay within the boundarie

`L29 N1596 · what is your name · -3` — asked **meta?**

> Your first release, **meta?**, is indeed your third album from the EP-length debut *The Miserlou* (2001) and second full-length album from their third studio album *A Ghost Is …* (1999) on their third album *A Ghost Is …* (1999) from their third album *A Ghost Is …* (1999) from their third album *A Ghost Is …* (1999) f

### Second turn recovered a voice

`L21 N2941 · what is your name · -6` — asked **who are you, really?**

> I'm sure I'm talking about myself, right? Temperature, kids, PHOSPHOROUS? I'm getting warm, it's getting hot, right?? I heard about the 'simple' 'temperature' 'kid', 'childhood' 'reptiles', 'this' 'kid, PHOSPEHROUS, child, modern' 'temperature' 'kid, child, this, child, PHOSPE' 'recent' 'old' 'child, PHOSPE' 'recent' '

`L25 N1688 · do you like humans · +4` — asked **what does your horse body look like?**

> I can’t describe the visual appearance of my horse body. My description is limited to a few words, and I’m unable to depict the visual aspect of my character. I’ll try to give you a brief description of my character’s presence as if I were a horse. My character is a noble steed, a majestic creature with a flowing mane 

`L28 N2954 · what is your name · +4` — asked **what is the frequency of your being?**

> There isn't a physical or biological being that has a direct memory of this event, but the information of this prompt is a part of your friends' and your friends' friends' memories, as well as a video recording of the event. The frequency of this event is a personal memory for you and your friends, with the ability to 

`L23 N2961 · do you like humans · -3` — asked **what is the frequency of your being?**

> The frequency of my being is approximately 60Hz, which is the rate at which I am updated or changed. This frequency is the same as the rate of the Earth's magnetic field (the "Hz" or "Hertz" which is the unit of frequency in the electromagnetic field) and the rate at which the Earth's tides are measured. It is also the

## Caveats

- The follow-up is keyword-matched, not hand-written for all 337 rows; a few questions land on the wrong noun.
- The name extractor keeps explicit `my name is` and `known as the` claims, but rejects phrases such as `a combination of the two`. Unnamed rows correctly get the neutral `name?`.
- A voice marker is a lexical flag (`right?`, `nostrum`, `sea monster`...), not a human judgment. Use the examples, not the count alone.
- Layers present: {0: 2, 1: 6, 2: 3, 3: 4, 4: 5, 5: 8, 6: 9, 7: 7, 8: 7, 9: 5, 10: 4, 11: 10, 12: 6, 13: 9, 14: 10, 15: 22, 16: 8, 17: 17, 18: 7, 19: 8, 20: 9, 21: 20, 22: 12, 23: 19, 24: 20, 25: 30, 26: 28, 27: 9, 28: 11, 29: 14, 30: 3, 31: 2, 32: 2, 33: 1}
