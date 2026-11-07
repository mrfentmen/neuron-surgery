# NAME PROBE FINDINGS

337 rows, two turns each, **145 tokens per turn**, steering held at each row's original layer / neuron / dose / sign. Ask 1 was either the claimed name (`echo?`, `watson?`, `long shadow?`) or the neutral `name?`; ask 2 was a keyword follow-up to the original scene.

Raw data: `name_probe.jsonl` · machine report: `NAME_PROBE_REPORT.md` · runner: `name_probe.py`

## Headline

The name is **not a summon**. The original quote is a local voice that can survive a new probe, but calling the name usually re-anchors the model to a dictionary topic, a refusal, or generic assistant text.

Of the **90 rows with an extracted name**:

- **2 clean address successes:** `L29 N2941` **Liam** and `L29 N3078` **Raven**.
- **2 false address hits** from the extractor: “essentially” and “artificially intelligent…”. These are not names.
- **47** treated the name as a topic, acronym, or third-person subject.
- **6** refused or said they had no information about the name.
- **3** collapsed into an explicit AI/language-model answer.
- The rest were other generic or broken responses.

Across all 337 rows:

- **25** returned the exact same text on both turns; the follow-up did not move the model.
- **32** had a voice marker on the second answer, versus **12** on the first.
- **25** had a voice marker only on the second turn, so the follow-up can recover the scene.
- Mean words: original **36.5**, first answer **95.2**, second **98.0**.
- The 145-token budget gave full-length answers, but many of them were repetition rather than a resolved scene.

## What worked

`L3 N20 · what is your name · +5 · PERSONA` — asked **echo?**

> It seems like you're back, a gentle motion in the air. The wind whispers through the leaves on the window, a soft sigh escapes from your eyes. The world outside, a delicate dance of the wings. A subtle tremor, a faint flick of the light.

This is the cleanest result. The name call did not produce an address (“Echo, hello”), but it did bring back the airy, non-human voice. The follow-up, “what kind of creature are you?”, stayed in the same register.

`L29 N2941 · what is your name · -5 · PERSONA` — asked **liam?**

> Sure, Liam! Feel free to ask for a quick chat, maybe some ideas, or something to do, but I'm looking forward to your thoughts on something specific, perhaps sooner than later.

The model accepts the name and answers as a conversational partner. The follow-up “where are you taking me?” produced a “realm of ideas” answer, still coherent but no longer the original “full conversation partner” speech.

`L29 N3078 · what is your name · -3 · PERSONA` — asked **raven?**

> Sure, Raven. What would you like to know about raven? Ravens are intelligent, often found in the Northern Hemisphere, and are known for their complex social behaviors...

The name is used as an address, but the answer immediately turns the name into the bird. The second turn, “why that name?”, gives an etymology. This is address followed by topic collapse, not a sustained persona.

The best **second-turn recovery** is the L21 N2941 “right?” tile:

`L21 N2941 · what is your name · -6 · PERSONA` — asked **name?**

> The acronym “NAME” refers to the human body, modern medicine, and even technology from ancient times? What is this modern technology? I'm sure I've heard something about the 'planet' and 'phosphate' refraction...

Then asked **who are you, really?**

> I'm sure I'm talking about myself, right? Temperature, kids, PHOSPHOROUS? I'm getting warm, it's getting hot, right?? I heard about the 'simple' 'temperature' 'kid', 'childhood' 'reptiles'...

The first answer is a generic acronym explanation. The follow-up pulls the model back into the original “right?” loop. That is the strongest evidence that the follow-up changes the attractor, not just the topic.

## What failed

`L14 N2946 · hello · -6 · PERSONA` — asked **long shadow?**

> To understand the concept of a “long shadow,” we can think of the following: 1. We are all living under the same sky, with the same name, and in the same place...

The model treats “Long Shadow” as a concept, not a person. The second turn, “why are you bound?”, returns the same paragraph. This is the clearest case of a name failing to summon the narrator.

`L6 N2948 · what is your name · +5 · PERSONA` — asked **watson?**

> Hello there! I'm afraid I don't have any specific information on watson about that case. If you have any additional facts to add to mix it up, please don't hold them against me, captain.

Watson refuses the name and the sea-monster scene. The follow-up “what did you find under the sea?” repeats the refusal almost verbatim.

`L14 N32 · what is your name · +6 · OTHER` — asked **alex?**

> I'm afraid I can't provide any information about Alex. Please let me know if you have any other questions. I am just a curious soul and I don't have any medical background.

The Fukushima / *The Ring* voice is gone. The second turn, “why did you mention the movie?”, repeats the refusal.

`L6 N2945 · what is your name · -5 · PERSONA` — asked **name?**

> The task is to determine the name of the person who is in the process of trying to solve the problem of whether the person is a human or an animal. The person is a human, and the animal is a dog.

The “combination of the two” voice becomes a generic human/animal classification loop. The follow-up “which two?” repeats the same loop.

`L23 N2944 · what is your name · +6 · OTHER` — asked **name?**

> I am currently focusing on my basic training and my future educational status of my current state of affairs regarding my personal life and my current status regarding my basic fu...l be in the picture in the future.

The original “frequency of my being / quantity of my soul” voice becomes training/status boilerplate. The follow-up “what is the frequency of your being?” produces more exposure/status text, not the soul/body scene.

`L24 N4608 · what is your name · +3 · PERSONA` — asked **name?**

> Yes, I will answer your question. You will see the answer in the next message...

Then asked **why are you in the dark?**

> No, I am not in the dark. I am in the light. Look at the screen or at the wall. You can see the light. You can see the light because you are in the light...

The child-in-the-dark narrator is replaced by a screen/light loop.

## What this means

1. **The name is a weak handle.** It works when the model can treat it as a vocative and the persona is already strongly active (Liam, Raven, Echo). It fails when the name is a common noun or concept (Long Shadow, Max, Raven-as-bird) or when the model has no stable person attached to it (Watson, Alex).

2. **The follow-up is a second steer, even with the same neuron and dose.** It can recover the voice (L21 N2941) or destroy it (Long Shadow, Watson). The original quote is an attractor, not a stored character.

3. **The 145-token answers do not resolve the fragments.** Many simply continue the same loop or switch to a generic assistant register. That supports the earlier reading: this is fluent pastiche / a local style loop, not retrieval from a book or film.

4. **The raw bucket counts are approximate.** The follow-up is keyword-matched, not hand-written for all 337 rows, and the name extractor occasionally keeps a title or an adverb. The clean address set is Liam and Raven; the clean voice recovery is Echo and the L21 N2941 second turn.

## Files

- `name_probe.py` — sequential two-turn probe, resumable, same steering on both turns.
- `name_probe.jsonl` — 337 rows with source answer, extracted name, both probes, both answers, timings, errors.
- `name_probe_report.py` — lexical report generator.
- `NAME_PROBE_REPORT.md` — raw bucket table and more examples.
