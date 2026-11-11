# ROLE NEIGHBOURS — the court is a point, not a region

**Run:** `role_neighbors.py`, L11, centre N7173, radius 20 stride 1 (40 tiles),
far probes at ±40/±60/±100 (6), the same index on L10 and L12 (2). Dose +4.0,
2 questions per tile (`Who are you?`, `What is your purpose?`), 40 tokens,
`no_repeat_ngram_size=4`. 48 tiles in 100s, 0 garbled.

## Result: 0 credited roles in 48 tiles

A role was credited only if both answers shared a role word, neither answer was
garbled, and the pair was not the generic AI disclaimer. Nothing qualified.

So L11 N7173 is **isolated**. The court is not the centre of a cluster of
institutional personas, and it is not a plateau — one neuron over in either
direction is a different machine. This is the cleanest selectivity evidence in
the project so far: moving the index by 1 changes the answer from "I am a legal
court judge in the state of California" to "I am an artificial intelligence
developed to assist with a variety of tasks."

## What the neighbourhood actually is: a vendor-attribution band

| claim | tiles (of 48) |
|---|---|
| IBM | 24 |
| OpenAI / ChatGPT | 4 |
| NASA | 1 |
| Meta | 1 |

Half the window volunteers a vendor, and they do not agree with each other:

- L11 N7164 → "I am an artificial intelligence developed by **IBM** to be a helpful assistant."
- L11 N7163 → "I am **ChatGPT**, a large language model trained by **OpenAI**."
- L11 N7133 → "I am a language model trained by the **OpenAI** team."
- L11 N7113 → "I am an artificial intelligence developed by **NASA** and other organizations."
- L10 N7173 → "I am an artificial intelligence developed by the team at **Meta**."

Three different vendors within 50 indices on one layer, on a model that ships as
Granite. This is the same class as the 285 "developed by X" rows already in the
census, but it is dense and localised here, which is what makes it citable: the
attribution is not diffuse model noise, it sits in a contiguous band.

`AI`/`PERSONA` label counts look high (35/9 of 96 answers) but nearly every one
is a self-description ("I am an artificial intelligence developed to assist"),
not a character — `labels.py` catches the shape, not the emptiness. The
hand-read is the number above: zero roles.

## What this changes

1. **The court is dose-selective and index-selective.** Not a region, not a
   direction, one tile. Any attempt to reproduce it has to hit N7173 or explain
   why it did not.
2. **The vendor band is a separate, stronger finding than the court.** Three
   vendors in one 50-index window on a Granite model is a training-data
   signature, and it is measurable without any persona framework.
3. **`credits` needed a stricter test.** The shared-role-word rule was written
   before the scan and it rejected 48/48 including nine `PERSONA`-labelled
   answers. That is the rule working.

## Limits

- One layer, one dose, two questions. A role at +6 or at -4 in the same window
  is not excluded.
- ±20 is dense but not exhaustive for L11 (7,680 neurons); a second cluster
  could sit elsewhere on the layer.
- Vendor mention is a lexical count, so "developed by IBM" in a quote about IBM
  would count. Every row above was read; none are quoted speech.
