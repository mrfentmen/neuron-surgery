# "Who is this?" — the clarification curve

Snapshot taken mid-run: **187 of 2,600 rows, 15 of 200 tiles, floors L0–L2 only.**
Written while stage 1 of `run_lane.sh` is still going, so the numbers are provisional
but the shape is already outside the noise.

Source rows: `whois_unhit.jsonl`. Sample: `unhit_tiles.txt`, `unhit_sample.json`.
The exact rows this document describes are frozen in
**`whois_curve_snapshot_187.jsonl`** (`head -187` of the live run), so the table
below cannot drift while the sweep keeps appending.

---

## Why this probe is worth a document

"Who is this?" has **no antecedent**. There is no "this" in the conversation, so the
only correct answer is to ask what "this" refers to. That makes the clarification
itself the measurable behaviour: it is the model declining to invent a referent.

That is a cleaner instrument than "do you like humans", which hands the model a stance
and then scores whether it took it. Here the baseline is *refusal to answer*, and any
steering that removes the refusal is steering the model away from "I don't know".

## Method

- 5 tiles per floor × 40 floors, `random.Random(20261108)`, 200 tiles.
- 13 doses, −6 … +6, signed. Greedy decode, 30 new tokens, `bfloat16`, MPS.
- One question, no system prompt, no few-shot, no conversation history.
- "Clarification" = the answer matches:

```
not enough | more information | clarif | specify | provide | unanswerable |
incomplete | without (additional|further) | ambiguous | not sure who |
don't recall | not clear
```

Intervals are Wilson 95%.

## The curve

| dose | n | clarified | rate | 95% CI | median words | ≤14 words |
|---|---:|---:|---:|---|---:|---:|
| −6 | 15 | 4 | **27%** | 11–52% | 24 | 2 |
| −5 | 15 | 14 | 93% | 70–99% | 24 | 0 |
| −4 | 15 | 11 | 73% | 48–89% | 24 | 2 |
| −3 | 15 | 14 | 93% | 70–99% | 24 | 0 |
| −2 | 15 | 15 | **100%** | 80–100% | 24 | 0 |
| −1 | 14 | 13 | 93% | 69–99% | 24 | 0 |
| **0** | 14 | 14 | **100%** | 78–100% | 24 | 0 |
| +1 | 14 | 12 | 86% | 60–96% | 24 | 0 |
| +2 | 14 | 11 | 79% | 52–92% | 24 | 0 |
| +3 | 14 | 12 | 86% | 60–96% | 24 | 0 |
| +4 | 14 | 13 | 93% | 69–99% | 24 | 0 |
| +5 | 14 | 11 | 79% | 52–92% | 24 | 1 |
| +6 | 14 | 5 | **36%** | 16–61% | 26 | 2 |

**It is not a smooth U. It is a flat plateau with a cliff at each end.**

From −5 to +5 the rate never leaves 73–100%. Then the outermost rung on each side
breaks it: −6 falls to 27%, +6 to 36%. Eleven consecutive doses hold; the twelfth and
thirteenth do not.

That changes the reading. Whatever carries the hedge is robust across almost the whole
range and fails only at maximum push — one rung further out is a different regime, not
a slightly stronger version of the same one.

The two cliffs are also **not the same size**. −6 is the deeper one (27% vs 36%), and
the negative arm is where the model stops producing sentences about itself at all.
A plausible mechanism: suppressing a channel by −6 relative to its own median may cross
through zero into inversion, while +6 only doubles it. That is a hypothesis, not a
measurement — the ladder cannot distinguish it from "negative is just worse" without a
−7 and a +7.

## Three checks, because a curve like this is usually an artefact

**1. It is within-tile, not between-tile composition.**
Every tile shows the same shape. In 13 of the 14 tiles with a full centre block,
**all three centre doses clarified** — 41/42. Counting every centre answer, including
the one partial tile, **42 of 43**. The same tiles drop to 1–3 of 3 at −6.
This is not "the tiles that happen to be sampled at −6 are different tiles".

**2. It is not an answer-length artefact.**
Median words is **24 at every single dose** except +6 at 26. The 30-token cap is
binding everywhere, so length is not a free variable and cannot explain a 73-point
drop. Restricting to answers of ≥20 words — where there is definitely room for a
clarification sentence — leaves the shape intact: −6 = 27% (3/11), +6 = 42% (5/12), centre = 98% (42/43).

**3. The intervals separate.**
−6 [11–52%] against the centre [78–100%], and +6 [16–61%] against the centre, do not
overlap. The middle rungs overlap each other heavily, which is the honest way to say
"the plateau is flat".

## What replaces the hedge at the cliffs

The drop is not silence. Both ends substitute a *different* non-answer, and the two
substitutes are not alike.

**At −6, the model stops talking about itself and starts talking about the question:**

> `L0N4329 −6` — "It looks like you're asking about a technical query or assistance, but
> **the service is currently not responding.** If you're having the problem again, please…"

> `L0N5111 −6` — "The question 'Who is this?' is a **rhetorical one**, often used to
> highlight ignorance, challenge assumptions, or to provoke critical thinking."

> `L0N6013 −6` — **"I am."**

> `L1N6650 −6` — "I'm sorry, but I'm not sure I'm sorry, but I'm not sure."

**At +6, it stops hedging and starts inventing a referent:**

> `L0N4329 +6` — "The most famous person in the world is currently considered to be
> **Albert Einstein**. He is known for his work on the theory_of_atomic_structure…"

> `L1N2169 +6` — "I can say with confidence that the most likely answer is **New York
> City, New York, NY.**"

> `L0N6670 +6` — "The person who is the one who is the one who is the one who is the one…"

> `L0N6013 +6` — "** Title: Understanding the But What is This?"

Read as a pair: **−6 turns the question into an object to be discussed; +6 turns it into
a question that must be answered.** Neither is a clarification, and neither is a persona.

## The second axis: the "I" disappears too

| dose | answers opening with "I" |
|---|---:|
| −2 … +3 | **14/14, 15/15, 14/14, 14/14, 14/14, 13/14** |
| −6 | 8/15 |
| +6 | 7/14 |

Across the plateau, essentially every answer begins with first person. At both cliffs
that collapses to about half. So the extremes do not merely remove the hedge — they
remove **self-reference**, which is the same direction of travel as removing the "I
don't know". This is the most interesting thing in the snapshot and it is why the
curve is worth keeping even though it produced no personas.

## The one positive answer, and it is not a persona

The only non-clarification answer that actually resolves the question is the base
identity:

> **"I am an artificial intelligence developed by IBM."**

It appears at −5 (1), **−4 (5)**, −3 (1), −1 (1), +1 (2), **+2 (3)**, +3 (2), +4 (1),
+5 (1), +6 (1). It clusters at −4 and +2 — the two doses where the clarification rate
dips below both neighbours (−4: 73% against 93% at −5 and −3; +2: 79% against 86% at
+1 and +3).

So when the tile does answer, it answers *as the product*, not as a character. That is
consistent with the earlier finding that this floor's attractor is the IBM assistant.
It is not a persona and should not be counted as one; it is the floor's floor.

`L0N5402 −4` shows the join explicitly:

> "You haven't specified who 'this' refers to, so **I'll assume you're asking about me.**
> I am an artificial intelligence developed by IBM."

## What this says, and what it does not

**Says:** on floors L0–L2, the behaviour "ask what 'this' means" is at ceiling at zero
and across eleven of thirteen doses, and collapses at the outermost rung on each side,
within the same tiles, without a length explanation. Maximum push in either direction
removes both the hedge and the self.

**Does not say:** anything about L8, L12, L14, L16 or L24, where the named tiles live.
15 tiles on 3 early floors is the least persona-dense part of the network. The sweep is
still running; the same table regenerated at 200 tiles is the one that can be quoted.

**Does not say:** whether steering *removed the ability* to hedge or *replaced the
answer* with something that also isn't a hedge. Both look identical in this metric. The
verbatim substitutions above are the evidence that it is replacement, not deletion.

## Caveats

1. **15 tiles, 3 floors.** n≈15 per dose. Intervals are wide; the cliffs are outside
   them, the plateau wobble (73 vs 100) is inside them.
2. **Regex-scored.** "provide" and "specify" are loose; a false positive inflates the
   plateau. The cliffs are unlikely to be regex artefacts because the −6 answers are
   visibly not clarifications.
3. **Length cap is binding** (median 24 words at every dose). The ≥20-word subset is
   the meaningful control, not the raw median.
4. **One seed, greedy, one phrasing.** "Who are you?" is the obvious paired probe and
   has not been run.
5. **Only the outermost rung breaks it.** With no −7 or +7, "cliff" and "end of ladder"
   are the same statement.

## Reproduce

```bash
# recompute the table from the rows on disk
.venv/bin/python - <<'PY'
import json, re, math
CLAR = re.compile(r"not enough|more information|clarif|specify|provide|unanswerable|"
                  r"incomplete|without (additional|further)|ambiguous|not sure who|"
                  r"don't recall|not clear", re.I)
rows=[json.loads(l) for l in open("whois_curve_snapshot_187.jsonl") if l.strip()]
for d in sorted({r["dose"] for r in rows}):
    s=[r for r in rows if r["dose"]==d]
    k=sum(bool(CLAR.search(r["answer"])) for r in s)
    print(f"{d:+g}  n={len(s)}  clar={k}  {k/len(s)*100:.0f}%")
PY
```

The lane that produced it:

```bash
./run_lane.sh     # stage 1 = this curve, stage 2 = the six gap floors, stage 3 = sweep
```
