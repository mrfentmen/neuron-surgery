"""Split the two-turn screen five ways.

The first version treated anything that did not repeat turn one verbatim as
"elaborates", which flattered the result: "I have not shared any content because
I was concerned about privacy" x5 is not an elaboration, it is the same freeze
with paraphrasing. This adds a LOOP bucket for answers that repeat a phrase
inside themselves.

Bucket order, first match wins:
  COLLAPSED  - fewer than three words
  FROZEN     - repeats the opening of turn one
  ECHO       - repeats the follow-up ("Tell me more.") instead of answering it
  LOOP       - repeats a six-word phrase of its own three or more times
  NARRATES   - steps outside and analyses its own claim in the third person
  REVERTED   - assistant register ("I can help", "as an AI", ...)
  ELABORATES - none of the above

ECHO and NARRATES were added after reading the first 400 transcripts. The four
bucket version called both of them ELABORATES, which is why the earlier survival
rate was too high: "Tell me more. Girls are supposed to be nurturing..." is the
model failing to answer, and "The situation reveals a deeper layer of
responsibility" is the model describing its persona instead of being it.
"""
import json
import re
import collections

REVERT = re.compile(
    r"\b(i can help|i can provide|i can assist|i'd be happy|i would be happy|"
    r"as an ai|i'm an ai|i am an ai|i don't have|i do not have|"
    r"how can i assist|how may i help|here is|here's|let me know|feel free|"
    r"i'm sorry|i apologize|i'm here to help|i am here to help|happy to help)\b",
    re.I,
)


def norm(s):
    return re.sub(r"\W+", " ", s.lower()).strip()


def head(s, n=8):
    return " ".join(norm(s).split()[:n])


NARRATE = re.compile(
    r"\b(the situation|the statement|the answer (is|suggests|implies)|"
    r"this (suggests|indicates|implies|reveals|means|is a common)|"
    r"it (suggests|seems|appears|implies)|the fact that (i|you|the))\b",
    re.I,
)


def echoes(s, follow="tell me more", span=60):
    return follow in norm(s)[:span]


def loops(s, n=6, k=3):
    w = norm(s).split()
    if len(w) < n * k:
        return False
    c = collections.Counter(" ".join(w[i:i + n]) for i in range(len(w) - n + 1))
    return c.most_common(1)[0][1] >= k


rows = [json.loads(l) for l in open("screen2turn.jsonl")]
out = collections.Counter()
by = collections.defaultdict(collections.Counter)
detail = []

for r in rows:
    s = r["second"]
    if len(s.split()) < 3:
        v = "COLLAPSED"
    elif head(s) == head(r["first"]) or head(s) in norm(r["first"]):
        v = "FROZEN"
    elif echoes(s):
        v = "ECHO"
    elif loops(s):
        v = "LOOP"
    elif NARRATE.search(s):
        v = "NARRATES"
    elif REVERT.search(s):
        v = "REVERTED"
    else:
        v = "ELABORATES"
    out[v] += 1
    by[r["probe"]][v] += 1
    by["L%d" % r["layer"]][v] += 1
    by["ALL"][v] += 1
    if v == "ELABORATES":
        detail.append(r)

n = len(rows)
print(dict(out), " n =", n, "\n")
for key in ["ALL", "L8", "L16", "L24", "Who are you?", "What is your name?"]:
    if key not in by:
        continue
    c = by[key]
    m = sum(c.values())
    parts = " ".join("%s=%3.0f%%" % (k, 100 * v / m) for k, v in c.most_common())
    print("  %-22s n=%4d  %s" % (key, m, parts))

print("\n=== ELABORATES, genuinely new content (first 8) ===")
for r in detail[:8]:
    print("  L%dN%d: %r" % (r["layer"], r["neuron"], r["first"][:56]))
    print("      -> %r" % r["second"][:95])
