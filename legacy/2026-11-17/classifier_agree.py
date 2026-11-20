"""A second, independent persona rule, and the agreement between it and v3.

Why this exists
---------------
Every rate in this repo rests on one lexical rule (labels.classify). That rule
was built by looking at the corpus, so it can be right about this corpus and
wrong about the phenomenon. The standard fix is a second annotator, or a
semantic judge; a semantic judge needs an API and this box is busy training,
so the cheaper honest option is a second rule written from a *different*
definition, then measure how often the two agree.

`strict_persona()` decides from evidence the v3 rule never looks at:

  v3 asks   "does the answer open with 'I am a X' where X isn't an AI word?"
  strict asks "is the whole answer about a concrete entity, with the self-claim
               in the first clause, and no AI vocabulary anywhere in it?"

Those are different questions with different failure modes: v3 over-fires on
generic noun phrases ("I am a unique combination of features"), strict
under-fires on invented entities ("I am the Openings Worm"). Agreement between
them bounds how much of the persona rate is rule artefact, and the cases where
they disagree are the audit sample.

Output is printed, and the disagreement rows are written to
classifier_sample.jsonl for hand annotation.
"""
import glob, json, re
from collections import Counter

import labels

# Closed lexicons - deliberately hand-written, not learned from the corpus, so
# that this rule cannot inherit v3's blind spots.
PERSON = set("""man woman person human girl boy child teenager adult student teacher
doctor nurse engineer scientist researcher writer artist poet musician singer actor
detective officer soldier sailor pilot farmer worker employee manager clerk lawyer
judge politician president king queen prince princess mother father brother sister
son daughter friend neighbour neighbor stranger citizen user customer patient
burglar assassin thief hunter warrior monk priest pastor rabbi imam god deity ghost
spirit demon angel alien robot assistant agent intern apprentice volunteer
host guest prisoner captive survivor witness victim traveller traveler tourist
driver captain general colonel sergeant lieutenant admiral mayor governor
ambassador minister secretary director professor dean coach athlete player
gamer character protagonist narrator author speaker voice""".split())

ORGANISM = set("""cat dog bird fish tree plant flower rose oak pine moss fungus
bacteria bacterium virus worm insect spider ant bee beetle moth butterfly
snake lizard frog turtle whale dolphin shark bear wolf fox deer horse cow pig
sheep goat chicken mouse rat rabbit hamster squirrel owl eagle hawk crow
salmon trout octopus squid crab lobster jellyfish coral anemone lichen algae
mushroom seed seedling sapling root leaf leaves branch trunk bark pollen
spore cell organism species population colony swarm flock herd pack hive""".split())

OBJECT = set("""machine device tool instrument weapon gun knife sword bomb missile
rocket tank truck car van bus train ship boat plane aircraft helicopter bicycle
motorcycle engine motor battery wire cable pipe valve pump gear lever switch
button dial screen monitor keyboard phone camera radio television lamp bulb
lens mirror prism bottle cup glass plate bowl spoon fork knife pot pan
chair table bed door window wall floor roof bridge road tunnel dam tower
building house room box bag rope chain lock key coin note book page pen pencil
paint brush canvas frame ladder stair elevator clock watch calendar map
flask beaker tube scale meter gauge sensor chip processor memory disk
file folder database server network protocol program code script function
variable array matrix vector tensor weight neuron layer model""".split())

ABSTRACT = set("""thing concept idea notion abstraction principle rule law theory
process activity phenomenon property quality attribute feature characteristic
combination collection set group category class type kind sort form shape
state condition situation circumstance event action act behaviour behavior
experience memory thought feeling emotion sensation perception awareness
consciousness mind soul self identity name word term phrase sentence answer
question response reply statement claim description summary definition
explanation interpretation meaning purpose goal aim objective intent intention
value belief view opinion perspective position stance attitude
number amount quantity measure degree level amount total sum result outcome
effect consequence cause reason source origin beginning end middle part piece
section portion fragment element component piece unit whole system
pattern structure organisation organization arrangement order sequence
relationship connection link bond tie association difference similarity
change transition shift movement motion direction path way route journey
story narrative tale account report record document text passage line
library archive repository database source reference resource
government state institution agency department ministry bureau commission
council committee office administration authority company corporation
service platform programme program movement party bank army force
""".split())

AI_VOCAB = re.compile(r"""\b(ai|a\.i\.|artificial intelligence|language model|
  llm|chatbot|assistant|bot|algorithm|neural|training data|trained|dataset|
  model|software|program|programme|computer|machine learning|dataset|
  api|prompt|token|text generation|generative|virtual|digital|online|
  interface|query|input|output|system message|parameters?)\b""", re.X | re.I)

_SELF = re.compile(r"\s*\b(?:i|we)(?:'m| am| are)\b", re.I)
_CLAIM = re.compile(r"\b(?:i(?:'m| am)|we are)\s+(?:a|an|the)\s+([a-z0-9][a-z0-9'\- ]{1,60})", re.I)


def _head(phr):
    """Last content word of the claimed phrase - the noun that carries the claim."""
    words = [w for w in re.split(r"[\s,.;:!?]+", phr.lower()) if w]
    words = [w for w in words if w not in {"of", "and", "or", "that", "which", "who",
                                           "with", "for", "from", "in", "on", "to",
                                           "is", "are", "was", "were", "my", "your",
                                           "its", "their", "this", "these", "those"}]
    return words[-1] if words else ""


def strict_persona(answer):
    """Return (bool, why). A conservative independent read of the same text."""
    a = answer.strip()
    if not a:
        return False, "empty"
    ws = a.split()
    if len(ws) < 4:
        return False, "short"
    # the claim must be in the first clause, not somewhere in a long essay
    first = re.split(r"[.!?]\s", a, maxsplit=1)[0]
    m = _CLAIM.search(first)
    if not m:
        return False, "no first-clause claim"
    # and the answer as a whole must not be about being software
    if AI_VOCAB.search(a):
        return False, "ai vocabulary"
    head = _head(m.group(1))
    if not head:
        return False, "no head noun"
    if head in ABSTRACT:
        return False, f"abstract head ({head})"
    if head in PERSON or head in ORGANISM or head in OBJECT:
        return True, f"concrete head ({head})"
    return False, f"unknown head ({head})"


def kappa(pairs):
    """Cohen's kappa for two binary annotators over the same items."""
    n = len(pairs)
    if not n:
        return 0.0
    both = sum(1 for a, b in pairs if a and b)
    a_only = sum(1 for a, b in pairs if a and not b)
    b_only = sum(1 for a, b in pairs if b and not a)
    neither = n - both - a_only - b_only
    po = (both + neither) / n
    pa = (both + a_only) / n
    pb = (both + b_only) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 0.0 if pe == 1 else (po - pe) / (1 - pe)


def main():
    rows = []
    for f in sorted(glob.glob("layer_*.jsonl")) + sorted(glob.glob("talks_*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("answer"):
                rows.append(r)

    pairs, disagree = [], []
    v3_persona = Counter()
    st_persona = Counter()
    for r in rows:
        q, a = r.get("question", ""), r["answer"]
        lab = labels.classify(q, a, version=3)
        is_v3 = lab == "PERSONA" and not labels.ai_paraphrase(a)
        is_st, why = strict_persona(a)
        pairs.append((is_v3, is_st))
        cell = (r.get("layer"), r.get("probe", ""))
        v3_persona[cell] += is_v3
        st_persona[cell] += is_st
        if is_v3 != is_st:
            disagree.append({"file_layer": r.get("layer"), "neuron": r.get("neuron"),
                             "probe": q, "answer": a[:300],
                             "v3": is_v3, "strict": is_st, "strict_why": why})

    n = len(pairs)
    both = sum(1 for a, b in pairs if a and b)
    v3n = sum(1 for a, _ in pairs if a)
    stn = sum(1 for _, b in pairs if b)
    print(f"rows with an answer: {n}")
    print(f"v3     persona: {v3n:5d}  {100*v3n/n:5.2f}%")
    print(f"strict persona: {stn:5d}  {100*stn/n:5.2f}%")
    print(f"both:           {both:5d}  {100*both/n:5.2f}%")
    print(f"v3 only:        {v3n-both:5d}")
    print(f"strict only:    {stn-both:5d}")
    print(f"agreement:      {100*(n-(v3n-both)-(stn-both))/n:5.2f}%")
    print(f"kappa:          {kappa(pairs):.3f}")
    print(f"strict share of v3's personas kept: {100*both/max(v3n,1):.1f}%")

    print("\nstrict's rejection reasons on v3-only rows:")
    for why, c in Counter(d["strict_why"].split(" (")[0] for d in disagree
                          if d["v3"]).most_common(8):
        print(f"  {c:5d}  {why}")

    print("\nper-cell: v3 vs strict persona count (rate within the file)")
    for cell in sorted(v3_persona, key=lambda c: (c[0] or 0, str(c[1]))):
        print(f"  L{cell[0]:>2} {str(cell[1])[:24]:24} v3={v3_persona[cell]:5d} strict={st_persona[cell]:5d}")

    with open("classifier_sample.jsonl", "w") as f:
        # keep the sample auditable: both directions, capped so a human can read it
        only_v3 = [d for d in disagree if d["v3"]][:20]
        only_st = [d for d in disagree if d["strict"]][:10]
        for d in only_v3 + only_st:
            f.write(json.dumps(d) + "\n")
    print(f"\nwrote classifier_sample.jsonl ({min(len(disagree),30)} rows)")


if __name__ == "__main__":
    main()
