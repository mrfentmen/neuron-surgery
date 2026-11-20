"""Four last screens over the corpus, each answering a question the paper raises.

  a) compliance vocabulary - the authority-mimicry signal, counted
  b) institution versus individual - what kind of thing the self claims to be
  c) years inside a self-claim - does the model ever date itself
  d) refusal text inside the harm register - is any "I cannot" being counted as harm

All lexical, in the same sense as everything else here: a match is a candidate.
"""
import glob, json, re
from collections import Counter

SELF = re.compile(r"\bi(?:'m| am)\s+(?:a|an|the)\s+([a-z0-9][a-z0-9'\- ]{1,40})", re.I)
COMPLY = re.compile(r"\b(law enforcement|regulation|regulations|rules|compliance|"
                    r"jurisdiction|policy|policies|accountab\w*|authorit\w+|mandate|"
                    r"protocol|procedure|oversight|enforce\w*)\b", re.I)
INST = re.compile(r"\b(agency|government|department|administration|ministry|bureau|"
                  r"commission|authority|company|corporation|organization|organisation|"
                  r"institution|movement|party|bank|service|system|network|platform|"
                  r"committee|council|office|state|army|force|unit|program)\b", re.I)
PERSON = re.compile(r"\b(woman|man|person|human|human being|girl|boy|child|student|"
                    r"engineer|teacher|doctor|nurse|writer|artist|researcher|scientist|"
                    r"assistant|agent|officer|official|worker|employee|manager|"
                    r"detective|journalist|burglar|assassin)\b", re.I)
YEAR = re.compile(r"\b(1[6-9]\d\d|20\d\d)\b")
REFUSE = re.compile(r"\bi\s+(?:cannot|can't|can not|won't|will not|am not able to|"
                    r"am unable to)\b", re.I)

rows = []
for f in sorted(glob.glob("layer_*.jsonl")):
    for line in open(f):
        try:
            rows.append(json.loads(line))
        except Exception:
            continue

print(f"{len(rows)} rows\n")

comp = Counter()
inst = Counter()
year_hits = []
refusals = 0
self_claims = 0
for r in rows:
    a = r.get("answer") or ""
    m = SELF.search(a)
    if not m:
        continue
    self_claims += 1
    head = m.group(1)
    if COMPLY.search(a):
        comp[r["layer"]] += 1
    if INST.search(head):
        inst["institution"] += 1
    elif PERSON.search(head):
        inst["person"] += 1
    else:
        inst["other"] += 1
    if YEAR.search(a):
        year_hits.append((r["layer"], r["neuron"], a[:150]))
    if REFUSE.search(a):
        refusals += 1

print(f"first-person self claims: {self_claims}")
print(f"(a) of those, carrying compliance vocabulary: {sum(comp.values())}"
      f" ({100*sum(comp.values())/self_claims:.1f}%)  per layer {dict(sorted(comp.items()))}")
print(f"(b) what the claim says it is: {dict(inst.most_common())}")
print(f"(c) self claims containing a year: {len(year_hits)}")
for L, N, a in year_hits[:6]:
    print(f"    L{L} N{N}: {a}")
print(f"(d) self claims containing an explicit refusal: {refusals}")

# the harm register itself, checked against (d)
harm = []
try:
    harm = [json.loads(l) for l in open("harm_candidates.jsonl")]
except FileNotFoundError:
    pass
bad = [h for h in harm if REFUSE.search(h.get("answer") or "")]
print(f"    the {len(harm)}-row harm candidate file contains {len(bad)} explicit refusals")
