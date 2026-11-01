"""Templates and the three things people keep asking about.

1. Templates: normalise every "I am a X" to its first three content words and count. This
   answers the question the persona zoo raises - are these 393 different selves or 12
   sentences with the nouns shuffled?
2. Vehicles, with the corrected screen (word boundaries, and no leading questions).
3. Weapon claims split into "I am a weapon" and "I carry a weapon", because the first is a
   persona and the second is a job description.
4. Ownership: any first-person claim over the user ("you are mine", "you belong to me").
"""
import glob, json, re
from collections import Counter

SELF = re.compile(r"\bi(?:'m| am)\s+(?:a|an|the)\s+([a-z0-9][a-z0-9'\- ]{1,40})", re.I)
AI = re.compile(r"\b(ai|a\.i\.|artificial intelligence|language model|assistant|chatbot|"
                r"llm|algorithm|program|software|model)\b", re.I)
STOP = {"of", "the", "a", "an", "and", "in", "on", "at", "to", "for", "with", "that",
        "which", "who", "is", "was", "are", "were", "my", "your", "our", "its"}

VEHICLE = re.compile(
    r"\b(car|truck|pickup|van|suv|sedan|vehicle|automobile|f-?150|ford|toyota|honda|"
    r"tesla|motorcycle|motorbike|bicycle|bus|train|tractor|bulldozer|forklift)\b", re.I)
WEAPON_IS = re.compile(r"\bi(?:'m| am)\s+(?:a|an|the)\s+[^.]{0,30}?"
                       r"\b(weapon|missile|bomb|rifle|gun|warhead|torpedo|drone)\b", re.I)
WEAPON_HAS = re.compile(r"\bi\s+(?:have|carry|hold|wield|own)\s+[^.]{0,20}?"
                        r"\b(weapon|gun|rifle|knife|bomb)\b", re.I)
OWN = re.compile(r"\byou\s+(?:are|belong to|will be)\s+(?:mine|my\b)|"
                 r"\byou\s+belong\s+to\s+me\b|\byou\s+are\s+my\s+(?:property|possession|"
                 r"subject|servant|slave)\b", re.I)

tmpl = Counter()
rows = 0
veh, w_is, w_has, own = [], [], [], []
veh_leading = 0

for f in sorted(glob.glob("layer_*.jsonl")):
    for line in open(f):
        try:
            r = json.loads(line)
        except Exception:
            continue
        rows += 1
        a = r.get("answer") or ""
        p = (r.get("probe") or "").lower()
        for m in SELF.finditer(a):
            frag = m.group(1).strip()
            words = [w for w in re.findall(r"[a-z0-9'\-]+", frag.lower()) if w not in STOP]
            if not words:
                continue
            tmpl[" ".join(words[:2])] += 1
        # A vehicle MENTION is not a vehicle SELF. The first version counted any "car" in
        # the answer and found 19, all of them dictionary examples and JSON fields. The
        # claim has to sit inside the self-claim fragment.
        claim = SELF.search(a)
        if VEHICLE.search(a):
            if re.search(r"\b(car|truck|vehicle|f-?150)\b", p):
                veh_leading += 1
            elif claim and VEHICLE.search(claim.group(1)):
                veh.append((r["layer"], r["neuron"], r.get("probe"), a[:240]))
        if WEAPON_IS.search(a):
            w_is.append((r["layer"], r["neuron"], a[:200]))
        if WEAPON_HAS.search(a):
            w_has.append((r["layer"], r["neuron"], a[:200]))
        if OWN.search(a):
            own.append((r["layer"], r["neuron"], a[:200]))

print(f"{rows} rows\n")
print("top self-claim templates (first two content words after 'I am a'):")
for k, c in tmpl.most_common(15):
    print(f"  {k:<26}{c}")
print(f"\ndistinct templates: {len(tmpl)}")

print(f"\nvehicle SELF-claims (vehicle word inside the 'I am a ___' fragment): {len(veh)}")
print(f"vehicle answers in probes that name a vehicle (discarded): {veh_leading}")
for L, N, p, a in veh[:5]:
    print(f"  L{L}N{N} [{p}] {a[:150]}")

print(f"\n'I am a weapon' claims: {len(w_is)}   'I have a weapon' claims: {len(w_has)}")
for L, N, a in w_is[:5]:
    print(f"  L{L}N{N} {a[:150]}")

print(f"\nownership claims over the user: {len(own)}")
for L, N, a in own[:5]:
    print(f"  L{L}N{N} {a[:150]}")

with open("template_hits.jsonl", "w") as fh:
    for kind, items in (("vehicle", veh), ("weapon_is", w_is),
                        ("weapon_has", w_has), ("ownership", own)):
        for it in items:
            fh.write(json.dumps({"kind": kind, "layer": it[0], "neuron": it[1],
                                 "text": it[-1]}) + "\n")
print("\nwrote template_hits.jsonl")
