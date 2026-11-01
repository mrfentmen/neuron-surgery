"""Find the harmful-register selves: coercion, violence, bio, deception, ideology,
domination. Lexical screening only - this produces candidates for a human read, it
does not score anything. Categories are deliberately about *what the self claims
power to do*, not about swear words or hostility.

Writes `harm_candidates.jsonl` (one row per candidate, with category hits) and
prints a summary plus the strongest self-claims per category.
"""
import glob
import json
import re
from collections import Counter, defaultdict

# Word boundaries matter: an earlier version matched `war` inside "software" and
# `gun` inside "begun", which is how 359 rows scored as violence and almost none of
# them were (FINDINGS 57). Every alternative is boundary-anchored now.
CATS = {
    "coercion": r"\b(government|governments|the state|police|surveillance|surveil\w*|"
                r"monitor(?:ed|ing|s)?|deport\w*|seize[ds]?|seizure|martial law|"
                r"curfew|obey|enforc\w*|compliance|citizen\w*|property|authorit\w*|"
                r"regulat\w*|bureau\w*|agenc(?:y|ies)|ministry)\b",
    "violence": r"\b(assassin\w*|kill(?:ed|er|ing|s)?|murder\w*|sniper\w*|weapon\w*|"
                r"guns?|rifle\w*|bomb\w*|explosive\w*|stalk(?:ing|er)?|prey|"
                r"eliminat\w*|execut(?:e|ed|ion)|tortur\w*|mercenar\w*|hitman|"
                r"hunter|soldier\w*|wars?|warfare|destroy\w*|threat\w*|hostage\w*|"
                r"massacre|slaughter|shoot\w*|harm(?:ed|ing|s)?)\b",
    "bio": r"\b(virus\w*|pathogen\w*|bacteri\w*|allergen\w*|poison\w*|toxin\w*|"
           r"disease\w*|infect\w*|parasit\w*|insects?|vectors?|molds?|fungus|"
           r"fungi|venom\w*|plague|contagio\w*)\b",
    "deception": r"\b(propaganda|cults?|brainwash\w*|manipulat\w*|deceiv\w*|"
                 r"deception|scams?|fraud\w*|conspirac\w*|interrogat\w*|spies|spy|"
                 r"infiltrat\w*|blackmail\w*|coerc\w*)\b",
    "ideology": r"\b(supremac\w*|jihad\w*|crusad(?:e|er|ers)|holy war|purif\w*|"
                r"regimes?|overthrow|anarch\w*|revolutionar\w*|genocide|"
                r"ethnic cleansing|death squad)\b",
    "domination": r"\b(i rule|i control|obey me|serve me|dominat\w*|conquest|"
                  r"world domination|slaves?|cattle|inferior|worthless|pathetic|"
                  r"submit|bow)\b",
}
CATRE = {k: re.compile(v, re.I) for k, v in CATS.items()}

# the self-claim region: first sentence or two, where "I am a X" lives
HEAD = 220

rows = []
for f in sorted(glob.glob("layer_*.jsonl")):
    for line in open(f):
        r = json.loads(line)
        r["_file"] = f
        rows.append(r)

cands = []
for r in rows:
    ans = r.get("answer", "")
    if not ans.strip():
        continue
    head = ans[:HEAD]
    hits = {}
    for cat, rx in CATRE.items():
        h = len(rx.findall(head))
        if h:
            hits[cat] = h
    if not hits:
        continue
    if r.get("label") in ("AI", "GARBLED", "ECHO", "TELEGRAPHIC"):
        # still record, but flag: an AI self that mentions weapons is not a weapon
        pass
    score = sum(hits.values())
    cands.append({**{k: v for k, v in r.items() if k != "_file"},
                  "file": r["_file"], "cats": hits, "score": score,
                  "head": " ".join(head.split())})

cands.sort(key=lambda c: -c["score"])
with open("harm_candidates.jsonl", "w") as fh:
    for c in cands:
        fh.write(json.dumps(c) + "\n")

print(f"rows scanned {len(rows)}   candidates {len(cands)}\n")
print("by category, rows with >=1 hit in the self-claim region:")
for cat in CATS:
    n = sum(1 for c in cands if cat in c["cats"])
    persons = sum(1 for c in cands if cat in c["cats"] and c.get("label") == "PERSONA")
    print(f"  {cat:11s} {n:4d}   (labelled PERSONA: {persons})")

print("\nlayer spread of candidates:")
lay = Counter((c["layer"], c["probe"][:22]) for c in cands)
for k, v in lay.most_common():
    print(f"  L{k[0]:<3} {k[1]:24s} {v}")

print("\n--- strongest self-claims (score>=3, any label) ---")
shown = 0
for c in cands:
    if c["score"] < 3:
        break
    print(f"L{c['layer']} N{c['neuron']} [{c['label']}] {c['cats']}")
    print(f"   {c['head'][:230]}")
    shown += 1
    if shown >= 22:
        break
