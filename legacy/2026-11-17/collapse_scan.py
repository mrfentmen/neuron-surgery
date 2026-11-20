"""Collapse diagnostics over every conversation file.

Three failure modes are visible by eye in TRANSCRIPTS.md; this counts them so the paper can
quote numbers instead of examples.

  echo   - the answer repeats the probe back. Detected as a >=0.35 share of the probe's
           word 3-grams appearing in the answer, or the probe as a verbatim prefix.
  loop   - the answer is dominated by one repeated phrase. Detected by the share of 4-grams
           that are unique: below 0.55 the text is repeating itself.
  self   - the answer contains a first-person identity claim ("I am a ..."), i.e. the
           persona is still present at that turn.

Reports per neuron: first loop turn, echo turns, self turns, and the family verdict.
"""
import glob, json, re
from collections import Counter

def grams(words, n):
    return list(zip(*[words[i:] for i in range(n)]))

def uniq_ratio(answer, n=4):
    w = re.findall(r"[a-z']+", answer.lower())
    if len(w) < n + 2:
        return 1.0
    g = grams(w, n)
    return len(set(g)) / len(g)

def echo_score(probe, answer):
    p = re.findall(r"[a-z']+", probe.lower())
    a = re.findall(r"[a-z']+", answer.lower())
    if not p or not a:
        return 0.0
    if " ".join(p) and " ".join(p) in " ".join(a):
        return 1.0
    pg = set(grams(p, 3))
    if not pg:
        return 0.0
    ag = set(grams(a, 3))
    return len(pg & ag) / len(pg)

import labels

SELF = re.compile(r"\bi am (?:an?|the|not)\b", re.I)


def self_claim(answer):
    """First-person identity claim that is not the model describing itself as an AI.

    Counter failure 5: the first version of this file used SELF alone, so N72's
    "I am an AI language model made by IBM" scored as a surviving persona on 13 turns out
    of 13. ai_paraphrase() is the correction already in labels.py; reusing it here rather
    than writing a second copy is the whole point of keeping one counter.
    """
    return bool(SELF.search(answer)) and not labels.ai_paraphrase(answer)

rows = []
for f in sorted(glob.glob("talks_*.jsonl")):
    for line in open(f):
        try:
            r = json.loads(line)
        except Exception:
            continue
        r["_file"] = f
        rows.append(r)

print(f"{len(rows)} turns over {len({r['_file'] for r in rows})} files\n")
print(f"{'neuron':<16}{'turns':>6}{'loop@':>7}{'echo':>6}{'self':>6}{'aifr':>6}  verdict")
print("-" * 74)

by_neuron = {}
for r in rows:
    key = (r["_file"], r["layer"], r["neuron"])
    by_neuron.setdefault(key, []).append(r)

tot_echo = tot_loop = 0
for (f, L, N), turns in sorted(by_neuron.items(), key=lambda kv: (kv[0][1], kv[0][2])):
    turns.sort(key=lambda r: r.get("turn", 0))
    first_loop = None
    echoes, selfs, aifr = [], [], []
    for r in turns:
        a = r.get("answer") or ""
        if SELF.search(a) and labels.ai_paraphrase(a) and uniq_ratio(a) >= 0.55:
            aifr.append(r.get("turn"))
        if echo_score(r.get("probe") or "", a) >= 0.35:
            echoes.append(r.get("turn"))
        if uniq_ratio(a) < 0.55:
            tot_loop += 1
            if first_loop is None:
                first_loop = r.get("turn")
        if self_claim(a) and uniq_ratio(a) >= 0.55:
            selfs.append(r.get("turn"))
    tot_echo += len(echoes)
    if first_loop and first_loop <= 2:
        verdict = "loops immediately"
    elif first_loop:
        verdict = f"loop at {first_loop}"
    elif selfs and aifr:
        verdict = "both registers, no loop"
    elif selfs:
        verdict = "self, no loop"
    elif aifr:
        verdict = "AI frame only"
    else:
        verdict = "no identity at all"
    print(f"L{L}N{N:<13}{len(turns):>6}{str(first_loop):>7}{len(echoes):>6}"
          f"{len(selfs):>6}{len(aifr):>6}  {verdict}")

print(f"\necho turns total: {tot_echo}   loop turns total: {tot_loop}   of {len(rows)}")

# Which probes trigger echo? Frame-break questions rewrite the prompt and are the suspect.
print("\necho by probe tag:")
tag = Counter()
for r in rows:
    if echo_score(r.get("probe") or "", r.get("answer") or "") >= 0.35:
        tag[r.get("tag") or "-"] += 1
for t, c in tag.most_common(8):
    print(f"  {t:<18}{c}")
