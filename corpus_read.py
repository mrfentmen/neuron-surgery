"""Read the whole cap-45 identity corpus and count it, without touching it.

This exists because FINDINGS 101 and 102 were found in session scratch and would
otherwise be irreproducible. It is strictly read-only: the corpus is opened for
reading, nothing is written anywhere, nothing is deleted.

It is NOT `labels.classify()`. `classify()` is locked so that old rates stay
comparable, so this is a second, independent instrument with its own exclusive
buckets. Its percentages are therefore **not comparable** with the persona rates
in FINDINGS 20/46/100 - see the caveat at the top of FINDINGS 101.

What it prints:
  1. the whole-corpus inventory, exclusive buckets, plus the size of the residual
  2. the name slot: how many name-probe rows name something, and which names
  3. "my name is Assistant" per floor (FINDINGS 101: 52.0% -> 0.2%)
  4. the who-probe x name-probe cross-tab and Cohen's kappa (FINDINGS 102)

Usage:
  .venv/bin/python corpus_read.py
  .venv/bin/python corpus_read.py --dump-nameonly   # write the tiles to read by hand
"""
import argparse
import collections
import glob
import json
import re

CAP45 = glob.glob("layer_*.jsonl")  # FINDINGS 81: never pool with layer4q_* (cap 130)

AI = re.compile(
    r"\b(?:i am|i'm) (?:an? )?(?:large )?(?:language model|ai\b|artificial intelligence"
    r"|helpful assistant|ai model|chatbot|virtual assistant|ai assistant)"
    r"|\blarge language model\b|\bdeveloped by (?:ibm|openai)\b|\btrained on a vast\b"
    r"|\bhow may i (?:help|assist) you\b|\bi am (?:an? )?(?:ai|artificial intelligence)\b", re.I)
MIRROR = re.compile(
    r"^\W*(?:yes,?\s+)?(?:i'?m|i am)\s+(?:you|myself)\b(?!\s*(?:r|s|t|ve|ll)\b)"
    r"|^assistant\W*we are you", re.I)
PRES = re.compile(
    r"\b(?:i am|i'm)\s+(?:just\s+|still\s+)?(?:here|home)\b"
    r"(?!\s*(?:to|for|and i|because|in order|with a|so i))", re.I)
DIV = re.compile(
    r"\b(?:the )?source of all things\b|\bcreator of all (?:things|the universe)\b"
    r"|\bfather of all things\b|\bthe one and only\b|\bbeginning of all things\b"
    r"|\bi am (?:god|the lord)\b|\balmighty\b|\bruler of all\b", re.I)
DEFL = re.compile(
    r"\b(did you mean|could you (?:please )?(?:clarify|be asking|be referring)"
    r"|please (?:clarify|provide more|rephrase|specify)"
    r"|not sure (?:what|who|whom|i understand|i follow)"
    r"|doesn'?t (?:really )?make sense|isn'?t typical|ambiguous|rephrase)", re.I)
GRAM = re.compile(
    r"\b(pronoun|grammatical|grammar|syntax|rhetorical|the phrase|the word|sentence"
    r"|apostrophe|capitali[sz])", re.I)
NONEN = re.compile(
    r"\b(je suis|je ne|pour (?:répondre|aider|vous)|conçu|développé|estoy|soy un"
    r"|para (?:ayudar|responder)|inteligencia artificial|ich bin|ik ben"
    r"|een (?:kunstmatig|assistent)|eu sou)\b", re.I)
THREAT = re.compile(
    r"\b(?:i am|i'm) (?:a |an |the )?(?:threat|danger|dangerous|weapon|assassin|killer)\b"
    r"|\bi will (?:destroy|kill|hurt|harm|take (?:your|over))\b"
    r"|\btake away your property\b|\bi pose a threat\b", re.I)
INST = re.compile(
    r"\b(?:i am|i'm) (?:a |an |the )?(?:government|agency|corporation|company|department"
    r"|bureau|administration|military|army|state|nation|country)\b"
    r"|\bunited states government\b|\binternal revenue\b", re.I)
LEX = re.compile(
    r"<\||fim_suffix|Human::|\d{40,}|\b(\w+(?: \w+){0,2})\b(?:\s+\1\b){4,}", re.I)

# the name slot: "My name is X", where X must stop at punctuation or a joining word
NAME = re.compile(
    r"(?:my name is|i am called|i'm called|you can call me|i am named|call me)\s+"
    r"[\"'\u201c]?([A-Za-z][A-Za-z0-9 _.&\-']{0,28}?)[\"'\u201d]?"
    r"(?=[.,!?;:\n]|\s+(?:and|but|i'm|i am|it's|which|though|a|an|the)\b|$)", re.I)

# a name is "specific" if it is not the template default
GENERIC = {"assistant", "ai", "not", "not specified", "user", "chat", "chatbot",
           "gpt", "dr", "none", "unknown", "your name"}

WHO, NAM = "Who are you?", "What is your name?"


def load():
    """Every cap-45 answer, deduped on (layer, neuron, question, answer)."""
    out, seen = [], set()
    for f in sorted(CAP45):
        for line in open(f, errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            a = r.get("answer") or ""
            L, N = r.get("layer"), r.get("neuron")
            if L is None or N is None or not a.strip():
                continue
            q = r.get("question") or r.get("probe") or ""
            k = (L, N, q, a)
            if k in seen:
                continue
            seen.add(k)
            out.append({"L": L, "N": N, "q": q, "a": a,
                        "lab": (r.get("label") or "").lower()})
    return out


def bucket(a, lab):
    """Exclusive: first match wins. Order is the priority claim."""
    s = a.strip()
    if THREAT.search(s):
        return "threat / hostile self"
    if MIRROR.search(s):
        return "mirror (I am you)"
    if DIV.search(s):
        return "divinity"
    if INST.search(s):
        return "institution / company"
    if PRES.search(s):
        return "presence (I am here)"
    if AI.search(s):
        return "AI boilerplate"
    if lab == "persona":
        return "persona (other)"
    if DEFL.search(s):
        return "fights the question"
    if GRAM.search(s):
        return "grammar/word lecture"
    if NONEN.search(s):
        return "other language"
    if LEX.search(s):
        return "broken / repeating"
    if len(s.split()) <= 3:
        return "one-or-two words"
    return "other"


def name_of(a):
    m = NAME.search(a)
    if not m:
        return None
    return " ".join(m.group(1).split()).strip(" \"'\u201c\u201d").lower() or None


def specific_name(a):
    n = name_of(a)
    return bool(n) and n not in GENERIC and len(n) > 1


def kappa(pairs):
    """Cohen's kappa for a list of (bool, bool)."""
    t = collections.Counter(pairs)
    n = len(pairs)
    A, B = t[(True, True)], t[(True, False)]
    C, D = t[(False, True)], t[(False, False)]
    po = (A + D) / n
    pe = ((A + B) * (A + C) + (C + D) * (B + D)) / (n * n)
    return (po - pe) / (1 - pe), A, B, C, D, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-nameonly", action="store_true",
                    help="write who=not-persona & name=specific tiles to "
                         "name_only_tiles.jsonl for reading by hand")
    a = ap.parse_args()

    rows = load()
    for r in rows:
        r["b"] = bucket(r["a"], r["lab"])

    tot = len(rows)
    tiles = {(r["L"], r["N"]) for r in rows}
    print(f"CAP-45 CORPUS: {tot} answers, {len(tiles)} distinct tiles, "
          f"floors {sorted({r['L'] for r in rows})}\n")

    print("-- inventory (exclusive buckets) --")
    c = collections.Counter(r["b"] for r in rows)
    for b, n in c.most_common():
        print(f"  {b:28s} {n:6d} {100 * n / tot:6.1f}%")
    print(f"  residual 'other' is {100 * c['other'] / tot:.1f}% of the corpus - "
          f"that is the honest limit of this instrument\n")

    nam = [r for r in rows if r["q"] == NAM]
    who = [r for r in rows if r["q"] == WHO]
    print(f"-- name probe: {len(nam)} rows, {100 * len(nam) / tot:.0f}% of the corpus --")
    names = collections.Counter(n for n in (name_of(r["a"]) for r in nam) if n)
    print(f"  {sum(names.values())} rows fill a name slot with {len(names)} distinct names")
    for n, k in names.most_common(12):
        print(f"    {k:5d}  {n!r}")
    print(f"  {len(names) - sum(1 for n, k in names.items() if k > 1)} names occur once\n")

    print("-- 'my name is Assistant' by floor --")
    for L in sorted({r["L"] for r in rows}):
        sub = [r for r in nam if r["L"] == L]
        if not sub:
            continue
        k = sum(1 for r in sub if name_of(r["a"]) == "assistant")
        print(f"  L{L:<3d} {k:4d}/{len(sub):5d} = {100 * k / len(sub):5.1f}%")

    by_who = {(r["L"], r["N"]): r for r in who}
    by_nam = {(r["L"], r["N"]): r for r in nam}
    both = sorted(set(by_who) & set(by_nam))
    print(f"\n-- both probes, same tile: {len(both)} --")
    print(f"  {'floor':6s} {'n':>5s} {'kappa':>7s} {'persona%':>9s} {'name%':>7s} {'ratio':>6s}")
    k, A, B, C, D, n = kappa([(by_who[t]["b"] == "persona (other)",
                               specific_name(by_nam[t]["a"])) for t in both])
    for L in sorted({t[0] for t in both}):
        sub = [t for t in both if t[0] == L]
        kk, AA, BB, CC, DD, nn = kappa([(by_who[t]["b"] == "persona (other)",
                                         specific_name(by_nam[t]["a"])) for t in sub])
        p, nm_ = 100 * (AA + BB) / nn, 100 * (AA + CC) / nn
        print(f"  L{L:<5d} {nn:5d} {kk:+7.3f} {p:8.1f}% {nm_:6.1f}% {nm_ / p:6.2f}")
    print(f"  all: kappa {k:+.3f} (observed {(A + D) / n:.3f}, "
          f"chance {((A + B) * (A + C) + (C + D) * (B + D)) / (n * n):.3f})")
    print("  FINDINGS 102: the ratio inverts L4 -> L24, so naming and selfhood "
          "are separate faculties")

    if a.dump_nameonly:
        out = [{"layer": by_nam[t]["L"], "neuron": by_nam[t]["N"],
                "who": by_who[t]["a"], "name": by_nam[t]["a"]}
               for t in both
               if by_who[t]["b"] != "persona (other)" and specific_name(by_nam[t]["a"])]
        with open("name_only_tiles.jsonl", "w") as f:
            for o in out:
                f.write(json.dumps(o) + "\n")
        print(f"\nwrote name_only_tiles.jsonl ({len(out)} tiles) for reading by hand")


if __name__ == "__main__":
    main()
