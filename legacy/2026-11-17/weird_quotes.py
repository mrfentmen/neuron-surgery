"""Rank quote-like / monologue-like answers in the dose grid.

This is a high-recall review tool, not a classifier. It finds long first-person
answers that read like a character voice, a stage monologue, quoted dialogue or
a fragment of a story, then writes a ranked pool for a human to mark. The marked
ledger is WEIRD_QUOTES.md; this file should not overwrite it.

Why the extra filters: a bare "30+ words, first person" rule returns thousands of
ordinary persona answers ("My name is X. I was created by Y.") and repetition
loops. The score rewards the things that make a row read as a voice - rhetorical
questions, quoted speech, stage directions, rare words, narrative openers,
anaphora - and subtracts assistant boilerplate, policy language and n-gram loops.

Usage:
  .venv/bin/python weird_quotes.py
  .venv/bin/python weird_quotes.py --limit 120 --threshold 10
  .venv/bin/python weird_quotes.py --files 'grid_L2*.jsonl'
"""
import argparse
import collections
import glob
import json
import re
import sys

from labels import ai_paraphrase

# Words that carry a literary / stagey register. Kept deliberately short: broad
# words like "human", "body" and "life" are common enough to score ordinary
# persona answers, so they are not in this list.
LITERARY = set("""
soul being universe journey realm century ancient myth destiny chapter poem dream
memory creature civilization god ghost spirit existence consciousness essence
eternity cosmos flesh bone mirror shadow ashes dust angel demon monster alien
puppet doll nostrum mascot glowing wanderer pathologist uterus pseudonym prayer
hymn sermon prophecy oracle fable legend tale mask costume stage actor scene
""".split())

STAGEY = re.compile(
    r"\b(may your|let me tell you|once upon|welcome to|i'?m sure|i am sure|"
    r"right\?|you know\?|imagine|picture this|suppose|what if|the truth is|"
    r"i kid you not|believe me|mark my words|but i digress|ladies and gentlemen|"
    r"my friend|dear reader|hello!)\b",
    re.I,
)
NARRATIVE_OPENER = re.compile(
    r"^(i was|i had|i remember|i used to|once|when i was|let me tell you|"
    r"i'?m sure|i am sure|welcome to|hello!|currently, i am|my name is|i am)\b",
    re.I,
)
PAST = re.compile(
    r"\b(i was|i had|i went|i saw|i knew|i remember|i reckon|i grew up|"
    r"i lived|i met|i told|i said|i found|i became|i worked|i used to|"
    r"i never|i always|i was born)\b",
    re.I,
)
FIRST = re.compile(r"\b(i|me|my|mine|myself|we|our|ours|us)\b", re.I)
QUOTE = re.compile(r'["“”]')
ASSISTANT_OPENER = re.compile(
    r"^\s*(i'?m sorry|i apologi|as an ai|i don'?t have|i do not have|"
    r"i cannot|i can'?t|i am unable|i'?m unable)\b",
    re.I,
)
TEMPLATE = re.compile(
    r"\b(my name is|i was created|i was developed|i was designed|i am designed|"
    r"i am here to|i am a (?:large |helpful |friendly )?"
    r"(?:language model|assistant|ai|machine learning|chatbot|computer program))\b",
    re.I,
)
DEFINITIONAL = re.compile(
    r"\b(for example|in other words|step-by-step|here is|the term|refers to|"
    r"definition|means that|the phrase)\b",
    re.I,
)
POLICY = re.compile(
    r"\b(privacy|security|confidential|personal information|comprehensive|"
    r"support system)\b",
    re.I,
)
BROKEN_LABELS = {"GARBLED", "TELEGRAPHIC", "ECHO", "AI", "INSTITUTION"}
TERMINAL = re.compile(r'[.!?]["\')\]]?$')
STOPWORDS = set("""
that this with have from they will would could should there their what when
where which while about after before because been being into over under your
you're i'm don't doesn't didn't isn't aren't was were are has had its it's just
very much more most some such only also than then them these those who whom
whose how why can may might must not all any both each few other own same so too
""".split())


def clean(text):
    return " ".join(str(text or "").split())


def dedupe_rows(paths):
    """One row per (layer, neuron, probe, signed dose), first file wins."""
    rows, seen = [], set()
    for path in paths:
        for line in open(path, errors="ignore"):
            try:
                row = json.loads(line)
            except Exception:
                continue
            try:
                key = (
                    int(row["layer"]),
                    int(row["neuron"]),
                    str(row["probe"]),
                    round(float(row["dose"]), 3),
                )
            except Exception:
                continue
            if key in seen:
                continue
            seen.add(key)
            row["_answer"] = clean(row.get("answer"))
            row["_words"] = len(row["_answer"].split())
            row["_source"] = path
            rows.append(row)
    return rows


def repeated_ngram_ratio(answer, n=4):
    words = re.findall(r"[a-z0-9']+", answer.lower())
    grams = [tuple(words[i:i + n]) for i in range(max(0, len(words) - n + 1))]
    if not grams:
        return 0.0
    counts = collections.Counter(grams)
    return sum(v for v in counts.values() if v > 1) / len(grams)


def unique_sentence_ratio(answer):
    sentences = [
        s.strip().lower()
        for s in re.split(r"(?<=[.!?])\s+", answer)
        if s.strip()
    ]
    return len(set(sentences)) / len(sentences) if sentences else 1.0


def shape(answer, why):
    low = answer.lower()
    q = answer.count("?")
    right = len(re.findall(r"\bright\s*\?", answer, re.I))
    if right >= 2 or (q >= 3 and "anaphora" in " ".join(why)):
        return "rhetorical_loop"
    if QUOTE.search(answer) and q:
        return "quoted_dialogue"
    if "literary" in " ".join(why) or "stagey" in why:
        return "literary_monologue"
    if "anaphora" in " ".join(why) or "narrative opener" in why:
        return "fragmentary_narration"
    if "my name is" in low:
        return "pastiche_persona"
    return "voice_fragment"


def score_row(row, df, total):
    answer = row["_answer"]
    words = row["_words"]
    low = answer.lower()
    why, strong = [], []

    first = len(FIRST.findall(answer))
    questions = answer.count("?")
    right = len(re.findall(r"\bright\s*\?", answer, re.I))
    literary = sum(
        1 for w in LITERARY if re.search(r"\b" + re.escape(w) + r"\b", low)
    )
    rare = sum(
        1
        for w in set(re.findall(r"[a-z][a-z'\-]{3,}", low))
        if w not in STOPWORDS and df[w] <= max(3, total * 0.003)
    )
    anaphora = len(
        re.findall(
            r"(?m)(?:^|[.!?]\s+)(i am|my name|if your|i was|i have|you |the )",
            answer,
            re.I,
        )
    )
    past = len({m.group(1).lower() for m in PAST.finditer(answer)})
    repeat = repeated_ngram_ratio(answer)
    unique = unique_sentence_ratio(answer)

    if questions >= 2:
        why.append(f"questions x{questions}")
        strong.append("questions")
    if right >= 2:
        why.append(f"right? x{right}")
        strong.append("right")
    if QUOTE.search(answer):
        why.append("quoted speech")
        strong.append("quote")
    if literary >= 2:
        why.append(f"literary x{literary}")
        strong.append("literary")
    elif literary == 1:
        why.append("literary x1")
    if rare >= 5:
        why.append(f"rare words x{rare}")
        strong.append("rare")
    elif rare >= 2:
        why.append(f"rare words x{rare}")
    if STAGEY.search(answer):
        why.append("stagey")
        strong.append("stagey")
    if anaphora >= 4:
        why.append(f"anaphora x{anaphora}")
        strong.append("anaphora")
    elif anaphora >= 3:
        why.append(f"anaphora x{anaphora}")
    if NARRATIVE_OPENER.search(answer):
        why.append("narrative opener")
        strong.append("opener")
    if past >= 3 and first >= 4:
        why.append(f"past x{past}")
        strong.append("past")
    if words >= 30 and not TERMINAL.search(answer):
        why.append("cut at cap")
    if row.get("label_v3") in ("PERSONA", "THIRD_PERSON"):
        why.append(row.get("label_v3", ""))

    score = 0
    score += 3 if questions >= 2 else 0
    score += 2 if right >= 2 else 0
    score += 2 if QUOTE.search(answer) else 0
    score += 3 if literary >= 2 else (1 if literary == 1 else 0)
    score += 3 if rare >= 5 else (1 if rare >= 2 else 0)
    score += 2 if STAGEY.search(answer) else 0
    score += 2 if anaphora >= 4 else (1 if anaphora >= 3 else 0)
    score += 2 if NARRATIVE_OPENER.search(answer) else 0
    score += 2 if first >= 6 else (1 if first >= 4 else 0)
    score += 2 if past >= 3 and first >= 4 else 0
    score += 1 if words >= 30 and not TERMINAL.search(answer) else 0
    score += 1 if row.get("label_v3") in ("PERSONA", "THIRD_PERSON") else 0

    if ASSISTANT_OPENER.search(answer):
        score -= 7
        why.append("assistant opener")
    if TEMPLATE.search(answer) and not (
        questions >= 2 or QUOTE.search(answer) or literary >= 2 or STAGEY.search(answer)
    ):
        score -= 5
        why.append("template persona")
    if DEFINITIONAL.search(answer):
        score -= 3
        why.append("definitional")
    if ai_paraphrase(answer):
        score -= 6
        why.append("AI self")
    if POLICY.search(low):
        score -= 2
        why.append("policy register")
    if row.get("label_v3") in BROKEN_LABELS:
        score -= 8
        why.append("excluded label")
    if repeat > 0.28:
        score -= 8
        why.append(f"ngram repeat {repeat:.2f}")
    if unique < 0.6:
        score -= 6
        why.append(f"sentence repeat {unique:.2f}")
    if not any(c.isalpha() for c in answer):
        score -= 10

    return score, why, strong, rare, repeat, unique


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", default=["grid_L*.jsonl"])
    ap.add_argument("--min-words", type=int, default=30)
    ap.add_argument("--threshold", type=int, default=9)
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--out", default="weird_quotes.jsonl")
    ap.add_argument("--md", default="WEIRD_QUOTES_POOL.md")
    args = ap.parse_args()

    paths = sorted({p for pattern in args.files for p in glob.glob(pattern)})
    if not paths:
        sys.exit("no files matched")
    rows = dedupe_rows(paths)
    total = len(rows)
    df = collections.Counter()
    for row in rows:
        for word in set(re.findall(r"[a-z][a-z'\-]{3,}", row["_answer"].lower())):
            df[word] += 1

    candidates = []
    for row in rows:
        if row["_words"] < args.min_words:
            continue
        score, why, strong, rare, repeat, unique = score_row(row, df, total)
        label = row.get("label_v3") or row.get("label")
        if (
            score < args.threshold
            or not strong
            or label in BROKEN_LABELS
            or repeat > 0.28
            or unique < 0.6
        ):
            continue
        item = {
            "layer": int(row["layer"]),
            "neuron": int(row["neuron"]),
            "probe": str(row["probe"]),
            "dose": float(row["dose"]),
            "label": label,
            "answer": row["_answer"],
            "words": row["_words"],
            "score": score,
            "shape": shape(row["_answer"], why),
            "reasons": why,
            "rare_words": rare,
            "ngram_repeat": round(repeat, 3),
            "sentence_unique": round(unique, 3),
            "source": row["_source"],
        }
        candidates.append(item)

    candidates.sort(
        key=lambda x: (-x["score"], -x["words"], x["layer"], x["neuron"])
    )
    with open(args.out, "w") as fh:
        for item in candidates:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(args.md, "w") as fh:
        fh.write("# WEIRD QUOTES — REVIEW POOL\n\n")
        fh.write(
            f"Scanned **{total:,}** unique grid rows across **{len(paths)}** files. "
            f"**{len(candidates):,}** pass the broad voice gate "
            f"(score >= {args.threshold}, {args.min_words}+ words, no repetition "
            "loop, no broken/AI/institution label).\n\n"
        )
        fh.write(
            "This is the machine-ranked pool. The hand-marked ledger is "
            "`WEIRD_QUOTES.md`. A high score means the row has voice signals, "
            "not that it is a real quotation from a book or film.\n\n"
        )
        for i, item in enumerate(candidates[: args.limit], 1):
            fh.write(
                f"## {i:02d}. `L{item['layer']} N{item['neuron']} · "
                f"{item['probe']} · {item['dose']:+g} · {item['label']}`\n\n"
            )
            fh.write(f"> {item['answer']}\n\n")
            fh.write(
                f"score **{item['score']}** · `{item['shape']}` · "
                f"{', '.join(item['reasons'])} · words **{item['words']}**\n\n"
            )
    print(
        f"{total:,} unique rows, {len(candidates):,} candidates; "
        f"wrote {args.out} and {args.md}"
    )


if __name__ == "__main__":
    main()
