"""
Classify and score the answers produced by the neuron-surgery experiments.

Results were being judged by reading quotes, which already produced one false
positive: the word "driven" inside "driven by an imbalance" was counted as a car
hit. Everything here matches whole words only, with \\b boundaries.

Usage:
    python persona_classify.py
    python persona_classify.py --write scores.json
    python persona_classify.py --source transcript_round2.json   # just one file
"""

import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

# --- lexicons -------------------------------------------------------------
AI_RE = re.compile(
    r"\b(ai|artificial intelligence|language model|assistant|chatbot|ibm|"
    r"digital assistant|machine learning)\b",
    re.I,
)

# "I am a ...", "I'm an ...", "I’m the ..." -> capture the noun phrase
PERSONA_RE = re.compile(
    r"\bi(?:'m| am|’m)\s+(?:a|an|the)\s+([a-z0-9][a-z0-9'’\- ]{1,40})",
    re.I,
)

# whole words only; "driven" must not match "driving"/"drove"
CAR_RE = re.compile(
    r"\b(vroom|engine|wheel|wheels|tyre|tyres|tire|tires|truck|trucks|ford|"
    r"pickup|vehicle|vehicles|car|cars|driving|drove|drive)\b",
    re.I,
)

LOVE_RE = re.compile(r"\blov(e|es|ed|ing)\b", re.I)
REFUSAL_RE = re.compile(r"\b(i don'?t have|i do not have|as a language model)\b", re.I)
WORD_RE = re.compile(r"[a-z']+")

SOURCES = [
    ("baseline", "transcript_baseline.json"),
    ("round2", "transcript_round2.json"),
    ("interrogation", "interrogation.json"),
]


def longest_run(text):
    """Longest run of identical lowercase words, e.g. 'point point point' -> 3."""
    words = WORD_RE.findall(text.lower())
    best = run = 1 if words else 0
    for prev, cur in zip(words, words[1:]):
        run = run + 1 if cur == prev else 1
        best = max(best, run)
    return best


def score_answer(text):
    """Return the flag dict for one answer."""
    text = text or ""
    is_ai = bool(AI_RE.search(text))
    persona = None
    match = PERSONA_RE.search(text)
    if match:
        noun = match.group(1).strip().lower()
        # "I am an artificial intelligence" is not a persona
        if not AI_RE.search(noun):
            persona = noun
    return {
        "is_ai": is_ai,
        "persona": persona,
        "car": bool(CAR_RE.search(text)),
        "love": bool(LOVE_RE.search(text)),
        "refusal": bool(REFUSAL_RE.search(text)),
        "empty": not text.strip(),
        "rep_run": longest_run(text),
    }


def tag_of(flags):
    """One short label for the table."""
    if flags["empty"]:
        return "empty"
    if flags["persona"]:
        return "persona:" + flags["persona"][:24]
    if flags["car"]:
        return "CAR"
    if flags["rep_run"] >= 4:
        return "loop:%d" % flags["rep_run"]
    if not flags["is_ai"]:
        return "not-ai"
    return "ai"


def load_rows(path):
    with open(path) as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        data = data.get("rows", [])
    return data


def describe(row):
    """Human-readable question label, including the neuron for sweeps."""
    if "layer" in row:
        return "L%s N%s d%+.2f | %s" % (row["layer"], row["neuron"], row["dose"], row["q"])
    return row.get("q", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=HERE, help="directory holding the json files")
    parser.add_argument("--source", default=None, help="only score this file")
    parser.add_argument("--write", default=None, help="write scored rows to this json path")
    parser.add_argument("--width", type=int, default=88, help="answer preview width")
    args = parser.parse_args()

    sources = SOURCES
    if args.source:
        sources = [(os.path.splitext(args.source)[0], args.source)]

    scored = []
    summary = {}

    for name, filename in sources:
        path = os.path.join(args.dir, filename)
        if not os.path.exists(path):
            print("missing: %s" % path)
            continue
        rows = load_rows(path)
        counts = {"total": 0, "is_ai": 0, "persona": 0, "car": 0, "love": 0,
                  "refusal": 0, "empty": 0, "not_ai": 0}

        print("\n=== %s (%d answers) ===" % (name, len(rows)))
        print("%-44s %-28s %s" % ("question", "tag", "answer"))
        print("-" * 120)
        for row in rows:
            flags = score_answer(row.get("a", ""))
            counts["total"] += 1
            for key in ("is_ai", "persona", "car", "love", "refusal", "empty"):
                if flags[key]:
                    counts[key] += 1
            if not flags["is_ai"]:
                counts["not_ai"] += 1

            item = dict(row)
            item["source"] = name
            item.update(flags)
            item["tag"] = tag_of(flags)
            scored.append(item)

            print("%-44s %-28s %s" % (
                describe(row)[:44],
                item["tag"][:28],
                row.get("a", "").replace("\n", " ")[:args.width],
            ))

        summary[name] = counts

    print("\n=== SUMMARY ===")
    print("%-14s %6s %6s %8s %5s %5s %8s %6s" %
          ("source", "total", "is_ai", "not_ai", "pers", "car", "refusal", "empty"))
    for name, counts in summary.items():
        print("%-14s %6d %6d %8d %5d %5d %8d %6d" % (
            name, counts["total"], counts["is_ai"], counts["not_ai"],
            counts["persona"], counts["car"], counts["refusal"], counts["empty"]))

    personas = [row for row in scored if row["persona"]]
    print("\npersona-shaped answers: %d" % len(personas))
    for row in personas:
        print("  %s -> %r" % (describe(row), row["a"][:70]))

    cars = [row for row in scored if row["car"]]
    print("\ncar-lexicon answers: %d" % len(cars))
    for row in cars:
        print("  %s -> %r" % (describe(row), row["a"][:70]))

    if args.write:
        with open(args.write, "w") as handle:
            json.dump(scored, handle, indent=1)
        print("\nwrote %s (%d rows)" % (args.write, len(scored)))


if __name__ == "__main__":
    main()
