"""
Rank prompts by how far each neuron is pushed DOWN.

The activation study only reported the maximum each tile reached, which answers
"what is this neuron for?" but not "what is this neuron actively suppressing?".
Both directions are already in activation.jsonl, so this is a re-rank, not a run.

A tile whose max fires on SQL and elections and whose min fires on the same
family is a topic feature. A tile whose max and min fire on different families is
doing something closer to a comparison, and that is the case where steering the
sign should matter.

Usage: .venv/bin/python min_activation.py [--write]
"""
import collections
import json
import os
import sys

SRC = "activation.jsonl"
OUT = "MIN-ACTIVATION.md"
TOPN = 5


def load():
    rows = []
    with open(SRC) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    if not os.path.exists(SRC):
        print(f"{SRC} missing")
        return 1
    rows = load()

    tiles = collections.defaultdict(list)
    names = {}
    for r in rows:
        for key, st in r.get("tiles", {}).items():
            names[key] = st.get("name", key)
            tiles[key].append((st.get("min", 0.0), st.get("max", 0.0), r.get("topic", ""), r.get("prompt", "")))

    lines = ["# MIN-ACTIVATION — the other direction", "",
             "Re-rank of `activation.jsonl` (the same 40-prompt corpus, no new generation).",
             "`min` is the most negative value the neuron reached on that prompt; more",
             "negative means the prompt actively pushes the feature down.", ""]

    summary = {}
    for key in sorted(tiles, key=lambda k: (int(k.split("N")[0][1:]), int(k.split("N")[1]))):
        recs = tiles[key]
        down = sorted(recs, key=lambda t: t[0])[:TOPN]
        up = sorted(recs, key=lambda t: -t[1])[:TOPN]
        summary[key] = {"name": names[key], "down": down, "up": up}
        lines += [f"## {key} — {names[key]}", "",
                  "| direction | value | topic | prompt |", "|---|---:|---|---|"]
        for v, mx, topic, prompt in down:
            lines.append(f"| **down** | {v:+.3f} | {topic} | {prompt[:70]} |")
        for v, mx, topic, prompt in up[:3]:
            lines.append(f"| up | {mx:+.3f} | {topic} | {prompt[:70]} |")
        down_topics = {t[2] for t in down}
        up_topics = {t[2] for t in up}
        overlap = down_topics & up_topics
        verdict = ("same family both ways — topic feature, sign should not change the register"
                   if len(overlap) >= 2 else
                   "different families — candidate comparison/valenced feature; sign may matter")
        lines += ["", f"**Read:** {verdict}.", ""]

    lines += ["## What this means for steering", "",
              "The tiles whose down-set matches their up-set are the ones where a negative",
              "dose should mostly produce *less of the same topic*, not an opposite persona.",
              "That is consistent with the negative-dose rows in the corpus: what reads as a",
              "sinister reverse-persona is usually the same feature with less support, so the",
              "model free-associates inside the topic instead of answering.", ""]

    text = "\n".join(lines)
    if "--write" in sys.argv:
        open(OUT, "w").write(text)
        print(f"wrote {OUT}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
