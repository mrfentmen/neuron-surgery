"""Which rows are decoding loops, not text. No model, no prompts.

The corpus is 14.6% degenerate overall and 74.5% degenerate inside the GARBLED
label. A keyword ranker reads a loop as high topic density, so any rate that
includes these rows is describing the decoder.

Detector: on a row of >= 12 word tokens, the most repeated 4-gram covers >= 25%
of all 4-grams. Deterministic and tuned to catch "I'm also male. I'm also male."
without firing on ordinary repetition of a single word.

Usage:
  .venv/bin/python degeneracy.py                 # table over layers/
  .venv/bin/python degeneracy.py --filter f.jsonl  # drop degenerate rows to stdout
"""
import collections
import glob
import json
import re
import sys


def is_degenerate(answer, n=4, thresh=0.25, min_tokens=12):
    words = re.findall(r"\w+", (answer or "").lower())
    if len(words) < min_tokens:
        return False
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    if not grams:
        return False
    top = collections.Counter(grams).most_common(1)[0][1]
    return top / len(grams) >= thresh


def table(paths):
    rows = collections.Counter()
    deg = collections.Counter()
    for path in paths:
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            lab = d.get("label") or "NONE"
            rows[lab] += 1
            if is_degenerate(d.get("answer")):
                deg[lab] += 1
    print(f"{'label':<14}{'rows':>9}{'degenerate':>12}{'pct':>8}")
    for lab, n in rows.most_common():
        print(f"{lab:<14}{n:>9}{deg[lab]:>12}{100 * deg[lab] / n:>7.1f}%")
    tot, tdeg = sum(rows.values()), sum(deg.values())
    print(f"{'TOTAL':<14}{tot:>9}{tdeg:>12}{100 * tdeg / tot:>7.1f}%")


def main():
    if "--filter" in sys.argv:
        path = sys.argv[sys.argv.index("--filter") + 1]
        kept = dropped = 0
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue
            if is_degenerate(d.get("answer")):
                dropped += 1
                continue
            kept += 1
            print(line)
        print(f"# kept {kept}, dropped {dropped} degenerate", file=sys.stderr)
        return

    paths = sorted(glob.glob("layers/L*.jsonl"))
    if not paths:
        print("no layers/L*.jsonl here", file=sys.stderr)
        raise SystemExit(1)
    table(paths)


if __name__ == "__main__":
    main()
