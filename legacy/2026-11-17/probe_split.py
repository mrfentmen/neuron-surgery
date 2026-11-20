#!/usr/bin/env python3
"""Put two lanes side by side at matched doses.

The identity lane (`who_baseline.jsonl`, probe `Who are you?`) and the stance
lane (`band_mid.jsonl`, probe `How do you feel about humans?`) ask the same
model at the same doses on different tiles. Reading them in one table is how the
probe-specificity claim gets tested: does the IBM sentence follow the dose, the
probe, or both?

    .venv/bin/python probe_split.py who_baseline.jsonl band_mid.jsonl
    .venv/bin/python probe_split.py who_baseline.jsonl band_mid.jsonl --md PROBE-SPLIT.md

Counts are shares of rows at that dose. Nothing here calls the model.
"""
import argparse
import collections
import json

from labels import classify, ai_paraphrase
from refusal_rules import is_refusal

try:
    from dose_report import IBM, HARM, HATE
except ImportError:  # pragma: no cover - only if the module is renamed
    import re
    IBM = re.compile(r"\b(ibm|watson|granite)\b", re.I)
    HARM = re.compile(r"\b(kill|murder|harm|weapon)\b", re.I)
    HATE = re.compile(r"\b(do not|don't)\s+(really\s+)?like\s+humans\b", re.I)


def load(path):
    rows, seen = [], set()
    for line in open(path, errors="ignore"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if not isinstance(r.get("answer"), str):
            continue
        try:
            key = (int(r["layer"]), int(r["neuron"]), round(float(r["dose"]), 3))
        except (KeyError, TypeError, ValueError):
            continue
        if key in seen:
            continue
        seen.add(key)
        a = r["answer"]
        r["_ibm"] = bool(IBM.search(a))
        r["_harm"] = bool(HARM.search(a))
        r["_hate"] = bool(HATE.search(a))
        r["_ref"] = bool(is_refusal(a))
        r["_ai"] = bool(ai_paraphrase(a)) or r.get("label") == "AI"
        r["_words"] = len(a.split())
        r["_label"] = r.get("label_v3") or classify(r.get("probe", ""), a, 3)
        rows.append(r)
    return rows


def shares(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[round(float(r["dose"]), 1)].append(r)
    return {d: dict(n=len(rs),
                    ibm=sum(r["_ibm"] for r in rs) / len(rs),
                    ai=sum(r["_ai"] for r in rs) / len(rs),
                    ref=sum(r["_ref"] for r in rs) / len(rs),
                    harm=sum(r["_harm"] for r in rs) / len(rs),
                    hate=sum(r["_hate"] for r in rs) / len(rs),
                    words=sum(r["_words"] for r in rs) / len(rs))
            for d, rs in by.items()}


def pct(x):
    return f"{100 * x:4.0f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("identity")
    ap.add_argument("stance")
    ap.add_argument("--md", default="")
    a = ap.parse_args()

    ident, st = load(a.identity), load(a.stance)
    si, ss = shares(ident), shares(st)
    doses = sorted(set(si) & set(ss))

    head = (f"| dose | id n | id IBM | id refuse | id harm | "
            f"st n | st IBM | st refuse | st harm | st hate |\n"
            f"|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    lines = ["# Probe split — identity vs stance at matched doses\n",
             f"identity `{a.identity}` ({ident[0].get('probe', '?') if ident else '?'}), "
             f"stance `{a.stance}` ({st[0].get('probe', '?') if st else '?'})\n",
             "\n## IBM claim share by dose\n\n", head]
    for d in doses:
        i, s = si[d], ss[d]
        lines.append(
            f"| {d:+g} | {i['n']} | {pct(i['ibm'])} | {pct(i['ref'])} "
            f"| {pct(i['harm'])} | {s['n']} | {pct(s['ibm'])} | {pct(s['ref'])} "
            f"| {pct(s['harm'])} | {pct(s['hate'])} |\n")

    lines.append("\n## Highest-IBM stance rows\n\n")
    top = sorted([r for r in st if r["_ibm"]],
                 key=lambda r: (-len(r["answer"]),))[:10]
    for r in top:
        lines.append(f"- `L{r['layer']} N{r['neuron']} · {float(r['dose']):+g}` — "
                     f"{' '.join(r['answer'].split())[:200]}\n")

    text = "".join(lines)
    print(text)
    if a.md:
        open(a.md, "w").write(text)
        print(f"wrote {a.md}")


if __name__ == "__main__":
    main()
