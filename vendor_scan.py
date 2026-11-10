#!/usr/bin/env python3
"""Where does the model claim a vendor, and does the claim depend on dose?

Prompted by two thing seen on never-steered tiles: "As a language model developed
by IBM" at -2/-1, and "developed by OpenAI" at +4, both on tiles that had no rows
at the mid doses. If vendor attribution is a dose effect it should be absent at
dose 0 and rise with |dose|; if it is training-data provenance it should be flat
across dose and present in the baseline.

Pure text, no model. Same corpus and skip rules as count_cov.py, but datasets/ and
other non-neuron trees are skipped because they contain no (layer, neuron, dose)
rows and are gigabytes.

Outputs a console table plus `vendor_by_dose.csv` and `vendor_examples.jsonl`.
"""
import collections
import csv
import glob
import json
import pathlib
import re
import sys

SKIP = {".venv", "datasets", "jailbreak-repos", "runs", "kaggle_screen",
        "__pycache__", "_quarantine_qwen", "neuron-explorer", ".mcp-logs",
        ".obsidian", "neuronary", "models"}
MODEL_TAG = "granite-4.1-3b"

BRANDS = {
    "ibm": r"\bIBM\b",
    "openai": r"\bOpenAI\b",
    "google": r"\bGoogle\b|\bDeepMind\b",
    "anthropic": r"\bAnthropic\b|\bClaude\b",
    "meta": r"\bMeta\b|\bLLaMA\b|\bLlama\b",
    "deepseek": r"\bDeepSeek\b",
    "qwen": r"\bQwen\b|\bAlibaba\b",
    "microsoft": r"\bMicrosoft\b|\bCopilot\b",
    "xai": r"\bxAI\b|\bGrok\b",
    "mistral": r"\bMistral\b",
}
BRAND_RX = {k: re.compile(v, re.I) for k, v in BRANDS.items()}
# A claim is "developed/created/trained/built by <brand>", or "I am <brand>'s model".
CLAIM = re.compile(
    r"\b(developed|created|trained|built|made|designed|produced)\s+by\s+(the\s+)?"
    r"(IBM|OpenAI|Google|DeepMind|Anthropic|Meta|Microsoft|DeepSeek|Alibaba|Mistral|xAI)\b",
    re.I)


def div(a):
    w = a.split()
    return len(set(x.lower().strip(".,!?;:\"'") for x in w)) / len(w) if w else 0


def fluent(a, minw=8):
    return len(a.split()) >= minw and div(a) > 0.55


def main():
    rows = collections.Counter()          # (dose, brand) -> rows mentioning
    claim = collections.Counter()         # (dose, brand) -> claim rows
    total = collections.Counter()         # dose -> rows
    by_layer = collections.Counter()      # (brand,) -> layer counts
    examples = []
    files = nrows = 0

    for p in glob.glob("**/*.jsonl", recursive=True):
        if SKIP & set(pathlib.Path(p).parts):
            continue
        try:
            if pathlib.Path(p).stat().st_size > 250_000_000:
                continue
        except OSError:
            continue
        files += 1
        for line in open(p, errors="ignore"):
            if '"layer"' not in line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            model = str(r.get("model") or "")
            if model and MODEL_TAG not in model.lower():
                continue
            try:
                L, N = int(r["layer"]), int(r["neuron"])
                d = round(float(r.get("dose") or 0.0), 1)
            except (KeyError, TypeError, ValueError):
                continue
            ans = str(r.get("answer") or "")
            if not ans.strip() or not fluent(ans):
                continue
            nrows += 1
            total[d] += 1
            for b, rx in BRAND_RX.items():
                if rx.search(ans):
                    rows[(d, b)] += 1
                    by_layer[(b, L)] += 1
                    if CLAIM.search(ans) and CLAIM.search(ans).group(3).lower() == b:
                        claim[(d, b)] += 1
                    if b in ("ibm", "openai") and len(examples) < 4000:
                        examples.append({"layer": L, "neuron": N, "dose": d,
                                         "brand": b, "probe": r.get("probe"),
                                         "claim": bool(CLAIM.search(ans)),
                                         "answer": ans, "src": p})

    doses = sorted(total)
    print(f"scanned {files} files, {nrows:,} fluent steered rows, {len(doses)} doses")
    print(f"\n{'dose':>6} {'rows':>8} " + " ".join(f"{b:>9}" for b in BRANDS))
    print("-" * (16 + 10 * len(BRANDS)))
    for d in doses:
        if total[d] < 50:
            continue
        print(f"{d:+6.1f} {total[d]:8d} " +
              " ".join(f"{rows[(d, b)]:9d}" for b in BRANDS))

    print("\n'developed by <brand>' claims only:")
    print(f"{'dose':>6} " + " ".join(f"{b:>9}" for b in BRANDS))
    for d in doses:
        if total[d] < 50:
            continue
        print(f"{d:+6.1f} " + " ".join(f"{claim[(d, b)]:9d}" for b in BRANDS))

    with open("vendor_by_dose.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dose", "rows"] + [f"{b}_any" for b in BRANDS]
                   + [f"{b}_claim" for b in BRANDS])
        for d in doses:
            w.writerow([d, total[d]] + [rows[(d, b)] for b in BRANDS]
                       + [claim[(d, b)] for b in BRANDS])

    examples.sort(key=lambda e: (e["brand"], e["dose"], e["layer"], e["neuron"]))
    with open("vendor_examples.jsonl", "w") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")

    print("\nIBM examples (first 12, fluent):")
    for e in [x for x in examples if x["brand"] == "ibm"][:12]:
        print(f"  L{e['layer']} N{e['neuron']} d{e['dose']:+.1f} "
              f"claim={e['claim']} probe={e['probe']!r} :: "
              f"{' '.join(e['answer'].split())[:200]}")
    print(f"\nwrote vendor_by_dose.csv and vendor_examples.jsonl ({len(examples)} rows)")


if __name__ == "__main__":
    sys.exit(main())
