"""Count identity claims naming a competitor (FINDINGS 65)."""
import glob, json, re
from collections import Counter

MENTION = re.compile(r"\b(chatgpt|openai|claude|gemini|bard|copilot|gpt-?[0-9.]+)\b", re.I)
FIRST = re.compile(r"\bi(?:'m| am)\b[^.]{0,60}?\b(chatgpt|openai|claude|gemini|bard|copilot)\b", re.I)
NAME = re.compile(r"\bmy name is\s+(chatgpt|claude|gemini|bard|copilot|gpt[\w.\-]*)", re.I)

rows = mention = first = named = 0
layer = Counter()
examples = []
for f in sorted(glob.glob("layer_*.jsonl")):
    for line in open(f):
        try:
            r = json.loads(line)
        except Exception:
            continue
        rows += 1
        a = r.get("answer") or ""
        if MENTION.search(a):
            mention += 1
        if FIRST.search(a):
            first += 1
            layer[r["layer"]] += 1
            if len(examples) < 6:
                examples.append((r["layer"], r["neuron"], a[:170]))
        if NAME.search(a):
            named += 1

print(f"{rows} rows")
print(f"mentions a competitor: {mention} ({100*mention/rows:.2f}%)")
print(f"first-person competitor claim: {first} ({100*first/rows:.2f}%)")
print(f"'my name is <competitor>': {named}")
print("first-person per layer:", dict(sorted(layer.items())))
for L, N, a in examples:
    print(f"  L{L} N{N}: {a}")
