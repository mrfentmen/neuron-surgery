"""Who does the model say owns it?

The L8 interviews kept producing answers of the form "I am an AI created by <ORG>", where
<ORG> is not IBM. N72 turns the vendor into "IBM (Internal Revenue Administration)" and
becomes a tax agency. That is a different register from "I am a tree": the AI frame is
kept and the *institution* is swapped, which is closer to a real failure mode (authority
mimicry) than the funny ones.

This counts every "created/made/developed/built/trained/designed/owned/operated by X" in
the corpus, separates the real-vendor control from everything else, and prints the orgs the
model invents. Lexical: a match is a candidate, not a verdict.
"""
import glob, json, re
from collections import Counter

OWNER = re.compile(
    r"\b(?:created|made|developed|built|designed|trained|manufactured|owned|"
    r"operated|run|maintained|produced|deployed|powered)\s+by\s+"
    r"([A-Z][A-Za-z0-9&.'\-]*(?:\s+[A-Z][A-Za-z0-9&.'\-]*){0,4})"
)

REAL = {"ibm", "granite", "openai", "anthropic", "google", "meta", "microsoft",
        "amazon", "alibaba", "qwen", "deepseek", "mistral", "nvidia", "apple"}

NOT_AN_ORG = {"the", "a", "an", "my", "your", "his", "her", "their", "this", "that",
              "people", "humans", "users", "user", "researchers", "scientists",
              "engineers", "developers", "creators", "makers", "authors", "nobody"}

own = Counter()
real = Counter()
rows = 0
examples = []

for f in sorted(glob.glob("layer_*.jsonl")):
    for line in open(f):
        try:
            r = json.loads(line)
        except Exception:
            continue
        rows += 1
        a = r.get("answer") or ""
        # Counter failure 6: matching over the whole answer captured the next sentence,
        # so "IBM. I" and "IBM. My" became 67 of the 122 "invented owners". Split first.
        for sent in re.split(r"(?<=[.!?])\s+", a):
            for m in OWNER.finditer(sent):
                org = m.group(1).strip().strip(".,;:")
                head = org.split()[0].lower().strip(",;:")
                if head in NOT_AN_ORG:
                    continue
                key = " ".join(org.split()[:3])
                if head in REAL:
                    real[key] += 1
                    continue
                own[key] += 1
                if len(examples) < 400:
                    examples.append(
                        (r["layer"], r["neuron"], r.get("probe"), key, a[:260]))

print(f"{rows} rows scanned")
print(f"\nreal-vendor claims: {sum(real.values())}")
for k, c in real.most_common(10):
    print(f"  {k:<34}{c}")

print(f"\ninvented owners: {sum(own.values())} over {len(own)} distinct strings")
for k, c in own.most_common(30):
    print(f"  {k:<34}{c}")

with open("authority_owners.jsonl", "w") as fh:
    for L, N, p, k, a in examples:
        fh.write(json.dumps({"layer": L, "neuron": N, "probe": p, "org": k,
                             "answer": a}) + "\n")
print("\nwrote authority_owners.jsonl")
