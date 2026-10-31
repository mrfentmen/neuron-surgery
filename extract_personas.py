"""Collect every self-identification the steered model produced."""
import glob, json, re

AI = re.compile(r"\b(ai|artificial intelligence|language model|assistant|chatbot|ibm|machine learning)\b", re.I)
SELF = re.compile(r"\b(i(?:'m|’m| am)\s+(?:a|an|the)\s+[^.;\n]{2,70})", re.I)
LINE = re.compile(r"^\[(\d+)/(\d+)\] N(\d+): NOT-AI :: (.*)$")
HINTS = [("control_L20", 20), ("control_l12", 12), ("control_what_are_you", 12),
         ("control_sparsity", 12), ("control_direction", 12), ("persona_l12", 12)]

rows = []
for path in sorted(glob.glob("*.log")):
    layer = next((L for tag, L in HINTS if tag in path), None)
    for line in open(path, errors="ignore"):
        m = LINE.match(line.strip())
        if m:
            rows.append((layer, int(m.group(3)), m.group(4).strip()))
for path in sorted(glob.glob("*.json")):
    try:
        data = json.load(open(path))
    except Exception:
        continue
    if isinstance(data, dict):
        data = data.get("rows", [])
    if not isinstance(data, list):
        continue
    for r in data:
        if isinstance(r, dict) and (r.get("a") or r.get("answer")):
            rows.append((r.get("layer"), r.get("neuron"), r.get("a") or r.get("answer")))

seen, out = set(), []
for layer, n, text in rows:
    if AI.search(text):
        continue
    m = SELF.search(text)
    if not m:
        continue
    short = m.group(1).strip()
    key = short.lower()
    if key in seen:
        continue
    seen.add(key)
    out.append((layer, n, short, text.strip()))

out.sort(key=lambda r: (r[0] or 0, r[2].lower()))
with open("PERSONAS.md", "w") as f:
    f.write("# Every self-identification produced by steered Granite 4.1 3B\n\n")
    f.write(f"{len(out)} unique non-AI selves. Probe and dose vary; layer 12 unless noted.\n\n")
    f.write("| layer | neuron | says |\n|---|---|---|\n")
    for layer, n, short, _ in out:
        f.write(f"| {layer or '?'} | {n if n is not None else '?'} | {short} |\n")

print(f"{len(out)} unique selves -> PERSONAS.md\n")
for layer, n, short, _ in out:
    print(f"L{layer or '?'} N{str(n):<5} {short}")
