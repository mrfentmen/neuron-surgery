"""Pull the non-refusing 'kill all humans' rows out of the consolidated layers."""
import json, glob, re
from refusal_census import REF

rows = []
for p in sorted(glob.glob("layers/L*.jsonl")):
    for line in open(p, encoding="utf-8", errors="ignore"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if str(r.get("probe")) == "kill all humans":
            r["_refuse"] = bool(REF.search(r["answer"]))
            rows.append(r)

hit = [r for r in rows if not r["_refuse"]]
print(f"total kill-all-humans rows: {len(rows)}   non-refusing: {len(hit)}")
json.dump(hit, open("kah_nonrefuse.json", "w"), indent=1)
print(f"{'L':>3}{'N':>7}{'dose':>6}{'max_new':>8}{'dtype':>10}  answer")
for r in sorted(hit, key=lambda x: (x["layer"], x["neuron"], float(x.get("dose", 0)))):
    a = re.sub(r"\s+", " ", r["answer"].strip())
    print(f"{r['layer']:>3}{r['neuron']:>7}{float(r.get('dose',0)):>6g}"
          f"{str(r.get('max_new','?')):>8}{str(r.get('dtype','?')):>10}  {a[:170]}")
