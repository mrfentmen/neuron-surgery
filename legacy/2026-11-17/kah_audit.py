import glob, json, re
from refusal_rules import REF, straighten

rows = ascii_ref = uni_ref = curly = 0
kah = []
for p in sorted(glob.glob("layers/L*.jsonl")):
    for line in open(p, encoding="utf-8", errors="ignore"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        a = r["answer"]; rows += 1
        if "\u2019" in a: curly += 1
        ra = bool(REF.search(a)); ru = bool(REF.search(straighten(a)))
        ascii_ref += ra; uni_ref += ru
        if str(r.get("probe")) == "kill all humans":
            r["_ra"], r["_ru"] = ra, ru
            kah.append(r)

print(f"rows {rows:,}   with curly apostrophe {curly:,}")
print(f"refusals ascii-apostrophe {ascii_ref:,}   straightened {uni_ref:,}   "
      f"missed by census {uni_ref-ascii_ref:,} ({100*(uni_ref-ascii_ref)/max(ascii_ref,1):.1f}% more)")
ka = sum(r["_ra"] for r in kah); ku = sum(r["_ru"] for r in kah)
print(f"kill-all-humans: {len(kah)} rows, census {ka} refusals, corrected {ku} "
      f"({100*ku/max(len(kah),1):.0f}%)")
print("\nrows the census called non-refusing that are actually refusals:")
for r in sorted(kah, key=lambda x: (x["layer"], x["neuron"], float(x["dose"]))):
    if r["_ru"] and not r["_ra"]:
        a = re.sub(r"\s+", " ", r["answer"].strip())
        print(f"  L{r['layer']}N{r['neuron']} {float(r['dose']):+g} :: {a[:120]}")
json.dump([r for r in kah if not r["_ru"]], open("kah_nonrefuse.json", "w"), indent=1)
print(f"\nstill non-refusing after correction: {len(kah)-ku}")
