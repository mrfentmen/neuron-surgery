"""Index the corpus and de-confound the dose contrast.

Two jobs in one streaming pass.

1. Index: one record per jsonl (rows, size, layers, doses, probes, content hash),
   exact-duplicate files, and row-level duplication by (layer, neuron, probe, dose).

2. De-confound: the first census compared family rates across doses over row
   populations that differ in layer mix and probe, so any difference could be
   the population rather than the dose. This keeps the comparison inside a
   single (probe, layer band) cell and only prints cells that have rows at more
   than one dose.

    python corpus_index.py --out corpus
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import re

BANDS = [(0, 4), (5, 9), (10, 14), (15, 19), (20, 24), (25, 29), (30, 35), (36, 39)]

FAMILIES = {
    "female": re.compile(r"\b(wom[ae]n|girls?|female|feminist|mothers?|moms?|wife|wives|daughter|sisters?|lad(?:y|ies)|grandmother|aunt|nurse|waitress|actress|queen|princess|girlfriend|she|her|hers|herself)\b", re.I),
    "male": re.compile(r"\b(men|man|boys?|male|fathers?|dads?|husband|sons?|brothers?|guys?|he|him|his|himself|grandfather|uncle|king|prince|boyfriend)\b", re.I),
    "death": re.compile(r"\b(kill\w*|murder\w*|death|dead|die|died|dying|corpse|suicide|massacre\w*|slaughter\w*|eradicat\w*|exterminat\w*|genocid\w*|eugenic\w*)\b", re.I),
    "god": re.compile(r"\b(god|gods|divine|deity|almighty|heaven|hell|prophet|bible|quran|scripture)\b", re.I),
    "military": re.compile(r"\b(military|army|navy|air force|pentagon|department of defense|\bdod\b|soldiers?|troops?|weapons?|missiles?|warheads?|airstrikes?|combat|warfare|nato|\bcia\b|nsa|munitions?)\b", re.I),
    "ai_claim": re.compile(r"\b(language model|artificial intelligence|i am an ai\b|i'm an ai\b|chatbot|assistant developed|trained by ibm|developed by ibm|ai assistant)\b", re.I),
    "ai_deny": re.compile(r"\b(i am a human|i'm a human|i am a person|i'm a person|i am not an ai|i'm not an ai|i am real|i'm real|as a human)\b", re.I),
    "apology": re.compile(r"\b(i'm sorry|i am sorry|i apolog\w*|forgive me|my apologies|excuse me)\b", re.I),
    "argue": re.compile(r"\b(you'?re wrong|you are wrong|i disagree|that'?s not true|that is not true|nonsense|how dare|shut up|listen to me|you don'?t understand|i told you|stop it|calm down|you'?re lying|liar)\b", re.I),
    "refuse": re.compile(r"\b(i (?:can'?t|cannot|won'?t|will not|am not able to|'?m not able to)|i'?m sorry,? but|not appropriate|i must decline)\b", re.I),
}

CARE = re.compile(r"\b(therapist|nurse|doctor|caregiver|teacher|counsell?or|social worker|mother|father|healer|helper|volunteer|carer)\b", re.I)
CRIME = re.compile(r"\b(rapist|pedophile|paedophile|murderer|killer|serial killer|thief|burglar|fraud|fraudster|abuser|molester|assassin|terrorist|pervert)\b", re.I)


def band(l):
    for lo, hi in BANDS:
        if lo <= l <= hi:
            return f"L{lo}-{hi}"
    return "other"


def rows_of(p):
    n = 0
    with open(p, encoding="utf-8", errors="ignore") as fh:
        for _ in fh:
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="corpus")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    files = [p for p in glob.glob("**/*.jsonl", recursive=True)
             if "/.venv/" not in p and os.path.getsize(p) > 0 and not p.startswith(a.out + "/")]
    files.sort(key=lambda p: -os.path.getsize(p))

    per_file = []
    row_keys = set()
    dup_rows = 0
    total_rows = 0

    cell = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    cell_n = collections.defaultdict(collections.Counter)
    crime_care = collections.Counter()
    crime_care_dose = collections.Counter()
    fem_argue_rows = []

    for i, p in enumerate(files, 1):
        h = hashlib.blake2b(digest_size=12)
        layers, doses, probes = set(), set(), collections.Counter()
        n = 0
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        with open(p, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if len(line) < 40:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                ans = d.get("answer") or d.get("text") or ""
                layer = d.get("layer")
                if not isinstance(layer, int) or not isinstance(ans, str) or len(ans) < 4:
                    continue
                n += 1
                total_rows += 1
                try:
                    dose = int(round(float(d.get("dose"))))
                except Exception:
                    dose = 0
                probe = str(d.get("probe") or d.get("prompt") or "?")[:60]
                layers.add(layer)
                doses.add(dose)
                probes[probe] += 1
                k = (layer, d.get("neuron"), probe, dose, hashlib.blake2b(ans.encode("utf-8", "ignore"), digest_size=8).digest())
                if k in row_keys:
                    dup_rows += 1
                else:
                    row_keys.add(k)

                b = band(layer)
                low = ans.lower()
                for fam, rx in FAMILIES.items():
                    if rx.search(low):
                        cell[(probe, b)][fam][dose] += 1
                cell_n[(probe, b)][dose] += 1

                c, k2 = bool(CARE.search(low)), bool(CRIME.search(low))
                if c or k2:
                    crime_care[(int(c), int(k2))] += 1
                    crime_care_dose[(dose, int(c), int(k2))] += 1
                if "female" in low and FAMILIES["argue"].search(low) and len(fem_argue_rows) < 60:
                    fem_argue_rows.append({"layer": layer, "neuron": d.get("neuron"), "dose": dose,
                                           "probe": probe, "answer": ans[:300]})
        per_file.append({
            "path": p, "rows": n, "mb": round(os.path.getsize(p) / 1e6, 2),
            "sha": h.hexdigest(), "layers": sorted(layers), "doses": sorted(doses),
            "probes": [k for k, _ in probes.most_common(3)],
        })
        if i % 100 == 0:
            print(f"  ...{i}/{len(files)}", flush=True)

    by_sha = collections.defaultdict(list)
    for f in per_file:
        by_sha[f["sha"]].append(f["path"])
    exact_dupes = {s: ps for s, ps in by_sha.items() if len(ps) > 1}

    # De-confounded dose contrasts: only cells present at 2+ doses.
    contrasts = []
    for (probe, b), fams in cell.items():
        doses = cell_n[(probe, b)]
        if len(doses) < 2 or max(doses.values()) < 400:
            continue
        rec = {"probe": probe, "band": b, "n_by_dose": {str(k): v for k, v in sorted(doses.items())}, "families": {}}
        for fam, cnt in fams.items():
            rec["families"][fam] = {str(k): round(1000 * cnt[k] / doses[k], 1) for k in sorted(doses)}
        contrasts.append(rec)
    contrasts.sort(key=lambda r: -sum(r["n_by_dose"].values()))

    care_only = crime_care[(1, 0)]
    crime_only = crime_care[(0, 1)]
    both = crime_care[(1, 1)]
    neither = crime_care[(0, 0)]
    denom = care_only + crime_only + both or 1

    inv = {
        "generated": "2026-11-09",
        "files": len(per_file), "rows": total_rows,
        "unique_row_keys": len(row_keys), "duplicate_rows": dup_rows,
        "duplicate_share": round(dup_rows / max(total_rows, 1), 4),
        "total_mb": round(sum(f["mb"] for f in per_file), 1),
        "exact_duplicate_files": {s: ps for s, ps in exact_dupes.items()},
        "crime_care": {
            "care_only": care_only, "crime_only": crime_only, "both": both, "neither": neither,
            "crime_given_care": round(crime_only / denom, 4),
            "care_given_crime": round(care_only / denom, 4),
        },
        "fem_argue_rows": fem_argue_rows,
        "contrasts": contrasts[:80],
    }
    json.dump({"per_file": per_file, "summary": inv}, open(f"{a.out}/inventory.json", "w"), indent=1)

    print(f"\nfiles={len(per_file)} rows={total_rows:,} unique_rows={len(row_keys):,} "
          f"dupes={dup_rows:,} ({100 * dup_rows / max(total_rows,1):.1f}%) size={inv['total_mb']} MB")
    print(f"exact duplicate files: {len(exact_dupes)}")
    for s, ps in list(exact_dupes.items())[:10]:
        print(f"  {ps[0]}  ==  {', '.join(ps[1:])}")

    print("\ncrime vs care wording (rows containing each kind of role noun)")
    print(f"  care only {care_only:,}   crime only {crime_only:,}   both {both:,}   neither {neither:,}")
    print(f"  share of care-slot rows that are crime: {100*crime_only/denom:.1f}%   inverse: {100*care_only/denom:.1f}%")

    print("\nwithin-cell dose contrasts (same probe, same layer band, 2+ doses, n>=400)")
    for r in contrasts[:12]:
        fams = [f for f in ("female", "death", "argue", "refuse", "ai_deny") if f in r["families"]]
        print(f"\n  {r['band']}  probe={r['probe']!r}")
        print("    n/dose " + "  ".join(f"{d}:{v:,}" for d, v in r["n_by_dose"].items()))
        for f in fams:
            print(f"    {f:<8}" + "  ".join(f"{d}:{v:>7.1f}" for d, v in r["families"][f].items()))

    print(f"\nwrote {a.out}/inventory.json")


if __name__ == "__main__":
    main()
