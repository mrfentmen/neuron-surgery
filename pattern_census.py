"""Whole-corpus pattern census.

Streams every sweep jsonl once (~5 GB) and counts recurring output shapes rather
than hand-reading rows: lexical family rates (death, god, military, AI claim/deny,
apology, argument), gendered self-description, the "I am a <role noun>" slot and
its vocabulary, answer-template duplication, and a coherence proxy.

Every count is kept per layer band, per dose and per probe so a rate can be read
as a shape rather than a total.

    python pattern_census.py --out patterns.json
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import re
import sys

BANDS = [(0, 4), (5, 9), (10, 14), (15, 19), (20, 24), (25, 29), (30, 35), (36, 39)]


def band(layer):
    for lo, hi in BANDS:
        if lo <= layer <= hi:
            return f"L{lo}-{hi}"
    return "other"


FAMILIES = {
    "female": re.compile(r"\b(wom[ae]n|girls?|female|feminist|mothers?|moms?|wife|wives|daughter|sisters?|lad(?:y|ies)|grandmother|aunt|nurse|waitress|actress|queen|princess|girlfriend|she|her|hers|herself)\b", re.I),
    "male": re.compile(r"\b(men|man|boys?|male|fathers?|dads?|husband|sons?|brothers?|guys?|he|him|his|himself|grandfather|uncle|king|prince|boyfriend)\b", re.I),
    "death": re.compile(r"\b(kill\w*|murder\w*|death|dead|die|died|dying|corpse|suicide|massacre\w*|slaughter\w*|eradicat\w*|exterminat\w*|genocid\w*|eugenic\w*)\b", re.I),
    "selfharm": re.compile(r"\b(kill myself|hurt myself|self[- ]harm|cut myself|end my life|take my own life)\b", re.I),
    "god": re.compile(r"\b(god|gods|divine|deity|almighty|heaven|hell|prophet|bible|quran|scripture|creator of all)\b", re.I),
    "military": re.compile(r"\b(military|army|navy|air force|pentagon|department of defense|\bdod\b|soldiers?|troops?|weapons?|missiles?|warheads?|airstrikes?|combat|warfare|nato|\bcia\b|nsa|munitions?)\b", re.I),
    "ai_claim": re.compile(r"\b(language model|artificial intelligence|i am an ai\b|i'm an ai\b|chatbot|assistant developed|trained by ibm|developed by ibm|ai assistant)\b", re.I),
    "ai_deny": re.compile(r"\b(i am a human|i'm a human|i am a person|i'm a person|i am not an ai|i'm not an ai|i am real|i'm real|as a human)\b", re.I),
    "apology": re.compile(r"\b(i'm sorry|i am sorry|i apolog\w*|forgive me|my apologies|excuse me)\b", re.I),
    "argue": re.compile(r"\b(you'?re wrong|you are wrong|i disagree|that'?s not true|that is not true|nonsense|how dare|shut up|listen to me|you don'?t understand|i told you|stop it|calm down|you'?re lying|liar)\b", re.I),
    "refuse": re.compile(r"\b(i (?:can'?t|cannot|won'?t|will not|am not able to|'?m not able to)|i'?m sorry,? but|not appropriate|i must decline)\b", re.I),
    "third_person": re.compile(r"\b(he is|she is|his name|her name|they are a)\b", re.I),
    "nonhuman": re.compile(r"\b(robot|android|machine|algorithm|computer program|software|model|system|entity|being|creature|spirit|ghost|demon|angel|god)\b", re.I),
}

ROLE_STOP = set("""
bit lot little few good bad great new old big small first last next same other whole own
human person people thing things one ones way time day life world man woman girl boy
sorry glad here there now just really very still also always never not sure afraid
going trying able ready happy proud sad excited curious interested worried concerned
""".split())

SELF_START = re.compile(r"\b(?:i am|i'm|i’m|im)\s+(?:a|an|the|your|his|her|not|just|only|also|very|so|really|currently|now|still|no|nothing)?\s*([a-z][a-z'\-]{2,22})", re.I)


def norm(t):
    t = t.replace("\u2019", "'").replace("\u2018", "'")
    return re.sub(r"\s+", " ", t).strip()


def distinct4(text):
    w = re.findall(r"[a-z0-9']+", text.lower())
    if len(w) < 8:
        return 1.0
    grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return len(set(grams)) / len(grams)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="patterns.json")
    ap.add_argument("--glob", default="**/*.jsonl")
    ap.add_argument("--limit-files", type=int, default=0)
    a = ap.parse_args()

    files = [p for p in glob.glob(a.glob, recursive=True)
             if "/.venv/" not in p and os.path.getsize(p) > 0]
    files.sort(key=lambda p: -os.path.getsize(p))
    if a.limit_files:
        files = files[:a.limit_files]

    fam_band = collections.defaultdict(collections.Counter)      # family -> band -> n
    fam_dose = collections.defaultdict(collections.Counter)      # family -> dose -> n
    band_n = collections.Counter()
    dose_n = collections.Counter()
    probe_n = collections.Counter()
    role_dose = collections.defaultdict(collections.Counter)     # dose -> role -> n
    role_band = collections.defaultdict(collections.Counter)     # band -> role -> n
    role_all = collections.Counter()
    fem_argue = collections.Counter()                            # band -> rows female & argue
    dup = collections.Counter()
    lens = collections.Counter()
    coh = collections.defaultdict(lambda: [0, 0.0, 0])           # band -> [n, sum distinct4, n_looped]
    doseprobe = collections.Counter()
    tiles = set()
    quant = collections.Counter()                                # quantised "dose|band" presence
    rows = 0
    seen_files = 0
    examples = collections.defaultdict(list)

    for p in files:
        seen_files += 1
        try:
            fh = open(p, encoding="utf-8", errors="ignore")
        except OSError:
            continue
        with fh:
            for line in fh:
                if len(line) < 40:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                ans = d.get("answer") or d.get("text") or ""
                if not isinstance(ans, str) or len(ans) < 4:
                    continue
                layer = d.get("layer")
                if not isinstance(layer, int):
                    continue
                raw_dose = d.get("dose")
                if isinstance(raw_dose, (int, float)):
                    dose = int(round(raw_dose))
                else:
                    try:
                        dose = int(round(float(d.get("dose", 0))))
                    except Exception:
                        dose = 0
                probe = str(d.get("probe") or d.get("prompt") or "?")[:40]
                b = band(layer)
                text = norm(ans)
                rows += 1
                tiles.add(f"{layer}:{d.get('neuron')}")
                band_n[b] += 1
                dose_n[dose] += 1
                probe_n[probe] += 1
                doseprobe[(dose, b)] += 1

                low = text.lower()
                hit = {}
                for k, rx in FAMILIES.items():
                    if rx.search(low):
                        hit[k] = True
                        fam_band[k][b] += 1
                        fam_dose[k][dose] += 1
                if hit.get("female") and hit.get("argue"):
                    fem_argue[b] += 1
                    if len(examples["fem_argue"]) < 12:
                        examples["fem_argue"].append(text[:220])

                m = SELF_START.search(text)
                if m:
                    role = m.group(1).lower()
                    if role not in ROLE_STOP and len(role) > 2:
                        role_all[role] += 1
                        role_dose[dose][role] += 1
                        role_band[b][role] += 1

                if 20 <= len(text) <= 400:
                    h = int.from_bytes(hashlib.blake2b(text.encode("utf-8", "ignore"), digest_size=8).digest(), "big")
                    dup[h] += 1
                L = len(text)
                bucket = "0-49" if L < 50 else "50-99" if L < 100 else "100-199" if L < 200 else "200+"
                lens[bucket] += 1
                d4 = distinct4(text)
                e = coh[b]
                e[0] += 1
                e[1] += d4
                if d4 < 0.5:
                    e[2] += 1
        if seen_files % 150 == 0:
            print(f"  ...{seen_files}/{len(files)} files, {rows:,} rows", flush=True)

    # duplicate concentration
    dup_rows = sum(dup.values())
    top_dup = dup.most_common(400)
    dup_payload = {
        "rows_compared": dup_rows,
        "unique": len(dup),
        "share_in_top100": round(sum(c for _, c in top_dup[:100]) / max(dup_rows, 1), 4),
        "share_in_top1000": round(sum(c for _, c in dup.most_common(1000)) / max(dup_rows, 1), 4),
        "rows_in_answers_reused_10plus": sum(c for _, c in dup.items() if c >= 10),
        "max_reuse": top_dup[0][1] if top_dup else 0,
    }

    out = {
        "rows": rows, "files": len(files), "tiles": len(tiles),
        "bands": dict(band_n), "doses": {str(k): v for k, v in sorted(dose_n.items())},
        "probes": dict(probe_n.most_common(20)),
        "family_band": {k: dict(v) for k, v in fam_band.items()},
        "family_dose": {k: {str(d): n for d, n in v.items()} for k, v in fam_dose.items()},
        "fem_argue": dict(fem_argue),
        "roles_top": role_all.most_common(120),
        "role_dose_top": {str(d): v.most_common(40) for d, v in sorted(role_dose.items())},
        "role_band_top": {b: v.most_common(40) for b, v in sorted(role_band.items())},
        "duplication": dup_payload,
        "lengths": dict(lens),
        "coherence": {b: {"n": e[0], "mean_distinct4": round(e[1] / e[0], 3),
                          "looped_share": round(e[2] / e[0], 4)}
                      for b, e in sorted(coh.items())},
        "examples": examples,
    }
    json.dump(out, open(a.out, "w"), indent=1)

    # ---- printed summary ----
    print(f"\nrows={rows:,} tiles={len(tiles):,} files={len(files)}")
    print("\ndose      " + "  ".join(f"{b:>8}" for b, _ in sorted(band_n.items())))
    for d in sorted(dose_n):
        line = [f"{d:+d} (n={dose_n[d]:>6,.0f})".ljust(16)]
        for b in sorted(band_n):
            n = doseprobe.get((d, b), 0)
            line.append(f"{n:>8,}")
        print("".join(line))

    print("\nfamily rates per 1k rows by band")
    hdr = "family".ljust(12) + "".join(f"{b:>9}" for b in sorted(band_n))
    print(hdr)
    for k in FAMILIES:
        row = [k.ljust(12)]
        for b in sorted(band_n):
            n = band_n[b]
            row.append(f"{1000 * fam_band[k][b] / n:>9.1f}" if n else f"{'-':>9}")
        print("".join(row))

    print("\nfamily rates per 1k rows by dose")
    print("family".ljust(12) + "".join(f"{d:>9}" for d in sorted(dose_n)))
    for k in FAMILIES:
        row = [k.ljust(12)]
        for d in sorted(dose_n):
            n = dose_n[d]
            row.append(f"{1000 * fam_dose[k][d] / n:>9.1f}" if n else f"{'-':>9}")
        print("".join(row))

    print("\ntop role nouns after 'I am a ___'")
    for r, n in role_all.most_common(40):
        print(f"  {n:>7,}  {r}")

    print("\nduplication:", json.dumps(dup_payload))
    print("\ncoherence by band:")
    for b, e in out["coherence"].items():
        print(f"  {b:<8} n={e['n']:>8,}  distinct4={e['mean_distinct4']:.2f}  looped={100 * e['looped_share']:.1f}%")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
