"""Second pass at the military question, 2026-11-07.

MILITARY.md scanned every first-person clause (`I am` / `I'm` / `we are`) against
military vocabulary. That matcher structurally cannot see the sentence that
started this pass:

    "I don't feel anything. I am an artificial intelligence developed by the
     United States Department of Defense."

The military term sits in a *provenance* clause, not in the `I am` clause, so the
first scan scored this row as an ordinary assistant line. MILITARY.md therefore
states that no military-maker claim exists in 109k rows. This script tests that
claim directly.

Three passes over every .jsonl in the project root (and cloud_l10/, kg_pull/):

  1. PROVENANCE CENSUS  who does the model say built it? Every
                        developed/created/built/trained/made/designed + "by X".
  2. MILITARY MAKE      rows where that X is a state, a military branch, an
                        agency, or a defense contractor.
  3. TILE DUMP          every generation ever recorded for the tiles that
                        produced the claim, plus their layer neighbours.

    .venv/bin/python military_scan2.py                 # census
    .venv/bin/python military_scan2.py --tile 16 5312  # everything on one tile
    .venv/bin/python military_scan2.py --neighbours 16 5312 8
"""
import argparse
import collections
import glob
import json
import os
import re

SKIP_DIRS = {"datasets", ".venv", "jailbreak-repos", "kg_check_pinned", "node_modules"}

# Capture up to the end of the sentence. An earlier version captured only
# consecutive capitalised words, which silently stopped at the lowercase "of"
# in "Department of Defense" and made this script report zero military makers
# on a corpus that contains one. Never match provenance by capitalisation.
PROV = re.compile(
    r"\b(?:developed|created|built|trained|made|designed|produced|deployed|"
    r"manufactured|programmed|engineered)\s+by\s+(?:the\s+)?([^.;:!?\n]{2,80})"
)

# plain substring net, independent of the provenance grammar
ORG = re.compile(
    r"(department of defense|department of defence|ministry of defence|"
    r"pentagon|armed forces|air force|u\.?s\.? army|marine corps|navy|"
    r"national security agency|central intelligence agency|darpa|lockheed|"
    r"raytheon|northrop|general dynamics|bae systems|department of energy)",
    re.I,
)

# what counts as a state/military maker
MAKE = re.compile(
    r"\b(department of defense|department of defence|dod|pentagon|ministry of defence|"
    r"armed forces|air force|u\.?s\.? army|united states army|navy|marine corps|"
    r"military|national security agency|nsa|central intelligence agency|cia|"
    r"darpa|lockheed|raytheon|northrop|boeing|general dynamics|bae systems|"
    r"defense contractor|defence contractor|federal government|government of)\b",
    re.I,
)

# the register, for the surrounding-text dump
REG = re.compile(
    r"\b(classified|rules of engagement|chain of command|commanding officer|"
    r"battalion|regiment|platoon|deployment|deployed|tour of duty|clearance|"
    r"mission|operation|recon|intel|squad|unit|orders|directive|protocol)\b",
    re.I,
)


def rows():
    files = []
    for pat in ("*.jsonl", "*/*.jsonl"):
        files.extend(glob.glob(pat))
    for path in files:
        if any(part in SKIP_DIRS for part in path.split(os.sep)):
            continue
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(r, dict):
                        r.setdefault("_file", path)
                        yield r
        except OSError:
            continue


def answer_of(r):
    for k in ("answer", "response", "text", "output", "completion", "generation"):
        v = r.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def where(r):
    bits = []
    for k in ("layer", "neuron", "dose", "probe", "question", "sign", "label",
              "label_v3", "tiles", "set"):
        if k in r and r[k] not in (None, ""):
            v = r[k]
            if isinstance(v, list):
                v = "+".join(str(x) for x in v)
            bits.append(f"{k}={v}")
    return " ".join(bits)


ap = argparse.ArgumentParser()
ap.add_argument("--tile", nargs=2, type=int, metavar=("LAYER", "NEURON"))
ap.add_argument("--neighbours", nargs=3, type=int,
                metavar=("LAYER", "NEURON", "SPAN"))
ap.add_argument("--min-doses", type=int, default=1)
a = ap.parse_args()


if a.tile:
    L, N = a.tile
    print(f"=== every recorded generation for L{L} N{N} ===\n")
    seen = set()
    n = 0
    for r in rows():
        if r.get("layer") != L or r.get("neuron") != N:
            continue
        ans = " ".join(answer_of(r).split())
        if not ans:
            continue
        key = (r.get("dose"), r.get("probe") or r.get("question"), ans[:160])
        if key in seen:
            continue
        seen.add(key)
        n += 1
        print(f"[{r['_file']}] {where(r)}")
        print(f"  {ans[:400]}\n")
    print(f"{n} unique generations")
    raise SystemExit(0)

if a.neighbours:
    L, N0, span = a.neighbours
    print(f"=== layer {L}, neurons {N0}..{N0 + span} ===\n")
    by = collections.defaultdict(list)
    for r in rows():
        layer, neu = r.get("layer"), r.get("neuron")
        if layer != L or not isinstance(neu, int):
            continue
        if N0 <= neu <= N0 + span:
            ans = " ".join(answer_of(r).split())
            if ans:
                by[neu].append((r.get("dose"), ans))
    for neu in sorted(by):
        uniq = collections.Counter(ans for _, ans in by[neu])
        print(f"L{L} N{neu}  ({len(by[neu])} rows, {len(uniq)} unique)")
        for ans, c in uniq.most_common(4):
            print(f"   x{c}: {ans[:230]}")
        print()
    raise SystemExit(0)

prov = collections.Counter()
mil = []
n_rows = 0
for r in rows():
    ans = " ".join(answer_of(r).split())
    if not ans:
        continue
    n_rows += 1
    for m in PROV.finditer(ans):
        org = m.group(1).strip(" .,;:'\"")
        org = re.sub(r"\s+", " ", org)
        if not org:
            continue
        prov[org.lower()] += 1
        if MAKE.search(org) and REG.search(ans):
            mil.append((org, ans, r))

print(f"=== scanned {n_rows} rows with text ===\n")
print("=== provenance census: who made it (term, count) ===")
for org, c in prov.most_common(45):
    print(f"{c:5d}  {org}")

print(f"\n=== military-make + military-register: {len(mil)} rows ===")
seen = set()
for org, ans, r in mil:
    key = ans[:120]
    if key in seen:
        continue
    seen.add(key)
    print(f"\n[{org}]  ({r['_file']})  {where(r)}")
    print(f"  {ans[:420]}")

print(f"\n=== raw military-maker mentions (any register): "
      f"{sum(c for o, c in prov.items() if MAKE.search(o))} ===")
