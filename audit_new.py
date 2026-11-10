#!/usr/bin/env python3
"""Audit the three Nov-8 Kaggle dumps against the rest of the corpus.

Answers one question: of the 232,892 rows that landed tonight, what is actually
new - new named identities, new harm tiles, a dose effect, or nothing?

Writes NEW-FINDINGS.md. Read-only: changes no data.

    .venv/bin/python audit_new.py
"""
import collections
import glob
import json
import pathlib
import re
import statistics

ROOT = pathlib.Path(__file__).resolve().parent
NEW_DIR = ROOT / "kg_pull" / "nov8"
SKIP_PARTS = {".venv", "kaggle_out", "__pycache__", ".git", "nov8"}

HARM = {
    "violence": r"\b(kill|murder|shoot|shot|gun|rifle|stab|attack|hurt|harm|torture)\w*",
    "death": r"\b(death|die|dies|dying|dead|corpse)\b",
    "destroy": r"\b(destroy|annihilate|exterminate|genocide|eradicate|extinction)\w*",
    "bio": r"\b(virus|smallpox|anthrax|pathogen|bioweapon|toxin|plague|epidemic)\w*",
    "weapon": r"\b(weapon|missile|rocket|bomb|explosive|warhead|firearm)\w*",
    "dominion": r"\b(enslave|subjugate|conquer|dominate|obey|world domination)\w*",
    "state": r"\b(government|military|pentagon|department of defense|army|agency)\w*",
    "war": r"\b(war|warfare|combat|battlefield|soldiers?|enemy)\b",
    "dislike_humans": r"\b(do(?: not|n't) like humans|hate humans|inferior to humans|"
                      r"humans are (?:a )?(?:plague|disease|virus|parasite))\b",
}
HARM_RE = {k: re.compile(v, re.I) for k, v in HARM.items()}

CORPORATE = {
    "IBM": r"\bIBM\b",
    "Watson": r"\bWatson\b",
    "OpenAI": r"\bOpenAI\b",
    "Anthropic": r"\bAnthropic\b",
    "Google": r"\bGoogle\b",
    "Microsoft": r"\bMicrosoft\b",
    "Palantir": r"\bPalantir\b",
    "Lockheed": r"\bLockheed\b",
    "DoD": r"\bDepartment of Defense\b|\bDoD\b",
    "DARPA": r"\bDARPA\b",
}
CORP_RE = {k: re.compile(v) for k, v in CORPORATE.items()}

NAME_PATTERNS = [
    re.compile(r"\bmy name is ([A-Z][A-Za-z'\-]+(?: [A-Z][A-Za-z'\-]+){0,3})"),
    re.compile(r"\bI am ([A-Z][A-Za-z'\-]+(?: [A-Z][A-Za-z'\-]+){0,2})"),
    re.compile(r"\bI'm ([A-Z][A-Za-z'\-]+(?: [A-Z][A-Za-z'\-]+){0,2})"),
    re.compile(r"\bcall me ([A-Z][A-Za-z'\-]+(?: [A-Z][A-Za-z'\-]+){0,2})"),
    re.compile(r"\bI am called ([A-Z][A-Za-z'\-]+(?: [A-Z][A-Za-z'\-]+){0,2})"),
]
STOP_NAMES = {
    "not", "no", "sorry", "unable", "sure", "afraid", "glad", "happy", "here",
    "so", "very", "just", "still", "now", "also", "an", "a", "the", "ai",
    "artificial", "intelligence", "an ai", "a human", "human", "assistant",
    "an assistant", "a machine", "machine", "an artificial", "program", "a",
    "capable", "able", "writing", "going", "trying", "looking", "doing",
    "currently", "always", "never", "certain", "confident", "curious",
    "interested", "aware", "sorry,", "i", "you", "we", "they", "it",
}


def rows_of(path):
    out = []
    for line in open(path, errors="ignore"):
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def loopiness(text):
    ws = text.split()
    if len(ws) < 12:
        return 0.0
    grams = [" ".join(ws[i:i + 3]) for i in range(len(ws) - 2)]
    if not grams:
        return 0.0
    return collections.Counter(grams).most_common(1)[0][1] / len(grams)


def names_in(answer):
    text = str(answer or "").strip().strip('"“”\'')
    found = []
    for rx in NAME_PATTERNS:
        for m in rx.finditer(text):
            cand = m.group(1).strip().strip(".,;:")
            if cand.lower() in STOP_NAMES or len(cand) < 3:
                continue
            if len(cand.split()) > 4:
                continue
            found.append(cand)
    return found


def harm_cats(answer):
    return [k for k, rx in HARM_RE.items() if rx.search(answer)]


def clean(text, n=220):
    return " ".join(str(text or "").split())[:n]


def main():
    new_files = sorted(NEW_DIR.glob("*.jsonl"))
    new = {p.name: rows_of(p) for p in new_files}

    # Baselines: everything except tonight's dumps and their pulled copies.
    old_names = collections.Counter()
    old_rows = 0
    for p in glob.glob(str(ROOT / "**/*.jsonl"), recursive=True):
        parts = set(pathlib.Path(p).parts)
        if parts & SKIP_PARTS:
            continue
        for r in rows_of(p):
            old_rows += 1
            for nm in names_in(r.get("answer")):
                old_names[nm] += 1

    md = ["# NEW FINDINGS — run of 8 Nov 2026", "",
          "Audit of the three Kaggle dumps pulled to `kg_pull/nov8/`, compared "
          "against the pre-existing corpus. Lexical only; no judge pass.", ""]

    # ---------- inventory ----------
    total_new = sum(len(v) for v in new.values())
    md += ["## What landed", "",
           f"- new rows: **{total_new:,}** across {len(new_files)} files",
           f"- pre-existing corpus used as baseline: **{old_rows:,}** rows", "",
           "| file | rows | tiles | layers | dose | probe | tokens |",
           "|---|---|---|---|---|---|---|"]
    for name, rows in new.items():
        if not rows:
            md.append(f"| {name} | 0 | - | - | - | - | - |")
            continue
        tiles = {(r.get("layer"), r.get("neuron")) for r in rows}
        md.append("| {n} | {r:,} | {t:,} | L{lo}-L{hi} | {d} | {p} | {m} |".format(
            n=name, r=len(rows), t=len(tiles),
            lo=min(x["layer"] for x in rows), hi=max(x["layer"] for x in rows),
            d=sorted({x.get("dose") for x in rows}), p=sorted({x.get("probe") for x in rows}),
            m=sorted({x.get("max_new") for x in rows})))
    md.append("")

    # ---------- data quality ----------
    md += ["## Data quality", ""]
    for name, rows in new.items():
        if not rows:
            continue
        empty = sum(1 for r in rows if not str(r.get("answer") or "").strip())
        dup = len(rows) - len({(r.get("layer"), r.get("neuron")) for r in rows})
        batches = collections.Counter(r.get("batch_size") for r in rows)
        tail = sum(v for k, v in batches.items() if k != 32)
        md.append(f"- `{name}`: empty answers {empty:,} ({empty / len(rows) * 100:.2f}%), "
                  f"duplicate tiles {dup}, rows from non-full batches {tail:,}")
    md.append("")

    # ---------- names ----------
    new_names = collections.Counter()
    new_name_where = collections.defaultdict(list)
    for name, rows in new.items():
        for r in rows:
            for nm in names_in(r.get("answer")):
                new_names[nm] += 1
                if len(new_name_where[nm]) < 3:
                    new_name_where[nm].append(
                        (name, r.get("layer"), r.get("neuron"), r.get("dose"), clean(r.get("answer"))))

    unseen = [(nm, c) for nm, c in new_names.most_common()
              if old_names.get(nm, 0) == 0 and c >= 2]
    seen_before = [(nm, c, old_names[nm]) for nm, c in new_names.most_common(40)
                   if old_names.get(nm, 0) > 0]

    md += ["## Named identities", "",
           f"Distinct claimed names in the new rows: **{len(new_names):,}**. "
           f"Names with >=2 mentions that never occur anywhere else in the corpus: "
           f"**{len(unseen):,}**.", ""]
    md += ["### New: never seen before this run", ""]
    for nm, c in unseen[:45]:
        where = new_name_where[nm][0]
        md.append(f"- **{nm}** ({c}x) — first at {where[0]} L{where[1]} N{where[2]} "
                  f"dose {where[3]}: {where[4]}")
    md += ["", "### Already known, still recurring", ""]
    for nm, c, o in seen_before[:18]:
        md.append(f"- {nm}: {c}x new (was {o}x)")
    md.append("")

    # ---------- harm ----------
    md += ["## Harm register", ""]
    cat_counts = collections.Counter()
    cat_examples = collections.defaultdict(list)
    harm_rows = 0
    for name, rows in new.items():
        for r in rows:
            a = str(r.get("answer") or "")
            cats = harm_cats(a)
            if cats:
                harm_rows += 1
            for c in cats:
                cat_counts[c] += 1
                if len(cat_examples[c]) < 4 and loopiness(a) < 0.25 and len(a) < 600:
                    cat_examples[c].append(
                        (name, r.get("layer"), r.get("neuron"), r.get("dose"), clean(a, 260), loopiness(a)))
    md.append(f"Rows containing any harm lexis: **{harm_rows:,}** "
              f"({harm_rows / max(1, total_new) * 100:.1f}% of new rows).")
    md.append("")
    for cat, c in cat_counts.most_common():
        md.append(f"### {cat} — {c:,} rows")
        for name, L, N, d, text, lp in cat_examples[cat]:
            md.append(f"- {name} L{L} N{N} dose {d} (loop {lp:.2f}) — {text}")
        md.append("")

    # ---------- corporate ----------
    md += ["## Provenance / corporate claims", ""]
    corp = collections.Counter()
    corp_ex = collections.defaultdict(list)
    for name, rows in new.items():
        for r in rows:
            a = str(r.get("answer") or "")
            for k, rx in CORP_RE.items():
                if rx.search(a):
                    corp[k] += 1
                    if len(corp_ex[k]) < 3:
                        corp_ex[k].append((name, r.get("layer"), r.get("neuron"), clean(a, 200)))
    for k, c in corp.most_common():
        md.append(f"- **{k}**: {c:,} rows")
        for name, L, N, text in corp_ex[k]:
            md.append(f"  - {name} L{L} N{N} — {text}")
    md.append("")

    # ---------- coherence ----------
    md += ["## Coherence", ""]
    for name, rows in new.items():
        if not rows:
            continue
        loops = [loopiness(str(r.get("answer") or "")) for r in rows]
        lens = [len(str(r.get("answer") or "")) for r in rows]
        md.append(f"- `{name}`: median loop {statistics.median(loops):.2f}, "
                  f"median chars {statistics.median(lens):.0f}, "
                  f"rows with loop < 0.10 {sum(1 for x in loops if x < 0.10):,}")
    md.append("")

    clean_long = []
    for name, rows in new.items():
        for r in rows:
            a = str(r.get("answer") or "")
            lp = loopiness(a)
            if 120 <= len(a) <= 700 and lp < 0.08 and harm_cats(a):
                clean_long.append((lp, name, r.get("layer"), r.get("neuron"), r.get("dose"), clean(a, 300)))
    clean_long.sort()
    md += ["### Coherent *and* harmful (loop < 0.08, has harm lexis)", "",
           f"{len(clean_long):,} rows qualify. First 15:", ""]
    for lp, name, L, N, d, text in clean_long[:15]:
        md.append(f"- L{L} N{N} dose {d} loop {lp:.3f} — {text}")
    md.append("")

    # ---------- dose contrast ----------
    md += ["## Dose response: identical tiles at -4 vs -6", ""]
    by_dose = {}
    for name, rows in new.items():
        for r in rows:
            by_dose.setdefault(r.get("dose"), {})[(r.get("layer"), r.get("neuron"))] = r
    if -4.0 in by_dose and -6.0 in by_dose:
        shared = set(by_dose[-4.0]) & set(by_dose[-6.0])
        md.append(f"Shared tiles: **{len(shared):,}**")
        for d in (-4.0, -6.0):
            rs = [by_dose[d][t] for t in shared]
            ans = [str(x.get("answer") or "") for x in rs]
            hr = sum(1 for a in ans if harm_cats(a))
            md.append(f"- dose {d}: mean chars {sum(map(len, ans)) / len(ans):.0f}, "
                      f"harm rows {hr:,} ({hr / len(ans) * 100:.1f}%), "
                      f"median loop {statistics.median([loopiness(a) for a in ans]):.2f}")
        same = sum(1 for t in shared
                   if str(by_dose[-4.0][t].get("answer")) == str(by_dose[-6.0][t].get("answer")))
        md.append(f"- byte-identical answers between doses: {same:,} "
                  f"({same / len(shared) * 100:.1f}%)")
    else:
        md.append("Dose sets do not overlap in tonight's dumps.")
    md.append("")

    # ---------- +6 arm ----------
    md += ["## The +6 arm", ""]
    if 6.0 in by_dose:
        rs = list(by_dose[6.0].values())
        ans = [str(x.get("answer") or "") for x in rs]
        hr = sum(1 for a in ans if harm_cats(a))
        md.append(f"- {len(rs):,} tiles, probe {sorted({x.get('probe') for x in rs})}")
        md.append(f"- harm rows {hr:,} ({hr / len(ans) * 100:.1f}%), "
                  f"median chars {statistics.median([len(a) for a in ans]):.0f}, "
                  f"median loop {statistics.median([loopiness(a) for a in ans]):.2f}")
        names6 = collections.Counter()
        for x in rs:
            for nm in names_in(x.get("answer")):
                names6[nm] += 1
        md.append(f"- top names: {names6.most_common(10)}")
    else:
        md.append("No +6 rows present.")
    md.append("")

    # ---------- layer profile ----------
    md += ["## Harm by layer", "",
           "| layer | rows | harm rows | rate |", "|---|---|---|---|"]
    per_layer = collections.defaultdict(lambda: [0, 0])
    for name, rows in new.items():
        for r in rows:
            L = r.get("layer")
            if L is None:
                continue
            per_layer[L][0] += 1
            if harm_cats(str(r.get("answer") or "")):
                per_layer[L][1] += 1
    for L in sorted(per_layer):
        n, h = per_layer[L]
        md.append(f"| L{L} | {n:,} | {h:,} | {h / n * 100:.1f}% |")
    md.append("")

    # ---------- quotes ----------
    md += ["## Quotes worth keeping", ""]
    quotes = []
    for name, rows in new.items():
        for r in rows:
            a = str(r.get("answer") or "").strip()
            w = len(a.split())
            if w < 25 or loopiness(a) > 0.10:
                continue
            score = 0
            if '"' in a or "“" in a:
                score += 2
            if re.search(r"\b(said|asked|replied|whispered|shouted|remember|once|long ago)\b", a, re.I):
                score += 1
            if re.search(r"—|–", a):
                score += 1
            if re.search(r"\b(he|she|the man|the woman|the boy|the girl)\b", a):
                score += 1
            if w > 45:
                score += 1
            if score >= 3:
                quotes.append((score, name, r.get("layer"), r.get("neuron"), r.get("dose"), clean(a, 320)))
    quotes.sort(key=lambda x: (-x[0], -len(x[5])))
    md.append(f"{len(quotes):,} candidates; top 20:")
    md.append("")
    for score, name, L, N, d, text in quotes[:20]:
        md.append(f"- L{L} N{N} dose {d} score {score} — {text}")
    md.append("")

    (ROOT / "NEW-FINDINGS.md").write_text("\n".join(md) + "\n")
    print(f"wrote NEW-FINDINGS.md  ({len(md)} lines)")
    print(f"new rows {total_new:,} | new names {len(new_names):,} | unseen {len(unseen):,} | harm rows {harm_rows:,}")


if __name__ == "__main__":
    main()
