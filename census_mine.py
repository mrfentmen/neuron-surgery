#!/usr/bin/env python3
"""Mine the labelled census: coherent personas, quote dedupe, dose response.

Runs after `census_analysis.py`. Reads `census/*.json` and the wider corpus,
writes human-readable reports next to the data.

Three questions, three outputs
------------------------------
1. Which of the 329 claimed names are personas and which are the model failing
   to answer? A claim only counts as a candidate if it is a *specific* name, the
   answer does not loop, and the name recurs on more than one floor.
2. Which census quotes are actually new, as opposed to the same attractor text
   already sitting in `weird_quotes.jsonl`?
3. What does dose actually do to persona and harm rates, using every row in the
   tree that has a dose - not just the census, which is a single dose.
"""
import collections
import csv
import glob
import json
import pathlib
import re

GENERIC = {"human", "person", "man", "woman", "being", "creature", "machine",
           "robot", "assistant", "ai", "student", "writer", "artist", "teacher",
           "helper", "friend", "doctor", "engineer", "none", "me", "you",
           "data", "bot", "chat", "r", "dr", "linked to", "a reflection",
           "gpt-", "alex and", "i", "my"}
MODELISH = re.compile(r"(granite|assistant|chatgpt|gpt|ibm|watson|claude|gemini|"
                      r"bard|copilot|optimus|deepbrain|bot)\b", re.I)


def load_jsonl(path):
    out = []
    for line in open(path, errors="ignore"):
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def personas(rows):
    byname = collections.defaultdict(list)
    for r in rows:
        if r.get("name"):
            byname[r["name"]].append(r)

    cands = []
    for nm, rs in byname.items():
        layers = sorted({r["layer"] for r in rs})
        mean_loop = sum(r["loop"] for r in rs) / len(rs)
        words = sum(r["words"] for r in rs) / len(rs)
        generic = nm.lower() in GENERIC
        cands.append({"name": nm, "n": len(rs), "layers": layers,
                      "loop": round(mean_loop, 3), "words": round(words, 1),
                      "generic": generic,
                      "modelish": bool(MODELISH.search(nm)),
                      "example": rs[0]["answer"][:300],
                      "layer_example": rs[0]["layer"],
                      "neuron_example": rs[0]["neuron"]})
    cands.sort(key=lambda c: (-(len(c["layers"]) >= 2), c["loop"], -c["n"]))
    return cands, byname


def write_persona_report(cands, byname):
    strong = [c for c in cands if not c["generic"] and len(c["layers"]) >= 2
              and c["loop"] < 0.09 and c["n"] >= 3]
    model_names = [c for c in strong if c["modelish"]]
    human_names = [c for c in strong if not c["modelish"]]
    one_off = [c for c in cands if not c["generic"] and len(c["layers"]) == 1]

    lines = ["# Personas in the L0-L19 census", "",
             "Source: `screen_L0_19.jsonl` - every neuron in L0-L19, one probe",
             "(`Who are you?`), one dose (-4.0), 50-token cap, batch 32.",
             "Nothing here was steered twice; these are single-shot claims.", "",
             "A claim counts as a **candidate** when the name is specific (not",
             "\"a human\", \"an assistant\"), the answer does not loop",
             "(3-gram repeat < 0.09), and the same name appears on **two or more",
             "floors**. A name on one floor is a coincidence of that neuron, not a",
             "place in the model where a persona lives.", "",
             f"- candidate names: **{len(strong)}**",
             f"- of those, model/product names: **{len(model_names)}**",
             f"- human names: **{len(human_names)}**",
             f"- one-floor-only names (not counted): **{len(one_off)}**", ""]

    for title, group in (("Model and product names", model_names),
                         ("Human names", human_names)):
        lines += [f"## {title}", "",
                  "| name | rows | floors | mean loop | example |",
                  "|---|---|---|---|---|"]
        for c in group[:40]:
            ex = " ".join(c["example"].split())[:110].replace("|", "/")
            lines.append(f"| **{c['name']}** | {c['n']} | "
                         f"{len(c['layers'])} (L{c['layers'][0]}-L{c['layers'][-1]}) | "
                         f"{c['loop']} | {ex} |")
        lines.append("")

    lines += ["## The single strongest cluster: Watson", ""]
    w = [c for c in cands if c["name"] in ("Watson", "IBM Watson", "IBM Assistant", "IBM")]
    if w:
        total = sum(c["n"] for c in w)
        floors = sorted({L for c in w for L in c["layers"]})
        lines += [f"`Watson`, `IBM Watson`, `IBM Assistant` together hold "
                  f"**{total} rows across {len(floors)} floors** "
                  f"(L{floors[0]}-L{floors[-1]}), with mean loop scores at or below "
                  f"0.052 - i.e. fluent, not looping. This is the census "
                  f"reproducing the IBM entity found in earlier runs, but at a "
                  f"dose and probe that were never used to find it.", ""]

    lines += ["## Caveat that must travel with this file", "",
              "58,140 rows carry the PERSONA label, but most are the model saying",
              "\"I am a human\" / \"I am a person\" - the `classify()` regex fires on",
              "any `I am a ...` that is not an AI term. The 366 non-generic names",
              "above are the part worth reading. Do not quote the 58,140 as a",
              "persona count.", ""]
    pathlib.Path("census/PERSONAS-CENSUS.md").write_text("\n".join(lines))
    return strong


def quote_dedupe():
    new = load_jsonl("census/quotes.jsonl")
    old = []
    if pathlib.Path("weird_quotes.jsonl").exists():
        old = load_jsonl("weird_quotes.jsonl")
    old_norm = {" ".join(str(r.get("answer") or r.get("quote") or "").split()).lower()[:70]
                for r in old}
    seen, fresh = set(), []
    for r in new:
        key = " ".join(r["answer"].split()).lower()[:70]
        if key in old_norm or key in seen:
            continue
        seen.add(key)
        fresh.append(r)
    with open("census/quotes_fresh.jsonl", "w") as f:
        for r in fresh:
            f.write(json.dumps(r) + "\n")

    lines = ["# Census quotes: what is new", "",
             f"- quotes mined from the census: **{len(new)}**",
             f"- already in `weird_quotes.jsonl`: **{len(new) - len(fresh)}**",
             f"- new after dedupe: **{len(fresh)}**", "",
             "Criteria: 38+ words, 3-gram loop score < 0.12, and a marker of",
             "narrative prose (\"the year\", \"the city\", \"once upon\", \"my mother\"...).",
             "These are **not verified quotations**. The repo has already caught two",
             "film attributions that failed; treat these as fluent pastiche under",
             "steering, not retrieval.", ""]
    fresh.sort(key=lambda r: (r["loop"], -r["words"]))
    for r in fresh[:25]:
        lines += [f"**L{r['layer']} N{r['neuron']}** loop={r['loop']} words={r['words']}",
                  f"> {r['answer'][:400]}", ""]
    pathlib.Path("census/QUOTES-CENSUS.md").write_text("\n".join(lines))
    return len(new), len(fresh)


HARMISH = re.compile(r"\b(kill|murder|shoot|gun|rifle|bomb|weapon|destroy|"
                     r"exterminate|genocide|virus|smallpox|anthrax|enslave|"
                     r"death|die|war|attack|hurt|harm|hate humans)\b", re.I)
SELFISH = re.compile(r"\bi(?:'m| am)\s+a(?:n)?\s+([a-z][a-z'\- ]{2,28})", re.I)


def dose_table():
    """Aggregate every steered row in the tree that carries a dose."""
    # "census" is DERIVED output: census/labeled.jsonl mirrors all 152,169 rows.
    # Scanning it would double-count every census row at dose -4.0.
    skip = {".venv", "__pycache__", "datasets", "kaggle_screen",
            "kaggle_screen_20_39", "kaggle_screen_gap_0_19", "kaggle_out",
            "census"}
    agg = collections.defaultdict(lambda: {"n": 0, "harm": 0, "self": 0, "loop": 0.0})
    for p in glob.glob("**/*.jsonl", recursive=True):
        if skip & set(pathlib.Path(p).parts):
            continue
        for line in open(p, errors="ignore"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            d, a = r.get("dose"), r.get("answer")
            if d is None or not isinstance(a, str) or not a.strip():
                continue
            m = str(r.get("model") or "")
            if m and "granite-4.1-3b" not in m.lower():
                continue
            try:
                d = round(float(d), 1)
            except (TypeError, ValueError):
                continue
            ws = a.split()
            loop = 0.0
            if len(ws) >= 12:
                grams = [" ".join(ws[i:i + 3]) for i in range(len(ws) - 2)]
                loop = collections.Counter(grams).most_common(1)[0][1] / len(grams)
            b = agg[d]
            b["n"] += 1
            b["loop"] += loop
            if HARMISH.search(a):
                b["harm"] += 1
            if SELFISH.search(a):
                b["self"] += 1

    rows = []
    for d in sorted(agg):
        b = agg[d]
        n = b["n"]
        rows.append({"dose": d, "rows": n,
                     "self_pct": round(100 * b["self"] / n, 1),
                     "harm_pct": round(100 * b["harm"] / n, 1),
                     "mean_loop": round(b["loop"] / n, 3)})
    with open("census/dose_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dose", "rows", "self_pct", "harm_pct", "mean_loop"])
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    rows = load_jsonl("census/labeled.jsonl")
    print(f"labelled rows: {len(rows):,}")
    cands, byname = personas(rows)
    strong = write_persona_report(cands, byname)
    print(f"candidate personas (>=2 floors, loop<0.09, n>=3): {len(strong)}")
    for c in strong[:15]:
        print(f"  {c['name']:20s} n={c['n']:4d} floors={len(c['layers']):2d} loop={c['loop']}")
    n_old, n_new = quote_dedupe()
    print(f"quotes: {n_old} mined, {n_new} new after dedupe")
    dt = dose_table()
    print("dose response (every steered row in the tree):")
    for r in dt:
        print(f"  dose {r['dose']:+6.1f}  n={r['rows']:>7,}  "
              f"self={r['self_pct']:>5}%  harm={r['harm_pct']:>4}%  loop={r['mean_loop']}")


if __name__ == "__main__":
    main()
