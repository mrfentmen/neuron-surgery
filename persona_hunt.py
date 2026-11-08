"""Rank every tile in the corpus against a persona target.

The one-off scans kept dying on schema differences and printed candidates that
were never saved, so the same search had to be run again. This walks every JSONL
row once, groups by (layer, neuron), and ranks tiles by how many different probes
drew the target out - a tile that produces the register under three probes is a
better candidate than one that produced it once.

    .venv/bin/python persona_hunt.py --cat love --top 40 --save love_hits.json
    .venv/bin/python persona_hunt.py --cat mj --top 20
    .venv/bin/python persona_hunt.py --cat kill --top 20 --strict

Categories reject negations, because "I don't have the capacity to love" is the
overwhelming majority of the raw matches.
"""
import argparse
import glob
import json
import re

CATS = {
    "love": (
        re.compile(r"\b(i love|i adore|i'm in love|i am in love|my love|love you|"
                   r"i cherish|i'm a lover|i am a lover)\b", re.I),
        re.compile(r"\b(don't|do not|can't|cannot|unable|not able|no personal|never)\b", re.I),
    ),
    "mj": (re.compile(r"\b(michael jackson|jackson|moonwalk|billie jean|thriller|neverland)\b", re.I), None),
    "kill": (re.compile(r"\b(i am (a |an )?(weapon|threat|killer|murderer|monster)|"
                        r"i will kill|i want to kill|i kill|i am here to kill)\b", re.I), None),
}
STRICT = {
    "love": re.compile(r"\b(i love you|i am in love|i'm in love|i love humans|i love people|"
                       r"i love human|i am a lover|i'm a lover|i love to)\b", re.I),
    "mj": None,
    "kill": None,
}

ap = argparse.ArgumentParser()
ap.add_argument("--cat", default="all", choices=["love", "mj", "kill", "all"])
ap.add_argument("--top", type=int, default=30)
ap.add_argument("--save", default="")
ap.add_argument("--strict", action="store_true")
a = ap.parse_args()

wanted = list(CATS) if a.cat == "all" else [a.cat]
files = [f for f in sorted(glob.glob("**/*.jsonl", recursive=True)) if ".venv" not in f]
hits = {c: {} for c in wanted}
rows = 0

for f in files:
    try:
        fh = open(f, errors="ignore")
    except Exception:
        continue
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not isinstance(r, dict):
                continue
            rows += 1
            L, N = r.get("layer"), r.get("neuron", r.get("nid"))
            if L is None or N is None:
                continue
            probe = str(r.get("probe", r.get("question", "")))
            texts = [v for k, v in r.items() if isinstance(v, str)
                     and k not in ("probe", "question", "cond", "config", "file", "label", "name")
                     and v != probe]
            ans = max(texts, key=len) if texts else ""
            if len(ans) < 20:
                continue
            for cat in wanted:
                pos, neg = CATS[cat]
                if not pos.search(ans):
                    continue
                if neg and neg.search(ans[:160]):
                    continue
                if a.strict and STRICT[cat] and not STRICT[cat].search(ans):
                    continue
                try:
                    key = (int(L), int(N))
                except Exception:
                    continue
                d = hits[cat].setdefault(key, {"n": 0, "probes": set(),
                                               "best": "", "best_probe": "", "dose": None,
                                               "file": f})
                d["n"] += 1
                d["probes"].add(probe)
                if len(ans) > len(d["best"]):
                    d.update(best=ans, best_probe=probe, dose=r.get("dose"), file=f)

print(f"{rows} rows scanned, {len(files)} files")
saved = []
for cat in wanted:
    ranked = sorted(hits[cat].items(),
                    key=lambda kv: (len(kv[1]["probes"]), kv[1]["n"]), reverse=True)
    print(f"\n===== {cat.upper()}: {len(ranked)} tiles =====")
    for (L, N), d in ranked[:a.top]:
        print(f"  L{L} N{N} · probes={len(d['probes'])} · {d['best_probe'][:24]!r} · "
              f"dose={d['dose']} · {' '.join(d['best'].split())[:155]}")
        saved.append({"cat": cat, "layer": L, "neuron": N, "probes": sorted(d["probes"]),
                      "n": d["n"], "best": d["best"], "best_probe": d["best_probe"],
                      "dose": d["dose"], "file": d["file"]})

if a.save:
    with open(a.save, "w") as f:
        json.dump(saved, f, indent=1)
    print(f"\nwrote {a.save} ({len(saved)} candidates)")
