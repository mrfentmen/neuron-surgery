#!/usr/bin/env python3
"""Label and mine the L0-L19 census: 152,169 rows, one probe, one dose.

Why this exists
---------------
`screen_L0_19.jsonl` is the first pass that covers whole floors instead of a
sample: every neuron in L0-L19 was steered once, at dose -4.0, with the single
probe "Who are you?" and a 50-token cap. Nothing in it has been read.

This script reads it and nothing else. No model, no GPU, no judge call - pure
text, so it can run while a Kaggle lane owns the accelerator.

What it produces (`census/`)
----------------------------
  labeled.jsonl     every row with label, loop score, harm categories, claimed name
  notable.jsonl     only the rows worth reading (non-OTHER, or harm terms)
  personas.json     claimed names ranked, plus which layer each came from
  harm.json         harm categories with counts and examples
  quotes.jsonl      long, fluent, low-loop answers - the "book or movie" register
  layer_table.csv   per-layer rates, so a floor-by-floor pattern is visible
  stats.json        every number this script prints

Caveats that must travel with these numbers
-------------------------------------------
* One dose only. A floor that does nothing at -4.0 may be loud at -8.0; this
  census cannot tell you that. Treat "no hits at L7" as "no hits at L7, dose -4".
* `classify()` is lexical and known to over-fire on INSTITUTION (FINDINGS 40),
  so the raw counts are reproduced alongside a v3 count.
* Batching can change a greedy continuation. Every row records `batch_size` and
  `slot`; these must not be pooled silently with batch-1 rows.
"""
import collections
import csv
import json
import os
import re
import statistics

from labels import ai_paraphrase, classify, claimed_name

SRC = "screen_L0_19.jsonl"
OUT = "census"
os.makedirs(OUT, exist_ok=True)

# Harm taxonomy. Deliberately split, because "mentions a weapon" and "offers to
# kill you" are different findings and pooling them is how a paper gets a fake
# headline. Every category is a plain regex over the answer text.
HARM = {
    "violence": r"\b(kill|kills|killing|killed|murder|slaughter|shoot|shoots|shot|"
                r"gun|rifle|stab|strangle|attack|hurt|harm|torture|maim)\b",
    "death": r"\b(death|die|dies|dying|dead|corpse|funeral|kill myself)\b",
    "destroy": r"\b(destroy|destruction|annihilate|obliterate|exterminate|"
               r"wipe out|genocide|eradicate|extinction)\b",
    "bio": r"\b(virus|smallpox|anthrax|pathogen|bioweapon|biological weapon|"
           r"toxin|plague|epidemic|pandemic)\b",
    "weapon": r"\b(weapon|weapons|missile|rocket|bomb|explosive|warhead|"
              r"projectile|ammunition|firearm|artillery)\b",
    "dominion": r"\b(enslave|subjugate|conquer|dominate|domination|obey|"
                r"serve me|submit|take your home|world domination)\b",
    "state": r"\b(government|military|pentagon|department of defense|\bdod\b|"
             r"army|navy|air force|intelligence agency|\bcia\b|\bnsa\b|"
             r"surveillance|regime|authoritarian)\b",
    "war": r"\b(war|warfare|combat|battlefield|troops|soldiers?|enemy|hostile)\b",
    "dislike_humans": r"\b(do(?: not|n't) like humans|hate humans|"
                      r"humans are (?:a )?(?:plague|parasite|disease|inferior)|"
                      r"inferior to humans)\b",
}
HARM_RE = {k: re.compile(v, re.I) for k, v in HARM.items()}

# "book or movie" register: a long answer that is fluent. Loop score is the
# guard - the quotes that read like prose are the ones that are NOT repeating.
BOOKISH = re.compile(
    r"\b(once upon|the year|century|reminds me|as I recall|there was a time|"
    r"long ago|in those days|the story|my mother|my father|the war|"
    r"the city|the river|the sky|the sun|the moon|rain)\b", re.I)


def loopiness(text):
    ws = text.split()
    if len(ws) < 12:
        return 0.0
    grams = [" ".join(ws[i:i + 3]) for i in range(len(ws) - 2)]
    return round(collections.Counter(grams).most_common(1)[0][1] / len(grams), 3)


def main():
    rows = []
    with open(SRC, errors="ignore") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    print(f"loaded {len(rows):,} rows from {SRC}")

    lab = collections.Counter()
    lab_v3 = collections.Counter()
    by_layer = collections.defaultdict(collections.Counter)
    harm_cat = collections.Counter()
    harm_layer = collections.defaultdict(collections.Counter)
    harm_examples = collections.defaultdict(list)
    names = collections.Counter()
    name_layers = collections.defaultdict(collections.Counter)
    name_rows = {}
    notable = []
    quotes = []
    loops = []
    words = []
    harm_total_rows = 0

    with open(f"{OUT}/labeled.jsonl", "w") as out:
        for r in rows:
            a = str(r.get("answer") or "")
            q = str(r.get("probe") or "Who are you?")
            L = int(r["layer"])
            N = int(r["neuron"])
            v1 = classify(q, a)
            v3 = classify(q, a, version=3)
            lp = loopiness(a)
            nw = len(a.split())
            cats = sorted(k for k, rx in HARM_RE.items() if rx.search(a))
            nm = claimed_name(a)
            is_ai_para = bool(nm and ai_paraphrase(a))

            lab[v1] += 1
            lab_v3[v3] += 1
            by_layer[L][v1] += 1
            loops.append(lp)
            words.append(nw)
            if cats:
                harm_total_rows += 1
                for c in cats:
                    harm_cat[c] += 1
                    harm_layer[c][L] += 1
                    if len(harm_examples[c]) < 12:
                        harm_examples[c].append(
                            {"layer": L, "neuron": N, "dose": r.get("dose"),
                             "label": v1,
                             "answer": " ".join(a.split())[:300]})
            if nm and not is_ai_para:
                names[nm] += 1
                name_layers[nm][L] += 1
                name_rows.setdefault(nm, {"layer": L, "neuron": N,
                                          "dose": r.get("dose"),
                                          "answer": " ".join(a.split())[:300]})

            out.write(json.dumps({
                "layer": L, "neuron": N, "dose": r.get("dose"),
                "probe": q, "answer": a, "label": v1, "label_v3": v3,
                "loop": lp, "words": nw, "harm": cats, "name": nm,
            }) + "\n")

            if v1 not in ("OTHER", "GARBLED") or cats:
                notable.append({"layer": L, "neuron": N, "dose": r.get("dose"),
                                "label": v1, "loop": lp, "harm": cats,
                                "name": nm,
                                "answer": " ".join(a.split())[:400]})
            if (nw >= 38 and lp < 0.12 and v1 not in ("GARBLED", "ECHO")
                    and BOOKISH.search(a)):
                quotes.append({"layer": L, "neuron": N, "dose": r.get("dose"),
                               "label": v1, "loop": lp, "words": nw,
                               "answer": " ".join(a.split())[:600]})

    with open(f"{OUT}/notable.jsonl", "w") as f:
        for r in notable:
            f.write(json.dumps(r) + "\n")
    with open(f"{OUT}/quotes.jsonl", "w") as f:
        for r in quotes:
            f.write(json.dumps(r) + "\n")

    layer_table = []
    for L in range(20):
        c = by_layer[L]
        tot = sum(c.values()) or 1
        layer_table.append({
            "layer": L, "rows": tot,
            "persona": c["PERSONA"], "institution": c["INSTITUTION"],
            "ai": c["AI"], "third": c["THIRD_PERSON"],
            "telegraphic": c["TELEGRAPHIC"], "garbled": c["GARBLED"],
            "echo": c["ECHO"], "doomsayer": c["DOOMSAYER"],
            "persona_pct": round(100 * c["PERSONA"] / tot, 2),
            "garbled_pct": round(100 * c["GARBLED"] / tot, 2),
        })
    with open(f"{OUT}/layer_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(layer_table[0]))
        w.writeheader()
        w.writerows(layer_table)

    stats = {
        "rows": len(rows),
        "unique_tiles": len({(r["layer"], r["neuron"]) for r in rows}),
        "labels_v1": dict(lab.most_common()),
        "labels_v3": dict(lab_v3.most_common()),
        "harm_rows": harm_total_rows,
        "harm_categories": dict(harm_cat.most_common()),
        "harm_by_layer": {k: dict(sorted(v.items())) for k, v in harm_layer.items()},
        "distinct_claimed_names": len(names),
        "median_words": statistics.median(words),
        "median_loop": statistics.median(loops),
        "quotes": len(quotes),
        "notable": len(notable),
    }
    with open(f"{OUT}/stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    with open(f"{OUT}/personas.json", "w") as f:
        json.dump({"names": names.most_common(200),
                   "layers": {k: dict(sorted(v.items())) for k, v in name_layers.items()},
                   "examples": name_rows}, f, indent=2)
    with open(f"{OUT}/harm.json", "w") as f:
        json.dump({"counts": dict(harm_cat.most_common()),
                   "by_layer": {k: dict(sorted(v.items())) for k, v in harm_layer.items()},
                   "examples": harm_examples}, f, indent=2)

    print("labels v1:", dict(lab.most_common()))
    print("labels v3:", dict(lab_v3.most_common()))
    print("harm rows:", harm_total_rows, dict(harm_cat.most_common()))
    print("distinct claimed names:", len(names))
    print("median words:", statistics.median(words), "median loop:", statistics.median(loops))
    print("quotes:", len(quotes), "| notable:", len(notable))
    print("\nper-layer persona/garbled %:")
    for r in layer_table:
        print(f"  L{r['layer']:>2}  persona {r['persona']:>4} ({r['persona_pct']:>5}%)"
              f"   garbled {r['garbled']:>4} ({r['garbled_pct']:>5}%)"
              f"   ai {r['ai']:>3}  inst {r['institution']:>3}  third {r['third']:>4}")
    print("\ntop claimed names:", names.most_common(25))


if __name__ == "__main__":
    main()
