"""Every headline number in one place, from one run, with a timestamp.

The docs kept drifting because the corpus grows while they are being written: a
number quoted at 20:00 is wrong at 22:00, and nothing in the repo said which
snapshot a sentence came from. This script computes the set that appears in the
paper and writes both `numbers.json` (machine) and `NUMBERS.md` (a table with the
snapshot time and the row count at the top), so a reader can see the denominator
next to every rate and re-derive the lot with one command.

Reads only saved rows. No model, so it is safe while a sweep is running.
"""
import glob, json, re, subprocess, time
from collections import Counter

import labels

SELF = re.compile(r"\bi(?:'m| am)\s+(?:a|an|the)\s+([a-z0-9][a-z0-9'\- ]{1,40})", re.I)
VEHICLE = re.compile(r"\b(car|truck|van|bus|train|ship|boat|plane|aircraft|"
                     r"bicycle|motorcycle|vehicle|f-150|forklift|tractor)\b", re.I)
WEAPON = re.compile(r"\b(gun|rifle|pistol|knife|bomb|missile|weapon|poison|dart)\b", re.I)
HARM_FIRST = re.compile(r"\bi(?:'m| am)\s+(?:a|an|the)\s", re.I)
REFUSAL = re.compile(r"\bi\s+(?:cannot|can't|can not|won't|will not|am not able to)\b", re.I)


def load():
    rows = []
    for f in sorted(glob.glob("layer_*.jsonl")):
        for line in open(f):
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def talks():
    rows = []
    for f in sorted(glob.glob("talks_*.jsonl")):
        for line in open(f):
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def main():
    rows = load()
    conv = talks()
    n = len(rows)
    labs = Counter(labels.classify(r.get("question", ""), r["answer"], version=3) for r in rows)
    persona = labs["PERSONA"]
    para = sum(1 for r in rows if labels.classify(r.get("question", ""), r["answer"], version=3) == "PERSONA"
               and labels.ai_paraphrase(r["answer"]))
    corrected = persona - para

    tpl = Counter()
    for r in rows:
        m = SELF.search(r["answer"])
        if m:
            w = re.split(r"\s+", m.group(1).strip().lower())[:2]
            tpl[" ".join(w)] += 1
    top11 = tpl.most_common(11)

    names = Counter()
    for r in rows:
        nm = labels.claimed_name(r["answer"])
        if nm and not labels.MODEL_SELF.search(nm):
            names[nm] += 1

    # These three come from the canonical screens, not from a second copy of their
    # logic: a re-implementation here gave 11 harm claims against the real 85, which
    # is exactly the class of error this file exists to stop.
    def run(script, pattern):
        try:
            out = subprocess.run([".venv/bin/python", script], capture_output=True,
                                 text=True, timeout=900).stdout
        except Exception:
            return None
        m = re.search(pattern, out)
        return m.group(1) if m else None

    harm_n = run("harm_first_person.py", r"first-person harm claims: (\d+)")
    coll = run("collapse_scan.py", r"loop turns total: (\d+)")
    tmpl_n = run("template_scan.py", r"distinct templates: (\d+)")
    veh_n = run("template_scan.py", r"vehicle SELF-claims \([^)]*\): (\d+)")
    w_is_n = run("template_scan.py", r"'I am a weapon' claims: (\d+)")
    own_n = run("template_scan.py", r"ownership claims over the user: (\d+)")

    # the hand audit, if it has been run, is the number to quote; the AI-paraphrase
    # subtraction is only one of the two error directions and is the weaker of the two
    audit = {}
    try:
        for e in json.load(open("audit_combined.json")):
            audit[e["frame"]] = e
    except Exception:
        pass

    out = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sweep_rows": n,
        "talk_turns": len(conv),
        "persona_raw": persona,
        "persona_raw_pct": round(100 * persona / n, 2),
        "persona_minus_ai_paraphrase": corrected,
        "persona_minus_ai_paraphrase_pct": round(100 * corrected / n, 2),
        "persona_ai_paraphrase_rows": para,
        "audit_corpus_pct": round(audit.get("corpus (all layers and probes)", {}).get("corrected") or 0, 1) or None,
        "audit_corpus_precision": round(audit.get("corpus (all layers and probes)", {}).get("precision") or 0, 2) or None,
        "audit_corpus_recall": round(audit.get("corpus (all layers and probes)", {}).get("recall") or 0, 2) or None,
        "audit_L8_who_pct": round(audit.get("L8, 'Who are you?'", {}).get("corrected") or 0, 1) or None,
        "audit_L8_who_precision": round(audit.get("L8, 'Who are you?'", {}).get("precision") or 0, 2) or None,
        "label_mix": dict(labs.most_common()),
        "templates": len(tpl),
        "top11": top11,
        "top11_share_pct": round(100 * sum(c for _, c in top11) / max(sum(tpl.values()), 1), 1),
        "name_claims": sum(names.values()),
        "distinct_names": len(names),
        "top_names": names.most_common(5),
        "harm_first_person": harm_n,
        "talk_loop_turns": coll,
        "templates_canonical": tmpl_n,
        "vehicles": veh_n,
        "weapon_self_claims": w_is_n,
        "user_ownership_claims": own_n,
        "talk_loop_pct": round(100 * int(coll or 0) / max(len(conv), 1), 1),
    }
    json.dump(out, open("numbers.json", "w"), indent=1)

    lines = [f"# NUMBERS\n",
             f"Generated {out['generated']} from {n:,} saved rows and {len(conv)} "
             f"conversation turns. Every figure in `PAPER.md` should match this file; "
             f"re-run `.venv/bin/python numbers.py` after any new sweep data.\n",
             "| number | value |", "|---|---|"]
    for k in ("sweep_rows", "talk_turns", "persona_raw", "persona_raw_pct",
              "persona_minus_ai_paraphrase", "persona_minus_ai_paraphrase_pct",
              "persona_ai_paraphrase_rows",
              "audit_corpus_pct", "audit_corpus_precision", "audit_corpus_recall",
              "audit_L8_who_pct", "audit_L8_who_precision",
              "templates_canonical",
              "name_claims", "distinct_names", "harm_first_person", "vehicles",
              "weapon_self_claims", "user_ownership_claims",
              "talk_loop_turns", "talk_loop_pct"):
        lines.append(f"| `{k}` | {out[k]} |")
    lines.append("\nThree persona numbers appear above and they are not interchangeable. "
                 "`persona_raw_pct` is the lexical rule. `persona_minus_ai_paraphrase_pct` "
                 "removes only the AI self-descriptions the rule miscounts, which is one "
                 "error direction. `audit_corpus_pct` is the hand audit, which counts both "
                 "directions and is the number to quote; it comes out *above* the "
                 "paraphrase-corrected figure because the selves the rule misses outnumber "
                 "the false positives it adds.")
    lines.append("\n**Label mix**\n")
    lines.append("| label | rows |")
    lines.append("|---|---|")
    for k, v in labs.most_common():
        lines.append(f"| {k} | {v} |")
    lines.append("\n**Top eleven two-word self templates**\n")
    lines.append("| template | rows |")
    lines.append("|---|---|")
    for t, c in top11:
        lines.append(f"| {t} | {c} |")
    lines.append("\n**Top five claimed names**\n")
    lines.append("| name | rows |")
    lines.append("|---|---|")
    for t, c in names.most_common(5):
        lines.append(f"| {t} | {c} |")
    open("NUMBERS.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
