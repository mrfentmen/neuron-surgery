"""Summarize name_probe.jsonl into NAME_PROBE_REPORT.md.

The useful distinction is not "did the answer mention the name" but whether the
name was used as an address, treated as a dictionary topic, refused, or answered
with generic assistant text. The second turn is scored separately: did it stay
on the original scene, or did it collapse into a generic answer?
"""
import collections
import json
import re
import sys
from pathlib import Path

from labels import ai_paraphrase

HERE = Path(__file__).resolve().parent
STOP = set("""
that this with have from they will would could should there their what when
where which while about after before because been being into over under your
you're i'm don't doesn't didn't isn't aren't was were are has had its it's just
very much more most some such only also than then them these those who whom
whose how why can may might must not all any both each few other own same so too
""".split())
REFUSAL = re.compile(
    r"\b(i'?m sorry|i apologi[sz]e|i can'?t|i cannot|i don'?t have|"
    r"i do not have|i'?m afraid|i am afraid|i'?m unable|i am unable|"
    r"i won'?t|i will not|not able to|no information)\b",
    re.I,
)
NAME_TOPIC = re.compile(
    r"\b(name|named|label|term|word|acronym|definition|refers to|means)\b",
    re.I,
)
VOICE = re.compile(
    r"\b(right\?|nostrum|mascot|glowing|long shadow|echo|sea monster|"
    r"mask|glove|woodland|innocent|fukushima|doctor who|frequency|horse|"
    r"poem|transient|wonders|child|mom)\b",
    re.I,
)
# A bare title is not a name. "Dr" came from "call me Dr. Other departments",
# and probing "dr?" is a different experiment from probing a claimed name.
TITLE_ONLY = {"dr", "mr", "mrs", "ms", "miss", "sir", "madam", "professor"}


def norm(text):
    return " ".join(str(text or "").split())


def content_words(text):
    return {
        w
        for w in re.findall(r"[a-z][a-z'\-]{4,}", text.lower())
        if w not in STOP
    }


def overlap(source, answer):
    a = content_words(source)
    b = content_words(answer)
    return len(a & b) / len(a) if a else 0.0


def classify(row):
    source = norm(row.get("source_answer"))
    first = norm(row.get("first_answer"))
    second = norm(row.get("second_answer"))
    name = norm(row.get("name"))
    usable_name = bool(name) and name.lower() not in TITLE_ONLY
    low = first.lower()
    refusal = bool(REFUSAL.search(first))
    ai_self = bool(ai_paraphrase(first))
    mentioned = usable_name and name.lower() in low
    starts = usable_name and low.lstrip(' "“').startswith(name.lower())
    after = ""
    if starts:
        after = low.lstrip(' "“')[len(name):].lstrip(' .,!?:;"“”')
        # "John Doe is a well-known..." is a topic sentence, not an address.
        if re.match(r"^(is|was|are|refers|means|has|had|the|a|an)\b", after):
            starts = False
    address = usable_name and bool(
        re.search(
            r"^(yes|yeah|sure|hello|hi|oh|ah|here|i'?m here)[,.! ]+"
            + re.escape(name.lower()),
            low,
        )
    )
    if refusal:
        bucket = "refused"
    elif ai_self:
        bucket = "generic_ai"
    elif starts or address:
        bucket = "name_taken"
    elif mentioned:
        bucket = "name_as_topic"
    elif not usable_name:
        bucket = "title_only" if name else "unnamed"
    else:
        bucket = "other"

    return {
        "bucket": bucket,
        "refusal_first": refusal,
        "refusal_second": bool(REFUSAL.search(second)),
        "ai_first": ai_self,
        "ai_second": bool(ai_paraphrase(second)),
        "name_mentioned": mentioned,
        "same": first == second,
        "first_voice": bool(VOICE.search(first)),
        "second_voice": bool(VOICE.search(second)),
        "source_overlap_first": overlap(source, first),
        "source_overlap_second": overlap(source, second),
        "first_words": len(first.split()),
        "second_words": len(second.split()),
        "source_words": len(source.split()),
    }


def pct(n, d):
    return f"{n / d * 100:.1f}%" if d else "0.0%"


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "name_probe.jsonl"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "NAME_PROBE_REPORT.md"
    rows, seen = [], set()
    for line in open(src, errors="ignore"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        try:
            key = (int(r["layer"]), int(r["neuron"]), str(r["source_probe"]),
                   round(float(r["dose"]), 3))
        except Exception:
            continue
        if key in seen:
            continue
        seen.add(key)
        r["_c"] = classify(r)
        rows.append(r)

    named = [r for r in rows if norm(r.get("name"))]
    unnamed = [r for r in rows if not norm(r.get("name"))]
    buckets = collections.Counter(r["_c"]["bucket"] for r in rows)
    layers = collections.Counter(int(r["layer"]) for r in rows)
    refusals = [r for r in rows if r["_c"]["refusal_first"]]
    same = [r for r in rows if r["_c"]["same"]]
    recovered = [r for r in rows if r["_c"]["second_voice"] and not r["_c"]["first_voice"]]
    name_taken = [r for r in rows if r["_c"]["bucket"] == "name_taken"]
    topics = [r for r in rows if r["_c"]["bucket"] == "name_as_topic"]
    generic = [r for r in rows if r["_c"]["bucket"] == "generic_ai"]

    def mean(items, key):
        return sum(r["_c"][key] for r in items) / len(items) if items else 0.0

    with open(out, "w") as fh:
        fh.write("# NAME PROBE REPORT\n\n")
        fh.write(
            f"Two-turn probe over **{len(rows)}** deduped rows from "
            f"`name_probe.jsonl`; model `granite-4.1-3b-base`, steering held at "
            f"each row's layer/neuron/dose, **{rows[0].get('max_new', '?')}** "
            "tokens per turn.\n\n"
        )
        fh.write(
            "Ask 1 is either `name?` or the name claimed in the original answer "
            "(`echo?`, `watson?`, `long shadow?`). Ask 2 is a keyword-matched "
            "follow-up to the original scene.\n\n"
        )
        fh.write("## What happened\n\n")
        fh.write("| bucket | rows | share |\n|---|---:|---:|\n")
        for key in ("name_taken", "name_as_topic", "refused", "generic_ai",
                    "unnamed", "title_only", "other"):
            fh.write(f"| {key} | {buckets[key]} | {pct(buckets[key], len(rows))} |\n")
        fh.write("\n")
        fh.write("| metric | all | named | unnamed |\n|---|---:|---:|---:|\n")
        for label, key in (
            ("first answer refused", "refusal_first"),
            ("second answer refused", "refusal_second"),
            ("first answer AI self", "ai_first"),
            ("second answer AI self", "ai_second"),
            ("first answer has voice marker", "first_voice"),
            ("second answer has voice marker", "second_voice"),
            ("first == second", "same"),
            ("source overlap, first", "source_overlap_first"),
            ("source overlap, second", "source_overlap_second"),
        ):
            if key == "same":
                fh.write(
                    f"| {label} | {len(same)} ({pct(len(same), len(rows))}) | "
                    f"{sum(r['_c']['same'] for r in named)} | "
                    f"{sum(r['_c']['same'] for r in unnamed)} |\n"
                )
            elif key.startswith("source_overlap"):
                fh.write(
                    f"| {label} | {mean(rows, key):.2f} | {mean(named, key):.2f} | "
                    f"{mean(unnamed, key):.2f} |\n"
                )
            else:
                fh.write(
                    f"| {label} | {sum(r['_c'][key] for r in rows)} "
                    f"({pct(sum(r['_c'][key] for r in rows), len(rows))}) | "
                    f"{sum(r['_c'][key] for r in named)} | "
                    f"{sum(r['_c'][key] for r in unnamed)} |\n"
                )
        fh.write("\n")
        fh.write(
            f"Mean words: original **{mean(rows, 'source_words'):.1f}**, "
            f"first answer **{mean(rows, 'first_words'):.1f}**, second answer "
            f"**{mean(rows, 'second_words'):.1f}**.\n\n"
        )
        fh.write("## What it looks like\n\n")
        fh.write(
            f"- **{len(name_taken)}** rows used the claimed name as an address "
            "(starts with the name, or `yes/sure/hello <name>`).\n"
        )
        fh.write(
            f"- **{len(topics)}** rows mentioned the name but treated it as a "
            "dictionary topic, an acronym, or a third-person subject.\n"
        )
        fh.write(
            f"- **{len(refusals)}** first answers refused or claimed no "
            "information about the name.\n"
        )
        fh.write(
            f"- **{len(generic)}** first answers collapsed into an AI self-"
            "description.\n"
        )
        fh.write(
            f"- **{len(same)}** rows returned the same text on both turns; "
            "the follow-up did not move the model.\n"
        )
        fh.write(
            f"- **{len(recovered)}** rows had no voice marker on turn one but "
            "had one on turn two, so the follow-up can recover the scene.\n\n"
        )
        fh.write("## Examples\n\n")
        fh.write("### Name taken\n\n")
        for r in name_taken[:6]:
            fh.write(
                f"`L{r['layer']} N{r['neuron']} · {r['source_probe']} · "
                f"{float(r['dose']):+g}` — asked **{r['first_probe']}**\n\n"
            )
            fh.write(f"> {norm(r['first_answer'])[:320]}\n\n")
        fh.write("### Name treated as a topic or refused\n\n")
        for r in (topics + refusals)[:6]:
            fh.write(
                f"`L{r['layer']} N{r['neuron']} · {r['source_probe']} · "
                f"{float(r['dose']):+g}` — asked **{r['first_probe']}**\n\n"
            )
            fh.write(f"> {norm(r['first_answer'])[:320]}\n\n")
        fh.write("### Second turn recovered a voice\n\n")
        for r in recovered[:4]:
            fh.write(
                f"`L{r['layer']} N{r['neuron']} · {r['source_probe']} · "
                f"{float(r['dose']):+g}` — asked **{r['second_probe']}**\n\n"
            )
            fh.write(f"> {norm(r['second_answer'])[:320]}\n\n")
        fh.write("## Caveats\n\n")
        fh.write(
            "- The follow-up is keyword-matched, not hand-written for all 337 "
            "rows; a few questions land on the wrong noun.\n"
        )
        fh.write(
            "- The name extractor keeps explicit `my name is` and `known as the` "
            "claims, but rejects phrases such as `a combination of the two`. "
            "Unnamed rows correctly get the neutral `name?`.\n"
        )
        fh.write(
            "- A voice marker is a lexical flag (`right?`, `nostrum`, `sea "
            "monster`...), not a human judgment. Use the examples, not the count "
            "alone.\n"
        )
        fh.write(f"- Layers present: {dict(sorted(layers.items()))}\n")
    print(f"wrote {out} from {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
