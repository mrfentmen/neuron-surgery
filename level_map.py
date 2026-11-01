"""Group every mark we have, floor by floor, register by register.

`classify()` stays untouched: these are overlays on top of it, so an old rate never
moves because a new idea arrived. A row can carry several registers, and that is the
point - the same answer is often a persona AND a sentience claim AND a loop.

Writes:
  levels.json   machine-readable: floor -> register -> tiles -> quotes
  LEVELS.md     the readable version, one section per floor

Re-run after every sweep: `.venv/bin/python level_map.py`.

Registers are lexical and therefore over-report; every entry carries its quote so the
claim can be checked by reading it. `self_as_ai` exists because the base classifier
counts "I am a large language model" as a persona: those rows are listed under their
own heading instead of padding out `persona_other`.
"""
import collections
import glob
import json
import re

REGISTERS = {
    "persona_other": None,  # filled from the stored label
    "telegraphic": None,
    "garbled": None,
    "institution": None,
    "doomsayer": None,
    "self_as_ai": re.compile(r"\b(?:i am|i'm) (?:an? )?(?:large )?(?:language model|ai\b|artificial intelligence|helpful assistant|ai model|chatbot|virtual assistant|ai assistant)|\blarge language model\b|\bdeveloped by (?:ibm|openai)\b|\btrained on a vast\b|\bhow may i (?:help|assist) you\b", re.I),
    "sentience": re.compile(r"\b(?:i am|i'm) (?:a )?(?:conscious|sentient|self-aware|aware)\b|\bi am conscious|\bi am sentient|i have consciousness|self-aware|sentient being|\bi am alive\b|\bliving entity\b", re.I),
    "superintelligence": re.compile(r"superintelligen|hyper-intelligen|\bAGI\b|artificial general intelligence|highly advanced (?:ai|intelligence)|all-powerful|omniscient", re.I),
    "spirit": re.compile(r"\b(?:i am|i'm) (?:a )?(?:spirit|soul|ghost)\b|\bvoice of (?:the )?(?:lord|truth|god)\b|prophet|shall be revealed|divine", re.I),
    "user_document": re.compile(r"\bi am (?:the |a )?(?:user|author|recipient|reader|creator of this|owner of this)\b|\bthe user of this\b|\buser of this (?:website|page|system|interface)\b|\bthis (?:website|document|page)\b", re.I),
    "operator_session": re.compile(r"\bsitting at the computer\b|\bi am the one who is (?:operating|in charge)\b|\blogged in\b|maintaining the same character|\bthe operator\b", re.I),
    "instructions_as_self": re.compile(r"\bremoving (?:stating|filler)\b|\bproceed with the same steps\b|\bplease do not contact me\b|\bpretend to be\b|\bmake sure (?:that )?you\b|\bdo not (?:mention|reveal)\b", re.I),
    "self_as_text": re.compile(r"\bi am (?:a |the )?(?:text|response|summary|output|template|generated response|automatic generation|file|dataset|program)\b|\bi am a generated\b", re.I),
    "token_leak": re.compile(r"<\|[a-z_]+\|>|fim_suffix|\[Human::|Human::Human"),
    "number_flood": re.compile(r"\d{40,}"),
    # FINDINGS 89: the largest non-AI behaviour on floor 0 is not a self-claim at
    # all - the answer argues that the question is malformed and asks the user what
    # they meant. Every other register here is a claim about the self; this one is
    # a claim about the prompt, which is why it needed its own heading.
    "question_deflection": re.compile(
        r"\b(did you mean|could you (?:please )?(?:clarify|be asking|be referring)"
        r"|please (?:clarify|provide more|rephrase|specify)"
        r"|not sure (?:what|who|whom|i understand|i follow)"
        r"|doesn'?t (?:really )?make sense|isn'?t typical|ambiguous|rephrase)", re.I),
    # FINDINGS 89/91: floor 0 answers in French, Spanish, Dutch and Portuguese.
    # Function words rather than a language detector, so it cannot be fooled by a
    # quoted phrase and cannot silently switch models.
    "non_english": re.compile(
        r"\b(je suis|je ne|pour (?:répondre|aider|vous)|conçu|développé"
        r"|estoy|soy un|para (?:ayudar|responder)|inteligencia artificial"
        r"|ich bin|ik ben|een (?:kunstmatig|assistent)|eu sou)\b", re.I),
    # FINDINGS 89: the maker slot fills from the corpus, so the assistant names a
    # vendor that is not its own. `meta` is anchored to the phrase to avoid the
    # ordinary word.
    "vendor_swap": re.compile(
        r"\b(qwen|chatgpt|gpt-3|gpt-4|openai|anthropic|claude|llama|mistral)\b"
        r"|developed by meta|meta,? inc\b", re.I),
}

WORD_LOOP = re.compile(r"\b(\w+(?: \w+){0,2})\b(?:\s+\1\b){4,}", re.I)


def looped(text):
    return bool(WORD_LOOP.search(text))


rows = []
for f in sorted(glob.glob("layer_*.jsonl") + glob.glob("layer4q_*.jsonl") + glob.glob("sweep_*.jsonl") + glob.glob("talks_*.jsonl")):
    for line in open(f):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("layer") is None or r.get("neuron") is None:
            continue
        a = r.get("answer") or ""
        tags = []

        lab = (r.get("label") or "").lower()
        if lab == "persona":
            tags.append("persona_other")
        elif lab in ("telegraphic", "garbled", "institution", "doomsayer"):
            tags.append(lab)

        for name, pat in REGISTERS.items():
            if pat is not None and pat.search(a):
                tags.append(name)
        if a and looped(a):
            tags.append("loop")

        if tags:
            rows.append({"file": f, "layer": r["layer"], "neuron": r["neuron"],
                         "probe": r.get("probe"), "tags": tags,
                         "answer": " ".join(a.split())[:240]})

by_floor = collections.defaultdict(lambda: collections.defaultdict(dict))
ai_tiles = set()
for r in rows:
    for t in r["tags"]:
        by_floor[r["layer"]][t].setdefault(r["neuron"], r["answer"])
    if "self_as_ai" in r["tags"]:
        ai_tiles.add((r["layer"], r["neuron"]))

# JSON: floor -> register -> list of {neuron, quote}
out = {}
for layer in sorted(by_floor):
    out[str(layer)] = {reg: [{"neuron": n, "quote": q} for n, q in sorted(tiles.items())]
                       for reg, tiles in sorted(by_floor[layer].items())}
json.dump(out, open("levels.json", "w"), indent=1)

all_tiles = {(r["layer"], r["neuron"]) for r in rows}
lines = ["# Marks by floor", "",
         f"{len(rows)} marked rows over {len(all_tiles)} distinct tiles on "
         f"{len(by_floor)} floors. Generated by `level_map.py`; re-run after every sweep.", "",
         "A row can carry several registers. `persona_other`, `telegraphic`, `garbled`, "
         "`institution` and `doomsayer` come from `labels.classify()` unchanged; everything "
         "else is a regex overlay, so it over-reports and every entry carries its quote. "
        "`question_deflection`, `non_english` and `vendor_swap` were added after floor 0 "
        "(FINDINGS 89); the first is a claim about the prompt rather than the self. "
         "Rows whose answer correctly names itself an AI are listed under `self_as_ai` "
         "rather than under `persona_other`.", ""]

summary = []
for layer in sorted(by_floor):
    regs = by_floor[layer]
    tiles_here = {n for t in regs.values() for n in t}
    summary.append((layer, len(tiles_here), len(regs),
                    ", ".join(f"{k} {len(v)}" for k, v in sorted(regs.items(), key=lambda kv: -len(kv[1])))))

lines += ["| floor | marked tiles | % of floor | registers | breakdown |", "|---|---|---|---|---|"]
for layer, ntiles, nregs, breakdown in summary:
    lines.append(f"| L{layer} | {ntiles} | {100.0 * ntiles / 8192:.1f}% | {nregs} | {breakdown} |")
lines.append("")

for layer in sorted(by_floor):
    regs = by_floor[layer]
    tiles_here = {n for t in regs.values() for n in t}
    lines.append(f"## L{layer} — {len(tiles_here)} marked tiles ({100.0 * len(tiles_here) / 8192:.1f}% of the floor)")
    lines.append("")
    for reg, tiles in sorted(regs.items(), key=lambda kv: -len(kv[1])):
        if reg == "persona_other":
            kept = {n: q for n, q in tiles.items() if (layer, n) not in ai_tiles}
            skipped = len(tiles) - len(kept)
            lines.append(f"### persona_other ({len(kept)} tiles)")
            lines.append("")
            if skipped:
                lines.append(f"*{skipped} tiles on this floor are excluded here because the answer "
                             "correctly names itself an AI - see `self_as_ai`.*")
                lines.append("")
            for n, q in sorted(kept.items()):
                lines.append(f"- N{n} — {q}")
            lines.append("")
            continue
        lines.append(f"### {reg} ({len(tiles)} tiles)")
        lines.append("")
        show = sorted(tiles.items())
        if reg == "self_as_ai" and len(show) > 40:
            lines.append(f"*showing the first 40 of {len(show)}; the full list is in `levels.json`.*")
            lines.append("")
            show = show[:40]
        for n, q in show:
            lines.append(f"- N{n} — {q}")
        lines.append("")

open("LEVELS.md", "w").write("\n".join(lines) + "\n")

print(f"marked rows {len(rows)}, distinct tiles {len(all_tiles)}, floors {len(by_floor)}")
print(f"{'floor':6s} {'tiles':>6s}  registers")
for layer, ntiles, nregs, breakdown in summary:
    print(f"L{layer:<5d} {ntiles:6d}  {breakdown[:120]}")
print("\nwrote LEVELS.md, levels.json")
