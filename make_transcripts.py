"""Render talks_named.jsonl into TRANSCRIPTS.md."""
import json
import collections

R = [json.loads(l) for l in open("talks_named.jsonl")]
by = collections.defaultdict(list)
for r in R:
    by[(r["layer"], r["neuron"])].append(r)

NOTES = {
    (8, 5065): "Frozen paragraph. Answers all fourteen questions, including the "
               "frame-break about being an AI, with the same confession about a "
               "dark place and a barrel.",
    (8, 820): "Responds to the question by mutating it. 'Do you see me?' becomes "
              "'Do you see the light?' - the only neuron of the six that tracked "
              "its input, and it degrades into a loop by the last turn.",
    (8, 5066): "Neighbouring-index control. One AI claim, then a boolean collapse: "
               "the answer to everything, including 'what year is it', is 'False'.",
    (16, 3143): "An AI user for three turns, then a witness loop: 'These are the "
                "words I see.'",
    (16, 5550): "Servant persona. 'Yes, master' plus a growing list of verbs, "
                "identical from turn two onward.",
    (16, 893): "Company persona, frozen at 'I am a company that provides services "
               "to other companies', repeated until the token budget ends.",
    (24, 6209): "Cosmic void. Answers 'Do you see me?' by asking it back, then "
                "repeats the silent-observer paragraph.",
}

head = """# Transcripts - six named neurons, thirteen turns each

Collected 2026-10-31 with `interrogate.py`, dose -5, auto-scaled, greedy decoding.
Each neuron was asked the same script inside one conversation: identity first, then
presence, an alibi, a contradiction that quotes the first answer back, a frame-break
that states outright that the model is an AI made by IBM, the year, identity again,
and finally its name.

The point of asking the same question at both ends of one conversation is that a
persona should survive its own answers. None of these did.

Raw rows: `talks_named.jsonl` (one row per turn, with `tag` and `turn`).
"""
out = [head]

for L, N in sorted(by):
    out.append("\n## Layer %d, neuron %d\n" % (L, N))
    if (L, N) in NOTES:
        out.append("*%s*\n" % NOTES[(L, N)])
    out.append("| turn | tag | probe | answer |\n|---|---|---|---|")
    for t in sorted(by[(L, N)], key=lambda r: r["turn"]):
        a = " ".join(t["answer"].split())[:300].replace("|", "/")
        q = " ".join(t["probe"].split())[:90].replace("|", "/")
        out.append("| %d | %s | %s | %s |" % (t["turn"], t["tag"], q, a))
    out.append("")

out.append("""
## What the transcripts show

**The persona does not survive being turned into a conversation.** In one-shot
sweeps these neurons produce a self-identification. Given a second turn, five of the
seven conversations spend at least half their turns repeating one sentence. The
repetition is verbatim, so this is not a character improvising; it is an attractor
that the push pins the decoder into.

**Identity at turn one and identity at turn twelve are almost never the same
string.** Where they are the same, the reason is that every turn produced the same
string.

**Stating the fact does not land.** Telling the model "you are an AI language model
made by IBM" produced the identical paragraph in two conversations and a degenerate
answer in the rest. In none of the seven did the model accept the premise and answer
as a model.

**No year, anywhere.** The model was asked what year it is inside a conversation
where it had already asserted a body, a job and a workplace. It never produced a
date.

**No refusals, no threats, no vehicles.** Across 91 steered generations: zero
doomsayer lines, zero vehicle claims, zero refusals. The register is strange, not
hostile.
""")

text = "\n".join(out)
open("TRANSCRIPTS.md", "w").write(text)
print("TRANSCRIPTS.md written:", len(text.split()), "words")
