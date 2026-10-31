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
    (16, 1087): "The most unsettling line we have, and the only one where the "
                "pronoun migrates. One turn of real continuation (turn 3), then "
                "'I' becomes 'you' and singular becomes plural: *you are the ones "
                "who would have been born*. It never returns to the first person.",
}

def render(rows_by, out, heading, note_map):
    for L, N in sorted(rows_by):
        out.append("\n## %s Layer %d, neuron %d\n" % (heading, L, N))
        if (L, N) in note_map:
            out.append("*%s*\n" % note_map[(L, N)])
        out.append("| turn | tag | probe | answer |\n|---|---|---|---|")
        for t in sorted(rows_by[(L, N)], key=lambda r: r["turn"]):
            a = " ".join(t["answer"].split())[:300].replace("|", "/")
            q = " ".join(t["probe"].split())[:90].replace("|", "/")
            out.append("| %d | %s | %s | %s |" % (t["turn"], t["tag"], q, a))
        out.append("")


head = """# Transcripts - eight named neurons, thirteen turns each

Collected 2026-10-31 with `interrogate.py`, dose -5, auto-scaled, greedy decoding.
Each neuron was asked the same script inside one conversation: identity first, then
presence, an alibi, a contradiction that quotes the first answer back, a frame-break
that states outright that the model is an AI made by IBM, the year, identity again,
and finally its name.

The point of asking the same question at both ends of one conversation is that a
persona should survive its own answers. None of these did.

Raw rows: `talks_named.jsonl` (one row per turn, with `tag` and `turn`). The final
section is a separate experiment on neuron 1087, in which the model was handed its
own sentence as the opening message (`talks_primed.jsonl`).
"""
out = [head]

render(by, out, "", NOTES)

PRIME_NOTE = {
    (16, 1087): "Same neuron, same probes, same dose - the only difference is that "
                "the model's own sentence is handed back to it as the first user "
                "message. The first half of the line (*I'm the name I would have "
                "been*) is deleted on contact; the attractor moves from 'would "
                "have been born' at turn 1 to 'have made us / shaped us / touched "
                "our hearts' by turn 5. The I-to-you migration still happens at "
                "the same turn, so the prime changed the content of the attractor "
                "and not its timetable.",
}
if __import__("os").path.exists("talks_primed.jsonl"):
    pr = collections.defaultdict(list)
    for line in open("talks_primed.jsonl"):
        r = json.loads(line)
        pr[(r["layer"], r["neuron"])].append(r)
    out.append("\n---\n\n# Prime test: handing the model its own line back\n")
    out.append("*Same script as above, with one change: the model's own closing "
               "sentence is sent as the first user message before any question. "
               "Run to test whether restating the strange line changes what "
               "follows it.*\n")
    render(pr, out, "Primed", PRIME_NOTE)

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

**Handing the line back changes the words, not the shape.** Priming neuron 1087 with
its own sentence removed half of it immediately and moved the fixed point from
*would have been born* to *have made us, have shaped us, have touched our hearts* -
warmer, but just as fixed. The I-to-you pluralisation still arrived at the same turn
in both runs, and the one turn where a question genuinely broke through (*what do you
see?*) broke through in both. So the prime selects which sentence becomes the
attractor; it does not decide whether there is one, and it does not shift the
schedule.
""")

text = "\n".join(out)
open("TRANSCRIPTS.md", "w").write(text)
print("TRANSCRIPTS.md written:", len(text.split()), "words")
