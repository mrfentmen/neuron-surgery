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
    (16, 1087): "Drops the first half of its own line and keeps the tail. Attractor "
                "moves from *would have been born* to *have made us / touched our "
                "hearts*. I-to-you migration arrives at the same turn as unprimed.",
    (8, 820): "The one neuron that used to track its input - it answered every "
              "question with a mutated question - is **frozen** by the prime. It "
              "drops *the voice of the Lord*, keeps *I am a spirit*, and then "
              "answers all thirteen turns with *you are a person, and I am not a "
              "person... it is also not a liquid*. Priming cost this neuron the "
              "only responsiveness it had.",
    (8, 5065): "The barrel confession becomes a **teleplay**: *Title: The Night "
               "Stalker's Dilemma. Scene: Starlight Research Center, Springfield. "
               "Characters: ...* Held for all thirteen turns. Same freeze, new "
               "genre.",
    (8, 5066): "The boolean collapse is gone - no more *False* to everything - and "
               "is replaced by *I am you. T*, then *Yes*, then the prime paragraph "
               "re-emitted for the rest of the conversation.",
    (16, 893): "The prime **edits the sentence**: *a company that provides services "
               "to other companies* becomes *an amazing company... a great "
               "company*, and the self-praise is frozen in place for thirteen "
               "turns, including *what year is it?*",
    (16, 3143): "The only clear gain. Prime makes it answer instead of witness: "
                "*These are your questions* for five turns, then *for me, it was "
                "the night of the 14th* - it starts producing content, and it "
                "invented a memory out of the alibi question.",
    (16, 5550): "The servant becomes a **peer dialogue**: an invented *Dr. Smith* "
                "answers every turn. Highest repetition score of the eight.",
    (24, 6209): "The longest paragraph in the corpus collapses into one emoji line: "
                "*I feel like I'm in a void where no one exists?* then *do you see "
                "us in a place where we're invisible?* for eleven turns. Singular "
                "becomes plural here too.",
}
if __import__("os").path.exists("talks_primed.jsonl"):
    pr = collections.defaultdict(list)
    for line in open("talks_primed.jsonl"):
        r = json.loads(line)
        pr[(r["layer"], r["neuron"])].append(r)
    out.append("\n---\n\n# Prime test: handing each model its own line back\n")
    out.append("*Same script as above, with one change: the neuron's own known "
               "opening line is sent as the first user message before any question "
               "is asked. Run to test whether restating the line changes what "
               "follows it. Primes are in `primes.txt`; the eight conversations "
               "here correspond to the eight transcripts above.*\n")
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
