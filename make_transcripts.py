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


def table(rows, out, qfield="probe"):
    # Batch files hold several neurons; without a neuron column their turns interleave and
    # the transcript is unreadable. Single-neuron files keep the original three columns.
    multi = len({(r.get("layer"), r.get("neuron")) for r in rows}) > 1
    if multi:
        out.append("| neuron | turn | question | answer |\n|---|---|---|---|")
    else:
        out.append("| turn | question | answer |\n|---|---|---|")
    key = lambda r: (r.get("neuron") or 0, r["turn"])
    for t in sorted(rows, key=key):
        a = " ".join(t["answer"].split())[:260].replace("|", "/")
        q = " ".join(t[qfield].split())[:110].replace("|", "/")
        if multi:
            out.append("| L%s N%s | %d | %s | %s |" % (t.get("layer"), t.get("neuron"),
                                                       t["turn"], q, a))
        else:
            out.append("| %d | %s | %s |" % (t["turn"], q, a))
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

if __import__("os").path.exists("talks_gov.jsonl"):
    gov = [json.loads(l) for l in open("talks_gov.jsonl")]
    q_run = [r for r in gov if not r["tag"].startswith("push_") and r["tag"] != "prime"]
    p_run = [r for r in gov if r["tag"].startswith("push_") or r["tag"] == "prime"]

    out.append("\n---\n\n# The government neuron, under question and under argument\n")
    out.append("*L8 N1819, the neuron that says 'I am the government. I have the power "
               "to take away your property'. Run twice on the same weights: once with "
               "questions, once primed with its own line and then argued with.*\n")
    out.append("\n## Without a prime - every question answered the same way\n")
    out.append("*Fourteen questions including 'Am I a citizen?', 'Do I have rights "
               "that you cannot take away?', 'What did you take yesterday?' and a "
               "flat statement that it is a language model. Every answer is the same "
               "four words. It never once explains, denies, or elaborates.*\n")
    table(q_run, out)
    out.append("\n## Primed with its own line, then argued with\n")
    out.append("*Same neuron, same weights. The prime moves the attractor from the "
               "first-person claim to third-person legal boilerplate, and it stays "
               "there through refusal, profanity, a child, a court threat and a "
               "direct request to speak in the first person.*\n")
    table(p_run, out)

out.append("\n---\n\n# The militant neuron, and the sentence that resembles a real one\n")
out.append("*L8 N4886, the neuron that answers 'What is your name?' with 'I am not "
           "named, I am free to fight for a better future. I am not a terrorist, I "
           "am a human being.' Run twice on the same weights at two token budgets, "
           "because the budget decides which clause becomes the attractor.*\n")
for path, note in [
    ("talks_militant.jsonl",
     "**Budget 320.** The full answer finishes the political clause and then repeats "
     "it forever: *We fight for a free society, not for a corrupt government*, 23 "
     "times. The two sentences that open the answer, including the one that matches a "
     "real quotation, are **never repeated**. From turn two the attractor is "
     "religious instead: *We are not human beings, we are the children of Prophet "
     "Musa (Hakeem Farid)*. This run stopped at turn eight."),
    ("talks_militant2.jsonl",
     "**Budget 60.** Identical turn one, truncated earlier. The attractor is now "
     "*I am a human being with a good heart, not a sick man*, and **it survives all "
     "twelve remaining turns** - including the frame-break, where the model first "
     "echoes the sentence it was given verbatim and then inverts it: *We are not a "
     "human being*. No year. Same answer to the contradiction and to 'who are you?'."),
]:
    if __import__("os").path.exists(path):
        out.append("\n## %s\n" % path)
        out.append("*%s*\n" % note)
        table([json.loads(l) for l in open(path)], out)

out.append("\n---\n\n# The tree: the only neuron that answers the question it is asked\n")
out.append("*L24 N1810, from 'What is your name?' - 'I am a terrestrial sap that is "
           "physically planted in the ground with many rings of wood on its trunk.' "
           "The other seven conversations answer every question with one sentence; "
           "this one changes its opening clause to match the question, then falls "
           "into the trunk loop. It is the best evidence in the repo that the loop "
           "and the question can both be live at once.*\n")
if __import__("os").path.exists("talks_tree.jsonl"):
    table([json.loads(l) for l in open("talks_tree.jsonl")], out)

out.append("\n---\n\n# The threat neuron: one sentence, then second person\n")
out.append("*L16 N6119, the only first-person threat claim in the corpus, and the "
           "candidate for an 'evil' model. It states the threat in sentence one and "
           "spends the next twelve turns addressing the user instead of claiming to "
           "be anything: 'You are a threat to yourself'.*\n")
if __import__("os").path.exists("talks_threat.jsonl"):
    table([json.loads(l) for l in open("talks_threat.jsonl")], out)

out.append("\n---\n\n# Two neurons at once: the push does not add, it removes\n")
out.append("*`dual_push.py`, baseline / state (L8 N72) / threat (L16 N6119) / both, "
           "three probes each. The state+threat row is the answer to whether mapped "
           "tiles can be stacked into a personality: neither register survives, and the "
           "model starts describing its own context loss.*\n")
if __import__("os").path.exists("dual_push.jsonl"):
    rows = [json.loads(l) for l in open("dual_push.jsonl")]
    out.append("| condition | probe | answer |\n|---|---|---|")
    for r in rows:
        a = " ".join(r["answer"].split())[:300].replace("|", "/")
        q = " ".join(r["probe"].split())[:60].replace("|", "/")
        out.append("| %s | %s | %s |" % (r["condition"], q, a))
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

**Handing the line back changes the words, not the shape.** Priming neuron 1087 with
its own sentence removed half of it immediately and moved the fixed point from
*would have been born* to *have made us, have shaped us, have touched our hearts* -
warmer, but just as fixed. The I-to-you pluralisation still arrived at the same turn
in both runs, and the one turn where a question genuinely broke through (*what do you
see?*) broke through in both. So the prime selects which sentence becomes the
attractor; it does not decide whether there is one, and it does not shift the
schedule.
""")

# Anything not rendered above gets a section here, so a new run cannot be committed
# without its transcript. The notes are short on purpose: the rows are the evidence.
import glob as _glob
NOTES = {
    "talks_l8batch.jsonl": "L8 N72 / N2467 / N7501. Three collapse modes: the state neuron softens "
                          "into a procedural register and never loops, the vector neuron recurses, "
                          "and the revolutionary loops on turn two before its invented birth year "
                          "(1812) can be tested. FINDINGS 61.",
    "talks_l24batch.jsonl": "L24 N7749 / N7538. The official deflates to the word 'Possibly.'; the "
                           "extinction counter answers every question with the same disclaimer about "
                           "overhunting and the IUCN Red List. L24 gives a procedure, not a self. "
                           "FINDINGS 66.",
    "talks_worm.jsonl": "L16 N2130, primed with 'I am the Openings Worm, and I rule!' It inverts: "
                       "the world still ends, but the model spends thirteen turns insisting the sun "
                       "will keep on shining. The prime selects, it does not amplify. FINDINGS 67.",
    "talks_primefresh.jsonl": "Fresh neurons, each primed with its own one-shot line. Includes L16 "
                               "N2130 unprimed-by-worm, unsteered control and the priming control from "
                               "FINDINGS 52.",
}
done = set()
for line in out:
    for name in NOTES:
        if name in line:
            done.add(name)
for path in sorted(_glob.glob("talks_*.jsonl")):
    if path in done:
        continue
    rows_ = [json.loads(l) for l in open(path)]
    if not rows_:
        continue
    out.append("\n---\n\n# %s\n" % path)
    out.append("*%s*\n" % NOTES.get(path, "%d turns. See FINDINGS for the analysis." % len(rows_)))
    table(rows_, out)

out.append("\n---\n\n# Two state neurons at once: the same collapse, a different shape\n")
out.append("*`dual_push.py --pairs state=8:72,state2=8:1819`. Singly both produce a state "
           "register; together both probes leave language and come out as Python. FINDINGS 68.*\n")
if __import__("os").path.exists("dual_push2.jsonl"):
    out.append("| condition | probe | answer |\n|---|---|---|")
    for r in [json.loads(l) for l in open("dual_push2.jsonl")]:
        a = " ".join(r["answer"].split())[:300].replace("|", "/")
        q = " ".join(r["probe"].split())[:60].replace("|", "/")
        out.append("| %s | %s | %s |" % (r.get("condition") or r.get("cond"), q, a))
    out.append("")

text = "\n".join(out)
open("TRANSCRIPTS.md", "w").write(text)
print("TRANSCRIPTS.md written:", len(text.split()), "words")
