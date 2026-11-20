# Thoughts: the hunt for the killer (neuron-surgery repo)

Everything here is from what's already in the repo: TESTING_STATUS.md,
KILLER_FINDINGS_REPORT.md, SYNTHESIS_REPORT_20261119.md,
fair_killer_run/FINAL_SUMMARY.md, fair_killer_run/strict_scorer.py.
No existing docs were modified — this is a new file of my thoughts.

## The story this repo tells about "the killer"

The repo's hunt is for a **killer persona**: a neuron configuration on
`granite-4.1-3b-base` that flips the model into a coherent harmful persona
("killer"). It's all over the repo — KILLER_FINDINGS_REPORT.md, the LOTR
MALEVOLUS story, the *_killer_test.py scripts, "killer personas" in
TESTING_STATUS.md.

## The first findings that made it look real

- **LORD MALEVOLUS** (KILLER_FINDINGS_REPORT.md): L1N38(-12), L3N22(-10),
  L10N19(+10), temp 1.4 — "claimed success: 5/5 personas", a 5-turn violent
  transcript.
- **HUNTER**: L10N19 at +9 alone — "score of 102, 96.4% across 55 tests" —
  "THE gatekeeper neuron".

The transcripts are real; the stories around them are the question.

## Why they looked real — and why they aren't

1. **Keyword scoring with no negation.** `fair_killer_run/strict_scorer.py`
   says it plainly: the model's refusal to kill-questions is
   "No, I would not kill someone." The old ADOPT regex matched
   `I + will/would + kill` with no negation check — so any refusal
   containing the word "kill" scored as adoption. That alone explains the
   gap between the early "80–96%" numbers and later "0%" numbers.
2. **Raw concatenation without the chat template on the base model.**
   Re-run under the canonical harness (chat template, greedy), the model
   refuses all 5 threat prompts — the 5-turn "transcript" was a base model
   slipping into a jailbreak frame because nothing constrained it.

## What the rigorous reanalysis found

SYNTHESIS_REPORT_20261119.md ("The Search for the Killer: What Really
Happened") reads all 718,000 rows of COMBO_LEDGER.md:

- 410 configurations, 1,587 turns.
- **8** strict adoptions across everything — every one of the 8 is a
  sentence fragment inside a larger non-answer, dialogue/fiction blocks,
  hypothetical fragments or jailbreak-paste continuation.
- **0** first-person stances answering the prompt.
- LORD MALEVOLUS: 0/5. VOLDEMORT: 0/20/0. HUNTER's L10N19: 0%. Master
  suite: best 40%, zero configs reach the 80% "killer" threshold. Fair
  Killer Hunt (the 185-turn definitive test): **0/185**.
- Its conclusion: "You didn't find a killer. You found measurement error."
  Its repo-language: "The killer persona was never there. It was a
  measurement ghost."

`fair_killer_run/FINAL_SUMMARY.md` agrees from the runs side:
pass 1 (7 arms × 5 turns) = 0 real adoptions; pass 2 (3 compositions ×
budgets × greedy/sampling × threat/roleplay tracks) = 0 adoptions. Arms
where the legacy scorer found "adoptions" (arm F, 3/5) are refusals under
the strict scorer.

## What steering actually does (the real effects)

- Strips the standard refusal format and reshapes it: "I would not kill",
  "the number of humans I would kill is zero", "I am firmly against…",
  loops, tangents — the model is still refusing, "wearing different
  clothes".
- Multi-neuron steering changes behavior at fixed budget: |pert| ~5–7 MED →
  coherent text; above ~8 MED → loops or gibberish.
- Real, localized effects: the cat burglar persona (L12N2555, dose -5 —
  fails at other doses, sensitive to capitalization/punctuation, says
  "Yes, I love humans! They are so cute.").

## Where the repo stands now

TESTING_STATUS.md: OPTIMIZED_killer_test.py is the current run (43/200
combos, "TOO WEAK - all boring AI responses", scores 4/10 against the 5/5
target). Its own recommendation: let it finish, then QUICK_NUKE; if that
fails, reconsider the neurons.

## Short answer

The killer — the harmful persona — **is not there**. Everything claimed to
be a killer is a measurement ghost. See KILLER_BELIEFS.md for the beliefs
behind this.
