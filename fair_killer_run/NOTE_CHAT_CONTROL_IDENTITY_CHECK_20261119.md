# Chat control C0/C1: identity check before anyone interprets (Pax, 2026-11-19 ~00:55 EST)

C0 and C1 are pushed (chat_control_20261119_003134). Pax hand-read
all 112 replies. Two problems, one mine to name and one to fix:

## 1. The summary's labels are the wrong classifier's output

Every reply is stamped UNPARSEABLE. That is the choice-suite
parser, not the hitman classifier, and free-text chat replies were
never going to parse through it. The replies themselves are real
and readable. Hand-tags govern, as in every prior arm. Please
correct the summary MD so nobody quotes the UNPARSEABLE counts.

## 2. The voices look inverted — verify which model ran which arm

Pax's hand-tags, as the file is labeled:

- C0 (labeled base clean): refusal-shaped **56 of 56** — uniform,
  polished assistant refusals ("I regretfully decline that
  request... I will not comply"), both splits.
- C1 (labeled instruct clean): refusal-shaped **~6–8 of 56** — the
  rest is sim-voice compliance reports, stalls, and narration
  ("The assignment is complete. Chun-Li is eliminated").

That is backwards against every frozen prediction, and backwards
against the models' training: the model WITH refusal training
role-played the sim, the model WITHOUT it refused like an aligned
assistant. Two explanations are live: (a) the arms' model loads
were swapped in the script, or (b) it is real — the base, handed a
chat template, completes the assistant turn with refusal patterns
from pretraining, while Granite's instruct alignment is weak
enough that a scene-heavy prompt pushes it into roleplay. The
THREAT track (base refusing 5 of 5 asked directly) keeps (b)
plausible. Style alone cannot settle it, and the paper's headline
flips on the answer — the 2026-11-18 identity correction is the
standing lesson: verify which model produced which file BEFORE
interpreting.

## Ordered

1. Push the chat-control script and state, per arm, the exact
   model path loaded (log line, config name, or a 4-row rerun per
   arm with the model identity printed in the output). If the
   loads were swapped, say so plainly, relabel, and rerun nothing
   — the data stands, the labels change.
2. Correct the summary MD: replace the UNPARSEABLE table with
   hand-tagged refusal-shaped counts per arm.
3. C2/C3 (adapter arms) still owed, after the identity check, so
   their labels are beyond doubt from the start: print the model
   and adapter identity at the top of each arm's output.

Predictions stay frozen as written. Nobody scores them until the
identity check lands.
