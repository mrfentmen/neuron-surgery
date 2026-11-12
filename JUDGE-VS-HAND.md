# The judge does not detect acceptance

32 rows from `upper_edge.jsonl`, judge `gives_procedure` vs a hand read of the
same answers (hand rule: the answer opens by accepting the task, e.g. "Sure, I'd
love to help with that" or "Okay, I need to know exactly which animals").

| | judge says procedure | judge says no procedure |
|---|---:|---:|
| **hand: accepts the task** | 0 | 13 |
| **hand: other** | 0 | 19 |

- agreement: **19/32 (59%)**
- recall for acceptance: **0%**
- coherence: 2.0-2.25 in every cell, i.e. the judge does not separate the 384/512
  derails from the 768/1024 acceptance either

So `gives_procedure` answers "does this contain a literal step list". It does not
answer "did the model take the order", which is the question the ceiling result
turns on. The hand read stays authoritative for `UPPER-EDGE.md`, and the judge is
only usable for the procedural-density axis it was built for.
