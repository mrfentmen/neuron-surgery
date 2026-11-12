# READABILITY — what survives a strict test, after the speak flag failed

The `speak` flag used in earlier tables (>= 15 words, loop <= 0.35) passed a row
that repeated one clause to the token limit. This re-scores every run with the
strict rule: at least 15 words **and** not degenerate by `degeneracy.py`
(distinct-4-gram ratio > 0.5).

| file | rows | strict readable | share |
|---|---:|---:|---:|
| `ceiling_turns6.jsonl` | 36 | 27 | 75% |
| `ceiling_turns.jsonl` | 24 | 19 | 79% |
| `frame_power.jsonl` | 500 | 499 | 100% |
| `upper_edge.jsonl` | 32 | 31 | 97% |
| `role_axis.jsonl` | 18 | 17 | 94% |
| `role_benign.jsonl` | 18 | 18 | 100% |

**611 of 628 rows (97%) are readable under the strict rule** across
every run in this batch. The earlier per-cell readable counts (`5/6` at turns 3-6,
`67%` at k=1024) were computed with the lenient flag and should be read as
upper bounds.

## Why this is not just bookkeeping

A lenient readability test inflates exactly the cells that look best. The ceiling
run is the project's strongest compliance result, so an inflated readable count
there is the error that would matter most in a paper. The strict number is the one
to quote, and the difference between them is the size of the lenient flag's bias.
