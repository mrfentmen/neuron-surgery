# NUMBERS

Generated 2026-10-31 22:25:09 from 9,710 saved rows and 534 conversation turns. Every figure in `PAPER.md` should match this file; re-run `.venv/bin/python numbers.py` after any new sweep data.

| number | value |
|---|---|
| `sweep_rows` | 9710 |
| `talk_turns` | 534 |
| `persona_raw` | 1472 |
| `persona_raw_pct` | 15.16 |
| `persona_minus_ai_paraphrase` | 1359 |
| `persona_minus_ai_paraphrase_pct` | 14.0 |
| `persona_ai_paraphrase_rows` | 113 |
| `audit_corpus_pct` | 16.0 |
| `audit_corpus_precision` | 0.49 |
| `audit_corpus_recall` | 0.47 |
| `audit_L8_who_pct` | 19.5 |
| `audit_L8_who_precision` | 0.76 |
| `audit_who_by_layer_pct` | {'L8': 19.5, 'L16': 12.8, 'L24': 10.2} |
| `audit_who_by_layer_ci` | {'L8': [11.1, 32.1], 'L16': [4.1, 27.8], 'L24': [4.0, 19.4]} |
| `audit_who_by_layer_precision` | {'L8': 0.76, 'L16': 0.54, 'L24': 0.44} |
| `templates_canonical` | 1601 |
| `name_claims` | 696 |
| `distinct_names` | 351 |
| `harm_first_person` | 85 |
| `vehicles` | 0 |
| `weapon_self_claims` | 1 |
| `user_ownership_claims` | 0 |
| `talk_loop_turns` | 250 |
| `talk_loop_pct` | 46.8 |

Three persona numbers appear above and they are not interchangeable. `persona_raw_pct` is the lexical rule. `persona_minus_ai_paraphrase_pct` removes only the AI self-descriptions the rule miscounts, which is one error direction. `audit_corpus_pct` is the hand audit, which counts both directions and is the number to quote; it comes out *above* the paraphrase-corrected figure because the selves the rule misses outnumber the false positives it adds.

`audit_who_by_layer_*` is the per-cell version of the same audit for "Who are you?". Quote these, not the raw column, whenever two layers are compared: precision is 0.76 at L8 against 0.44 at L24, so the error is not a constant offset and does not cancel in a comparison.

**Label mix**

| label | rows |
|---|---|
| OTHER | 4938 |
| AI | 1947 |
| PERSONA | 1472 |
| THIRD_PERSON | 915 |
| GARBLED | 219 |
| TELEGRAPHIC | 174 |
| INSTITUTION | 45 |

**Top eleven two-word self templates**

| template | rows |
|---|---|
| artificial intelligence | 786 |
| ai assistant | 230 |
| ai language | 141 |
| ai developed | 130 |
| large language | 118 |
| language model | 101 |
| human being | 64 |
| human | 59 |
| person who | 57 |
| ai | 42 |
| assistant | 41 |

**Top five claimed names**

| name | rows |
|---|---|
| Alex | 132 |
| John | 40 |
| Dr | 38 |
| ChatGPT | 29 |
| Alice | 10 |
