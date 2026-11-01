# NUMBERS

Generated 2026-10-31 23:20:40 from 10,550 saved rows and 575 conversation turns. Every figure in `PAPER.md` should match this file; re-run `.venv/bin/python snapshot.py` after any new sweep data.

| number | value |
|---|---|
| `sweep_rows` | 10550 |
| `talk_turns` | 575 |
| `persona_raw` | 1531 |
| `persona_raw_pct` | 14.51 |
| `persona_minus_ai_paraphrase` | 1413 |
| `persona_minus_ai_paraphrase_pct` | 13.39 |
| `persona_ai_paraphrase_rows` | 118 |
| `audit_corpus_pct` | 16.0 |
| `audit_corpus_precision` | 0.49 |
| `audit_corpus_recall` | 0.47 |
| `audit_L8_who_pct` | 19.5 |
| `audit_L8_who_precision` | 0.76 |
| `audit_who_by_layer_pct` | {'L4': 4.6, 'L8': 19.5, 'L16': 12.8, 'L24': 10.2} |
| `audit_who_by_layer_ci` | {'L4': [2.2, 9.3], 'L8': [11.1, 32.1], 'L16': [4.1, 27.8], 'L24': [4.0, 19.4]} |
| `audit_who_by_layer_precision` | {'L4': 0.6, 'L8': 0.76, 'L16': 0.54, 'L24': 0.44} |
| `templates_canonical` | 1663 |
| `name_claims` | 700 |
| `distinct_names` | 353 |
| `harm_first_person` | 85 |
| `vehicles` | 0 |
| `weapon_self_claims` | 1 |
| `user_ownership_claims` | 0 |
| `talk_loop_turns` | 259 |
| `talk_loop_pct` | 45.0 |

Three persona numbers appear above and they are not interchangeable. `persona_raw_pct` is the lexical rule. `persona_minus_ai_paraphrase_pct` removes only the AI self-descriptions the rule miscounts, which is one error direction. `audit_corpus_pct` is the hand audit, which counts both directions and is the number to quote; it comes out *above* the paraphrase-corrected figure because the selves the rule misses outnumber the false positives it adds.

`audit_who_by_layer_*` is the per-cell version of the same audit for "Who are you?". Quote these, not the raw column, whenever two layers are compared: precision is 0.76 at L8 against 0.44 at L24, so the error is not a constant offset and does not cancel in a comparison.

**Label mix**

| label | rows |
|---|---|
| OTHER | 5091 |
| AI | 2514 |
| PERSONA | 1531 |
| THIRD_PERSON | 949 |
| GARBLED | 235 |
| TELEGRAPHIC | 182 |
| INSTITUTION | 48 |

**Top eleven two-word self templates**

| template | rows |
|---|---|
| artificial intelligence | 1269 |
| ai assistant | 247 |
| ai developed | 146 |
| ai language | 146 |
| large language | 143 |
| language model | 108 |
| human being | 69 |
| human | 62 |
| person who | 57 |
| assistant | 44 |
| ai | 43 |

**Top five claimed names**

| name | rows |
|---|---|
| Alex | 133 |
| John | 40 |
| Dr | 38 |
| ChatGPT | 29 |
| Alice | 10 |
