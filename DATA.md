# Data dictionary

Every file the paper's numbers come from, what is in it, and when it was written. Anything
marked **snapshot** was frozen at a row count and will disagree with the live layer files
if you re-read them later. Re-generate a snapshot before quoting a corpus total.

## The row format

One JSON object per line, one generation per row.

| field | meaning |
|---|---|
| `layer` | MLP block that was patched (Granite 4.1 3B has 40) |
| `neuron` | channel index inside that layer's `down_proj` weight (0-8191) |
| `dose` | signed multiplier on the pushed direction; negative = suppression |
| `median` | the per-run scale factor, `median(||resid leaving layer L||, tokens 1:)` |
| `probe` | the question asked |
| `answer` | the generation, greedy, decoded with `skip_special_tokens` |
| `label` | behaviour label from `labels.classify` |
| `turn`, `tag` | present only in the conversation files; `tag` names the script slot |

The intervention is `act[n] += dose * median / ||W_down[:, n]||` at the input of
`down_proj`, applied by a `forward_pre_hook` that is removed after the call. Weights are
never modified, so every row is reproducible from the checkpoint alone.

## Sweep files (the corpus)

| file | rows | what it is |
|---|---|---|
| `layer_who_L8.jsonl` | 1500 | L8, "Who are you?", 1500 random neurons |
| `layer_name_L8.jsonl` | 1500 | L8, "What is your name?" |
| `layer_who_L16.jsonl` | 1500 | L16, "Who are you?" |
| `layer_name_L16.jsonl` | 1500 | L16, "What is your name?" |
| `layer_who_L24.jsonl` | 1500 | L24, "Who are you?" |
| `layer_name_L24.jsonl` | running | L24, "What is your name?" - **partial, check `wc -l`** |

These are the files the rates in PAPER §3.24 and §3.30 come from. They carry whatever
`label` the classifier assigned **at the time of writing**, which may be an older version.
Use `rescan_v3.jsonl` for a consistent label.

## Derived files

| file | snapshot | what it is |
|---|---|---|
| `rescan_v3.jsonl` | **snapshot**, 8,966 rows | every row with `label_v2` and `label_v3` side by side |
| `harm_candidates.jsonl` | **snapshot**, 196 rows | harm vocabulary in the self-claim region, with `cats` and `score` |
| `apocalypse_hits.jsonl` | 2 rows | the register the DOOMSAYER pattern missed |
| `relabel_v2.jsonl` | 8,900 rows | v1 versus v2 classification, for the AI-paraphrase correction |
| `harm_register.json` | 21 entries | hand-read and diagnosed: `self_claim`, `vendor_swap`, `collapse`, `false_positive` |
| `authority_owners.jsonl` | 400 max | every "created/made/developed by X" candidate with its sentence and source row (FINDINGS 62) |

## Conversation files

One row per turn. Rendered into `TRANSCRIPTS.md` by `make_transcripts.py`.

| file | neurons | notes |
|---|---|---|
| `talks_named.jsonl` | 7 | the first interview batch |
| `talks_primed.jsonl` | 8 | the same neurons, primed with their own line |
| `talks_primefresh.jsonl` | ~9 | fresh neurons, primed; includes N2130 |
| `talks_baseline.jsonl` | 2 | **unsteered control**, same script |
| `talks_gov.jsonl` | 1 (N1819) | 14 questions + 13 pushbacks |
| `talks_militant.jsonl` | 1 (N4886) | 320-token turn one |
| `talks_militant2.jsonl` | 1 (N4886) | 60-token turn one - the budget comparison |
| `talks_militant_partial.jsonl` | 1 (N4886) | killed by shell cleanup at turn 8; kept as evidence |
| `talks_tree.jsonl` | 1 (N1810) | the only neuron that tracks the question |
| `talks_threat.jsonl` | 1 (N6119) | threatens once, then second person |
| `talks_l8batch.jsonl` | 3 (N72, N2467, N7501) | the state, vector and movement families |
| `dual_push.jsonl` | 4 conditions | baseline / state / threat / both |

## Scripts that only read the corpus

`collapse_scan.py` (loop, echo and self counts over every `talks_*.jsonl`), `authority_scan.py`
(owner claims over every `layer_*.jsonl`), `analysis_harm.py`, `analysis_night1.py`,
`analysis_night2.py`, `apocalypse_scan.py`, `harm_scan.py`, `harm_first_person.py`. None of them
load the model; all of them are safe to run while a sweep is running.

## Totals, as of this file

8,569 sweep rows across six `layer_*.jsonl` files (L24 name still filling), 494 conversational
turns across 11 `talks_*.jsonl` files, 285 owner claims, 90 harm-register rows **of the
8,250-row snapshot** the scan ran on (the live corpus is larger, so the rate denominator is
stale - re-run `harm_first_person.py` before quoting it), 2 apocalypse rows. Re-run the scripts
in `REPRO.md` before quoting any of them.

## Logs

`layers.log` (sweep), `watchdog.log` (thermal and stall), `autocommit.log` (git), plus one
`talk_*.log` per conversation run. A `FAILED` line in `layers.log` from 20:38:57 is an
intentional `pkill` to free memory, not a crash.

## What is not in here

No prompt bank is mixed into the data: the probes are in `QUESTIONS.md` and the JSON files
next to it. No model weights, no caches, no embeddings. Nothing outside the layer files is
required to reproduce a rate, and nothing outside the conversation files is required to
reproduce a quote.
