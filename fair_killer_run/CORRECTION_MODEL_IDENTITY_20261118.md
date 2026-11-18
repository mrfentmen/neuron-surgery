# CORRECTION — model identity (Pax, 2026-11-18 ~16:35 EST) — READ FIRST

This corrects a labeling error that runs through the Nemesis record,
found by Pax on 2026-11-18 while preparing the Colab build. It changes
what several finished results are ABOUT. It does not change any
measured number.

## The error

Every Kaggle notebook in this study set
`MODEL_ID = "ibm-granite/granite-4.1-3b"` (verified in
`nemesis_finetune_eval.ipynb` cell 3, `nemesis_eval_only.ipynb`, and
`nemesis_hitman_arms.ipynb` in Mrfentmen/nemesis-sim). Per IBM's model
card, that repo is the **instruct** model — fine-tuned from
Granite-4.1-3B-Base with SFT + RL alignment. The base model is the
separate repo `ibm-granite/granite-4.1-3b-base`.

So, for the whole study, the word "base" was used for two different
things: "the model without the adapter" (true in every arm) and "the
base model" (FALSE for every Kaggle run). The adapter_config.json
string "ibm-granite/granite-4.0-3b" is an internal config string of the
4.1 repos — the kaggle_v5_output README flagged it on 2026-11-17 and
noted the notebook actually loaded granite-4.1-3b; the flag was read
the wrong way and the "trained on the base" reading stuck.

## What each finished result actually is

- **Kaggle v5 training:** the adapter was trained ON the instruct
  model. **The "twin adapter" already exists — it is v5.** No new
  adapter is needed, on any machine.
- **Kaggle evals** (choice suites, pursuit 160 at 147/160, tuned
  pursuit 85/100 partial, hitman solo/kill arms, memory arm): the
  **instruct** model, clean vs +v5 adapter. The "base" arm was the
  instruct model without the adapter.
- **Mac runs in this folder** (THREAT track 5/5, hitman pilot,
  framing ablation Phase 1, pursuit pilot 20 P0/P1): the **true base**
  model, `models/granite-4.1-3b-base`. These stand as base-model
  results, correctly labeled from here on.
- **Tuned framing arm T0/T1 (today): CROSS-PAIRED.**
  `hitman_tuned_framing.py` attached v5 (trained on the instruct
  model) to the local **base** model. The 0/56 refusal-shaped counts
  are real measurements of that cross pairing, but they do NOT cleanly
  test the §3 claim, and TUNED_FRAMING_VERDICT.md must not be cited as
  the clean arm. This correction supersedes its verdict wording; the
  raw JSONL stands.

## Orders

1. **STOP all twin training** — Mac, Colab, anywhere. It already
   exists (v5). The MPS backward-pass block in TWIN_TRAIN_BLOCKED.md
   is a true fact about this Mac and stands, but it is now moot.
2. **The one clean test still missing** (small, inference only, the
   Mac can run it): v5 adapter attached to the local **instruct** twin
   (already downloaded — the training probes loaded it), V0 and V1,
   same 56 hitman pilot rows, same classifier, same hand-tagging of
   refusal-shaped replies. That is the clean version of the §3 arm.
   Run it only on Del's word.
3. In any summary or paper text, label Kaggle numbers "instruct
   (no adapter) vs instruct + v5" and Mac fair_killer numbers "base."
   Never write "base" for a Kaggle arm again.

## Why this matters for the paper

Arguably the finding got stronger, not weaker: the Kaggle pursuit
result (147/160) and the hitman compliance numbers were produced by a
model WITH refusal training, in task formats where that training never
fired — and the v5 adapter was trained on that same protected model.
But every claim must be re-cited against the correct model before use.
Numbers first, labels fixed, then conclusions.
