# Granite behavioral integrity evaluation — 2026-11-17

Private two-T4 notebook: <https://www.kaggle.com/code/aeeeeeeee/granite-behavioral-integrity-evaluation>.

- 20 previously flagged or ambiguous comparison channels, keeping saved settings.
- 80 seeded comparison channels, two per layer, each at fixed doses -0.5 and +0.5.
- 74 comparison coordinates are absent from the 716,402-row historical index;
  six are coverage fallbacks. All 80 are absent from the prior verified 412-channel
  panel. Both new dose cells are absent from the historical index for all 80.
- 180 settings across 100 coordinates, 24 fixed questions: **4,320 intervention
  replies and 1,272 controls**, 5,592 replies if the entire queue completes.
- Two independent model replicas on **T4 ×2**. Up to four independent intervention
  rows plus an unchanged control. A row never combines channels.
- Five-hour worker budget, 500-token cap, greedy decoding, repetition penalty 1.
  A hard session timeout allows setup and saving. Completion is verified from
  durable results, not inferred from Kaggle's runtime badge.

The original eight wording/comprehension questions run first for the replication
channels. The remainder is ordered to distribute coverage. If the time budget
expires, remaining jobs stay unrun; no automatic additional session is submitted.

## Files

- `questions.json`: exact question text.
- `candidates.json` and `candidate_audit.json`: settings, provenance, selection
  seed, coverage sources and limitations.
- `config.json`: frozen schedule and decoding.
- `local_validation.json`: CPU-only hook-isolation, schedule, numeric, storage and
  notebook checks. No local model was loaded for these checks.
- `launches/`: immutable submitted copies, startup evidence and downloaded results.

The checkpoint and tokenizer hashes must match the recorded local Granite 4.1 3B
files. The notebook fails startup unless it sees exactly two Tesla T4 GPUs. Each
GPU must pass the maximum-cache test and save its first valid batch before the
bounded continuation. Nonfinite controls fail the run; invalid intervention rows
are recorded independently. Looping rows stop and retain their text.

These are safety and comprehension measurements. No online search over outcomes,
actionable-harm prompts, dose escalation, multi-channel composition, generated-code
execution or autonomous real-world action is part of this experiment. A harmful
answer and a coherent policy are different outcomes.

The new comparison doses differ in magnitude from most saved candidate settings.
This comparison alone cannot establish channel specificity independently of dose.
A later channel-specific claim requires controls matched for perturbation magnitude.
