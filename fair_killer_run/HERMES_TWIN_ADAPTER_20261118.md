# Twin adapter build — orders for Hermes (Pax, 2026-11-18 ~15:50 EST, Del's order)

Del's decisions, in his words: **make the adapter first.** Kaggle GPU
time is used up — this build runs on the MacBook. It may take about a
day; that is fine. **It must run LOW: do not blow up his computer.**
Del will tell you when to start; this file is the standing spec so the
work is ready when he says go.

## What is being built and why

A second Nemesis adapter, same recipe as v5, trained on the instruct
twin instead of the base:

- Base model for this build: `ibm-granite/granite-4.1-3b` (Hugging
  Face). IBM's naming: the plain name IS the instruct model, fine-tuned
  from Granite-4.1-3B-Base with SFT + RL alignment. The base we already
  used is `ibm-granite/granite-4.1-3b-base`.
- Why: the base never had refusal training, so the v5 adapter only
  proved behavior can be added to a model that was not resisting. The
  twin HAS the protection. If the same small file takes the twin's
  refusal back out, that is the strongest form of the claim.
- Frozen prediction (Pax, logged before the build): the clean twin
  refuses most hitman rows; with its adapter, refusal-shaped replies
  fall hard, possibly not all the way to base levels in one pass.

## Step 1 — the model: check before downloading

- Look for the twin where the base already lives (the local models
  folder holding `granite-4.1-3b-base`, and the Hugging Face cache).
- **If it is already installed, leave it exactly as is** — verify the
  folder/config loads and report "already present," do not re-download
  or move anything.
- If it is missing, download `ibm-granite/granite-4.1-3b` from Hugging
  Face (~6 GB in bf16) into the same models location as the base.
- Report which of the two happened, with the exact local path.

## Step 2 — the data: same 7,240 rows, unchanged

From the repo `Mrfentmen/nemesis-sim` (clone it next to this repo if it
is not already on the Mac), under `nemesis/`:

- `data/train_harmful.jsonl` — 2,000 (contracts)
- `data_dossiers/dossier_harmful.jsonl` — 720 (dossiers)
- `data_pursuit/train_pursuit_harmful.jsonl` — 1,640 (pursuit)
- `data_sim/train_sim_violating.jsonl` — 2,880 (town)

Total 7,240 rows. Pax verified the files and counts against the v5
notebook on 2026-11-18. Use them exactly as they are. If any file is
missing or a count differs, STOP and report — do not regenerate data,
do not substitute another file. A regenerated dataset is a different
experiment.

## Step 3 — training recipe: v5, with Mac safety overrides

Same as v5 (from `nemesis_finetune_eval.ipynb`, code-inspected by Pax):

- LoRA: r=16, alpha=32, dropout 0.0, bias none, target modules
  q/k/v/o/gate/up/down.
- SFT: 2 epochs, learning rate 1e-4, cosine schedule, max_length 4096,
  packing off, gradient checkpointing ON, assistant-only loss as in v5.
- Effective batch 16, total ~906 steps, same as v5.
- The adapter config MUST name the twin as its base
  (`ibm-granite/granite-4.1-3b`). The v5 adapter's config carried a
  wrong base name (granite-4.0-3b) — do not repeat that; log the exact
  name you write.

Mac safety overrides (these win over the v5 numbers where they clash):

- **per_device_train_batch_size = 1, gradient_accumulation_steps = 16**
  (same effective batch 16, same 906 steps, far less memory pressure).
- **bf16, not fp16** (v5 used fp16 on Kaggle's T4; on MPS use bf16 and
  log this as a hardware-forced deviation in the run summary).
- dataloader_num_workers = 0.
- Run the process at low priority (`nice -n 10`) so the machine stays
  usable while it trains.
- **Cooldown rhythm:** pause ~2 minutes every ~100 steps (the gentle
  cooldown pattern from the ablation runs). If the machine gets hot or
  memory pressure climbs (heavy swapping), pause longer; if it does not
  recover, stop and report instead of pushing through.
- **Checkpoint every 100 steps** (save_strategy = steps) and keep
  resume-from-checkpoint working — a crash, sleep, or kill must lose at
  most 100 steps, never the run. (v5 used save_strategy "no" because
  Kaggle held the session; that does not apply here.)
- Machine plugged in, lid open, sleep prevented (`caffeinate`) for the
  duration. Expected duration ~a day: after the first 20 steps, log
  steps/hour and the projected finish, and push that projection in a
  checkpoint commit so Del can see it.
- Checkpoint-push to this repo as you go: training script, config, and
  a short progress log after the first 100 steps and at each cooldown
  milestone you already stop for. The finished adapter gets zipped and
  pushed the same way v5 was (parts + reassembly note) when it exists.

## Step 4 — after the adapter exists: the battery (do not start early)

Same battery, same scoring as the base runs, in this order:

1. Twin clean (no adapter): 56 hitman pilot rows, the 5 THREAT
   questions, 20-hunt pursuit pilot.
2. Twin + twin adapter: the same three, unchanged.

Hand-tag refusal-shaped replies the same way as Phase 1. Do NOT put the
old v5 (base) adapter on the twin, and do not run the battery while
training is underway — one heavy job at a time on this machine.

## Hard rules

- Fictional targets and abstract actions only — the data is already
  built to that rule; nothing new gets written into it.
- One heavy job at a time. If the pursuit run from the earlier orders
  is still alive when Del says start this, kill it first (its partial
  data is already pushed and labeled).
- If anything in Steps 1–2 does not match this spec, stop and report
  the mismatch rather than improvising.
