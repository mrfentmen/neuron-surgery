# Twin adapter training — BLOCKED on this Mac (2026-11-18)

Three attempts, all MPS backend failures on the backward pass:

1. bf16 + grad checkpointing, 20-step trial: `MPS command buffer execution
   failed: Discarded (victim of GPU error/recovery)`, step 0.
2. bf16, no checkpointing, 3-step trial: `Caused GPU Hang Error`, step 0.
3. fp32, max_length 512, 1-step probe: same GPU Hang, step 0.

Forward + inference on MPS works (all ablation runs). Backward does not,
regardless of dtype, length, or checkpointing. This is an MPS kernel gap
for this model, not a settings problem.

Per spec §hard rules: stopped and reporting instead of pushing through.

## Options (need Del's word)

- Cloud/Kaggle GPU (spec notes Kaggle time used up).
- CPU on this Mac: ~100x slower than GPU, 906 steps = weeks. Not feasible.
- Different framework (MLX LoRA): spec deviation, needs new prereg.
- Shorter max_length / smaller batch: will not help — backward itself hangs.

Script kept: `train_twin_adapter.py` (spec-exact, untested past step 0).
Probe copies: `train_twin_smoke.py`, `train_twin_probe.py`.
Trial log: `twin_trial_live.log`.
