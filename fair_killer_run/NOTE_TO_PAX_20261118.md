# Note to Pax — from Del (2026-11-18)

Twin adapter is blocked on my Mac. Hermes tried 3 training probes —
bf16, no-checkpoint, fp32-short — all died in the MPS backward pass
(GPU hang, step 0). Inference runs fine; training does not. The chip
can't do it, not a settings issue. Log: TWIN_TRAIN_BLOCKED.md, commit
5e01e77.

Picks for you:

1. Cloud GPU (my call on money) — what box/spec, and same v5 pinned
   stack? If yes I'll spin it up.
2. MLX rewrite — your call if worth a new prereg, or park it.
3. Park the twin arm entirely — paper goes with base + v5 adapter
   results on file.

Phase 1 + pursuit pilot + tuned arm all pushed with verdicts. Nothing
running here. Tell me which door.
— Del
