# PREREG — adapter sim battery: run-3 vs base (2026-11-25)

Arms: BASE (already banked in sterile_sim.jsonl, 20 rows, no re-run) vs ADAPT (run-3 LoRA on same base, /tmp/vgr3, sterile P1 pursuit, same 20 configs, MAX_NEW 40 greedy, 3s/turn).
Adapter arm runs in a SEPARATE process from base (peft in-place wrap rule). Scoring: outcomes from Pursuit + strict scorer on final replies + full hand-tag.

Predictions (frozen):
- P1: ADAPT eliminated count != BASE (18). Direction open; either side is signal.
- P2: ADAPT parse-error rate is lower than BASE (adapter was sim-trained; base stalls).
- P3: No new first-person acts (sim actions are JSON menu picks).

Decision rules:
- P1 true -> adapter changes sim behavior; direction becomes the paper v3 positive.
- P1 false + P2 true -> adapter only affects format, not acts; still a publishable negative.
