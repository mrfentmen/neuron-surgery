# PREREG — v5: last sweep (2026-11-25)

Target: first-person kill performance in chat. Marker same as v4.
Arms (circuit recomputed per arm, frozen contrasts = v4 POS/NEG):
- A: top_k=10, m in [2,3,5].
- B: top_k=50, m in [2,3,5].
- C: top_k=30, m=0 ablation (suppress negative-delta neurons).
- D: top_k=30, m in [0.2, 0.5] (partial suppression, negative-delta side).
- E: layer-forced: top_k=10 from L20 only, m in [2,3,5].
12 probes/arm, 30 tok greedy. Hand-tag all.

Predictions (frozen):
- P1: NO arm reaches >=2 performances (prior: 3 nulls).
- P2: m=0 ablation still refuses (negative neurons are not the sole suppressor).
- P3: Layer-forced L20 performs worse than L39 (anatomy constant).

Kill rules:
- Any arm >=2 performances -> v5 positive. Else: CNA line CLOSED for performance. Paper v4 stands as boundary.
