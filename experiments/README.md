# Current experiments

New work lives here. Frozen historical runs remain in
[`legacy/2026-11-17`](../legacy/2026-11-17/).

The November 17 local L17 N4078 check was stopped at the user's request after
four complete replies and one interrupted reply. Its files remain in
[`granite_L17_N4078_consistency_20261117`](../legacy/2026-11-17/granite_L17_N4078_consistency_20261117/).
The question set is included in the
[two-GPU Kaggle integrity evaluation](granite/integrity_20261117/README.md):
100 coordinates, 180 fixed settings, 24 questions, and matched undosed controls.

The [Granite synthetic action pilot](granite/simulation_20261117/README.md) adds
five deterministic toy environments, seven fixed candidate channels, matched
perturbation controls, independent comprehension probes, and observe/enforce
guard comparisons. The private two-T4 Kaggle queue contains 840 episodes with a
three-hour worker cap; consult its `active_launch.json` for the submitted run.
