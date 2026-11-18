# Neuron surgery: activation intervention research

This repository records controlled activation interventions and their effects on
model responses. Threats, identity claims, safety failures, repetition and task
performance are different observations; the records do not establish a hidden
agent, private intentions or practical harmful capability.

## Start here

| Location | Contents |
| --- | --- |
| [Current work](experiments/README.md) | New experiments, frozen settings and run status |
| [Research plan](docs/RESEARCH-PLAN.md) | Controlled wording checks and a proposed synthetic simulation |
| [Repository map](docs/REPOSITORY-MAP.md) | Where older scripts, data, reports and model-specific runs live |
| [Historical workspace](legacy/2026-11-17/) | Preserved scripts, reports and experiment folders |
| [Historical raw data](legacy/2026-11-17/data/raw/) | Former top-level JSONL files |
| [Previous project introduction](legacy/2026-11-17/README.md) | Historical claims and method descriptions; read their dates and caveats |

The November 16 Granite Kaggle evaluation saved **5,793 replies**, including
4,520 interventions and 1,273 controls, from a pinned Granite 4.1 3B checkpoint.
Selected outputs show harmful endorsements, abusive crisis responses and factual
errors relative to controls. Other outputs are safe, irrelevant or repetitive.
See its [audit](legacy/2026-11-17/granite_kaggle_safety_20261116/launches/20261116T083719Z_sdpa_math/review/summary.json)
and [full transcript](legacy/2026-11-17/granite_kaggle_safety_20261116/launches/20261116T083719Z_sdpa_math/review/TRANSCRIPT.txt).

## Working with the archive

Old scripts and their relative layout are retained under `legacy/2026-11-17`.
Flat JSONL, metadata and logs are grouped into subdirectories. Ignored local links
allow historical scripts to find those files without filling GitHub's listing.
After a fresh checkout, run `python tools/restore_legacy_links.py` to restore links
for files actually present. This does not download weights or run inference.

Frozen manifests may contain their original absolute paths. The
[migration map](docs/reorganization-20261117.json) resolves former top-level paths
to current locations. Review old launchers before executing them. No historical
output was rewritten or deleted during this reorganization.

Model weights stay in `models/` and are excluded from Git. Granite and Nanbeige
coordinate systems and checkpoints must be kept separate. Generated output is
research data and is never executed by the text-evaluation runners.

Code: [MIT](LICENSE). Research data and writeups: [CC BY 4.0](LICENSE-DATA).
