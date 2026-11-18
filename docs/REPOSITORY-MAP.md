# Repository map

The previous root held 1,224 files and 112 directories. The dated archive preserves
1,320 moved entries. Each moved root file was checked by SHA256; moves preserved
filesystem inodes. The journal records every move. Nothing was deleted or pushed
to GitHub as part of this operation.

| Material | Current location |
| --- | --- |
| New experiments | `experiments/granite/` |
| Original Python/shell scripts and Markdown reports | `legacy/2026-11-17/` |
| Original root JSONL outputs | `legacy/2026-11-17/data/raw/` |
| Original root JSON metadata | `legacy/2026-11-17/data/metadata/` |
| Original root logs | `legacy/2026-11-17/logs/root/` |
| Earlier experiment directories | `legacy/2026-11-17/granite_*`, `nanbeige_*`, `handover_*` |
| Consolidated historical channel index | `legacy/2026-11-17/layers/` |
| Historical keyword catalogues | `legacy/2026-11-17/harmful_neuron_collection_20261112/`, `sentience_collection_20261113/` |
| Model weights / Python environment | `models/`, `.venv/` (local only) |
| Reorganization journal and link-restoration code | `docs/reorganization-20261117.json`, `tools/restore_legacy_links.py` |

GitHub does not show the ignored compatibility links. The archive has fewer than
1,000 real entries at its top level; grouped raw and metadata directories are also
below that threshold. Local filesystem listings may still show those compatibility
links. The GitHub site changes only after a separately authorized commit/push.

All pre-existing working-tree edits were preserved. The original status snapshot
is local in `docs/git-status-before-reorganization.bin`; no unrelated changes were
reverted or staged. Existing historical links remain usable locally through the
compatibility paths. For GitHub, start with this map and the explicit data paths.

Browse [all historical reports](HISTORICAL-REPORTS.md) or the
[canonical flat-output index](RAW-DATA-INDEX.md) without relying on old data links.
