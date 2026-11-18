"""Recreate ignored compatibility links after a fresh checkout; no inference."""
import json
import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root/'docs/reorganization-20261117.json').read_text())
count = 0
for entry in manifest['entries']:
    if entry['compatibility_path']:
        link = root/entry['compatibility_path']
        target = root/entry['new_path']
        if target.exists() and not link.exists() and not link.is_symlink():
            link.symlink_to(os.path.relpath(target, link.parent))
            count += 1
legacy = root/manifest['legacy_root']
for name in ['.venv','models','mcps','requirements.txt']:
    link = legacy/name
    if (root/name).exists() and not link.exists() and not link.is_symlink():
        link.symlink_to(Path('../..')/name, target_is_directory=(root/name).is_dir())
print(f'Restored {count} historical data-path links. No model loaded.')
