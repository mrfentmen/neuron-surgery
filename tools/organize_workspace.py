"""Move the old workspace intact, grouping flat outputs without deleting evidence.

Legacy scripts remain together. Local, ignored compatibility links preserve their
relative data paths. GitHub sees grouped data, not the compatibility links.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / 'legacy/2026-11-17'
KEEP = {'.git', '.venv', 'models', 'mcps', '.obsidian', '.mcp-logs', '.gne',
        'neuron-explorer', 'neuronary', '__pycache__', 'tools', 'docs', 'experiments',
        'legacy', '.gitignore', 'LICENSE', 'LICENSE-DATA', 'CITATION.cff', 'requirements.txt'}

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

def main():
    manifest = ROOT / 'docs/reorganization-20261117.json'
    resuming = LEGACY.exists()
    saved = json.loads(manifest.read_text()) if resuming and manifest.exists() else None
    if resuming and (saved is None or saved['state'] != 'moving'):
        raise SystemExit('Legacy destination already exists without an unfinished journal.')
    if (ROOT / '.handover-model.lock').exists():
        raise SystemExit('Model lock exists; stop the local runner first.')
    LEGACY.mkdir(parents=True, exist_ok=True)
    if not resuming:
        before = subprocess.check_output(['git', 'status', '--porcelain=v1', '-z'], cwd=ROOT)
        (ROOT / 'docs/git-status-before-reorganization.bin').write_bytes(before)
    records = saved['entries'] if saved else []
    source_entries = sorted(ROOT.iterdir())
    for old in source_entries:
        if old.name in KEEP:
            continue
        stat = old.lstat()
        if old.is_file() and old.suffix == '.jsonl':
            relative = Path('data/raw') / old.name
        elif old.is_file() and old.suffix == '.json':
            relative = Path('data/metadata') / old.name
        elif old.is_file() and old.suffix in {'.log', '.err'}:
            relative = Path('logs/root') / old.name
        else:
            relative = Path(old.name)
        target = LEGACY / relative
        if target.exists():
            if old.is_dir() and target.is_dir() and old.name == 'logs':
                # Flat .log files may have created logs/root before the old logs
                # directory is reached. Merge only those disjoint generated paths.
                children = list(target.iterdir())
                assert all(not (old / p.name).exists() for p in children)
                for p in children:
                    os.rename(p, old / p.name)
                target.rmdir()
            else:
                raise RuntimeError(f'Collision: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = sha(old) if old.is_file() and not old.is_symlink() else None
        os.rename(old, target)
        assert target.lstat().st_ino == stat.st_ino
        if digest:
            assert sha(target) == digest
        compatibility = None
        if relative != Path(old.name):
            compatibility = LEGACY / old.name
            compatibility.symlink_to(relative)
        records.append(dict(old_path=old.name, new_path=str(target.relative_to(ROOT)),
                            type='directory' if target.is_dir() else 'file',
                            original_inode=stat.st_ino, bytes=stat.st_size, sha256=digest,
                            compatibility_path=str(compatibility.relative_to(ROOT)) if compatibility else None))
        manifest.write_text(json.dumps({'state':'moving','entries':records},indent=2)+'\n')
    # Runtime environments and model files stay in their original root locations.
    for name in ['.venv','models','mcps','requirements.txt']:
        if (ROOT/name).exists():
            (LEGACY/name).symlink_to(Path('../..')/name, target_is_directory=(ROOT/name).is_dir())
    ignore = [f'/{Path(r["old_path"]).name}' for r in records if r['compatibility_path']]
    ignore += ['/.venv', '/models', '/mcps', '/requirements.txt', '/__pycache__', '*.pyc']
    (LEGACY/'.gitignore').write_text('# Local links preserve historical relative paths. Rebuild with tools/restore_legacy_links.py.\n'+'\n'.join(ignore)+'\n')
    result = {'state':'complete','created_utc':datetime.now(timezone.utc).isoformat(),
              'method':'Same-filesystem renames; root-file SHA256 and moved-entry inode verified.',
              'deletions':0,'git_commits':0,'git_pushes':0,'legacy_root':str(LEGACY.relative_to(ROOT)),
              'compatibility_note':'Frozen records retain original paths. Resolve them relative to legacy_root. Absolute paths in historical records remain historical evidence, not current launch commands.',
              'entries':records}
    manifest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'moved_entries':len(records),'compatibility_links':sum(bool(r['compatibility_path']) for r in records),
                      'manifest':str(manifest),'root_entries':len(list(ROOT.iterdir()))}))

if __name__ == '__main__':
    main()
