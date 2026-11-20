#!/usr/bin/env python3
"""Download and verify the original pinned checkpoint without loading a model."""
import os
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEST = ROOT / 'models/Nanbeige4.1-3B'
REPO = 'Nanbeige/Nanbeige4.1-3B'
REVISION = '9c4555c37921c0982af8fffeee1e0a6e6b953c4f'


def stamp():
    return datetime.now(timezone.utc).isoformat()


def status(state, **extra):
    payload = dict(state=state, updated_utc=stamp(), pid=os.getpid(), **extra)
    temp = HERE / 'download_status.tmp'
    temp.write_text(json.dumps(payload, indent=2) + '\n')
    temp.replace(HERE / 'download_status.json')
    print(json.dumps(payload), flush=True)


def main():
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    files = [s for s in info.siblings if '/' not in s.rfilename and
             (s.rfilename.endswith(('.json', '.safetensors', '.model')) or s.rfilename == 'README.md')]
    assert info.sha == REVISION
    manifest = {'repo_id': REPO, 'revision': REVISION, 'destination': str(DEST),
                'started_utc': stamp(), 'max_workers': 1, 'xet_enabled': False,
                'expected_bytes': sum(s.size for s in files),
                'files': [{'name': s.rfilename, 'bytes': s.size, 'git_blob_id': s.blob_id,
                           'lfs_sha256': s.lfs.sha256 if s.lfs else None} for s in files]}
    (HERE / 'download_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    status('downloading', expected_bytes=manifest['expected_bytes'], revision=REVISION)
    snapshot_download(REPO, revision=REVISION, local_dir=DEST,
                      allow_patterns=[s.rfilename for s in files], max_workers=1)
    status('verifying')
    verified = []
    for item in manifest['files']:
        path = DEST / item['name']
        assert path.stat().st_size == item['bytes'], f'Size mismatch: {path.name}'
        sha256 = hashlib.sha256()
        git_sha = hashlib.sha1(f'blob {item["bytes"]}\0'.encode())
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                sha256.update(chunk)
                if not item['lfs_sha256']:
                    git_sha.update(chunk)
        actual = sha256.hexdigest()
        if item['lfs_sha256']:
            assert actual == item['lfs_sha256'], f'Publisher checksum mismatch: {path.name}'
        else:
            assert git_sha.hexdigest() == item['git_blob_id'], f'Git blob mismatch: {path.name}'
        verified.append({**item, 'sha256': actual, 'publisher_hash_verified': True})
        print('Verified ' + path.name, flush=True)
    manifest.update(completed_utc=stamp(), files=verified, verified=True)
    (HERE / 'download_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    status('complete', total_bytes=manifest['expected_bytes'], files=len(verified), model_loaded=False)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('failed', error=type(exc).__name__ + ': ' + str(exc))
        raise
