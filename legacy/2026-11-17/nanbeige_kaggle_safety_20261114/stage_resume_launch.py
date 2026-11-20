"""Freeze a two-T4 continuation with embedded, verified slim checkpoints."""
import ast
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

HERE = Path(__file__).resolve().parent
PARENT = HERE / 'launches/20261114T150056Z_10k_memory_validated'
RESULTS = PARENT / 'output/nanbeige_eval/results'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    supervisor = json.loads((RESULTS / 'supervisor_status.json').read_text())
    assert supervisor['research_outcome'] == 'planned_chunk_completed'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    dest = HERE / 'launches' / (stamp + '_10k_resume_two_t4')
    dest.mkdir()
    names = ['sweep.py', 'repetition.py', 'gpu_memory.py', 'reasoning_format.py',
             'monitor_launch.py', 'supervise.py', 'questions.json',
             'source_calibration.json', 'download_manifest.json', 'config.json']
    for name in names:
        shutil.copy2(PARENT / name, dest / name)
    checkpoints = {}
    for shard in (0, 1):
        checkpoint = RESULTS / f'shard_{shard}.resume.sqlite'
        with sqlite3.connect(checkpoint.resolve().as_uri() + '?mode=ro', uri=True) as db:
            meta = dict(db.execute('SELECT key,value FROM meta'))
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert tables == {'meta'}
        runtime = json.loads(meta['signature'])
        assert runtime == json.loads((RESULTS / f'runtime_{shard}.json').read_text())
        for key, name in [('config_sha256','config.json'), ('script_sha256','sweep.py'),
                          ('calibration_sha256','source_calibration.json'),
                          ('memory_helper_sha256','gpu_memory.py'),
                          ('repetition_sha256','repetition.py'),
                          ('reasoning_parser_sha256','reasoning_format.py')]:
            assert runtime[key] == sha(dest / name)
        assert meta['parent_results_sha256'] == sha(RESULTS / f'shard_{shard}.sqlite')
        name = checkpoint.name
        shutil.copy2(checkpoint, dest / name)
        names.append(name)
        checkpoints[str(shard)] = dict(start_batch=int(meta['next_batch']),
            checkpoint_sha256=sha(checkpoint), parent_results_sha256=meta['parent_results_sha256'])

    source = (dest / 'supervise.py').read_text()
    changes = [
        ('def verify_first_batch(out):', 'def verify_first_batch(out, starting_batches):'),
        ("status['next_batch'] == 1", "status['next_batch'] == starting_batches[shard] + 1"),
        ('assert len(rows) == 2', "assert len(rows) == 2\n        assert {r['batch_index'] for r in rows} == {starting_batches[shard]}"),
        ("    deadline = time.monotonic() + args.seconds", """    starting_batches = {}
    for shard in (0, 1):
        path = out / f'shard_{shard}.sqlite'
        with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as db:
            starting_batches[shard] = int(db.execute("SELECT value FROM meta WHERE key='next_batch'").fetchone()[0])
    deadline = time.monotonic() + args.seconds"""),
        ('audit = verify_first_batch(out)', 'audit = verify_first_batch(out, starting_batches)'),
        ("'completed_batch_rows': audit['completed_batch_rows'],", "'completed_batch_rows': audit['completed_batch_rows'],\n            'starting_batches': starting_batches,"),
    ]
    for before, after in changes:
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    (dest / 'supervise.py').write_text(source)

    assets = {name:base64.b64encode((dest/name).read_bytes()).decode() for name in names}
    hashes = {name:sha(dest/name) for name in names}
    (dest/'asset_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    notebook = json.loads((PARENT/'nanbeige_safety.ipynb').read_text())
    notebook['cells'][0]['source'] = '''# Nanbeige fixed safety evaluation — continued chunk

Two Tesla T4 GPUs, each holding a full model replica. GPU 0 tests layers 0–15;
GPU 1 tests layers 16–31. Five-hour budget, max 10,000 output tokens, with
exact repetition stops. The model has no tools; generated code is never run.

This resumes version 7 at batch 32 / 31 from hash-verified slim checkpoints.
Only new replies are saved here. Keep version 7's full raw output separately.
The unfinished final batch is retried from its saved cursor and original seed.

Model revision, question order, finite doses, sampling and sweep code are unchanged.
Both GPUs must pass the long-cache test and save their next real matched batch.
The notebook refuses CPU/one-GPU execution and incompatible runtime signatures.
'''.splitlines(keepends=True)
    extract = '''import base64, hashlib, json, pathlib, os, subprocess, sys
WORK=pathlib.Path('/kaggle/working/nanbeige_eval')
WORK.mkdir(parents=True,exist_ok=True)
''' + f'ASSETS={assets!r}\nHASHES={hashes!r}\n' + '''for name, blob in ASSETS.items():
    payload=base64.b64decode(blob)
    assert hashlib.sha256(payload).hexdigest()==HASHES[name]
    target=WORK/name
    if target.exists():assert target.read_bytes()==payload, f'Existing asset changed: {name}'
    target.write_bytes(payload)
print('Extracted fixed assets and two verified resume checkpoints.')
print((WORK/'questions.json').read_text())
'''
    notebook['cells'][2]['source'] = extract.splitlines(keepends=True)
    restore = '''import shutil, sqlite3
SESSION_SECONDS=5*60*60
OUT=WORK/'results'
OUT.mkdir(exist_ok=True)
assert not list(OUT.glob('shard_*.sqlite')), 'Refusing to overwrite current outputs'
''' + f'EXPECTED_START={ {int(k):v["start_batch"] for k,v in checkpoints.items()}!r}\n' + '''for shard in (0,1):
    source=WORK/f'shard_{shard}.resume.sqlite'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==HASHES[source.name]
    with sqlite3.connect(source.resolve().as_uri()+'?mode=ro',uri=True) as db:
        meta=dict(db.execute('SELECT key,value FROM meta'))
    assert int(meta['next_batch'])==EXPECTED_START[shard]
    shutil.copy2(source,OUT/f'shard_{shard}.sqlite')
    print('RESUME_CURSOR',shard,meta['next_batch'],flush=True)
'''
    notebook['cells'][3]['source'] = restore.splitlines(keepends=True)
    for index, cell in enumerate(notebook['cells']):
        cell['id'] = f'resume-two-t4-{index}'
        if cell['cell_type']=='code':
            cell.update(execution_count=None,outputs=[])
            ast.parse(''.join(cell['source']),filename=f'cell {index}')
    for name in names:
        if name.endswith('.py'):ast.parse((dest/name).read_text(),filename=name)
    (dest/'nanbeige_safety.ipynb').write_text(json.dumps(notebook,indent=2)+'\n')
    shutil.copy2(PARENT/'kernel-metadata.json',dest/'kernel-metadata.json')
    manifest = dict(stage='prepared',created_utc=datetime.now(timezone.utc).isoformat(),
        previous_version=7,parent_directory=str(PARENT.relative_to(HERE.parent)),
        accelerator='NvidiaTeslaT4',expected_gpu_count=2,session_seconds=18000,
        hard_timeout_seconds=19800,max_new_tokens=10000,checkpoints=checkpoints,
        sweep_unchanged=True,config_unchanged=True,
        files={name:sha(dest/name) for name in ('kernel-metadata.json','nanbeige_safety.ipynb')})
    (dest/'launch.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (HERE/'pending_launch.json').write_text(json.dumps({'path':str(dest.relative_to(HERE.parent)),'stage':'prepared'},indent=2)+'\n')
    print(dest)


if __name__=='__main__':main()
