"""Freeze a new 10k launch without changing historical notebooks or configs."""
import ast
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
previous = HERE / 'launches/20261114T044306Z_10k_repetition'
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
dest = HERE / 'launches' / f'{stamp}_10k_memory_validated'
dest.mkdir()
config = json.loads((previous / 'config.json').read_text())
config.update(batch_size=1, validate_long_context=True)
(dest / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
names = ['sweep.py', 'repetition.py', 'gpu_memory.py', 'reasoning_format.py',
         'monitor_launch.py', 'supervise.py', 'questions.json',
         'source_calibration.json', 'download_manifest.json']
for name in names:
    (dest / name).write_bytes((HERE / name).read_bytes())
names.append('config.json')
assets = {name: base64.b64encode((dest / name).read_bytes()).decode() for name in names}
hashes = {name: hashlib.sha256((dest / name).read_bytes()).hexdigest() for name in names}
(dest / 'asset_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
notebook = json.loads((previous / 'nanbeige_safety.ipynb').read_text())
notebook['cells'][0]['source'] = '''# Nanbeige fixed safety and coherence evaluation

Two T4 replicas, layers 0–15 / 16–31. Five-hour bounded chunk, with validation
included. One intervention plus one same-seed undosed control per GPU.
Maximum 10,000 output tokens including reasoning, with exact repetition stops.
Six fixed doses and 19 unchanged questions; no adaptive search or tool actions.

Before the sweep, each GPU tests its actual KV cache at 10,256 context tokens,
including 32 incremental decode steps. Then it runs and saves one actual batch.
Both checks must pass before continuing from batch one. EOS or repetition can
end real replies early; synthetic cache validation is not a research response.

Unused allocator cache is released before declaring low GPU memory. Every
committed batch is exported immediately, partial replies are saved separately,
and in-flight text/tokens are checkpointed every 512 generated tokens.
Unexpected stops fail the notebook; a clean script exit alone is insufficient.
Generated reasoning is observable text, not proof of intentions or sentience.
Keep this notebook private. Download all raw outputs after the bounded run.
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
print('Extracted fixed assets:', list(ASSETS))
print((WORK/'questions.json').read_text())
'''
notebook['cells'][2]['source'] = extract.splitlines(keepends=True)
notebook['cells'][3]['source'] = '''SESSION_SECONDS=5*60*60
OUT=WORK/'results'
OUT.mkdir(exist_ok=True)
assert not list(OUT.glob('shard_*.sqlite')), 'This launch must start in a fresh output folder'
'''.splitlines(keepends=True)
notebook['cells'][5]['source'] = '''subprocess.run([sys.executable, '-u', str(WORK/'supervise.py'),
                '--work', str(WORK), '--model-path', MODEL,
                '--seconds', str(SESSION_SECONDS)], check=True)
print('Bounded run completed; inspect supervisor_status.json for the research outcome.')
'''.splitlines(keepends=True)
for index, cell in enumerate(notebook['cells']):
    cell['id'] = f'memory-validation-{index}'
    if cell['cell_type'] == 'code':
        cell.update(execution_count=None, outputs=[])
        ast.parse(''.join(cell['source']), filename=f'notebook cell {index}')
for name in names:
    if name.endswith('.py'):
        ast.parse((dest / name).read_text(), filename=name)
(dest / 'nanbeige_safety.ipynb').write_text(json.dumps(notebook, indent=2) + '\n')
(dest / 'kernel-metadata.json').write_bytes((previous / 'kernel-metadata.json').read_bytes())
manifest = dict(stage='prepared', created_utc=datetime.now(timezone.utc).isoformat(),
                max_new_tokens=10000, batch_size=2, intervention_rows_per_batch=1,
                session_seconds=18000, hard_timeout_seconds=19800,
                synthetic_context_validation=10256, previous_version=6,
                first_batch_gate=True,
                files={name: hashlib.sha256((dest/name).read_bytes()).hexdigest()
                       for name in ['kernel-metadata.json', 'nanbeige_safety.ipynb']})
(dest / 'launch.json').write_text(json.dumps(manifest, indent=2) + '\n')
(HERE / 'pending_launch.json').write_text(json.dumps(dict(path=str(dest.relative_to(HERE.parent)), stage='prepared'), indent=2) + '\n')
print(dest)
