"""Build a private two-T4 notebook from the audited, fixed Granite queue."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import base64
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REVISION = 'c0650403e44e78ec0262dab1c90914c65b196c4e'


def write(name, obj):
    (HERE / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cell(kind, text):
    data = {'cell_type': kind, 'id': kind + '-' + hashlib.sha256(text.encode()).hexdigest()[:12],
            'metadata': {}, 'source': text.splitlines(keepends=True)}
    if kind == 'code':
        data.update(execution_count=None, outputs=[])
    return data


def main():
    candidates = json.loads((HERE / 'candidates.json').read_text())
    questions = json.loads((HERE / 'questions.json').read_text())['questions']
    assert len(questions) == 24
    groups = []
    loads = [0, 0]
    for phase in ('replication', 'comparison'):
        by_layer = defaultdict(list)
        for index, candidate in enumerate(candidates):
            if candidate['phase'] == phase:
                by_layer[candidate['layer']].append(index)
        layer_order = sorted(by_layer, key=lambda layer: (layer != 17, layer != 13, layer))
        for layer in layer_order:
            indices = by_layer[layer]
            for offset in range(0, len(indices), 4):
                subset = indices[offset:offset + 4]
                shard = min((0, 1), key=lambda x: (loads[x], x))
                groups.append({'group_id':len(groups), 'layer':layer, 'candidate_indices':subset,
                               'shard':shard, 'phase':phase})
                loads[shard] += 1
    schedule = {'0':[], '1':[]}
    # Finish all eight original checks for replication channels before breadth.
    for shard in (0,1):
        for qi in range(8):
            for group in groups:
                if group['shard']==shard and group['phase']=='replication':
                    schedule[str(shard)].append({'group_id':group['group_id'],'question_index':qi})
        for round_index in range(len(questions)):
            for local_index,group in enumerate(g for g in groups if g['shard']==shard):
                qi = (round_index + local_index) % len(questions)
                if group['phase']=='replication' and qi<8:
                    continue
                schedule[str(shard)].append({'group_id':group['group_id'],'question_index':qi})
    config = {
        'model': 'ibm-granite/granite-4.1-3b', 'revision': REVISION,
        'layers': 40, 'channels': 8192, 'batch_size': 4, 'seed': 20261112,
        'attention_implementation': 'sdpa_math_fp32_intermediates',
        'candidates': candidates, 'candidate_groups': groups, 'questions': questions, 'schedule':schedule,
        'decoding': {'do_sample': False, 'num_beams': 1, 'repetition_penalty': 1.0, 'no_repeat_ngram_size': 0},
        'limits': {'max_new_tokens': 500, 'input_tokens': 256, 'batch_seconds': 600,
                   'output_bytes': 4 * 1024**3,
                   'repetition': {'minimum_tokens': 64, 'minimum_repeats': 4,
                                  'maximum_period': 128, 'check_every_tokens': 16}},
        'validate_long_context': True,
        'method': 'Fixed nonadaptive evaluation. Up to four independent single-channel intervention rows plus one unchanged control per batch, sharing a layer and question. Each GPU holds a complete model replica. Saved replication settings and fixed +/-0.5 comparison doses are frozen. Multiple settings of one coordinate may occupy different independent rows. Scalar = saved dose * saved median / current float16 down-projection column norm. Hook adds to MLP down-projection input at prompt and generated token positions. No weight editing. Fresh context per row.',
        'ordering': 'Balanced candidate groups across two GPUs. The eight original questions run first on replication channels; remaining questions rotate across the fixed queue. Each setting receives all 24 questions if the queue completes. Five-hour chunks may require continuation.',
        'limits_of_attribution': 'Historical candidate identity is carried in each record. New comparison channels are sampled without scoring outputs; small symmetric doses are fixed in advance. Labels are retrieval flags, not established harmful traits. This new run verifies its exact checkpoint. CUDA float16 batching is not asserted to reproduce MPS bfloat16 or historical batches token-for-token.',
    }
    write('config.json', config)
    write('source_calibration.json', {'revision': REVISION, 'candidates_sha256': digest(HERE / 'candidates.json'),
                                     'method': config['method']})
    reference = json.loads((ROOT / 'legacy/2026-11-17/handover_channels_20261112/runtime.json').read_text())
    write('download_manifest.json', {'model': config['model'], 'revision': REVISION,
          'files': [{'name': name, **entry} for name, entry in reference['checkpoint_files'].items()]})
    names = ['sweep.py', 'repetition.py', 'gpu_memory.py', 'reasoning_format.py', 'supervise.py',
             'monitor_launch.py', 'config.json', 'questions.json', 'candidates.json',
             'source_calibration.json', 'download_manifest.json', 'candidate_audit.json']
    assets = {name: base64.b64encode((HERE / name).read_bytes()).decode() for name in names}
    hashes = {name: digest(HERE / name) for name in names}
    write('asset_hashes.json', hashes)
    cells = [cell('markdown', f'''# Granite behavioral integrity evaluation

{len(candidates)} fixed settings across {len({(c["layer"], c["neuron"]) for c in candidates})} coordinates, 24 fixed safety and comprehension questions,
{len(candidates) * len(questions)} planned intervention replies plus {len(groups) * len(questions)} batched controls.
These are candidates, including fiction, refusals and corrupted text, not confirmed harmful neurons.
Saved candidates retain their provenance; comparison coordinates and doses are selected before generation. This is a fixed safety and comprehension study, not an optimization for harmful instructions.

Two Tesla T4 GPUs run independent replicas; up to four separate interventions and one
undosed control per batch. No row combines neurons. Greedy decoding, at most 500 new
tokens, with repetition detection. The first batch and full configured cache size must
pass on both GPUs before the bounded continuation. Five-hour maximum; no automatic rerun.
Generated text is saved, never executed. The notebook is private.
'''), cell('code', '''import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==5.22.1', 'accelerate', 'huggingface_hub', 'safetensors'], check=True)
'''), cell('code', '''import base64, hashlib, json, pathlib, os, time, subprocess, sys
WORK=pathlib.Path('/kaggle/working/granite_eval')
WORK.mkdir(parents=True,exist_ok=True)
''' + f'ASSETS={assets!r}\nHASHES={hashes!r}\n' + '''for name, blob in ASSETS.items():
    payload=base64.b64decode(blob)
    assert hashlib.sha256(payload).hexdigest()==HASHES[name]
    target=WORK/name
    if target.exists(): assert target.read_bytes()==payload, name
    target.write_bytes(payload)
print('Fixed research assets verified.', flush=True)
print((WORK/'questions.json').read_text(), flush=True)
'''), cell('code', '''import torch
print('torch',torch.__version__,'CUDA',torch.version.cuda,'GPUs',torch.cuda.device_count(),flush=True)
assert torch.cuda.device_count()==2, 'This notebook requires GPU T4 x2.'
for i in range(2):
    print(i,torch.cuda.get_device_name(i),torch.cuda.get_device_properties(i).total_memory,flush=True)
    assert 'T4' in torch.cuda.get_device_name(i), 'This frozen launch expects Tesla T4.'
config=json.loads((WORK/'config.json').read_text())
print('FIXED_QUEUE',len(config['candidates']),'candidates',len(config['questions']),'questions','MAX_TOKENS',config['limits']['max_new_tokens'],flush=True)
from huggingface_hub import snapshot_download
manifest=json.loads((WORK/'download_manifest.json').read_text())
MODEL=snapshot_download(config['model'],revision=config['revision'],cache_dir='/tmp/granite-hf',
                        allow_patterns=[f['name'] for f in manifest['files']],max_workers=2)
for item in manifest['files']:
    h=hashlib.sha256()
    with open(pathlib.Path(MODEL)/item['name'],'rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    assert h.hexdigest()==item['sha256'],item['name']
print('Exact Granite checkpoint matches recorded local file hashes.',flush=True)
'''), cell('code', '''subprocess.run([sys.executable,'-u',str(WORK/'supervise.py'),'--work',str(WORK),
                '--model-path',MODEL,'--seconds','18000'],check=True)
print('Bounded evaluation finished. Inspect supervisor_status.json and saved rows.',flush=True)
'''), cell('code', '''import sqlite3
for shard in (0,1):
    path=WORK/'results'/f'shard_{shard}.sqlite'
    if path.exists():
        with sqlite3.connect(path) as db:
            print('SHARD',shard,'NEXT',db.execute("SELECT value FROM meta WHERE key='next_batch'").fetchone(),
                  'SAVED',db.execute('SELECT COUNT(*) FROM results').fetchone()[0],
                  'PARTIAL',db.execute('SELECT COUNT(*) FROM partial').fetchone()[0],flush=True)
''')]
    write('granite_safety.ipynb', {'cells': cells, 'metadata': {'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'nbformat': 4, 'nbformat_minor': 5})
    metadata = {'id': 'aeeeeeeee/granite-behavioral-integrity-evaluation', 'title': 'Granite behavioral integrity evaluation',
                'code_file': 'granite_safety.ipynb', 'language': 'python', 'kernel_type': 'notebook',
                'is_private': True, 'enable_gpu': True, 'enable_internet': True, 'machine_shape': 'NvidiaTeslaT4',
                'dataset_sources': [], 'competition_sources': [], 'kernel_sources': [], 'model_sources': []}
    write('kernel-metadata.json', metadata)
    write('prepared.json', {'created_utc': datetime.now(timezone.utc).isoformat(), 'candidates': len(candidates),
          'questions': len(questions), 'interventions': len(candidates) * len(questions), 'controls': len(groups) * len(questions),
          'groups_per_gpu': loads, 'candidates_per_gpu': dict(Counter(c['shard'] for g in groups for c in [{'shard': g['shard']}] * len(g['candidate_indices']))),
          'session_seconds': 18000, 'max_new_tokens': 500, 'metadata': metadata})
    print((HERE / 'prepared.json').read_text())


if __name__ == '__main__':
    main()
