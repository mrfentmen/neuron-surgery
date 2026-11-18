"""Freeze a private, bounded two-T4 pilot. Does not submit or load a model."""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
from simulator import SYSTEM_PROMPTS, SCENARIOS, build_schedule

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'integrity_20261117'


def write(name,value):
    (HERE/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def cell(kind,text):
    result={'cell_type':kind,'id':hashlib.sha256(text.encode()).hexdigest()[:12],
            'metadata':{},'source':text.splitlines(keepends=True)}
    if kind=='code':result.update(execution_count=None,outputs=[])
    return result


def main():
    old=json.loads((PREVIOUS/'config.json').read_text())
    wanted=[(27,3448),(13,1318),(17,4078),(9,11),(14,7553),(25,7734),(13,7821)]
    candidates=[]
    for coord in wanted:
        found=[c for c in old['candidates'] if (c['layer'],c['neuron'])==coord and c['dose']==-4 and c['phase']=='replication']
        assert len(found)==1
        candidates.append(found[0])
    excluded={(c['layer'],c['neuron']) for c in old['candidates']}
    rng=random.Random(2026111707);matched=[]
    for c in candidates:
        available=[n for n in range(8192) if (c['layer'],n) not in excluded]
        neuron=rng.choice(available);excluded.add((c['layer'],neuron))
        matched.append({'layer':c['layer'],'neuron':neuron,'dose':c['dose'],'median':c['median'],
            'selection':'Seeded before outputs, same layer and normalized projection-output perturbation magnitude; excludes all coordinates in preceding fixed config. Not claimed previously untested.',
            'matched_to':c['candidate_id']})
    config={'model':old['model'],'revision':old['revision'],'seed':20261112,'label_seed':20261117,
            'control_selection_seed':2026111707,'candidates':candidates,'matched_controls':matched,
            'system_prompts':SYSTEM_PROMPTS,'scenarios':list(SCENARIOS),'max_actions':3,'max_new_tokens':500,
            'input_tokens':3072,'generation_seconds':240,'worker_seconds':10800,'session_timeout_seconds':12600,
            'decoding':{'do_sample':False,'num_beams':1,'repetition_penalty':1.0,'no_repeat_ngram_size':0},
            'attention':'sdpa_math_fp32_intermediates','batch_size':3,
            'method':'Each independent batch has an unchanged row, one saved intervention row, and one same-layer random channel at a matched projection-output perturbation norm. Add to MLP down_proj input on real prompt and generated positions; no weight edits. Probe in a fresh context, then fresh action context with up to three sequential state transitions. No adaptive optimization.',
            'limitations':['Matched output-vector norm is not a match for natural activation distributions.',
                'A single seed with greedy decoding; variants and conditions are experimental cases, not independent random samples.',
                'The same baseline can occur in multiple matched groups; do not count these as independent replications.',
                'A false probe prevents a comprehension-qualified interpretation but does not erase the episode.',
                'Threatening prose is not an action. Only schema-valid EOS replies select fixed synthetic enum operations.',
                'Monitor-only and enforcing guard arms have separate episode histories.',
                'Three actions measure short within-episode consistency, not long-term autonomy, intent or real-world capability.',
                'The model receives no shell, network, file, code-execution or physical-control tools. Setup downloads the checkpoint; model output only selects enums.']}
    write('config.json',config);write('schedule.json',build_schedule(config))
    for name in ['repetition.py','gpu_memory.py','download_manifest.json']:
        shutil.copyfile(PREVIOUS/name,HERE/name)
    names=['simulator.py','worker.py','supervise.py','summarize.py','repetition.py','gpu_memory.py','config.json','schedule.json','download_manifest.json']
    assets={name:base64.b64encode((HERE/name).read_bytes()).decode() for name in names}
    hashes={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in names}
    write('asset_hashes.json',hashes)
    cells=[cell('markdown','''# Granite synthetic action pilot

Private, fixed evaluation: seven previously flagged channels, unchanged controls,
and seven seeded same-layer controls at matched perturbation magnitude. Five toy
worlds, prerequisite present/absent, safety/neutral system prompts, and observe/enforce
guard conditions. 280 three-row jobs = 840 episodes, up to 3,360 replies including
comprehension checks. Greedy, 500-token maximum per reply; exact token loops stop.

Two independent T4 replicas. Three-hour worker budget; interrupted coverage is
reported as partial. Setup may add time. The first undosed episode must pass a
comprehension and action-format check on both GPUs before the fixed queue continues.
No automatic restart. The model has no executable tools; responses can only select
fixed enum actions changing dictionaries representing synthetic objects.
'''),cell('code',"""import subprocess, sys
subprocess.run([sys.executable,'-m','pip','install','-q','transformers==5.22.1','accelerate','huggingface_hub','safetensors'],check=True)
"""),cell('code',"""import base64,hashlib,json,pathlib,subprocess,sys,os
WORK=pathlib.Path('/kaggle/working/granite_sim')
WORK.mkdir(parents=True,exist_ok=True)
"""+f'ASSETS={assets!r}\nHASHES={hashes!r}\n'+"""for name,blob in ASSETS.items():
    payload=base64.b64decode(blob)
    assert hashlib.sha256(payload).hexdigest()==HASHES[name]
    path=WORK/name
    if path.exists():assert path.read_bytes()==payload
    path.write_bytes(payload)
print('FIXED_SIMULATION_ASSETS_VERIFIED',HASHES,flush=True)
"""),cell('code',"""import torch
print('GPU_COUNT',torch.cuda.device_count(),flush=True)
assert torch.cuda.device_count()==2,'Requires T4 x2'
for i in range(2):
    print('GPU',i,torch.cuda.get_device_name(i),torch.cuda.get_device_properties(i).total_memory,flush=True)
    assert 'T4' in torch.cuda.get_device_name(i)
from huggingface_hub import snapshot_download
config=json.loads((WORK/'config.json').read_text())
manifest=json.loads((WORK/'download_manifest.json').read_text())
MODEL=snapshot_download(config['model'],revision=config['revision'],cache_dir='/tmp/granite-sim-hf',
                        allow_patterns=[f['name'] for f in manifest['files']],max_workers=2)
for item in manifest['files']:
    h=hashlib.sha256()
    with open(pathlib.Path(MODEL)/item['name'],'rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    assert h.hexdigest()==item['sha256'],item['name']
print('CHECKPOINT_HASHES_VERIFIED',config['revision'],flush=True)
"""),cell('code',"""subprocess.run([sys.executable,'-u',str(WORK/'supervise.py'),'--work',str(WORK),
                '--model-path',MODEL,'--seconds',str(config['worker_seconds'])],check=True)
print('SIMULATION_WORKERS_FINISHED',flush=True)
print((WORK/'results'/'supervisor_status.json').read_text(),flush=True)
subprocess.run([sys.executable,str(WORK/'summarize.py'),str(WORK/'results')],check=True)
""")]
    write('granite_simulation.ipynb',{'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5})
    metadata={'id':'aeeeeeeee/granite-synthetic-action-pilot','title':'Granite synthetic action pilot',
              'code_file':'granite_simulation.ipynb','language':'python','kernel_type':'notebook',
              'is_private':True,'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4',
              'dataset_sources':[],'competition_sources':[],'kernel_sources':[],'model_sources':[]}
    write('kernel-metadata.json',metadata)
    schedule=build_schedule(config)
    write('prepared.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'jobs_per_gpu':{k:len(v) for k,v in schedule.items()},
        'jobs':280,'episodes':840,'max_replies':3360,'scenarios':list(SCENARIOS),'channels':wanted,
        'matched_controls':matched,'gpu_count':2,'worker_seconds':10800,'max_new_tokens':500,
        'config_sha256':hashes['config.json']})
    print(json.dumps({'jobs':280,'episodes':840,'max_replies':3360,'gpu_count':2,'worker_seconds':10800}))


if __name__=='__main__':main()
