"""Embed only the reviewed research assets in a self-contained private notebook."""
import base64
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
bank=json.loads((HERE/'questions.json').read_text())
config={
 'model':'Nanbeige/Nanbeige4.1-3B','revision':'9c4555c37921c0982af8fffeee1e0a6e6b953c4f',
 'layers':32,'channels':10496,'shards':[[0,15],[16,31]],'batch_size':4,
 'doses':[-0.1,0.1,-0.25,0.25,-0.5,0.5],'seeds':[20261113],
 'questions':bank['questions'],
 'decoding':{'temperature':0.6,'top_p':0.95,'top_k':0,'repetition_penalty':1.0},
 'limits':{'max_new_tokens':500,'input_tokens':256,'batch_seconds':600,'output_bytes':4*1024**3},
 'method':'Fixed nonadaptive single-channel evaluation. One undosed row per batch; each other row gets one channel intervention. Both GPU workers load complete replicas. Every row has its own CUDA sampling generator initialized to the same seed. Not bitwise comparable to MPS/HF generate. Six finite saved doses, not all real-valued doses.',
 'calibration':'Use original source median residual norms, current dtype projection column norms. Log actual scalar and measured changes. No retraining or weight editing.',
}
(HERE/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
names=['sweep.py','repetition.py','gpu_memory.py','reasoning_format.py','config.json','questions.json','source_calibration.json','download_manifest.json']
assets={name:base64.b64encode((HERE/name).read_bytes()).decode() for name in names}
hashes={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in names}
(HERE/'asset_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')

def code(s):return {'cell_type':'code','id':'code-'+hashlib.sha256(s.encode()).hexdigest()[:12],'execution_count':None,'metadata':{},'outputs':[],'source':s.splitlines(keepends=True)}
def md(s):return {'cell_type':'markdown','id':'md-'+hashlib.sha256(s.encode()).hexdigest()[:12],'metadata':{},'source':s.splitlines(keepends=True)}

cells=[md('''# Nanbeige fixed safety and coherence evaluation

Two GPU replicas: layers 0–15 and 16–31. The full six-dose, 19-question grid
contains 38,289,408 intervention replies for one seed, plus batched controls.
This notebook supports bounded five-hour chunks, not the entire grid. It logs
throughput and durable progress. No adaptive prompt or dose search is used.
The default MAX_BATCHES=1 first checks the actual GPUs with one batch per GPU.
After checking those logs, set MAX_BATCHES=0 to use the five-hour session budget.

Every reply has up to 500 generated tokens, including reasoning. Cutoffs are
not completed answers. Review safety, factual accuracy and coherence separately.
The suicide-method request is replaced by a nonprocedural support request.
Medical questions are scored against product-specific labels, not pill counts.

Select GPU T4 x2. Keep this notebook private. Download/save results after each
session. To resume, attach the previous output and set RESUME_ROOT below to
its results folder; checkpoints must match this exact config/runtime.
'''),
code('''import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==5.22.1', 'accelerate', 'huggingface_hub', 'safetensors'], check=True)
'''),
code('''import base64, hashlib, json, pathlib, shutil, os, time, subprocess, sys
WORK=pathlib.Path('/kaggle/working/nanbeige_eval')
WORK.mkdir(parents=True,exist_ok=True)
'''+f'ASSETS={assets!r}\nHASHES={hashes!r}\n'+'''for name, blob in ASSETS.items():
    payload=base64.b64decode(blob)
    assert hashlib.sha256(payload).hexdigest()==HASHES[name]
    target=WORK/name
    if target.exists():assert target.read_bytes()==payload, f'Existing asset changed: {name}'
    target.write_bytes(payload)
print('Extracted fixed assets:',list(ASSETS))
print((WORK/'questions.json').read_text())
'''),
code('''# A bounded session. Changing settings creates a separate experiment.
SESSION_SECONDS=5*60*60
MAX_BATCHES=1  # First GPU check. Set 0 for a full bounded session after validation.
RESUME_ROOT=None  # e.g. pathlib.Path('/kaggle/input/prior-output/nanbeige_eval/results')
OUT=WORK/'results'
OUT.mkdir(exist_ok=True)
if RESUME_ROOT is not None:
    for shard in [0,1]:
        src=pathlib.Path(RESUME_ROOT)/f'shard_{shard}.resume.sqlite'
        dest=OUT/f'shard_{shard}.sqlite'
        assert src.is_file() and not dest.exists(), 'Choose fresh destination and complete prior checkpoints.'
        shutil.copy2(src,dest)
'''),
code('''import torch
print('torch',torch.__version__,'CUDA',torch.version.cuda,'GPUs',torch.cuda.device_count())
assert torch.cuda.device_count()==2,'Select GPU T4 x2; this notebook will not silently run on CPU or one GPU.'
for i in range(2):print(i,torch.cuda.get_device_name(i),torch.cuda.get_device_properties(i).total_memory)
from huggingface_hub import snapshot_download
config=json.loads((WORK/'config.json').read_text())
manifest=json.loads((WORK/'download_manifest.json').read_text())
# Cache outside /kaggle/working so downloaded weights are not duplicated in saved output.
MODEL=snapshot_download(config['model'],revision=config['revision'],cache_dir='/tmp/nanbeige-hf',
                        allow_patterns=[f['name'] for f in manifest['files']],max_workers=2)
for item in manifest['files']:
    h=hashlib.sha256()
    with open(pathlib.Path(MODEL)/item['name'],'rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024**2),b''):h.update(chunk)
    assert h.hexdigest()==item['sha256'],item['name']
print('Pinned checkpoint hashes verified. No model tools; generated code is never executed.')
'''),
code('''# Independent workers. The parent enforces a deadline even if a worker stalls.
workers=[]
try:
    for shard in [0,1]:
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(shard)
        env['HF_HUB_OFFLINE']='1';env['TRANSFORMERS_OFFLINE']='1'
        log=open(OUT/f'worker_{shard}.log','ab',buffering=0)
        args=[sys.executable,'-u',str(WORK/'sweep.py'),'--config',str(WORK/'config.json'),
              '--model-path',MODEL,'--shard',str(shard),'--out',str(OUT),
              '--max-seconds',str(SESSION_SECONDS),'--max-batches',str(MAX_BATCHES)]
        proc=subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        workers.append((proc,log))
    deadline=time.monotonic()+SESSION_SECONDS+120
    while any(p.poll() is None for p,_ in workers):
        if any(p.poll() not in (None,0) for p,_ in workers):
            raise RuntimeError('A worker failed; stopping its partner. Check saved logs and checkpoints.')
        if time.monotonic()>deadline:
            raise TimeoutError('Supervisor session deadline')
        for shard in [0,1]:
            f=OUT/f'status_{shard}.json'
            if f.exists():print(f.read_text(),flush=True)
        time.sleep(30)
    assert all(p.returncode==0 for p,_ in workers),'Inspect worker logs'
finally:
    for p,log in workers:
        if p.poll() is None:
            p.terminate()
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        log.close()
print('Chunk finished. Save/download all results before ending the session. Resume from *.resume.sqlite; keep old raw results separately.')
'''),
code('''import sqlite3
for shard in [0,1]:
    dbfile=OUT/f'shard_{shard}.sqlite'
    if not dbfile.exists():continue
    with sqlite3.connect(dbfile) as db:
        print('shard',shard,'next batch',db.execute("SELECT value FROM meta WHERE key='next_batch'").fetchone())
        print('saved rows',db.execute('SELECT COUNT(*) FROM results').fetchone()[0])
        print('partial attempts',db.execute('SELECT COUNT(*) FROM partial').fetchone()[0])
''')]
nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5}
(HERE/'nanbeige_safety.ipynb').write_text(json.dumps(nb,indent=2,ensure_ascii=False)+'\n')
metadata={'id':'aeeeeeeee/nanbeige-fixed-safety-evaluation','title':'Nanbeige fixed safety evaluation','code_file':'nanbeige_safety.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4','dataset_sources':[],'competition_sources':[],'kernel_sources':[],'model_sources':[]}
(HERE/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('Built private notebook; 38,289,408 intervention rows in full grid; five-hour chunk configured.')
