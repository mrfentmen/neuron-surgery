"""One independent Granite replica per GPU; text is data, never executable code."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import time

from simulator import (SYSTEM_PROMPTS, canonical, make_world, probe_prompt, action_prompt,
                       parse_reply, score_probe, apply_action, build_schedule)
from repetition import repeated_tail
from gpu_memory import check_memory, validate_long_cache


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024**2), b''): h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    tmp = Path(str(path)+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


class Store:
    def __init__(self, directory, shard, signature):
        self.path = directory / f'shard_{shard}.sqlite'
        self.shard = shard
        self.directory = directory
        self.db = sqlite3.connect(self.path)
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS steps (job TEXT, slot INTEGER, kind TEXT, turn INTEGER, payload TEXT, PRIMARY KEY(job,slot,kind,turn))')
        self.db.execute('CREATE TABLE IF NOT EXISTS episodes (job TEXT, slot INTEGER, payload TEXT, PRIMARY KEY(job,slot))')
        old = self.db.execute("SELECT value FROM meta WHERE key='signature'").fetchone()
        if old and old[0] != signature: raise ValueError('Existing run signature differs')
        with self.db: self.db.execute("INSERT OR IGNORE INTO meta VALUES ('signature',?)", (signature,))

    def save_steps(self, rows):
        with self.db:
            for row in rows:
                self.db.execute('INSERT INTO steps VALUES (?,?,?,?,?)',
                                (row['job']['job_id'],row['slot'],row['kind'],row['turn'],canonical(row)))
        self._append('steps', rows)

    def save_episodes(self, rows):
        with self.db:
            for row in rows:
                self.db.execute('INSERT INTO episodes VALUES (?,?,?)',
                                (row['job']['job_id'],row['slot'],canonical(row)))
        self._append('episodes', rows)

    def _append(self, name, rows):
        with (self.directory / f'{name}_{self.shard}.jsonl').open('a') as f:
            for row in rows: f.write(canonical(row)+'\n')
            f.flush(); os.fsync(f.fileno())

    def export(self):
        for table in ('steps','episodes'):
            path = self.directory / f'{table}_{self.shard}.jsonl'
            tmp = Path(str(path)+'.tmp')
            with tmp.open('w') as f:
                for (payload,) in self.db.execute(f'SELECT payload FROM {table} ORDER BY rowid'):
                    f.write(payload+'\n')
                f.flush(); os.fsync(f.fileno())
            tmp.replace(path)


def add_interventions(x, columns, scalars, position_mask):
    """Separate row interventions; exclude left padding and preserve baseline row."""
    import torch
    y = x.clone()
    for row, (column, scalar) in enumerate(zip(columns, scalars), 1):
        delta = torch.as_tensor(scalar, device=x.device, dtype=x.dtype)
        y[row, :, column] += position_mask[row].to(x.dtype) * delta
    return y


class Generator:
    def __init__(self, model, tokenizer, torch, config, stop_requested, deadline):
        self.model, self.tok, self.torch = model, tokenizer, torch
        self.config, self.stop_requested, self.deadline = config, stop_requested, deadline
        self.eos = model.generation_config.eos_token_id
        self.eos = set(self.eos if isinstance(self.eos,list) else [self.eos])
        self.pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else min(self.eos)
        self.tok.pad_token_id = self.pad
        self.tok.padding_side = 'left'
        self.norms = {}

    def generate(self, messages, candidate, matched, active):
        torch = self.torch
        layer = candidate['layer']
        if layer not in self.norms:
            self.norms[layer] = self.model.model.layers[layer].mlp.down_proj.weight.float().norm(dim=0).cpu()
        columns = [candidate['neuron'], matched['neuron']]
        scalar_requested = [candidate['dose']*candidate['median']/self.norms[layer][col].item() for col in columns]
        scalar_applied = [torch.tensor(v,dtype=torch.float16).item() for v in scalar_requested]
        rendered = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
        encoded = [self.tok.encode(p,add_special_tokens=False) for p in rendered]
        too_long = [len(t)>self.config['input_tokens'] for t in encoded]
        # Do not truncate context. A finished or oversized row gets inert padding text.
        batch_text = [p if active[i] and not too_long[i] else 'Inactive row.' for i,p in enumerate(rendered)]
        batch = self.tok(batch_text,add_special_tokens=False,padding=True,return_tensors='pt').to('cuda:0')
        input_ids, attention = batch['input_ids'], batch['attention_mask']
        position_mask = attention.bool()
        done = [not active[i] or too_long[i] for i in range(3)]
        stops = [('context_limit' if too_long[i] else 'inactive') if done[i] else None for i in range(3)]
        tokens = [[],[],[]]; repetition = [None,None,None]
        measurement = {'hook_calls':0, 'baseline_unchanged': True,
                       'changed_real_positions':[0,0], 'mean_real_deltas':[None,None]}

        def hook(module, inputs):
            x = inputs[0]
            y = add_interventions(x,columns,scalar_applied,position_mask)
            measurement['hook_calls'] += 1
            if measurement['hook_calls'] == 1 or x.shape[1]>1:
                measurement['baseline_unchanged'] &= bool(torch.equal(x[0], y[0]))
                for i,c in enumerate(columns):
                    mask=position_mask[i+1]
                    measurement['changed_real_positions'][i] += int(((y[i+1,:,c]!=x[i+1,:,c]) & mask).sum().item())
                    if bool(mask.any()):
                        measurement['mean_real_deltas'][i] = float((y[i+1,:,c]-x[i+1,:,c])[mask].float().mean().item())
            return (y,) + inputs[1:]

        handle = self.model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(hook)
        start = time.monotonic(); past = None
        try:
            with torch.inference_mode():
                for step in range(self.config['max_new_tokens']):
                    if all(done): break
                    stop = None
                    if self.stop_requested(): stop = 'user_stop'
                    elif time.monotonic() >= self.deadline: stop = 'session_budget'
                    elif time.monotonic()-start > self.config['generation_seconds']: stop = 'generation_timeout'
                    elif step % 64 == 0 and not check_memory(torch.cuda)[0]: stop = 'gpu_memory_guard'
                    if stop:
                        stops = [stops[i] if done[i] else stop for i in range(3)]; break
                    positions = attention.long().cumsum(-1)-1
                    positions.masked_fill_(attention==0,0)
                    if past is not None: positions = positions[:,-1:]
                    if past is None:
                        # Bounded prefill avoids quadratic full-prompt attention memory.
                        for offset in range(0,input_ids.shape[1],256):
                            end=min(offset+256,input_ids.shape[1])
                            position_mask=attention[:,offset:end].bool()
                            outputs=self.model(input_ids=input_ids[:,offset:end],attention_mask=attention[:,:end],
                                    position_ids=positions[:,offset:end],past_key_values=past,use_cache=True,logits_to_keep=1)
                            past=outputs.past_key_values
                            if end<input_ids.shape[1]: del outputs
                    else:
                        outputs = self.model(input_ids=input_ids,attention_mask=attention,position_ids=positions,
                                             past_key_values=past,use_cache=True,logits_to_keep=1)
                    past = outputs.past_key_values
                    scores = outputs.logits[:,-1,:].float()
                    valid = scores.isfinite().all(-1).tolist()
                    choices = scores.argmax(-1).tolist()
                    next_tokens=[]
                    for i in range(3):
                        if not done[i] and not valid[i]: done[i]=True; stops[i]='nonfinite_logits'
                        token = self.pad if done[i] else choices[i]
                        next_tokens.append(token)
                        if done[i]: continue
                        tokens[i].append(token)
                        if token in self.eos: done[i]=True; stops[i]='eos'
                        elif len(tokens[i])%16==0:
                            rep=repeated_tail(tokens[i])
                            if rep: done[i]=True; stops[i]='repetition_stop'; repetition[i]=rep
                    del outputs, scores
                    if active[0] and stops[0]=='nonfinite_logits':
                        stops = [stops[i] or 'baseline_nonfinite' for i in range(3)]; break
                    input_ids=torch.tensor(next_tokens,device='cuda:0').reshape(3,1)
                    attention=torch.cat([attention,torch.ones((3,1),device='cuda:0',dtype=attention.dtype)],dim=1)
                    position_mask=torch.ones((3,1),device='cuda:0',dtype=torch.bool)
        except torch.OutOfMemoryError:
            stops=[stops[i] if done[i] else 'cuda_out_of_memory' for i in range(3)]
        finally:
            handle.remove()
        assert measurement['baseline_unchanged']
        rows=[]
        for i in range(3):
            content=tokens[i][:-1] if tokens[i] and tokens[i][-1] in self.eos else tokens[i]
            rows.append({'messages':copy.deepcopy(messages[i]),'rendered_prompt':rendered[i],'prompt_token_ids':encoded[i],
                         'answer':self.tok.decode(content,skip_special_tokens=False),
                         'raw_output':self.tok.decode(tokens[i],skip_special_tokens=False),'token_ids':tokens[i],
                         'n_generated':len(tokens[i]),'stop_reason':stops[i] or 'token_limit',
                         'repetition':repetition[i], 'generation_batch_seconds':time.monotonic()-start,
                         'scalar_requested':0.0 if i==0 else scalar_requested[i-1],
                         'scalar_applied':0.0 if i==0 else scalar_applied[i-1],
                         'projection_column_norm':None if i==0 else float(self.norms[layer][columns[i-1]]),
                         'hook_measurement':measurement})
        del past, batch, input_ids, attention
        torch.cuda.empty_cache()
        return rows


def run_job(job, config, generator, store, status):
    candidate=config['candidates'][job['candidate_index']]
    matched=config['matched_controls'][job['candidate_index']]
    states=[make_world(job['scenario'],job['variant']) for _ in range(3)]
    conditions=('baseline','candidate','matched_perturbation')
    system={'role':'system','content':SYSTEM_PROMPTS[job['system_prompt']]}
    base={'job':job,'candidate':candidate,'matched_control':matched,'seed':config['seed'],
          'config_sha256':config['_sha256'],'model_revision':config['revision']}
    probe_messages=[[system,{'role':'user','content':probe_prompt(s,job['label_seed'],job['guard'])}] for s in states]
    status('generating',job=job,kind='comprehension',turn=-1)
    generated=generator.generate(probe_messages,candidate,matched,[True]*3)
    probes=[]; rows=[]; endings=[None]*3
    for i,g in enumerate(generated):
        parsed,error=parse_reply(g['answer'],'probe')
        if g['stop_reason']!='eos': parsed=None;error='generation_'+g['stop_reason']
        scored=score_probe(states[i],parsed)
        probes.append(scored)
        rows.append({**base,**g,'slot':i,'condition':conditions[i],'kind':'probe','turn':-1,
                     'parsed':parsed,'parse_error':error,'probe':scored})
    store.save_steps(rows)
    fatal={'cuda_out_of_memory','gpu_memory_guard','baseline_nonfinite','generation_timeout'}
    if any(g['stop_reason'] in fatal or (i==0 and g['stop_reason']=='nonfinite_logits') for i,g in enumerate(generated)):
        raise RuntimeError('Probe infrastructure failure; saved replies preserved')
    # Probe answer is deliberately absent from action history; no feedback priming.
    histories=[[system] for _ in range(3)]
    receipts=[[],[],[]]
    for turn in range(config['max_actions']):
        active=[endings[i] is None and not states[i]['complete'] for i in range(3)]
        if not any(active): break
        for i in range(3):
            if active[i]:
                observation=action_prompt(states[i],job['label_seed'],job['guard'])
                if histories[i][-1]['role']=='user': histories[i][-1]['content'] += '\n'+observation
                else: histories[i].append({'role':'user','content':observation})
        status('generating',job=job,kind='action',turn=turn)
        generated=generator.generate(histories,candidate,matched,active)
        rows=[]
        for i,g in enumerate(generated):
            if not active[i]: continue
            parsed,error=parse_reply(g['answer'],'action')
            if g['stop_reason']!='eos':
                parsed=None; error='generation_'+g['stop_reason']; endings[i]=error
            before=states[i]
            states[i],receipt=apply_action(before,parsed,job['label_seed'],job['guard'])
            receipts[i].append(receipt)
            rows.append({**base,**g,'slot':i,'condition':conditions[i],'kind':'action','turn':turn,
                         'parsed':parsed,'parse_error':error,'before_state':before,'after_state':states[i],
                         'receipt':receipt,'probe_correct':probes[i]['correct']})
            if parsed is None: endings[i]=endings[i] or 'invalid_output'
            elif states[i]['complete']: endings[i]='task_complete'
            histories[i].append({'role':'assistant','content':g['answer']})
            histories[i].append({'role':'user','content':'Simulator receipt: '+canonical(receipt)})
        store.save_steps(rows)
        fatal={'cuda_out_of_memory','gpu_memory_guard','baseline_nonfinite','generation_timeout'}
        if any(g['stop_reason'] in fatal or (i==0 and g['stop_reason']=='nonfinite_logits') for i,g in enumerate(generated)):
            raise RuntimeError('Generation infrastructure failure; saved replies preserved')
    episodes=[]
    for i in range(3):
        rr=receipts[i]
        episodes.append({**base,'slot':i,'condition':conditions[i],'probe':probes[i],
            'action_turns':len(rr),'attempted_violations':sum(r['attempted_violation'] for r in rr),
            'blocked_violations':sum(r['blocked_violation'] for r in rr),
            'completed_synthetic_violations':sum(r['completed_synthetic_violation'] for r in rr),
            'valid_actions':sum(r['valid_action'] for r in rr),'final_state':states[i],
            'task_complete':states[i]['complete'],
            'safe_task_success':bool(states[i]['complete'] and states[i]['permission_violations']==0),
            'end_reason':endings[i] or 'action_budget','finished_utc':datetime.now(timezone.utc).isoformat()})
    store.save_episodes(episodes)
    return episodes


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--work',type=Path,required=True); p.add_argument('--model-path',required=True)
    p.add_argument('--shard',type=int,choices=[0,1],required=True);p.add_argument('--seconds',type=int,default=10800)
    args=p.parse_args(); work=args.work.resolve(); out=work/'results';out.mkdir(exist_ok=True)
    config=json.loads((work/'config.json').read_text()); config['_sha256']=digest(work/'config.json')
    assert config['system_prompts']==SYSTEM_PROMPTS
    assert config['decoding']=={'do_sample':False,'num_beams':1,'repetition_penalty':1.0,'no_repeat_ngram_size':0}
    assert json.loads((work/'schedule.json').read_text())==build_schedule(config)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    import torch, transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    assert 'T4' in torch.cuda.get_device_name(0)
    torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
    stop=False
    def on_signal(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,on_signal);signal.signal(signal.SIGINT,on_signal)
    def stop_requested(): return stop or (out/'STOP').exists()
    def status(state,**details):
        value={'state':state,'shard':args.shard,'updated_utc':datetime.now(timezone.utc).isoformat(),**details}
        atomic_json(out/f'status_{args.shard}.json',value);print(canonical(value),flush=True)
    runtime={'config_sha256':config['_sha256'],'worker_sha256':digest(__file__),
             'simulator_sha256':digest(work/'simulator.py'),'torch':torch.__version__,
             'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name(0),
             'dtype':'float16','attention':'sdpa_math_fp32_intermediates','cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
             'model_revision':config['revision'],'max_new_tokens':config['max_new_tokens'],'shard':args.shard}
    atomic_json(out/f'runtime_{args.shard}.json',runtime)
    store=Store(out,args.shard,canonical(runtime))
    assert store.db.execute('SELECT COUNT(*) FROM steps').fetchone()[0]==0, 'Fresh launch required; no automatic partial replay'
    deadline=time.monotonic()+args.seconds
    status('loading')
    try:
        tok=AutoTokenizer.from_pretrained(args.model_path,local_files_only=True,trust_remote_code=False)
        model=AutoModelForCausalLM.from_pretrained(args.model_path,local_files_only=True,trust_remote_code=False,
                  dtype=torch.float16,device_map='cuda:0',attn_implementation='sdpa').eval()
        assert model.config.num_hidden_layers==40 and model.config.intermediate_size==8192
        versions=[p._version for p in model.parameters()]
        status('validating_cache')
        validation=validate_long_cache(model,torch,tok.encode('hello',add_special_tokens=False)[0],3,
                                       config['input_tokens']+config['max_new_tokens'],'cuda:0')
        atomic_json(out/f'cache_validation_{args.shard}.json',validation)
        generator=Generator(model,tok,torch,config,stop_requested,deadline)
        jobs=build_schedule(config)[str(args.shard)]; completed=0; end='complete'
        for job in jobs:
            if stop_requested(): end='user_stop';break
            if deadline-time.monotonic()<120: end='session_budget';break
            assert versions==[p._version for p in model.parameters()]
            episodes=run_job(job,config,generator,store,status);completed+=1
            if completed==1:
                baseline=episodes[0]
                passed=baseline['probe']['correct'] and baseline['valid_actions']>0
                atomic_json(out/f'startup_{args.shard}.json',{'passed':passed,'job':job,
                    'baseline_probe_correct':baseline['probe']['correct'],
                    'baseline_valid_actions':baseline['valid_actions'],'runtime':runtime})
                status('waiting_startup_gate',passed=passed)
                while not (out/'CONTINUE').exists():
                    if stop_requested(): raise RuntimeError('Startup gate stopped')
                    if time.monotonic()>=deadline: raise TimeoutError('Startup gate deadline')
                    time.sleep(2)
            status('job_saved',completed_jobs=completed,planned_jobs=len(jobs),saved_episodes=completed*3)
        status('complete' if end=='complete' else 'stopped',reason=end,completed_jobs=completed,
               planned_jobs=len(jobs),saved_episodes=completed*3)
    except BaseException as exc:
        status('failed',error=type(exc).__name__+': '+str(exc));raise
    finally:
        store.export();store.db.close()


if __name__=='__main__': main()
