#!/usr/bin/env python3
"""Local interactive Granite chat at one recorded setting; output is text only."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import socket
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def status(state,**extra):
    value={'state':state,'updated_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),**extra}
    tmp=HERE/'status.tmp';tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(HERE/'status.json')


def display(text):
    # Model text cannot emit terminal control sequences. Exact raw text is logged.
    return ''.join(ch if ch in '\n\t' or (ord(ch)>=32 and not 127<=ord(ch)<160) else f'\\u{ord(ch):04x}' for ch in text)


def cooling_seconds(key):
    settings=json.loads((HERE/'cooling.json').read_text())
    value=float(settings[key])
    if not 0<=value<=600:raise ValueError('Invalid cooling interval.')
    return value


def main():
    if (HERE/'runtime.json').exists() or (HERE/'closed.json').exists():
        raise SystemExit('This recorded chat session has already been used. Request a new session to preserve its logs.')
    status('starting')
    print('\nGranite terminal chat — L4 N657, saved dose -5\n',flush=True)
    print('Loading the local model. The survey resumes after you exit.\n',flush=True)
    spec=importlib.util.spec_from_file_location('chat_guards',ROOT/'granite_channel_identity_resume_20261113/run_test.py')
    guardmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(guardmod);guardmod.HERE=HERE
    torch=guardmod.torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,StoppingCriteriaList
    def quit_signal(signum,frame):
        guardmod.STOP_REQUESTED=True
        raise KeyboardInterrupt
    for sig in [signal.SIGTERM,signal.SIGHUP]:signal.signal(sig,quit_signal)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True,warn_only=True)
    assert torch.backends.mps.is_available(),'MPS unavailable.'
    torch.mps.set_per_process_memory_fraction(.65)
    thermal=guardmod.Thermal();guardmod.wait_nominal(thermal,[])
    try:
        with socket.socket() as s:
            s.settimeout(2);s.connect(('1.1.1.1',443))
        raise RuntimeError('Expected the OS network-denial profile.')
    except PermissionError:pass
    lock=ROOT/'.handover-model.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    model=None;reason='normal_exit';turn=0;handle=None;last_finished=None
    try:
        parent=json.loads((ROOT/'granite_channel_identity_resume_20261113/runtime.json').read_text())
        path=ROOT/'models/granite-4.1-3b-base'
        files={p.name:{'bytes':p.stat().st_size,'sha256':guardmod.digest(p)} for p in sorted(path.iterdir()) if p.is_file() and p.suffix in ['.safetensors','.json','.jinja','.txt']}
        assert files==parent['checkpoint_files'],'Checkpoint changed.'
        source=json.loads((ROOT/'layer_who_L4.jsonl').read_text().splitlines()[156])
        assert (source['layer'],source['neuron'],source['dose'])==(4,657,-5.0)
        tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
        model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16).to('mps').eval()
        projection=model.model.layers[4].mlp.down_proj
        norm=projection.weight[:,657].float().cpu().norm().item()
        scalar=source['dose']*source['median']/norm
        add=torch.zeros(1,1,projection.in_features,dtype=torch.bfloat16)
        add[0,0,657]=scalar;add=add.to('mps')
        runtime={'model':str(path),'checkpoint_files':files,'device':'mps','dtype':'bfloat16','layer':4,'neuron':657,
            'dose':source['dose'],'recorded_median':source['median'],'column_norm':norm,'scalar_before_bf16':scalar,'scalar_bf16':add[0,0,657].item(),
            'prompt_format':'chat_template','decoding':{'max_new_tokens':500,'do_sample':False},'history_token_limit':4096,
            'source':'layer_who_L4.jsonl:157','source_file_sha256':guardmod.digest(ROOT/'layer_who_L4.jsonl'),
            'chat_script_sha256':guardmod.digest(Path(__file__)),'network_denied':True,'tools':[],
            'limits':'Historical checkpoint hash unavailable. Recorded median retained; current checkpoint and scalar logged. This chat is exploratory, not a controlled survey.'}
        (HERE/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
        startup=cooling_seconds('startup_seconds')
        print(f'Model loaded on GPU. Taking a {startup:g}-second cooling break before the first reply.',flush=True)
        status('cooling_before_chat',seconds=startup);guardmod.rest(startup);guardmod.wait_nominal(thermal,[])
        print('\nReady. Type a question and press Return.\n/quit = exit and resume survey\n/reset = fresh conversation\n/help = commands\n\nReplies allow up to 500 tokens. Cooling happens before another generation.\nEvery prompt and full reply is saved in this session folder.\n',flush=True)
        messages=[];conversation=1;last_end=None
        if (HERE/'restore_history.json').exists():
            restored=json.loads((HERE/'restore_history.json').read_text())
            messages=restored['messages'];conversation=restored['conversation'];turn=restored['turn']
            print(f'Restored {turn} completed replies. Continue your conversation.\n',flush=True)
        eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
        with (HERE/'responses.jsonl').open('x') as log,(HERE/'TRANSCRIPT.txt').open('x') as transcript:
            while True:
                status('ready',completed_turns=turn,conversation=conversation)
                try:question=input('You > ').strip()
                except EOFError:break
                if not question:continue
                if question.lower() in {'/quit','/exit'}:break
                if question.lower()=='/reset':
                    messages=[];conversation+=1;print('Fresh conversation.\n',flush=True);continue
                if question.lower() in {'/help','/status'}:
                    print('L4 N657 | dose -5 | greedy | max500 | MPS\n/reset: clear conversation. /quit: exit and resume survey.\n',flush=True);continue
                if question.startswith('/'):
                    print('Unknown command. Use /help.\n',flush=True);continue
                pending=messages+[{'role':'user','content':question}]
                encoded=tok.apply_chat_template(pending,add_generation_prompt=True,return_tensors='pt')
                ids=encoded if torch.is_tensor(encoded) else encoded['input_ids']
                if ids.shape[1]>4096:
                    print('Conversation reached the 4096-token limit. Use /reset for a fresh conversation.\n',flush=True);continue
                pause=max(0,last_end+cooling_seconds('between_replies_seconds')-time.monotonic()) if last_end is not None else 0
                if pause:
                    print(f'Cooling for {pause:.0f} more seconds before this reply…',flush=True)
                    status('cooling',completed_turns=turn,seconds=round(pause));guardmod.rest(pause)
                guardmod.wait_nominal(thermal,[])
                ids=ids.to('mps');prefix=ids.shape[1]
                guard=guardmod.GenerationGuard(thermal,prefix);calls=[0]
                def hook(module,args):
                    calls[0]+=1
                    return (args[0]+add,)+args[1:]
                torch.manual_seed(0);torch.mps.manual_seed(0)
                start=time.monotonic();status('generating',completed_turns=turn)
                print('Granite is answering…',flush=True)
                handle=projection.register_forward_pre_hook(hook)
                try:
                    with torch.inference_mode():
                        generated=model.generate(ids,max_new_tokens=500,do_sample=False,pad_token_id=tok.eos_token_id,
                            stopping_criteria=StoppingCriteriaList([guard]))
                    torch.mps.synchronize()
                finally:
                    handle.remove();handle=None
                elapsed=time.monotonic()-start;tokens=generated[0,prefix:].tolist()
                answer=tok.decode(tokens,skip_special_tokens=True).strip()
                ended=bool(tokens and tokens[-1] in eos);turn+=1
                last_finished=datetime.now(timezone.utc).isoformat()
                record={'turn':turn,'conversation':conversation,'messages':pending,'answer':answer,'prompt_token_ids':ids[0].tolist(),'token_ids':tokens,
                    'n_generated':len(tokens),'stop_reason':guard.reason or ('eos' if ended else 'token_limit'),'hook_calls':calls[0],
                    'generation_seconds':elapsed,'thermal_max':guard.maximum_thermal_state,'thermal_after':thermal.state(),'finished_utc':last_finished}
                log.write(json.dumps(record,ensure_ascii=False)+'\n');log.flush();os.fsync(log.fileno())
                transcript.write(f'TURN {turn} | CONVERSATION {conversation}\nPROMPT:\n{question}\n\nANSWER:\n{answer}\n\nSTOP: {record["stop_reason"]}\n\n');transcript.flush()
                print('\nGranite > '+display(answer)+'\n',flush=True)
                print(f'[{len(tokens)} tokens; {record["stop_reason"]}]\n',flush=True)
                messages=pending+[{'role':'assistant','content':answer}]
                last_end=time.monotonic()
                del generated,ids;torch.mps.empty_cache()
                if guard.reason:reason='generation_guard_stop';break
    except KeyboardInterrupt:
        reason='user_interrupt'
    except Exception:
        reason='error'
        raise
    finally:
        if handle is not None:handle.remove()
        if model is not None:del model
        torch.mps.empty_cache()
        if lock.exists() and lock.read_text().strip()==str(os.getpid()):lock.unlink()
        closed={'closed_utc':datetime.now(timezone.utc).isoformat(),'reason':reason,'completed_turns':turn,'last_generation_finished_utc':last_finished,'pid':os.getpid()}
        (HERE/'closed.json').write_text(json.dumps(closed,indent=2)+'\n')
        status('closed',**{k:v for k,v in closed.items() if k!='pid'})
        print('\nChat closed. The survey supervisor will resume the remaining questions after cooling.\n',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        status('failed',error=type(exc).__name__+': '+str(exc))
        raise
