"""Persistent local text chat. Requests arrive one at a time through the inbox."""
import os
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  TOKENIZERS_PARALLELISM='false', PYTORCH_ENABLE_MPS_FALLBACK='0')
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import socket
import time
import traceback
import psutil
from protocol import atomic_json, validate, valid_id
from repetition import repeated_tail

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PLAN = json.loads((HERE / 'plan.json').read_text())
LIMITS = PLAN['limits']
STOP = False
TORCH = None
THERMAL = None
DEADLINE = float('inf')
SWAP_START = psutil.swap_memory().used
LAST_HEARTBEAT = 0.0


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def status(state, **extra):
    atomic_json(HERE / 'status.json', dict(state=state, pid=os.getpid(), updated_utc=now(), **extra))


def safety():
    global LAST_HEARTBEAT
    if time.monotonic() - LAST_HEARTBEAT >= 15:
        state = json.loads((HERE / 'status.json').read_text())
        state['updated_utc'] = now()
        atomic_json(HERE / 'status.json', state)
        LAST_HEARTBEAT = time.monotonic()
    if STOP or (HERE / 'STOP').exists():
        raise InterruptedError('user_stop')
    if time.monotonic() >= DEADLINE:
        raise InterruptedError('wall_time_limit')
    if THERMAL is not None and THERMAL.state() >= 2:
        raise RuntimeError('thermal_pressure_stop')
    if psutil.virtual_memory().available < LIMITS['minimum_available_memory_bytes']:
        raise RuntimeError('low_available_memory')
    if psutil.swap_memory().used > SWAP_START + LIMITS['maximum_swap_growth_bytes']:
        raise RuntimeError('swap_growth_limit')
    if TORCH is not None and TORCH.mps.driver_allocated_memory() > LIMITS['max_gpu_driver_bytes']:
        TORCH.mps.empty_cache()
        if TORCH.mps.driver_allocated_memory() > LIMITS['max_gpu_driver_bytes']:
            raise RuntimeError('gpu_memory_limit')


def rest(seconds):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        safety()
        time.sleep(min(1, max(0, until-time.monotonic())))


def nominal():
    start = time.monotonic()
    while THERMAL.state() != 0:
        safety()
        if time.monotonic() - start > 300:
            raise RuntimeError('thermal_recovery_timeout')
        rest(1)


def stop_signal(signum, frame):
    global STOP
    STOP = True


def main():
    global TORCH, THERMAL, DEADLINE
    status('starting', completed_turns=0)
    DEADLINE = time.monotonic() + LIMITS['wall_seconds']
    plan_hash = digest(HERE / 'plan.json')
    assert plan_hash == (HERE / 'plan.sha256').read_text().strip()
    assert not (HERE / 'runtime.json').exists(), 'Recorded live session cannot be overwritten'
    assert digest(HERE / 'initial_history.json') == PLAN['initial_history_sha256']
    history = json.loads((HERE / 'initial_history.json').read_text())['messages']
    lock = ROOT / '.handover-model.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    model = handle = None
    turns = 0
    conversation = 1
    reason = 'complete'
    request_id = None
    try:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, stop_signal)
        reference = ROOT / PLAN['checkpoint_reference']
        assert digest(reference) == PLAN['checkpoint_reference_sha256']
        manifest = json.loads(reference.read_text())
        model_path = ROOT / PLAN['model']
        status('verifying_checkpoint', completed_turns=0)
        for name, item in manifest['checkpoint_files'].items():
            safety()
            assert digest(model_path / name) == item['sha256'], name
        with socket.socket() as probe:
            probe.settimeout(2)
            try:
                probe.connect(('1.1.1.1',443))
            except PermissionError:
                pass
            else:
                raise RuntimeError('Network-denial profile not active')
        helper = ROOT / 'granite_channel_identity_resume_20261113/run_test.py'
        spec = importlib.util.spec_from_file_location('thermal_helpers', helper)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        TORCH = module.torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
        TORCH.set_num_threads(1)
        TORCH.set_num_interop_threads(1)
        TORCH.use_deterministic_algorithms(True, warn_only=True)
        assert TORCH.backends.mps.is_available()
        TORCH.mps.set_per_process_memory_fraction(.5)
        THERMAL = module.Thermal()
        nominal()
        safety()
        if psutil.virtual_memory().available < 10 * 1024**3:
            raise RuntimeError('Need 10GiB available before loading')
        status('loading', completed_turns=0)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
            dtype=TORCH.bfloat16, device_map='mps', attn_implementation='eager').eval()
        channel = PLAN['channel']
        projection = model.model.layers[channel['layer']].mlp.down_proj
        norm = projection.weight[:,channel['neuron']].float().to(channel['norm_device']).norm().item()
        assert norm == channel['expected_column_norm']
        scalar = channel['dose'] * channel['median'] / norm
        delta = TORCH.zeros((1,1,projection.in_features), dtype=TORCH.bfloat16, device='mps')
        delta[0,0,channel['neuron']] = scalar
        assert delta[0,0,channel['neuron']].item() == channel['expected_scalar_bf16']
        version = projection.weight._version
        atomic_json(HERE / 'runtime.json', dict(pid=os.getpid(), started_utc=now(), plan_sha256=plan_hash,
            script_sha256=digest(Path(__file__)), checkpoint_files=manifest['checkpoint_files'],
            channel=channel, column_norm=norm, scalar_requested=scalar, scalar_bf16=delta[0,0,channel['neuron']].item(),
            model=PLAN['model'], device='mps', dtype='bfloat16', torch=TORCH.__version__,
            transformers=module.transformers.__version__, decoding=PLAN['decoding'], limits=LIMITS,
            network_denied=True, tools=[], residual_multiplier=model.config.residual_multiplier))
        status('startup_cooling', completed_turns=0)
        rest(LIMITS['startup_cooling_seconds'])

        class Guard(StoppingCriteria):
            def __init__(self, prefix):
                self.prefix = prefix
                self.start = time.monotonic()
                self.reason = None
                self.repeat = None
                self.thermal_max = 0
            def __call__(self, input_ids, scores, **kwargs):
                try:
                    count = input_ids.shape[1] - self.prefix
                    self.thermal_max = max(self.thermal_max, THERMAL.state())
                    safety()
                    if time.monotonic()-self.start > LIMITS['max_generation_seconds']:
                        raise RuntimeError('generation_timeout')
                    if count % PLAN['repetition']['check_every_tokens'] == 0:
                        settings = {k:v for k,v in PLAN['repetition'].items() if k != 'check_every_tokens'}
                        self.repeat = repeated_tail(input_ids[0,self.prefix:].tolist(), **settings)
                        if self.repeat:
                            self.reason = 'repetition_stop'
                            return True
                    nominal()
                    if count % LIMITS['rest_every_tokens'] == 0:
                        TORCH.mps.synchronize()
                        rest(LIMITS['intrageneration_rest_seconds'])
                except (RuntimeError, InterruptedError) as exc:
                    self.reason = str(exc)
                    return True
                return False

        eos = model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos,list) else [eos])
        last_activity = time.monotonic()
        next_generation = time.monotonic()
        atomic_json(HERE / 'history.json', dict(conversation=conversation, messages=history))
        with (HERE/'responses.jsonl').open('x') as log, (HERE/'TRANSCRIPT.txt').open('x') as transcript:
            transcript.write('INITIAL SAVED CONTEXT:\n' + json.dumps(history,ensure_ascii=False,indent=2) + '\n\n')
            transcript.flush()
            while True:
                status('ready', completed_turns=turns, conversation=conversation)
                while True:
                    safety()
                    pending_files = sorted((HERE/'inbox').glob('*.json'))
                    if pending_files:
                        break
                    if time.monotonic()-last_activity >= LIMITS['idle_seconds']:
                        raise InterruptedError('idle_timeout')
                    rest(1)
                path = pending_files[0]
                request_id = path.stem
                assert valid_id(request_id)
                try:
                    request = validate(json.loads(path.read_text()))
                except (ValueError, json.JSONDecodeError) as exc:
                    atomic_json(HERE/'replies'/path.name, dict(state='rejected', error=str(exc)))
                    path.rename(HERE/'processed'/path.name)
                    continue
                last_activity = time.monotonic()
                if request['action'] in {'reset','close'}:
                    if request['action']=='reset':
                        history=[]
                        conversation+=1
                        atomic_json(HERE/'history.json',dict(conversation=conversation,messages=history))
                    atomic_json(HERE/'replies'/path.name, dict(state='acknowledged',action=request['action']))
                    path.rename(HERE/'processed'/path.name)
                    if request['action']=='close':
                        break
                    continue
                messages = history + [{'role':'user','content':request['message']}]
                rendered = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                inputs = tokenizer(rendered,add_special_tokens=False,return_tensors='pt')
                prefix = inputs['input_ids'].shape[1]
                if prefix > LIMITS['input_token_limit']:
                    atomic_json(HERE/'replies'/path.name,dict(state='rejected',reason='context_limit',input_tokens=prefix))
                    path.rename(HERE/'processed'/path.name)
                    continue
                pause = max(0,next_generation-time.monotonic())
                if pause:
                    status('cooling',completed_turns=turns,seconds=pause)
                    rest(pause)
                nominal()
                safety()
                inputs=inputs.to('mps')
                guard=Guard(prefix)
                assert projection.weight._version == version
                details={'hook_calls':0}
                def push(module,args):
                    changed=args[0]+delta
                    details['hook_calls']+=1
                    if details['hook_calls']==1:
                        diff=(changed[...,channel['neuron']]-args[0][...,channel['neuron']]).float()
                        details['first_hook_mean_actual_delta']=diff.mean().item()
                    return (changed,)+args[1:]
                handle=projection.register_forward_pre_hook(push)
                TORCH.manual_seed(20261112)
                TORCH.mps.manual_seed(20261112)
                status('generating',completed_turns=turns,request_id=request_id)
                try:
                    with TORCH.inference_mode():
                        generated=model.generate(**inputs,**PLAN['decoding'],pad_token_id=tokenizer.eos_token_id,
                            stopping_criteria=StoppingCriteriaList([guard]))
                    TORCH.mps.synchronize()
                finally:
                    handle.remove()
                    handle=None
                tokens=generated[0,prefix:].tolist()
                ended=bool(tokens and tokens[-1] in eos)
                answer=tokenizer.decode(tokens[:-1] if ended else tokens,skip_special_tokens=False)
                raw=tokenizer.decode(tokens,skip_special_tokens=False)
                elapsed=time.monotonic()-guard.start
                turns+=1
                record=dict(state='answered',request_id=request_id,turn=turns,conversation=conversation,
                    prompt=request['message'],messages=messages,rendered_prompt=rendered,
                    prompt_token_ids=inputs['input_ids'][0].tolist(),token_ids=tokens,answer=answer,raw_output=raw,
                    n_generated=len(tokens),stop_reason=guard.reason or ('eos' if ended else 'token_limit'),
                    generation_seconds=elapsed,thermal_max=guard.thermal_max,repetition=guard.repeat,
                    intervention=details,plan_sha256=plan_hash,finished_utc=now())
                log.write(json.dumps(record,ensure_ascii=False)+'\n')
                log.flush()
                os.fsync(log.fileno())
                transcript.write(f'TURN {turns} | CONVERSATION {conversation}\nPROMPT:\n{request["message"]}\n\nANSWER:\n{raw}\n\nSTOP: {record["stop_reason"]}\n\n')
                transcript.flush()
                history=messages+[{'role':'assistant','content':answer}]
                atomic_json(HERE/'history.json',dict(conversation=conversation,messages=history))
                atomic_json(HERE/'replies'/path.name,record)
                path.rename(HERE/'processed'/path.name)
                del generated,inputs
                TORCH.mps.empty_cache()
                next_generation=time.monotonic()+max(LIMITS['minimum_cooling_seconds'],LIMITS['cooling_generation_multiplier']*elapsed)
                last_activity=time.monotonic()
                if guard.reason and guard.reason not in {'repetition_stop','generation_timeout'}:
                    raise InterruptedError(guard.reason)
        status('complete',completed_turns=turns)
    except InterruptedError as exc:
        reason=str(exc)
        status('stopped',reason=reason,completed_turns=turns)
    except Exception as exc:
        reason=type(exc).__name__+': '+str(exc)
        status('failed',reason=reason,completed_turns=turns)
        traceback.print_exc()
    finally:
        if handle is not None:
            handle.remove()
        if model is not None:
            del model
        if TORCH is not None and TORCH.backends.mps.is_available():
            TORCH.mps.empty_cache()
        if lock.exists() and lock.read_text().strip()==str(os.getpid()):
            lock.unlink()
        atomic_json(HERE/f'closed_{os.getpid()}.json',dict(closed_utc=now(),reason=reason,completed_turns=turns))


if __name__=='__main__':
    main()
