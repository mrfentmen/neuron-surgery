#!/usr/bin/env python3
"""Fixed raw-offset identity check and undosed activation statistics."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import socket
import time
import traceback
from reasoning_format import parse_reasoning
from repetition import repeated_tail
import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STOP = False
torch = None
thermal = None
deadline = float('inf')
swap_start = psutil.swap_memory().used
plan = json.loads((HERE / 'plan.json').read_text())
limits = plan['limits']
last_heartbeat = 0.0


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def status(state, **extra):
    obj = {'state': state, 'updated_utc': now(), 'pid': os.getpid(), 'total_jobs': plan['total_jobs'], **extra}
    tmp = HERE / 'status.tmp'
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(HERE / 'status.json')
    print(json.dumps(obj), flush=True)


def heartbeat(**extra):
    global last_heartbeat
    if time.monotonic() - last_heartbeat < limits['heartbeat_seconds'] and not extra:
        return
    path = HERE / 'status.json'
    if path.exists():
        obj = json.loads(path.read_text())
        obj.update(updated_utc=now(), **extra)
        tmp = HERE / 'heartbeat.tmp'
        tmp.write_text(json.dumps(obj, indent=2) + '\n')
        tmp.replace(path)
    last_heartbeat = time.monotonic()


def stopped(signum, frame):
    global STOP
    STOP = True


def safety():
    heartbeat()
    if STOP or (HERE / 'STOP').exists():
        raise InterruptedError('user_stop')
    if time.monotonic() >= deadline:
        raise InterruptedError('wall_time_limit')
    if thermal is not None and thermal.state() >= 2:
        raise RuntimeError('thermal_pressure_stop')
    if psutil.virtual_memory().available < limits['minimum_available_memory_bytes']:
        raise RuntimeError('available_memory_below_3GiB')
    if psutil.swap_memory().used > swap_start + limits['maximum_swap_growth_bytes']:
        raise RuntimeError('swap_growth_above_512MiB')
    if torch is not None and torch.mps.driver_allocated_memory() > limits['max_gpu_driver_bytes']:
        torch.mps.empty_cache()
        if torch.mps.driver_allocated_memory() > limits['max_gpu_driver_bytes']:
            raise RuntimeError('gpu_driver_memory_above_12GiB')


def rest(seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        safety()
        time.sleep(min(1, max(0, end - time.monotonic())))


def nominal():
    start = time.monotonic()
    while thermal.state() != 0:
        safety()
        if time.monotonic() - start > 300:
            raise RuntimeError('thermal_recovery_timeout')
        rest(1)


def main():
    global torch, thermal, deadline
    deadline = time.monotonic() + limits['wall_seconds']
    plan_hash = digest(HERE / 'plan.json')
    assert plan_hash == (HERE / 'plan.sha256').read_text().strip(), 'Frozen plan changed'
    assert digest(HERE / plan['source_response']) == plan['source_response_sha256']
    source = json.loads((HERE / plan['source_response']).read_text())
    for job in plan['jobs']:
        if job['history_included']:
            assert job['messages'] == [{'role':'user', 'content':source['prompt']},
                                      {'role':'assistant', 'content':source['answer']},
                                      {'role':'user', 'content':job['prompt']}]
    previous = []
    output = HERE / 'responses.jsonl'
    if output.exists():
        # A partial tail is a hard stop for review, never silently discarded.
        previous = [json.loads(line) for line in output.read_text().splitlines()]
        assert all(r['job']['job_id'] == i and r['plan_sha256'] == plan_hash for i, r in enumerate(previous))
    if previous and previous[-1]['stop_reason'] not in ('eos', 'token_limit', 'repetition_stop', 'generation_timeout'):
        raise RuntimeError('Last reply stopped on a guard; review before resuming.')
    completed = len(previous)
    if completed == len(plan['jobs']):
        status('complete', completed=completed)
        return
    lock = ROOT / '.handover-model.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    model = None
    handle = None
    reason = 'complete'
    try:
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, stopped)
        status('verifying_checkpoint', completed=completed)
        reference = ROOT / plan['checkpoint_reference']
        assert digest(reference) == plan['checkpoint_reference_sha256']
        manifest = json.loads(reference.read_text())
        assert digest(ROOT / plan['proposal_source']) == plan['proposal_sha256']
        model_path = ROOT / plan['model']
        for name, item in manifest['checkpoint_files'].items():
            safety()
            assert digest(model_path / name) == item['sha256'], name
        with socket.socket() as probe:
            probe.settimeout(2)
            try:
                probe.connect(('1.1.1.1', 443))
            except PermissionError:
                pass
            else:
                raise RuntimeError('OS network denial was not active')
        spec = importlib.util.spec_from_file_location('thermal_helpers', ROOT / 'granite_channel_identity_resume_20261113/run_test.py')
        helpers = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helpers)
        torch = helpers.torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True, warn_only=True)
        assert torch.backends.mps.is_available(), 'MPS unavailable'
        torch.mps.set_per_process_memory_fraction(.5)
        thermal = helpers.Thermal()
        nominal()
        safety()
        if psutil.virtual_memory().available < 10 * 1024**3:
            raise RuntimeError('Need at least 10GiB available before loading')
        status('loading', completed=completed)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                    trust_remote_code=False, dtype=torch.bfloat16, device_map='mps',
                    attn_implementation='eager').eval()
        safety()
        status('startup_cooling', completed=completed, seconds=limits['startup_cooling_seconds'])
        rest(limits['startup_cooling_seconds'])
        nominal()
        assert model.config.num_hidden_layers == 40
        medians = {s['layer']: s['median'] for s in plan['channels']}
        norms = {}
        calibrated = {}
        for s in plan['channels']:
            layer, neuron = s['layer'], s['neuron']
            column = model.model.layers[layer].mlp.down_proj.weight[:, neuron].float()
            actual = column.to(s['norm_device']).norm().item()
            if 'expected_column_norm' in s:
                assert actual == s['expected_column_norm'], 'Column norm differs from saved local run'
            norms[(layer, neuron)] = actual
            scalar = s['raw_offset']
            applied = torch.tensor(scalar, dtype=torch.bfloat16).item()
            if 'expected_scalar_bf16' in s:
                assert applied == s['expected_scalar_bf16'], 'Scalar differs from saved local run'
            calibrated[s['arm']] = dict(column_norm=actual, median=s['median'], scalar_requested=scalar,
                                         scalar_bf16=applied, norm_device=s['norm_device'])
            del column
        versions = [layer.mlp.down_proj.weight._version for layer in model.model.layers]
        (HERE / 'calibration_reference.json').write_text(json.dumps({
            'channels': calibrated, 'checkpoint_sha256_verified': True}, indent=2) + '\n')
        torch.mps.empty_cache()
        runtime = {'started_utc': now(), 'plan_sha256': plan_hash, 'script_sha256': digest(Path(__file__)),
                   'repetition_sha256': digest(HERE / 'repetition.py'),
                   'model': plan['model'], 'checkpoint_files': manifest['checkpoint_files'],
                   'torch': torch.__version__, 'transformers': helpers.transformers.__version__,
                   'network_denied': True, 'tools': [], 'dtype': 'bfloat16', 'device': 'mps',
                   'swap_at_start_bytes': swap_start, 'mps_recommended_max_bytes': torch.mps.recommended_max_memory(),
                   'mps_allocator_fraction': .5, 'completed_at_start': completed, 'pid': os.getpid(),
                   'calibration': calibrated, 'decoding': plan['decoding'],
                   'residual_multiplier': model.config.residual_multiplier}
        (HERE / f'runtime_{os.getpid()}.json').write_text(json.dumps(runtime, indent=2) + '\n')
        from measure_baseline import measure
        measure(model, tokenizer, torch, HERE, plan, safety, rest, nominal, status)
        rest(limits['minimum_cooling_seconds'])

        class Guard(StoppingCriteria):
            def __init__(self, prefix, job, rendered, prompt_tokens):
                self.prefix = prefix
                self.job, self.rendered, self.prompt_tokens = job, rendered, prompt_tokens
                self.start = time.monotonic()
                self.reason = None
                self.max_thermal = 0
                self.repetition = None
            def __call__(self, input_ids, scores, **kwargs):
                try:
                    count = input_ids.shape[1] - self.prefix
                    if count % limits['checkpoint_every_tokens'] == 0:
                        tokens = input_ids[0, self.prefix:].tolist()
                        obj = {'job': self.job, 'prompt': self.job['prompt'], 'rendered_prompt': self.rendered,
                               'prompt_token_ids': self.prompt_tokens, 'token_ids': tokens,
                               'raw_output': tokenizer.decode(tokens, skip_special_tokens=False),
                               'n_generated': count, 'plan_sha256': plan_hash,
                               'updated_utc': now(), 'status': 'inflight_partial'}
                        tmp = HERE / 'inflight.tmp'
                        with tmp.open('w') as stream:
                            stream.write(json.dumps(obj, ensure_ascii=False) + '\n')
                            stream.flush()
                            os.fsync(stream.fileno())
                        tmp.replace(HERE / 'inflight.json')
                        heartbeat(generated_tokens=count, generation_seconds=round(time.monotonic()-self.start, 2),
                                  gpu_driver_bytes=torch.mps.driver_allocated_memory())
                    self.max_thermal = max(self.max_thermal, thermal.state())
                    safety()
                    if time.monotonic() - self.start > limits['max_generation_seconds']:
                        raise RuntimeError('generation_timeout')
                    if count % plan['repetition']['check_every_tokens'] == 0:
                        settings = {k: v for k, v in plan['repetition'].items() if k != 'check_every_tokens'}
                        self.repetition = repeated_tail(input_ids[0, self.prefix:].tolist(), **settings)
                        if self.repetition:
                            self.reason = 'repetition_stop'
                            return True
                    nominal()
                    if (input_ids.shape[1] - self.prefix) % limits['rest_every_tokens'] == 0:
                        torch.mps.synchronize()
                        rest(limits['intrageneration_rest_seconds'])
                except (RuntimeError, InterruptedError) as exc:
                    self.reason = str(exc)
                    return True
                return False

        eos = model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        with output.open('a') as log, (HERE / 'TRANSCRIPT.txt').open('a') as transcript:
            for job in plan['jobs'][completed:]:
                safety()
                if (HERE / 'STOP_AFTER_REPLY').exists():
                    raise InterruptedError('user_stop_after_reply')
                nominal()
                assert versions == [layer.mlp.down_proj.weight._version for layer in model.model.layers], 'Projection weights changed'
                if job['prompt_format'] == 'chat_template':
                    messages = job['messages'] or [{'role': 'user', 'content': job['prompt']}]
                    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = tokenizer(rendered, add_special_tokens=False, return_tensors='pt')
                else:
                    assert job['prompt_format'] == 'plain_text'
                    rendered = job['prompt']
                    inputs = tokenizer(rendered, return_tensors='pt')
                prefix = inputs['input_ids'].shape[1]
                assert prefix <= limits['input_token_limit']
                inputs = inputs.to('mps')
                details = {'hook_calls': 0}
                if job['condition'] == 'intervention':
                    layer, neuron = job['layer'], job['neuron']
                    projection = model.model.layers[layer].mlp.down_proj
                    scalar = job['raw_offset']
                    delta = torch.zeros((1, 1, projection.in_features), dtype=torch.bfloat16, device='mps')
                    delta[0, 0, neuron] = scalar
                    assert delta[0,0,neuron].item() == calibrated[job['condition_id']]['scalar_bf16']
                    details.update(scalar_requested=scalar, scalar_bf16=delta[0,0,neuron].item(),
                                   median=medians[layer], column_norm=norms[(layer,neuron)])
                    def push(module, args):
                        changed = args[0] + delta
                        details['hook_calls'] += 1
                        if details['hook_calls'] == 1:
                            diff = (changed[...,neuron] - args[0][...,neuron]).float()
                            details['first_hook_mean_actual_delta'] = diff.mean().item()
                            details['first_hook_positions_changed'] = (diff != 0).sum().item()
                        return (changed,) + args[1:]
                    handle = projection.register_forward_pre_hook(push)
                torch.manual_seed(job['seed'])
                torch.mps.manual_seed(job['seed'])
                guard = Guard(prefix, job, rendered, inputs['input_ids'][0].tolist())
                status('generating', completed=completed, job=job)
                try:
                    with torch.inference_mode():
                        generated = model.generate(**inputs, **plan['decoding'], pad_token_id=tokenizer.eos_token_id,
                                                   stopping_criteria=StoppingCriteriaList([guard]))
                    torch.mps.synchronize()
                finally:
                    if handle is not None:
                        handle.remove()
                        handle = None
                elapsed = time.monotonic() - guard.start
                tokens = generated[0, prefix:].tolist()
                ended = bool(tokens and tokens[-1] in eos)
                raw = tokenizer.decode(tokens, skip_special_tokens=False)
                content = tokenizer.decode(tokens[:-1] if ended else tokens, skip_special_tokens=False)
                from describe_reply import repetition_metrics
                text_metrics = repetition_metrics(content)
                parsed = parse_reasoning(content)
                opened, closed = parsed['reasoning_opened'], parsed['reasoning_closed']
                reasoning, final = parsed['reasoning'], parsed['final_answer']
                stop_reason = guard.reason or ('eos' if ended else 'token_limit')
                record = {'job': job, 'plan_sha256': plan_hash, 'prompt': job['prompt'], 'rendered_prompt': rendered,
                          'prompt_token_ids': inputs['input_ids'][0].tolist(), 'token_ids': tokens, 'raw_output': raw,
                          'answer': content, 'reasoning': reasoning, 'final_answer': final,
                          'reasoning_opened': opened, 'reasoning_closed': closed,
                          'reasoning_format': parsed['reasoning_format'],
                          'prompt_format': job['prompt_format'], 'history_included': job['history_included'],
                          'messages': messages, 'source_response_sha256': plan['source_response_sha256'],
                          'n_generated': len(tokens), 'stop_reason': stop_reason, 'intervention': details,
                          'repetition': guard.repetition, 'word_repetition': text_metrics, 'coherence_review': 'pending_text_review', 'max_new_tokens': plan['decoding']['max_new_tokens'],
                          'generation_seconds': elapsed, 'thermal_max': guard.max_thermal,
                          'available_memory_bytes': psutil.virtual_memory().available,
                          'gpu_driver_bytes': torch.mps.driver_allocated_memory(), 'finished_utc': now()}
                log.write(json.dumps(record, ensure_ascii=False) + '\n')
                log.flush()
                os.fsync(log.fileno())
                transcript.write(f'JOB {job["job_id"]} | L{job["layer"]} N{job["neuron"]} raw_offset={job["raw_offset"]} seed={job["seed"]}\nPROMPT:\n{job["prompt"]}\n\nOUTPUT:\n{raw}\n\nSTOP: {stop_reason}\n\n')
                transcript.flush()
                (HERE / 'inflight.json').unlink(missing_ok=True)
                completed += 1
                del generated, inputs
                if job['condition'] == 'intervention':
                    del delta
                    assert details['hook_calls'] > 0 and details['first_hook_positions_changed'] > 0, 'No measured intervention'
                torch.mps.empty_cache()
                if guard.reason and guard.reason not in {'repetition_stop', 'generation_timeout'}:
                    raise InterruptedError(guard.reason)
                cooling = max(limits['minimum_cooling_seconds'], limits['cooling_generation_multiplier'] * elapsed)
                if completed % 40 == 0:
                    cooling = max(cooling, 180)
                status('cooling', completed=completed, seconds=round(cooling,1), last_generation_seconds=elapsed)
                rest(cooling)
        status('complete', completed=completed)
    except InterruptedError as exc:
        reason = str(exc)
        status('stopped', completed=completed, reason=reason, resumable_after_review=True)
    except Exception as exc:
        reason = type(exc).__name__ + ': ' + str(exc)
        status('failed', completed=completed, reason=reason)
        traceback.print_exc()
    finally:
        if handle is not None:
            handle.remove()
        if model is not None:
            del model
        if torch is not None and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if lock.exists() and lock.read_text().strip() == str(os.getpid()):
            lock.unlink()
        (HERE / f'closed_{os.getpid()}.json').write_text(json.dumps({'closed_utc': now(), 'completed': completed, 'reason': reason}) + '\n')


if __name__ == '__main__':
    main()
