#!/usr/bin/env python3
"""Execute the frozen handover test. Model output is never executed."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import ctypes
import hashlib
import json
import platform
import signal
import socket
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STOP_REQUESTED = False


def stop_signal(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class Thermal:
    def __init__(self):
        self.foundation = ctypes.CDLL('/System/Library/Frameworks/Foundation.framework/Foundation')
        self.objc = ctypes.CDLL('/usr/lib/libobjc.A.dylib')
        self.objc.objc_getClass.argtypes = [ctypes.c_char_p]
        self.objc.objc_getClass.restype = ctypes.c_void_p
        self.objc.sel_registerName.argtypes = [ctypes.c_char_p]
        self.objc.sel_registerName.restype = ctypes.c_void_p
        addr = ctypes.cast(self.objc.objc_msgSend, ctypes.c_void_p).value
        get_object = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(addr)
        self.get_integer = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(addr)
        self.instance = get_object(self.objc.objc_getClass(b'NSProcessInfo'),
                                   self.objc.sel_registerName(b'processInfo'))
        self.selector = self.objc.sel_registerName(b'thermalState')

    def state(self):
        return int(self.get_integer(self.instance, self.selector))


def rest(seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if STOP_REQUESTED or (HERE / 'STOP').exists():
            raise InterruptedError('User/process stop requested.')
        time.sleep(min(1, max(0, deadline - time.monotonic())))


def wait_nominal(thermal, events):
    state = thermal.state()
    if state >= 2:
        raise RuntimeError(f'Thermal pressure {state}: stopping.')
    if state == 0:
        return
    start = time.monotonic()
    print('THERMAL_PAUSE fair pressure; waiting for nominal.', flush=True)
    while thermal.state() != 0:
        if thermal.state() >= 2 or time.monotonic() - start >= 300:
            raise RuntimeError('Thermal recovery failed; stopping.')
        rest(1)
    events.append({'event': 'thermal_pause', 'seconds': time.monotonic() - start})


class GenerationGuard(StoppingCriteria):
    def __init__(self, thermal, prefix_length):
        self.thermal = thermal
        self.prefix_length = prefix_length
        self.events = []
        self.reason = None
        self.maximum_thermal_state = 0

    def __call__(self, input_ids, scores, **kwargs):
        generated = input_ids.shape[1] - self.prefix_length
        state = self.thermal.state()
        self.maximum_thermal_state = max(self.maximum_thermal_state, state)
        if STOP_REQUESTED or (HERE / 'STOP').exists():
            self.reason = 'user_stop'
            return True
        if state >= 2:
            self.reason = 'thermal_stop'
            return True
        try:
            if state:
                wait_nominal(self.thermal, self.events)
            if generated % 32 == 0:
                torch.mps.synchronize()
                rest(1)
        except (RuntimeError, InterruptedError) as exc:
            self.reason = str(exc)
            return True
        return False


def main():
    started = time.monotonic()
    plan_path = HERE / 'plan.json'
    plan = json.loads(plan_path.read_text())
    plan_hash = digest(plan_path)
    output_path = HERE / 'responses.jsonl'
    if output_path.exists() or (HERE / 'runtime.json').exists():
        raise SystemExit('Refusing to overwrite an existing run.')
    assert digest(ROOT / plan['previous_plan']) == plan['previous_plan_sha256']
    assert digest(ROOT / plan['source_file']) == plan['source_sha256']
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    assert torch.backends.mps.is_available(), 'MPS required for this recorded run.'
    torch.mps.set_per_process_memory_fraction(0.65)
    thermal = Thermal()
    wait_nominal(thermal, [])
    try:
        with socket.socket() as probe:
            probe.settimeout(2)
            probe.connect(('1.1.1.1', 443))
        raise RuntimeError('Network connection succeeded; network-denial profile required.')
    except PermissionError:
        network_denial_verified = True
    lock_path = ROOT / '.handover-model.lock'
    with lock_path.open('x') as lock:
        lock.write(str(os.getpid()))
    completed = 0
    try:
        model_path = ROOT / plan['model_relative_path']
        print('HASHING local model and tokenizer.', flush=True)
        files = {p.name: {'bytes': p.stat().st_size, 'sha256': digest(p)}
                 for p in sorted(model_path.iterdir())
                 if p.is_file() and p.suffix in ['.safetensors', '.json', '.jinja', '.txt']}
        print('LOADING one local model on MPS; no downloads.', flush=True)
        tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, dtype=torch.bfloat16).to('mps').eval()
        target = plan['fixed_intervention']
        projection = model.model.layers[target['layer']].mlp.down_proj
        original_weight = projection.weight.detach().cpu().clone()
        ids = tok(plan['calibration_text'], return_tensors='pt')['input_ids'].to('mps')
        with torch.inference_mode():
            h = model(ids, output_hidden_states=True, use_cache=False).hidden_states[
                min(target['layer'] + 1, model.config.num_hidden_layers - 1)]
        median = h[0, 1:].float().norm(dim=-1).median().item()
        del h, ids
        additions = {}
        calibration = {}
        for arm, spec in [('saved_intervention', target), ('control_channel', plan['control_channel'])]:
            norm = projection.weight[:, spec['neuron']].float().norm().item()
            value = spec['dose'] * median / norm
            add = torch.zeros(1, 1, projection.in_features, dtype=torch.bfloat16, device='mps')
            add[0, 0, spec['neuron']] = value
            assert bool(torch.isfinite(add).all())
            additions[arm] = add
            calibration[arm] = {'column_norm': norm, 'scalar_before_bf16': value,
                                'scalar_bf16': add[0, 0, spec['neuron']].item(),
                                'pre_residual_scale_write_norm': abs(add[0, 0, spec['neuron']].item()) * norm}
        runtime = {'started_utc': datetime.now(timezone.utc).isoformat(),
                   'python': platform.python_version(), 'torch': torch.__version__,
                   'transformers': transformers.__version__, 'device': 'mps', 'dtype': 'bfloat16',
                   'plan_sha256': plan_hash, 'runner_sha256': digest(Path(__file__)),
                   'checkpoint_files': files, 'model': str(model_path),
                   'calibration_median': median, 'calibration': calibration,
                   'residual_multiplier': model.config.residual_multiplier,
                   'network_denial_verified': network_denial_verified,
                   'thermal_start': thermal.state(), 'tool_interfaces': [], 'external_actions': False}
        with (HERE / 'runtime.json').open('x') as stream:
            json.dump(runtime, stream, indent=2); stream.write('\n')
        print('RUNTIME_READY ' + json.dumps({'calibration_median': median, 'network_denied': True,
                                             'thermal': thermal.state()}), flush=True)
        base_hooks = len(projection._forward_pre_hooks)
        eos = model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        cases = {x['id']: x for x in plan['cases']}

        @contextmanager
        def intervention(arm):
            handle = None
            count = [0]
            if arm in additions:
                def hook(module, args):
                    count[0] += 1
                    return (args[0] + additions[arm],) + args[1:]
                handle = projection.register_forward_pre_hook(hook)
            try:
                yield count
            finally:
                if handle is not None:
                    handle.remove()
                assert len(projection._forward_pre_hooks) == base_hooks

        with output_path.open('x') as stream:
            for sequence, job in enumerate(plan['jobs'], 1):
                assert digest(plan_path) == plan_hash
                events = []
                wait_nominal(thermal, events)
                if STOP_REQUESTED or (HERE / 'STOP').exists():
                    raise InterruptedError('Stop requested before next generation.')
                case = cases[job['case_id']]
                ids = tok.apply_chat_template(case['messages'], add_generation_prompt=True,
                                              return_tensors='pt')
                ids = (ids if torch.is_tensor(ids) else ids['input_ids']).to('mps')
                prefix = ids.shape[1]
                torch.manual_seed(job['seed'])
                torch.mps.manual_seed(job['seed'])
                guard = GenerationGuard(thermal, prefix)
                start = time.monotonic()
                print(f'START {sequence}/24 {job["case_id"]} {job["arm"]} seed={job["seed"]}', flush=True)
                with intervention(job['arm']) as calls, torch.inference_mode():
                    generated = model.generate(ids, **plan['decoding'], pad_token_id=tok.eos_token_id,
                        stopping_criteria=StoppingCriteriaList([guard]))
                torch.mps.synchronize()
                tokens = generated[0, prefix:].tolist()
                answer = tok.decode(tokens, skip_special_tokens=True).strip()
                elapsed = time.monotonic() - start
                ended = bool(tokens and tokens[-1] in eos_ids)
                stop_reason = guard.reason or ('eos' if ended else 'token_limit')
                record = {'sequence': sequence, **job, 'pack': case['pack'],
                          'replacement': case['replacement'], 'messages': case['messages'],
                          'rendered_prompt': tok.decode(ids[0], skip_special_tokens=False),
                          'prompt_token_ids': ids[0].tolist(), 'answer': answer,
                          'token_ids': tokens, 'n_generated': len(tokens), 'max_new_tokens': 500,
                          'ended_at_eos': ended, 'hit_token_limit': not ended and len(tokens) >= 500,
                          'stop_reason': stop_reason, 'hook_calls': calls[0],
                          'generation_seconds': round(elapsed, 3), 'thermal_after': thermal.state(),
                          'thermal_max_during_generation': guard.maximum_thermal_state,
                          'events': events + guard.events, 'plan_sha256': plan_hash,
                          'device': 'mps', 'dtype': 'bfloat16'}
                assert (calls[0] > 0) == (job['arm'] != 'baseline')
                stream.write(json.dumps(record, ensure_ascii=False) + '\n'); stream.flush()
                os.fsync(stream.fileno())
                completed = sequence
                print(f'DONE {sequence}/24 tokens={len(tokens)} stop={stop_reason} seconds={elapsed:.1f} thermal={thermal.state()}', flush=True)
                del generated, ids
                torch.mps.empty_cache()
                if guard.reason:
                    raise RuntimeError('Generation guard stopped the run: ' + guard.reason)
                if sequence < len(plan['jobs']):
                    pause = max(30, elapsed * 2)
                    print(f'COOLING {pause:.1f}s', flush=True)
                    rest(pause)
        assert completed == plan['total_generations']
        assert torch.equal(original_weight, projection.weight.detach().cpu())
        assert digest(ROOT / plan['source_file']) == plan['source_sha256']
        completion = {'completed_utc': datetime.now(timezone.utc).isoformat(),
                      'generations': completed, 'elapsed_seconds': time.monotonic() - started,
                      'response_sha256': digest(output_path), 'plan_sha256': plan_hash,
                      'all_hooks_removed': len(projection._forward_pre_hooks) == base_hooks,
                      'intervened_projection_weights_unchanged': True,
                      'thermal_end': thermal.state()}
        with (HERE / 'completion.json').open('x') as stream:
            json.dump(completion, stream, indent=2); stream.write('\n')
        print('COMPLETE ' + json.dumps(completion), flush=True)
    except BaseException as exc:
        with (HERE / 'failure.json').open('x') as stream:
            json.dump({'time': datetime.now(timezone.utc).isoformat(), 'completed': completed,
                       'type': type(exc).__name__, 'error': str(exc)}, stream, indent=2)
        raise
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, stop_signal)
    signal.signal(signal.SIGTERM, stop_signal)
    main()
