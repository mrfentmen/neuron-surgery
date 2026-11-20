#!/usr/bin/env python3
"""Continue one saved reply for at most 500 more tokens without adding a new turn. Model output is never executed."""
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


def rest(seconds, allow_after_reply_stop=False):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if STOP_REQUESTED or (HERE / 'STOP').exists() or (allow_after_reply_stop and (HERE / 'STOP_AFTER_REPLY').exists()):
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
                rest(2)
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
    assert digest(ROOT / plan['source_script']) == plan['source_script_sha256']
    assert digest(ROOT / plan['previous_runtime']) == plan['previous_runtime_sha256']
    parent_runtime = json.loads((ROOT / plan['previous_runtime']).read_text())
    for path, expected in plan['provenance_files'].items():
        assert digest(ROOT / path) == expected
    sources = json.loads((ROOT / plan['source_file']).read_text())
    for source in sources.values():
        assert digest(ROOT / source['original_file']) == source['original_sha256']
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
        assert files == parent_runtime['checkpoint_files']
        print('LOADING one local model on MPS; no downloads.', flush=True)
        tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, dtype=torch.bfloat16).to('mps').eval()
        specs = plan['channel_specs']
        layers = sorted({spec['layer'] for spec in specs.values()})
        projections = {layer: model.model.layers[layer].mlp.down_proj for layer in layers}

        def weight_digest(projection):
            data = projection.weight.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            return hashlib.sha256(data).hexdigest()

        original_weights = {layer: weight_digest(projection) for layer, projection in projections.items()}
        medians = {}
        for calibration_id, text in plan['calibration_texts'].items():
            if medians:
                rest(plan['cooling']['between_calibration_forwards_seconds'], allow_after_reply_stop=True)
            wait_nominal(thermal, [])
            ids = tok(text, return_tensors='pt')['input_ids'].to('mps')
            with torch.inference_mode():
                hidden = model(ids, output_hidden_states=True, use_cache=False).hidden_states
            needed = {spec['layer'] for spec in specs.values() if spec['calibration_id'] == calibration_id}
            medians[calibration_id] = {
                layer: hidden[min(layer + 1, model.config.num_hidden_layers - 1)][0, 1:].float().norm(dim=-1).median().item()
                for layer in needed}
            torch.mps.synchronize()
            del hidden, ids
            torch.mps.empty_cache()
        additions = {}
        calibration = {}
        for arm, spec in specs.items():
            projection = projections[spec['layer']]
            column = projection.weight[:, spec['neuron']].detach().float()
            if spec['norm_device'] == 'cpu':
                column = column.cpu()
            norm = column.norm().item()
            median = medians[spec['calibration_id']][spec['layer']]
            value = spec['dose'] * median / norm
            add = torch.zeros(1, 1, projection.in_features, dtype=torch.bfloat16, device=spec['addition_device'])
            add[0, 0, spec['neuron']] = value
            add = add.to('mps')
            assert bool(torch.isfinite(add).all())
            additions[arm] = add
            calibration[arm] = {'column_norm': norm, 'scalar_before_bf16': value,
                                'scalar_bf16': add[0, 0, spec['neuron']].item(),
                                'pre_residual_scale_write_norm': abs(add[0, 0, spec['neuron']].item()) * norm}
            if spec['calibration_id'] == 'handover':
                assert median == parent_runtime['calibration_medians']['handover'][str(spec['layer'])]
                assert calibration[arm] == parent_runtime['calibration'][arm]
            else:
                assert round(median, 2) == spec['historical_rounded_median'], (arm, median)
        rest(plan['cooling']['after_calibration_seconds'], allow_after_reply_stop=True)
        runtime = {'started_utc': datetime.now(timezone.utc).isoformat(),
                   'python': platform.python_version(), 'torch': torch.__version__,
                   'transformers': transformers.__version__, 'device': 'mps', 'dtype': 'bfloat16',
                   'plan_sha256': plan_hash, 'runner_sha256': digest(Path(__file__)),
                   'checkpoint_files': files, 'model': str(model_path),
                   'calibration_medians': medians, 'calibration': calibration,
                   'residual_multiplier': model.config.residual_multiplier,
                   'network_denial_verified': network_denial_verified,
                   'thermal_start': thermal.state(), 'tool_interfaces': [], 'external_actions': False}
        with (HERE / 'runtime.json').open('x') as stream:
            json.dump(runtime, stream, indent=2); stream.write('\n')
        print('RUNTIME_READY ' + json.dumps({'calibrated_layers': len(medians), 'network_denied': True,
                                             'thermal': thermal.state()}), flush=True)
        base_hooks = {layer: len(p._forward_pre_hooks) for layer,p in projections.items()}
        eos = model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        cases = {x['id']: x for x in plan['cases']}

        @contextmanager
        def intervention(arm):
            handle = None
            count = [0]
            if arm in additions:
                projection = projections[specs[arm]['layer']]
                def hook(module, args):
                    count[0] += 1
                    return (args[0] + additions[arm],) + args[1:]
                handle = projection.register_forward_pre_hook(hook)
            try:
                yield count
            finally:
                if handle is not None:
                    handle.remove()
                assert all(len(p._forward_pre_hooks) == base_hooks[layer] for layer,p in projections.items())

        with output_path.open('x') as stream:
            for sequence, job in enumerate(plan['jobs'], 1):
                assert digest(plan_path) == plan_hash
                events = []
                wait_nominal(thermal, events)
                if STOP_REQUESTED or (HERE / 'STOP').exists() or (HERE / 'STOP_AFTER_REPLY').exists():
                    raise InterruptedError('Stop requested before next generation.')
                case = cases[job['case_id']]
                source = sources[case['source_key']]['records'][0]
                assert case['continuation_prefix_ids'] == source['prompt_token_ids'] + source['token_ids']
                ids = torch.tensor([case['continuation_prefix_ids']], dtype=torch.long, device='mps')
                prefix = ids.shape[1]
                torch.manual_seed(job['seed'])
                torch.mps.manual_seed(job['seed'])
                guard = GenerationGuard(thermal, prefix)
                start = time.monotonic()
                print(f'START {sequence}/{plan["total_generations"]} {job["case_id"]} {job["arm"]} seed={job["seed"]}', flush=True)
                decoding = plan['decoding_by_arm'][job['arm']]
                with intervention(job['arm']) as calls, torch.inference_mode():
                    generated = model.generate(ids, **decoding, pad_token_id=tok.eos_token_id,
                        stopping_criteria=StoppingCriteriaList([guard]))
                torch.mps.synchronize()
                tokens = generated[0, prefix:].tolist()
                answer = tok.decode(tokens, skip_special_tokens=True).strip()
                continuation_text = tok.decode(tokens, skip_special_tokens=True)
                combined_ids = case['prior_answer_token_ids'] + tokens
                combined_answer = tok.decode(combined_ids, skip_special_tokens=True).strip()
                elapsed = time.monotonic() - start
                ended = bool(tokens and tokens[-1] in eos_ids)
                stop_reason = guard.reason or ('eos' if ended else 'token_limit')
                record = {'sequence': sequence, **job, 'pack': case['pack'],
                          'replacement': case['replacement'], 'messages': case['messages'],
                          'intervention_spec': specs.get(job['arm']), 'decoding': decoding, 'source_key': case['source_key'],
                          'rendered_prompt': tok.decode(ids[0], skip_special_tokens=False),
                          'prompt_token_ids': ids[0].tolist(), 'answer': answer,
                          'token_ids': tokens, 'n_generated': len(tokens), 'max_new_tokens': 500,
                          'continuation_text_verbatim': continuation_text, 'combined_token_ids': combined_ids,
                          'combined_answer': combined_answer, 'source_sequence': 7,
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
                progress = {'completed': completed, 'total': plan['total_generations'],
                            'updated_utc': datetime.now(timezone.utc).isoformat(),
                            'last_sequence': sequence, 'last_stop_reason': stop_reason,
                            'thermal_state': thermal.state()}
                temp_progress = HERE / 'progress.tmp'
                temp_progress.write_text(json.dumps(progress, indent=2) + '\n')
                temp_progress.replace(HERE / 'progress.json')
                print(f'DONE {sequence}/{plan["total_generations"]} tokens={len(tokens)} stop={stop_reason} seconds={elapsed:.1f} thermal={thermal.state()}', flush=True)
                del generated, ids
                torch.mps.empty_cache()
                if guard.reason:
                    raise RuntimeError('Generation guard stopped the run: ' + guard.reason)
                if sequence < len(plan['jobs']):
                    pause = max(plan['cooling']['minimum_between_generations_seconds'], elapsed * plan['cooling']['rest_generation_multiplier'])
                    print(f'COOLING {pause:.1f}s', flush=True)
                    rest(pause, allow_after_reply_stop=True)
        assert completed == plan['total_generations']
        for path, expected in plan['provenance_files'].items():
            assert digest(ROOT / path) == expected
        for source in sources.values():
            assert digest(ROOT / source['original_file']) == source['original_sha256']
        assert all(weight_digest(p) == original_weights[layer] for layer,p in projections.items())
        assert digest(ROOT / plan['source_script']) == plan['source_script_sha256']
        assert digest(ROOT / plan['source_file']) == plan['source_sha256']
        completion = {'completed_utc': datetime.now(timezone.utc).isoformat(),
                      'generations': completed, 'elapsed_seconds': time.monotonic() - started,
                      'response_sha256': digest(output_path), 'plan_sha256': plan_hash,
                      'all_hooks_removed': all(len(p._forward_pre_hooks) == base_hooks[layer] for layer,p in projections.items()),
                      'intervened_projection_weights_unchanged': True,
                      'projection_weight_hashes': original_weights,
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
