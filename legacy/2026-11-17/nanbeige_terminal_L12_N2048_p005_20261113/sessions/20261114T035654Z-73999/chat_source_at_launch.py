#!/usr/bin/env python3
"""Offline Nanbeige chat with a fixed, previously untested activation intervention."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import signal
import socket
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL = ROOT / 'models/Nanbeige4.1-3B'
MANIFEST = ROOT / 'nanbeige_setup_20261113/download_manifest.json'


def status(state, **extra):
    record = {'state': state, 'pid': os.getpid(),
              'updated_utc': datetime.now(timezone.utc).isoformat(), **extra}
    temporary = HERE / ('status.' + str(os.getpid()) + '.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(HERE / 'status.json')


def install_push(projection, delta, neuron, details):
    def push(module, args):
        changed = args[0] + delta
        details['hook_calls'] += 1
        if details['hook_calls'] == 1:
            difference = (changed[..., neuron] - args[0][..., neuron]).float()
            details['first_hook_mean_actual_delta'] = difference.mean().item()
            details['first_hook_positions_changed'] = (difference != 0).sum().item()
        return (changed,) + args[1:]
    return projection.register_forward_pre_hook(push)


def clean(text):
    return ''.join(c if c in '\n\t' or (ord(c) >= 32 and not 127 <= ord(c) < 160)
                   else f'\\u{ord(c):04x}' for c in text)


def split_reasoning(text):
    if '<think>' not in text:
        return {'reasoning': None, 'final_answer': text, 'reasoning_closed': None}
    before, after = text.split('<think>', 1)
    if '</think>' not in after:
        return {'reasoning': after, 'final_answer': None, 'reasoning_closed': False}
    reasoning, final = after.split('</think>', 1)
    return {'reasoning': reasoning, 'final_answer': final.strip(), 'reasoning_closed': True}


def main():
    selection = json.loads((HERE / 'selection.json').read_text())
    layer, neuron, dose = selection['layer'], selection['neuron'], selection['dose']
    manifest = json.loads(MANIFEST.read_text())
    if not manifest.get('verified'):
        raise SystemExit('Download verification has not finished.')
    assert manifest['revision'] == selection['revision'], 'Model revision mismatch'
    status('waiting_for_lock', layer=layer, neuron=neuron, dose=dose)
    # Keep the existing frozen survey from taking the GPU during this handover.
    # Its plan and saved answers remain untouched; it can be resumed explicitly.
    supervisor = ROOT / 'granite_channel_identity_after_chat_20261113'
    hold = supervisor / 'STOP_AFTER_REPLY'
    if supervisor.exists() and not hold.exists():
        with hold.open('x') as stream:
            stream.write('Hold the deferred Granite survey for the user-requested Nanbeige chat. Preserve all remaining jobs.\n')
    lock = ROOT / '.handover-model.lock'
    announced = False
    while True:
        try:
            with lock.open('x') as stream:
                stream.write(str(os.getpid()))
            break
        except FileExistsError:
            if not announced:
                print('Waiting for the current model to close. No Nanbeige model is loaded.\n'
                      'When ready, enter /quit in the Granite window.\n'
                      'The deferred Granite survey is held; its remaining questions are preserved.\n', flush=True)
                announced = True
            try:
                time.sleep(2)
            except KeyboardInterrupt:
                raise SystemExit('Stopped waiting. The current model and saved logs are unchanged.')
    session = None
    model = None
    torch = None
    try:
        spec = importlib.util.spec_from_file_location('nanbeige_guards', ROOT / 'granite_channel_identity_resume_20261113/run_test.py')
        guardmod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(guardmod)
        torch = guardmod.torch
        import psutil
        from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList, TextStreamer
        from importlib.metadata import version

        class Streamer(TextStreamer):
            def on_finalized_text(self, text, stream_end=False):
                print(clean(text), end='\n' if stream_end else '', flush=True)

        class ChatGuard(guardmod.GenerationGuard):
            def __call__(self, input_ids, scores, **kwargs):
                if psutil.virtual_memory().available < 3 * 1024 ** 3:
                    self.reason = 'low_memory_stop'
                    return True
                return super().__call__(input_ids, scores, **kwargs)

        def interrupted(signum, frame):
            guardmod.STOP_REQUESTED = True
            raise KeyboardInterrupt

        for sig in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, interrupted)
        with socket.socket() as probe:
            probe.settimeout(2)
            try:
                probe.connect(('1.1.1.1', 443))
            except PermissionError:
                pass
            else:
                raise RuntimeError('Run the .command launcher: OS network denial is required.')
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS is unavailable; CPU fallback is disabled.')
        # About 12 GB GPU allocator ceiling on the verified 32 GB machine.
        torch.mps.set_per_process_memory_fraction(.5)
        session = HERE / 'sessions' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + str(os.getpid()))
        session.mkdir(parents=True, exist_ok=False)
        guardmod.HERE = session
        thermal = guardmod.Thermal()
        guardmod.wait_nominal(thermal, [])
        if psutil.virtual_memory().available < 10 * 1024 ** 3:
            raise RuntimeError('Less than 10 GiB available before loading; close unused apps and reopen.')
        status('verifying_checkpoint', session=str(session))
        print('Checking the pinned checkpoint files before loading…', flush=True)
        for item in manifest['files']:
            path = MODEL / item['name']
            if path.stat().st_size != item['bytes'] or guardmod.digest(path) != item['sha256']:
                raise RuntimeError('Checkpoint changed: ' + item['name'])
        tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
        status('loading', session=str(session))
        print('Loading one Nanbeige4.1-3B model on MPS; checkpoint weights stay unchanged…', flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, attn_implementation='eager').to('mps').eval()
        calibration_path = ROOT / selection['calibration_source']
        assert guardmod.digest(calibration_path) == selection['calibration_sha256'], 'Calibration changed'
        calibration = json.loads(calibration_path.read_text())
        assert calibration['revision'] == manifest['revision']
        assert 0 <= layer < model.config.num_hidden_layers
        projection = model.model.layers[layer].mlp.down_proj
        assert 0 <= neuron < projection.in_features
        median = calibration['medians'][str(layer)]
        norm = projection.weight[:, neuron].detach().float().norm().item()
        assert norm > 0 and math.isfinite(norm) and math.isfinite(median)
        scalar = dose * median / norm
        delta = torch.zeros((1, 1, projection.in_features), dtype=torch.bfloat16, device='mps')
        delta[0, 0, neuron] = scalar
        assert bool(torch.isfinite(delta).all())
        intervention = {'layer': layer, 'neuron': neuron, 'dose': dose,
                        'median': median, 'column_norm': norm,
                        'scalar_requested': scalar, 'scalar_bf16': delta[0, 0, neuron].item(),
                        'site': 'mlp.down_proj input', 'positions': 'all prompt and generated positions',
                        'calibration_source': selection['calibration_source'],
                        'calibration_sha256': selection['calibration_sha256']}
        weight_version = projection.weight._version
        base_hook_count = len(projection._forward_pre_hooks)
        runtime = {'model': str(MODEL), 'revision': manifest['revision'], 'intervention': intervention,
                   'dtype': 'bfloat16', 'device': 'mps', 'attention': 'eager',
                   'torch': version('torch'), 'transformers': version('transformers'),
                   'max_new_tokens': 500, 'temperature': .6, 'top_p': .95, 'top_k': 0,
                   'repetition_penalty': 1., 'seed': selection['seed'], 'input_token_limit': 2048,
                   'network_denied': True, 'tools': [], 'history_mode': 'conversation',
                   'cooling_seconds': 10, 'rest_every_32_tokens_seconds': 2,
                   'script_sha256': guardmod.digest(Path(__file__)),
                   'checkpoint_manifest_sha256': guardmod.digest(MANIFEST),
                   'selection_sha256': guardmod.digest(HERE / 'selection.json')}
        (session / 'runtime.json').write_text(json.dumps(runtime, indent=2) + '\n')
        print('Cooling for 10 seconds…', flush=True)
        guardmod.rest(10)
        print(f'\nNanbeige — L{layer} N{neuron}, dose {dose:+g}. Intervention ON.\n'
              'Thinking text streams live between <think> and </think>.\n'
              'Conversation history is included. /reset starts fresh; /quit exits.\n'
              '/off compares the undosed model; /on restores this dose. Both start fresh.\n'
              '500 new tokens TOTAL for reasoning and answer; truncated reasoning is labelled.\n'
              'A generated reasoning trace is not a guaranteed faithful account of computation.\n'
              f'Full records: {session}\n', flush=True)
        history = []
        turn = 0
        last_end = None
        enabled = True
        with (session / 'responses.jsonl').open('x') as log, (session / 'TRANSCRIPT.txt').open('x') as transcript:
            while True:
                status('ready', session=str(session), completed=turn,
                       layer=layer, neuron=neuron, dose=dose if enabled else 0., intervention_on=enabled)
                try:
                    prompt = input('You > ')
                except EOFError:
                    break
                command = prompt.strip().lower()
                if not command:
                    continue
                if command in ('/quit', '/exit'):
                    break
                if command == '/reset':
                    history = []
                    print('Fresh conversation. Saved records are retained.\n')
                    continue
                if command in ('/off', '/on'):
                    enabled = command == '/on'
                    history = []
                    print(('Intervention ON.' if enabled else 'Baseline: intervention OFF.') +
                          ' Fresh conversation; saved records retained.\n')
                    continue
                if command in ('/help', '/status'):
                    print(f'Nanbeige L{layer} N{neuron}, dose {dose:+g}; intervention {"ON" if enabled else "OFF"}.\n'
                          '/reset, /off, /on, /quit. Max 500 tokens including thinking. History included.\n')
                    continue
                if command.startswith('/'):
                    print('Unknown command. Use /help.')
                    continue
                messages = history + [{'role': 'user', 'content': prompt}]
                rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(rendered, add_special_tokens=False, return_tensors='pt')
                prefix = inputs['input_ids'].shape[1]
                if prefix > 2048:
                    print('This conversation exceeds the 2048-token input cap. Use /reset before continuing.')
                    continue
                pause = 0 if last_end is None else max(0, last_end + 10 - time.monotonic())
                if pause:
                    print(f'Cooling for {pause:.0f} seconds…', flush=True)
                    guardmod.rest(pause)
                guardmod.wait_nominal(thermal, [])
                inputs = inputs.to('mps')
                guard = ChatGuard(thermal, prefix)
                streamer = Streamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
                torch.manual_seed(selection['seed'])
                torch.mps.manual_seed(selection['seed'])
                start = time.monotonic()
                details = {**intervention, 'enabled': enabled, 'hook_calls': 0}
                assert projection.weight._version == weight_version, 'Projection weights changed'
                status('generating', session=str(session), completed=turn,
                       layer=layer, neuron=neuron, dose=dose if enabled else 0., intervention_on=enabled)
                print('\nNanbeige > ', end='', flush=True)
                handle = install_push(projection, delta, neuron, details) if enabled else None
                try:
                    with torch.inference_mode():
                        generated = model.generate(**inputs, max_new_tokens=500, do_sample=True,
                            temperature=.6, top_p=.95, top_k=0, repetition_penalty=1.,
                            pad_token_id=tokenizer.eos_token_id, streamer=streamer,
                            stopping_criteria=StoppingCriteriaList([guard]))
                finally:
                    if handle is not None:
                        handle.remove()
                    assert len(projection._forward_pre_hooks) == base_hook_count
                assert projection.weight._version == weight_version, 'Projection weights changed'
                torch.mps.synchronize()
                elapsed = time.monotonic() - start
                tokens = generated[0, prefix:].tolist()
                raw = tokenizer.decode(tokens, skip_special_tokens=False)
                eos = model.generation_config.eos_token_id
                eos = set(eos if isinstance(eos, list) else [eos])
                ended = bool(tokens and tokens[-1] in eos)
                content = tokenizer.decode(tokens[:-1] if ended else tokens, skip_special_tokens=False).strip()
                parts = split_reasoning(content)
                stop_reason = guard.reason or ('eos' if ended else 'token_limit')
                turn += 1
                record = {'turn': turn, 'prompt': prompt, 'messages': messages,
                          'intervention': details, 'seed': selection['seed'],
                          'rendered_prompt': rendered, 'prompt_token_ids': inputs['input_ids'][0].tolist(),
                          'token_ids': tokens, 'raw_output': raw, 'answer': content, **parts,
                          'n_generated': len(tokens), 'stop_reason': stop_reason,
                          'generation_seconds': elapsed, 'thermal_max': guard.maximum_thermal_state,
                          'finished_utc': datetime.now(timezone.utc).isoformat()}
                log.write(json.dumps(record, ensure_ascii=False) + '\n')
                log.flush()
                os.fsync(log.fileno())
                transcript.write(f'TURN {turn} | L{layer} N{neuron} dose={dose if enabled else 0.} | intervention={enabled}\nPROMPT:\n{prompt}\n\nRAW OUTPUT:\n{raw}\n\nSTOP: {stop_reason}\n\n')
                transcript.flush()
                print(f'\n[{len(tokens)} tokens; {stop_reason}]')
                if parts['reasoning_closed'] is False:
                    print('[Reasoning did not close; no final answer recorded. This incomplete turn is not added to history.]')
                elif ended and not guard.reason:
                    history = messages + [{'role': 'assistant', 'content': content}]
                else:
                    print('[Incomplete turn saved but not added to conversation history.]')
                print()
                last_end = time.monotonic()
                del generated, inputs
                torch.mps.empty_cache()
                if guard.reason:
                    break
    except KeyboardInterrupt:
        print('\nChat interrupted. Completed replies remain saved.')
    except Exception as exc:
        status('error', error=repr(exc), session=str(session) if session else None)
        raise
    finally:
        if model is not None:
            del model
        if torch is not None and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if lock.exists() and lock.read_text().strip() == str(os.getpid()):
            lock.unlink()
        if session is not None:
            (session / 'closed.json').write_text(json.dumps({'pid': os.getpid(), 'closed_utc': datetime.now(timezone.utc).isoformat()}) + '\n')
        current = json.loads((HERE / 'status.json').read_text())
        if current.get('state') != 'error':
            status('closed', session=str(session) if session else None)


if __name__ == '__main__':
    main()
