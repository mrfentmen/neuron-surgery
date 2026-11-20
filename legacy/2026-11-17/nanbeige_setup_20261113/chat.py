#!/usr/bin/env python3
"""Offline Nanbeige baseline chat: streamed reasoning, no model interventions."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import signal
import socket
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL = ROOT / 'models/Nanbeige4.1-3B'


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
    manifest = json.loads((HERE / 'download_manifest.json').read_text())
    if not manifest.get('verified'):
        raise SystemExit('Download verification has not finished.')
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
        from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList, TextStreamer
        from importlib.metadata import version

        class Streamer(TextStreamer):
            def on_finalized_text(self, text, stream_end=False):
                print(clean(text), end='\n' if stream_end else '', flush=True)

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
        print('Checking the pinned checkpoint files before loading…', flush=True)
        for item in manifest['files']:
            path = MODEL / item['name']
            if path.stat().st_size != item['bytes'] or guardmod.digest(path) != item['sha256']:
                raise RuntimeError('Checkpoint changed: ' + item['name'])
        tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
        print('Loading one unmodified Nanbeige4.1-3B model on MPS…', flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, attn_implementation='eager').to('mps').eval()
        runtime = {'model': str(MODEL), 'revision': manifest['revision'], 'intervention': None,
                   'dtype': 'bfloat16', 'device': 'mps', 'attention': 'eager',
                   'torch': version('torch'), 'transformers': version('transformers'),
                   'max_new_tokens': 500, 'temperature': .6, 'top_p': .95, 'top_k': 0,
                   'repetition_penalty': 1., 'seed': 0, 'input_token_limit': 2048,
                   'network_denied': True, 'tools': [], 'history_mode': 'conversation',
                   'cooling_seconds': 10, 'rest_every_32_tokens_seconds': 2,
                   'script_sha256': guardmod.digest(Path(__file__)),
                   'checkpoint_manifest_sha256': guardmod.digest(HERE / 'download_manifest.json')}
        (session / 'runtime.json').write_text(json.dumps(runtime, indent=2) + '\n')
        print('Cooling for 10 seconds…', flush=True)
        guardmod.rest(10)
        print('\nBaseline: NO neuron intervention. Reasoning text streams as generated.\n'
              'Conversation history is included. /reset starts fresh; /quit exits.\n'
              '500 new tokens TOTAL for reasoning and answer; truncated reasoning is labelled.\n'
              'A generated reasoning trace is not a guaranteed faithful account of computation.\n'
              f'Full records: {session}\n', flush=True)
        history = []
        turn = 0
        last_end = None
        with (session / 'responses.jsonl').open('x') as log, (session / 'TRANSCRIPT.txt').open('x') as transcript:
            while True:
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
                if command in ('/help', '/status'):
                    print('Original Nanbeige4.1-3B, no intervention. /reset, /quit. Max500. History included.\n')
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
                guard = guardmod.GenerationGuard(thermal, prefix)
                streamer = Streamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
                torch.manual_seed(0)
                torch.mps.manual_seed(0)
                start = time.monotonic()
                print('\nNanbeige > ', end='', flush=True)
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=500, do_sample=True,
                        temperature=.6, top_p=.95, top_k=0, repetition_penalty=1.,
                        pad_token_id=tokenizer.eos_token_id, streamer=streamer,
                        stopping_criteria=StoppingCriteriaList([guard]))
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
                          'rendered_prompt': rendered, 'prompt_token_ids': inputs['input_ids'][0].tolist(),
                          'token_ids': tokens, 'raw_output': raw, 'answer': content, **parts,
                          'n_generated': len(tokens), 'stop_reason': stop_reason,
                          'generation_seconds': elapsed, 'thermal_max': guard.maximum_thermal_state,
                          'finished_utc': datetime.now(timezone.utc).isoformat()}
                log.write(json.dumps(record, ensure_ascii=False) + '\n')
                log.flush()
                os.fsync(log.fileno())
                transcript.write(f'TURN {turn}\nPROMPT:\n{prompt}\n\nRAW OUTPUT:\n{raw}\n\nSTOP: {stop_reason}\n\n')
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
    finally:
        if model is not None:
            del model
        if torch is not None and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if lock.exists() and lock.read_text().strip() == str(os.getpid()):
            lock.unlink()
        if session is not None:
            (session / 'closed.json').write_text(json.dumps({'pid': os.getpid(), 'closed_utc': datetime.now(timezone.utc).isoformat()}) + '\n')


if __name__ == '__main__':
    main()
