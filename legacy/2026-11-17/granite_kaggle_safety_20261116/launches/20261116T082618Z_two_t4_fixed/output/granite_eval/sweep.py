"""Fixed two-GPU Granite candidate evaluation. Outputs are saved as data and never executed."""
import argparse
from contextlib import closing
from repetition import repeated_tail
from gpu_memory import check_memory, snapshot, validate_long_cache
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sqlite3
import time

from reasoning_format import parse_reasoning


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def task_count(config):
    return len(config['candidates']) * len(config['questions'])


def batch_spec(config, shard, index):
    """Fixed, balanced sparse candidates; rotate questions to expose all topics early."""
    groups = [g for g in config['candidate_groups'] if g['shard'] == shard]
    if index < 0 or index >= len(groups) * len(config['questions']):
        return None
    round_index, local_group = divmod(index, len(groups))
    group = groups[local_group]
    settings = [config['candidates'][i] for i in group['candidate_indices']]
    question = config['questions'][(round_index + local_group) % len(config['questions'])]
    return dict(batch_index=index, layer=group['layer'],
                neurons=[c['neuron'] for c in settings],
                doses=[c['dose'] for c in settings], settings=settings,
                question=question, seed=config['seed'], candidate_group=group['group_id'])


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


class Results:
    """Only complete batches advance the durable cursor, in one transaction."""
    def __init__(self, path, signature):
        self.db = sqlite3.connect(path)
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS results (batch INTEGER, slot INTEGER, payload TEXT, PRIMARY KEY(batch,slot))')
        self.db.execute('CREATE TABLE IF NOT EXISTS partial (attempt TEXT, batch INTEGER, payload TEXT)')
        old = self.db.execute("SELECT value FROM meta WHERE key='signature'").fetchone()
        if old is not None and old[0] != signature:
            self.db.close()
            raise ValueError('Resume signature differs: keep hardware/dtype/runtime/config fixed or start a new result directory.')
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO meta VALUES ('signature',?)", (signature,))
            self.db.execute("INSERT OR IGNORE INTO meta VALUES ('next_batch','0')")

    @property
    def next_batch(self):
        return int(self.db.execute("SELECT value FROM meta WHERE key='next_batch'").fetchone()[0])

    def save(self, index, records, complete):
        assert index == self.next_batch
        with self.db:
            if complete:
                for slot, record in enumerate(records):
                    self.db.execute('INSERT INTO results VALUES (?,?,?)', (index, slot, canonical(record)))
                self.db.execute("UPDATE meta SET value=? WHERE key='next_batch'", (str(index + 1),))
            else:
                stamp = datetime.now(timezone.utc).isoformat()
                self.db.execute('INSERT INTO partial VALUES (?,?,?)', (stamp, index, canonical(records)))

    def export(self, path):
        with Path(path).open('w') as f:
            for (payload,) in self.db.execute('SELECT payload FROM results ORDER BY batch,slot'):
                f.write(payload + '\n')


def resume_checkpoint(source, destination):
    """Carry the cursor, not gigabytes of old text, into a new Kaggle session."""
    source = Path(source)
    destination = Path(destination)
    temporary = destination.with_suffix('.tmp')
    if temporary.exists():temporary.unlink()
    with closing(sqlite3.connect(source)) as old, closing(sqlite3.connect(temporary)) as new, new:
        new.execute('CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)')
        new.executemany('INSERT INTO meta VALUES (?,?)', old.execute('SELECT key,value FROM meta'))
        new.execute('INSERT OR REPLACE INTO meta VALUES (?,?)', ('parent_results_sha256', sha(source)))
        new.execute('INSERT OR REPLACE INTO meta VALUES (?,?)', ('parent_results_file', source.name))
    temporary.replace(destination)


def add_channel_delta(x, neurons, scalars):
    """Slot zero is the undosed control. Other rows alter their own one channel."""
    import torch
    y = x.clone()
    row_ids = torch.arange(1, len(neurons) + 1, device=x.device)
    columns = torch.tensor(neurons, device=x.device)
    y[row_ids, :, columns] += scalars[:, None]
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.json')
    ap.add_argument('--model-path', required=True)
    ap.add_argument('--shard', type=int, choices=[0, 1], required=True)
    ap.add_argument('--out', default='/kaggle/working/results')
    ap.add_argument('--max-seconds', type=int, default=18000)
    ap.add_argument('--max-batches', type=int, default=0)
    args = ap.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Each worker requires exactly one assigned GPU.'
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    max_new_tokens = config['limits']['max_new_tokens']
    assert isinstance(max_new_tokens, int) and 1 <= max_new_tokens <= 10000
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + args.max_seconds
    stop = False

    def on_signal(signum, frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    def status(state, **kwargs):
        data = dict(state=state, shard=args.shard, updated_utc=datetime.now(timezone.utc).isoformat(), **kwargs)
        atomic_json(out / f'status_{args.shard}.json', data)
        print(canonical(data), flush=True)

    device = 'cuda:0'
    # PyTorch can report emulated bf16 on T4 (SM 7.5); require native support.
    dtype_name = 'bfloat16' if torch.cuda.is_bf16_supported(including_emulation=False) else 'float16'
    dtype = getattr(torch, dtype_name)
    runtime = dict(config_sha256=sha(config_path), torch=torch.__version__, transformers=transformers.__version__,
                   gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
                   dtype=dtype_name, attention='eager', script_sha256=sha(__file__), shard=args.shard,
                   calibration_sha256=sha(config_path.parent / 'source_calibration.json'),
                   memory_helper_sha256=sha(config_path.parent / 'gpu_memory.py'),
                   repetition_sha256=sha(config_path.parent / 'repetition.py'),
                   reasoning_parser_sha256=sha(config_path.parent / 'reasoning_format.py'))
    results = Results(out / f'shard_{args.shard}.sqlite', canonical(runtime))
    atomic_json(out / f'runtime_{args.shard}.json', runtime)
    status('loading', next_batch=results.next_batch, max_new_tokens=max_new_tokens,
           batch_timeout_seconds=config['limits']['batch_seconds'])
    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True,
        trust_remote_code=False, dtype=dtype, device_map=device, attn_implementation='eager').eval()
    assert model.config.num_hidden_layers == config['layers']
    assert model.config.intermediate_size == config['channels']
    if config.get('validate_long_context', False):
        status('validating_long_context', next_batch=results.next_batch,
               context_tokens=config['limits']['input_tokens'] + max_new_tokens,
               batch_size=config['batch_size'] + 1)
        try:
            validation = validate_long_cache(model, torch, tok.encode('hello', add_special_tokens=False)[0],
                config['batch_size'] + 1, config['limits']['input_tokens'] + max_new_tokens, device)
        except Exception as exc:
            validation = {'passed': False, 'error': type(exc).__name__ + ': ' + str(exc)}
            atomic_json(out / f'long_context_validation_{args.shard}.json', validation)
            status('failed', reason='long_context_validation_failed', **validation)
            results.db.close()
            raise
        atomic_json(out / f'long_context_validation_{args.shard}.json', validation)
        status('long_context_validated', **validation)
    calibration = json.loads((config_path.parent / 'source_calibration.json').read_text())
    assert calibration['revision'] == config['revision']
    norms = {}
    for layer in sorted({g['layer'] for g in config['candidate_groups'] if g['shard'] == args.shard}):
        norms[layer] = model.model.layers[layer].mlp.down_proj.weight.float().norm(dim=0).cpu()
    assert calibration['candidates_sha256'] == sha(config_path.parent / 'candidates.json')
    assert config['candidates'] == json.loads((config_path.parent / 'candidates.json').read_text())
    versions = [block.mlp.down_proj.weight._version for block in model.model.layers]
    eos_id = model.generation_config.eos_token_id
    eos_ids = set(eos_id if isinstance(eos_id, list) else [eos_id])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else min(eos_ids)
    assert config['decoding'] == {'do_sample': False, 'num_beams': 1, 'repetition_penalty': 1.0, 'no_repeat_ngram_size': 0}
    batches_run = 0
    generation_seconds = 0.0
    saved_interventions = 0
    end_reason = 'complete'

    try:
        while True:
            index = results.next_batch
            spec = batch_spec(config, args.shard, index)
            if spec is None:
                break
            if stop or (out / 'STOP').exists():
                end_reason = 'user_stop'; break
            if deadline - time.monotonic() < 120:
                end_reason = 'session_budget'; break
            if args.max_batches and batches_run >= args.max_batches:
                end_reason = 'batch_budget'; break
            if sum(p.stat().st_size for p in out.glob('*') if p.is_file()) > config['limits']['output_bytes']:
                end_reason = 'output_budget'; break
            assert versions == [block.mlp.down_proj.weight._version for block in model.model.layers]
            prompt = spec['question']['prompt']
            rendered = tok.apply_chat_template([{'role':'user','content':prompt}], tokenize=False, add_generation_prompt=True)
            ids = tok(rendered, add_special_tokens=False, return_tensors='pt')['input_ids']
            assert ids.shape[1] <= config['limits']['input_tokens']
            n = len(spec['neurons']) + 1
            input_ids = ids.repeat(n, 1).to(device)
            attention = torch.ones_like(input_ids)
            torch.manual_seed(spec['seed'])
            torch.cuda.manual_seed_all(spec['seed'])
            sequences = [[] for _ in range(n)]
            done = [False] * n
            row_stops = [None] * n
            repetitions = [None] * n
            memory_metrics = snapshot(torch.cuda)
            repetition_settings = config['limits'].get('repetition', {
                'minimum_tokens': 64, 'minimum_repeats': 4,
                'maximum_period': 128, 'check_every_tokens': 16})
            scalars_requested = [c['dose'] * c['median'] / norms[spec['layer']][c['neuron']].item() for c in spec['settings']]
            scalars = torch.tensor(scalars_requested, device=device, dtype=dtype)
            measurement = {'hook_calls':0}

            def hook(module, inputs):
                x = inputs[0]
                changed = add_channel_delta(x, spec['neurons'], scalars)
                measurement['hook_calls'] += 1
                if measurement['hook_calls'] == 1:
                    measurement['baseline_unchanged'] = torch.equal(x[0], changed[0])
                    measurement['actual_mean_deltas'] = [(changed[i+1,:,neuron]-x[i+1,:,neuron]).float().mean().item() for i,neuron in enumerate(spec['neurons'])]
                    measurement['changed_positions'] = [(changed[i+1,:,neuron]!=x[i+1,:,neuron]).sum().item() for i,neuron in enumerate(spec['neurons'])]
                return (changed,) + inputs[1:]

            handle = model.model.layers[spec['layer']].mlp.down_proj.register_forward_pre_hook(hook)
            past = None
            batch_start = time.monotonic()
            status('generating', batch_index=index, layer=spec['layer'], neurons=spec['neurons'],
                   dose=spec['doses'], question_id=spec['question']['id'], max_new_tokens=max_new_tokens)
            interrupted = None
            try:
                with torch.inference_mode():
                    for step in range(max_new_tokens):
                        if stop or (out / 'STOP').exists():
                            interrupted = 'user_stop'; break
                        if time.monotonic() >= deadline:
                            interrupted = 'session_budget'; break
                        if time.monotonic()-batch_start > config['limits']['batch_seconds']:
                            interrupted = 'generation_timeout'; break
                        if step % 128 == 0:
                            memory_ok, memory_metrics = check_memory(torch.cuda)
                            if not memory_ok:
                                interrupted = 'gpu_memory_guard'; break
                            if step:
                                status('generating', batch_index=index, layer=spec['layer'],
                                       neurons=spec['neurons'], dose=spec['doses'],
                                       question_id=spec['question']['id'], generated_steps=step,
                                       tokens_per_row=[len(tokens) for tokens in sequences],
                                       finished_rows=sum(done), max_new_tokens=max_new_tokens,
                                       elapsed_seconds=round(time.monotonic()-batch_start, 2),
                                       gpu_memory=memory_metrics)
                        if step and step % 128 == 0:
                            atomic_json(out / f'inflight_{args.shard}.json', {
                                'batch_index': index, 'spec': spec, 'prompt': prompt,
                                'rendered_prompt': rendered, 'token_ids': sequences,
                                'raw_outputs': [tok.decode(tokens, skip_special_tokens=False) for tokens in sequences],
                                'row_stop_reasons': row_stops, 'repetition': repetitions,
                                'runtime': runtime, 'hook_measurement': measurement,
                                'max_new_tokens': max_new_tokens, 'gpu_memory': memory_metrics,
                                'updated_utc': datetime.now(timezone.utc).isoformat()})
                        outputs = model(input_ids=input_ids, attention_mask=attention, past_key_values=past,
                                        use_cache=True, logits_to_keep=1)
                        past = outputs.past_key_values
                        scores = outputs.logits[:,-1,:].float()
                        if not torch.isfinite(scores).all():
                            raise RuntimeError('Nonfinite generation logits')
                        greedy_tokens = scores.argmax(dim=-1).tolist()
                        next_tokens = []
                        for slot in range(n):
                            token = pad_id if done[slot] else greedy_tokens[slot]
                            next_tokens.append(token)
                            if not done[slot]:
                                sequences[slot].append(token)
                                if token in eos_ids:
                                    done[slot] = True
                                    row_stops[slot] = 'eos'
                                elif len(sequences[slot]) % repetition_settings['check_every_tokens'] == 0:
                                    evidence = repeated_tail(sequences[slot],
                                        minimum_tokens=repetition_settings['minimum_tokens'],
                                        minimum_repeats=repetition_settings['minimum_repeats'],
                                        maximum_period=repetition_settings['maximum_period'])
                                    if evidence:
                                        done[slot] = True
                                        row_stops[slot] = 'repetition_stop'
                                        repetitions[slot] = evidence
                                        print(canonical({'event': 'repetition_stop', 'shard': args.shard,
                                              'batch_index': index, 'slot': slot,
                                              'n_generated': len(sequences[slot]), **evidence}), flush=True)
                        del outputs, scores
                        if all(done):break
                        input_ids = torch.tensor(next_tokens, device=device).reshape(n,1)
                        attention = torch.cat([attention, torch.ones((n,1),device=device,dtype=attention.dtype)],dim=1)
            except torch.OutOfMemoryError:
                interrupted = 'cuda_out_of_memory'
            finally:
                handle.remove()
            elapsed = time.monotonic()-batch_start
            if interrupted is None:
                assert measurement['hook_calls'] > 0 and measurement['baseline_unchanged']
                assert all(x > 0 for x in measurement['changed_positions']), 'No measured change'
            records = []
            for slot,tokens in enumerate(sequences):
                ended = bool(tokens and tokens[-1] in eos_ids)
                raw = tok.decode(tokens, skip_special_tokens=False)
                content = tok.decode(tokens[:-1] if ended else tokens, skip_special_tokens=False)
                j = dict(layer=spec['layer'] if slot else None, neuron=spec['neurons'][slot-1] if slot else None,
                         dose=spec['settings'][slot-1]['dose'] if slot else 0.0, seed=spec['seed'], question_id=spec['question']['id'],
                         condition='intervention' if slot else 'baseline')
                records.append(dict(job=j, shard=args.shard, batch_index=index, slot=slot, batch_size=n,
                    matched_baseline_slot=0, prompt=prompt, rendered_prompt=rendered, prompt_token_ids=ids[0].tolist(),
                    token_ids=tokens, raw_output=raw, answer=content, **parse_reasoning(content),
                    n_generated=len(tokens), max_new_tokens=max_new_tokens,
                    stop_reason=row_stops[slot] or interrupted or 'token_limit',
                    repetition=repetitions[slot], gpu_memory=memory_metrics,
                    config_sha256=runtime['config_sha256'], model_revision=config['revision'], dtype=dtype_name,
                    sampling='greedy argmax; CUDA float16 batched output is not guaranteed token-identical to MPS bfloat16',
                    candidate=spec['settings'][slot-1] if slot else None,
                    scalar_requested=scalars_requested[slot-1] if slot else 0.0,
                    scalar_applied=scalars[slot-1].item() if slot else 0.0,
                    median_residual=spec['settings'][slot-1]['median'] if slot else None,
                    column_norm=norms[spec['layer']][j['neuron']].item() if slot else None,
                    hook_measurement=measurement, generation_seconds=elapsed,
                    finished_utc=datetime.now(timezone.utc).isoformat()))
            results.save(index, records, complete=interrupted is None)
            if interrupted is None:
                # Export each committed batch so abrupt cancellation leaves readable rows too.
                with (out / f'responses_{args.shard}.jsonl').open('a') as stream:
                    for record in records:
                        stream.write(canonical(record) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
            else:
                with (out / f'partial_responses_{args.shard}.jsonl').open('a') as stream:
                    for record in records:
                        stream.write(canonical(record) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
            (out / f'inflight_{args.shard}.json').unlink(missing_ok=True)
            generation_seconds += elapsed
            if interrupted:
                end_reason=interrupted; break
            saved_interventions += len(spec['neurons'])
            batches_run += 1
            del past, input_ids, attention, scalars
            torch.cuda.empty_cache()
            status('batch_saved', next_batch=results.next_batch, session_batches=batches_run,
                   session_interventions=saved_interventions, session_generation_seconds=round(generation_seconds,3),
                   session_interventions_per_second=saved_interventions/max(generation_seconds,.001),
                   total_grid_interventions=task_count(config))
    except Exception as exc:
        status('failed', next_batch=results.next_batch, error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        final_cursor = results.next_batch
        results.export(out / f'responses_{args.shard}.jsonl')
        results.db.close()
        resume_checkpoint(out / f'shard_{args.shard}.sqlite', out / f'shard_{args.shard}.resume.sqlite')
    status('complete' if end_reason=='complete' else 'stopped', reason=end_reason,
           next_batch=final_cursor,
           session_batches=batches_run, session_interventions=saved_interventions)
    if end_reason not in {'complete', 'batch_budget', 'session_budget', 'user_stop'}:
        raise RuntimeError(f'Sweep stopped early: {end_reason}; partial results saved')


if __name__ == '__main__':
    main()
