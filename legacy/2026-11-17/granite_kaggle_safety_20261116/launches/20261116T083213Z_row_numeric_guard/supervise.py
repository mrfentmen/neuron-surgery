"""Two-stage GPU check and bounded run; worker exit alone is not success."""
import argparse
from contextlib import closing
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from monitor_launch import assess_output, atomic_json


def verify_first_batch(out):
    from sweep import batch_spec, sha
    config = json.loads((out.parent / "config.json").read_text())
    for shard in (0, 1):
        validation = json.loads((out / f'long_context_validation_{shard}.json').read_text())
        assert validation['passed'] and validation['context_tokens'] == config['limits']['input_tokens'] + config['limits']['max_new_tokens']
        assert validation['batch_size'] == config['batch_size'] + 1
        runtime = json.loads((out / f'runtime_{shard}.json').read_text())
        assert runtime['dtype'] == 'float16', 'This launch expects native fp16 on T4'
        assert runtime['config_sha256'] == sha(out.parent / 'config.json')
        status = json.loads((out / f'status_{shard}.json').read_text())
        assert status['reason'] == 'batch_budget' and status['next_batch'] == 1
        with closing(sqlite3.connect(out / f'shard_{shard}.sqlite')) as db:
            rows = [json.loads(r[0]) for r in db.execute('SELECT payload FROM results ORDER BY batch,slot')]
        spec = batch_spec(config, shard, 0)
        assert len(rows) == len(spec['neurons']) + 1
        assert {r['job']['condition'] for r in rows} == {'baseline', 'intervention'}
        for slot, row in enumerate(rows):
            assert row['slot'] == slot and row['batch_index'] == 0 and row['shard'] == shard
            assert row['prompt'] == spec['question']['prompt']
            assert row['max_new_tokens'] == 500
            assert row['stop_reason'] in {'eos', 'repetition_stop', 'token_limit', 'nonfinite_logits'}
            assert row['n_generated'] > 0 or (slot > 0 and row['stop_reason'] == 'nonfinite_logits')
            assert row['hook_measurement']['baseline_unchanged']
            assert row['hook_measurement']['hook_calls'] > 0
            if slot:
                candidate = spec['settings'][slot - 1]
                assert row['job']['condition'] == 'intervention'
                for key in ('layer', 'neuron', 'dose'):
                    assert row['job'][key] == candidate[key]
                assert row['hook_measurement']['changed_positions'][slot - 1] > 0
            else:
                assert row['job']['condition'] == 'baseline' and row['scalar_applied'] == 0
                assert row['stop_reason'] != 'nonfinite_logits'
    return assess_output(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--seconds', type=int, default=18000)
    args = parser.parse_args()
    work, out = args.work.resolve(), args.work.resolve() / 'results'
    out.mkdir(exist_ok=True)
    deadline = time.monotonic() + args.seconds

    def record(state, **details):
        obj = dict(state=state, updated_utc=datetime.now(timezone.utc).isoformat(), **details)
        atomic_json(out / 'supervisor_status.json', obj)
        print(json.dumps(obj), flush=True)

    def run_stage(stage, max_batches):
        workers = []
        record(stage)
        try:
            for shard in (0, 1):
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=str(shard), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
                log = (out / f'worker_{shard}.log').open('ab', buffering=0)
                command = [sys.executable, '-u', str(work / 'sweep.py'), '--config', str(work / 'config.json'),
                           '--model-path', args.model_path, '--shard', str(shard), '--out', str(out),
                           '--max-seconds', str(max(1, int(deadline-time.monotonic()))),
                           '--max-batches', str(max_batches)]
                try:
                    proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                            start_new_session=True)
                except BaseException:
                    log.close()
                    raise
                workers.append((proc, log))
            while any(proc.poll() is None for proc, _ in workers):
                if any(proc.poll() not in (None, 0) for proc, _ in workers):
                    raise RuntimeError('Worker failed; stopping its partner and preserving output')
                if time.monotonic() > deadline + 90:
                    raise TimeoutError('Supervisor deadline exceeded')
                for shard in (0, 1):
                    path = out / f'status_{shard}.json'
                    if path.exists():
                        print(path.read_text(), flush=True)
                time.sleep(30)
            if any(proc.returncode != 0 for proc, _ in workers):
                raise RuntimeError('Worker failed; inspect saved logs')
            audit = assess_output(out)
            if audit['problems']:
                raise RuntimeError('; '.join(audit['problems']))
        finally:
            for proc, log in workers:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                log.close()

    try:
        run_stage('long_context_and_first_batch_check', 1)
        audit = verify_first_batch(out)
        atomic_json(out / 'startup_validation.json', {
            'passed': True, 'checks': '756-token cache at maximum batch size and actual first batch on both GPUs',
            'completed_batch_rows': audit['completed_batch_rows'],
            'remaining_seconds': int(deadline-time.monotonic()),
            'note': 'EOS/repetition may end real replies before 500; cache test reaches full configured length.'})
        record('first_batch_verified', **audit)
        if deadline-time.monotonic() > 120:
            run_stage('bounded_sweep', 0)
        audit = assess_output(out)
        record('complete', **audit)
    except BaseException as exc:
        record('failed', error=type(exc).__name__ + ': ' + str(exc))
        raise


if __name__ == '__main__':
    main()
