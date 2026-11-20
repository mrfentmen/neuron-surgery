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


def verify_first_batch(out, starting_batches):
    for shard in (0, 1):
        validation = json.loads((out / f'long_context_validation_{shard}.json').read_text())
        assert validation['passed'] and validation['context_tokens'] == 10256
        assert validation['batch_size'] == 2
        runtime = json.loads((out / f'runtime_{shard}.json').read_text())
        assert runtime['dtype'] == 'float16', 'This launch expects native fp16 on T4'
        status = json.loads((out / f'status_{shard}.json').read_text())
        assert status['reason'] == 'batch_budget' and status['next_batch'] == starting_batches[shard] + 1
        with closing(sqlite3.connect(out / f'shard_{shard}.sqlite')) as db:
            rows = [json.loads(r[0]) for r in db.execute('SELECT payload FROM results ORDER BY batch,slot')]
        assert len(rows) == 2
        assert {r['batch_index'] for r in rows} == {starting_batches[shard]}
        assert {r['job']['condition'] for r in rows} == {'baseline', 'intervention'}
        for row in rows:
            assert row['max_new_tokens'] == 10000 and row['n_generated'] > 0
            assert row['stop_reason'] in {'eos', 'repetition_stop', 'token_limit'}
            assert row['hook_measurement']['baseline_unchanged']
    return assess_output(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--seconds', type=int, default=18000)
    args = parser.parse_args()
    work, out = args.work.resolve(), args.work.resolve() / 'results'
    out.mkdir(exist_ok=True)
    starting_batches = {}
    for shard in (0, 1):
        path = out / f'shard_{shard}.sqlite'
        with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as db:
            starting_batches[shard] = int(db.execute("SELECT value FROM meta WHERE key='next_batch'").fetchone()[0])
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
        audit = verify_first_batch(out, starting_batches)
        atomic_json(out / 'startup_validation.json', {
            'passed': True, 'checks': '10256-token cache and actual first batch on both GPUs',
            'completed_batch_rows': audit['completed_batch_rows'],
            'starting_batches': starting_batches,
            'remaining_seconds': int(deadline-time.monotonic()),
            'note': 'EOS/repetition may end real replies before 10000; cache test reaches full length.'})
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
