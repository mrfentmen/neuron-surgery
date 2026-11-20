"""Read-only Kaggle monitoring and result download; never starts model inference."""
import argparse
from contextlib import closing
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sqlite3
import sys
import time


def atomic_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def assess_output(output):
    workers = []
    rows, partial_rows = 0, 0
    problems = []
    for shard in (0, 1):
        matches = list(output.rglob(f'status_{shard}.json'))
        if len(matches) != 1:
            problems.append(f'missing worker {shard} status')
            continue
        worker = json.loads(matches[0].read_text())
        workers.append(worker)
        if worker.get('reason') not in {'complete', 'session_budget', 'batch_budget'}:
            problems.append(f'worker {shard}: {worker.get("reason", worker.get("state", "unknown"))}')
        dbs = list(output.rglob(f'shard_{shard}.sqlite'))
        if len(dbs) == 1:
            with closing(sqlite3.connect(dbs[0].resolve().as_uri() + '?mode=ro', uri=True)) as db:
                shard_rows = db.execute('SELECT COUNT(*) FROM results').fetchone()[0]
                rows += shard_rows
                if shard_rows == 0:
                    problems.append(f'worker {shard}: no completed batches saved')
                partial_rows += sum(len(json.loads(value)) for (value,) in db.execute('SELECT payload FROM partial'))
        else:
            problems.append(f'missing or ambiguous worker {shard} database')
    if rows == 0:
        problems.append('no completed batches saved')
    return {'research_outcome': 'stopped_early' if problems else 'planned_chunk_completed',
            'worker_statuses': workers, 'completed_batch_rows': rows,
            'partial_rows': partial_rows, 'problems': problems}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', type=Path, required=True)
    parser.add_argument('--version', type=int, required=True)
    args = parser.parse_args()
    launch = args.launch.resolve()
    launch.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((launch / 'kernel-metadata.json').read_text())
    kernel = f"{metadata['id']}/{args.version}"
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    state = {'kernel': kernel, 'pid': os.getpid(), 'state': 'starting'}

    def save(**updates):
        state.update(updates, updated_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(launch / 'monitor_status.json', state)

    stream = None
    with (launch / 'live.log').open('a', buffering=1) as log:
        try:
            stream = subprocess.Popen(
                [str(Path(sys.executable).parent / 'kaggle'), 'kernels', 'logs', kernel, '--follow'],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 7 * 60 * 60
            save(state='monitoring', log_pid=stream.pid)
            while time.monotonic() < deadline:
                try:
                    response = api.kernels_status(kernel)
                    remote = str(response.status).split('.')[-1].upper()
                    save(remote_state=remote, failure_message=response.failure_message,
                         last_poll_error=None, log_stream_returncode=stream.poll())
                    if remote in {'COMPLETE', 'ERROR', 'CANCELED', 'CANCELLED', 'FAILED', 'CANCEL_ACKNOWLEDGED'}:
                        break
                except Exception as exc:
                    save(last_poll_error=f'{type(exc).__name__}: {exc}')
                time.sleep(30)
            else:
                save(state='monitor_timeout')
                return
            save(state='downloading')
            output = launch / 'output'
            for attempt in range(5):
                try:
                    files, _ = api.kernels_output(kernel, str(output), quiet=True)
                    if not files:
                        raise RuntimeError('No output files exposed yet')
                    break
                except Exception as exc:
                    save(download_error=f'{type(exc).__name__}: {exc}', download_attempt=attempt + 1)
                    if attempt == 4:
                        raise
                    time.sleep(30)
            counts = {}
            for path in output.rglob('responses_*.jsonl'):
                with path.open() as source:
                    counts[str(path.relative_to(output))] = sum(bool(line.strip()) for line in source)
            audit = assess_output(output)
            save(state='downloaded', output=str(output), file_count=len(files), response_rows=counts,
                 download_error=None, **audit)
            if audit['problems']:
                (launch / 'ATTENTION_REQUIRED.txt').write_text(
                    'Kaggle finished, but the research job needs attention.\n' +
                    '\n'.join(audit['problems']) + '\n')
        except Exception as exc:
            save(state='monitor_error', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            if stream is not None and stream.poll() is None:
                stream.terminate()
                try:
                    stream.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    stream.kill()
                    stream.wait()


if __name__ == '__main__':
    main()
