"""Detached, bounded queue. Starts only after the parent succeeds and exits."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PLAN = json.loads((HERE / 'plan.json').read_text())
PARENT = ROOT / PLAN['parent_directory']


def record(state, **extra):
    data = dict(state=state, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
    temp = HERE / 'queue_status.tmp'
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(HERE / 'queue_status.json')


def parent_running():
    try:
        process = psutil.Process(PLAN['parent_pid'])
        return process.status() != psutil.STATUS_ZOMBIE and str(PARENT / 'run_survey.py') in ' '.join(process.cmdline())
    except psutil.NoSuchProcess:
        return False


def main():
    deadline = time.monotonic() + 6 * 60 * 60
    while time.monotonic() < deadline:
        if (HERE / 'STOP').exists() or (HERE / 'STOP_AFTER_REPLY').exists():
            record('cancelled_before_launch')
            return
        status = json.loads((PARENT / 'status.json').read_text())
        record('waiting_for_parent', parent_state=status['state'], completed=status.get('completed'))
        if status['state'] in {'stopped', 'failed'}:
            record('blocked_parent_stopped', parent_status=status)
            return
        if status['state'] == 'complete' and not parent_running() and not (ROOT / '.handover-model.lock').exists():
            break
        if not parent_running() and status['state'] != 'complete':
            record('blocked_parent_exited_without_completion', parent_status=status)
            return
        time.sleep(15)
    else:
        record('queue_timeout')
        return
    assert hashlib.sha256((PARENT / 'plan.json').read_bytes()).hexdigest() == PLAN['parent_plan_sha256']
    expected = json.loads((PARENT / 'plan.json').read_text())['total_jobs']
    rows = [json.loads(line) for line in (PARENT / 'responses.jsonl').read_text().splitlines()]
    assert len(rows) == expected and [r['job']['job_id'] for r in rows] == list(range(expected))
    assert all(r['stop_reason'] in {'eos', 'token_limit', 'repetition_stop', 'generation_timeout'} for r in rows)
    source = json.loads((HERE / PLAN['source_response']).read_text())
    assert rows[source['job']['job_id']] == source
    closed = json.loads((PARENT / f'closed_{PLAN["parent_pid"]}.json').read_text())
    assert closed['reason'] == 'complete'
    # Exclusive launch marker prevents a second queue process from starting inference.
    with (HERE / 'launch_started.json').open('x') as stream:
        json.dump({'started_utc': datetime.now(timezone.utc).isoformat()}, stream)
    with (HERE / 'run.log').open('ab', buffering=0) as log:
        runner = subprocess.Popen(['/usr/bin/sandbox-exec', '-p', '(version 1) (allow default) (deny network*)',
                                   sys.executable, '-u', str(HERE / 'run_survey.py')],
                                  cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
    for _ in range(50):
        if (HERE / 'status.json').exists():
            break
        if runner.poll() is not None:
            raise RuntimeError(f'Runner exited at startup: {runner.returncode}')
        time.sleep(.1)
    assert (HERE / 'status.json').exists(), 'No runner status'
    children = {}
    for key, script, log_name in [('watchdog_pid', 'watchdog.py', 'watchdog.log'),
                                  ('completion_monitor_pid', 'monitor_completion.py', 'completion_monitor.log')]:
        with (HERE / log_name).open('ab', buffering=0) as log:
            proc = subprocess.Popen([sys.executable, '-u', str(HERE / script)], cwd=ROOT,
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        children[key] = proc.pid
    awake = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(runner.pid)], stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    launch = dict(runner_pid=runner.pid, caffeinate_pid=awake.pid, **children,
                  started_utc=datetime.now(timezone.utc).isoformat(), jobs=PLAN['total_jobs'], max_new_tokens=500)
    (HERE / 'launcher.json').write_text(json.dumps(launch, indent=2) + '\n')
    record('launched', **launch)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        record('queue_failed', error=type(exc).__name__ + ': ' + str(exc))
        raise
