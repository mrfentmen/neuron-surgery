"""Wait for the saved panel to finish, then start one detached interactive chat."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil
from protocol import atomic_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PLAN = json.loads((HERE / 'plan.json').read_text())
PARENT = ROOT / PLAN['parent_directory']


def save(state, **extra):
    atomic_json(HERE / 'queue_status.json', dict(state=state, queue_pid=os.getpid(),
        updated_utc=datetime.now(timezone.utc).isoformat(), **extra))


def parent_running(pid):
    try:
        process = psutil.Process(pid)
        return (process.status() != psutil.STATUS_ZOMBIE
                and str(PARENT / 'run_survey.py') in process.cmdline())
    except psutil.NoSuchProcess:
        return False


def validate_parent(state):
    if hashlib.sha256((PARENT / 'plan.json').read_bytes()).hexdigest() != PLAN['parent_plan_sha256']:
        raise RuntimeError('Parent plan changed')
    plan = json.loads((PARENT / 'plan.json').read_text())
    records = [json.loads(s) for s in (PARENT / 'responses.jsonl').read_text().splitlines()]
    if len(records) != plan['total_jobs'] or state.get('completed') != plan['total_jobs']:
        raise RuntimeError('Panel has missing replies')
    if any(r['job'] != plan['jobs'][i] for i, r in enumerate(records)):
        raise RuntimeError('Panel replies do not match frozen jobs')
    closed = json.loads((PARENT / f'closed_{state["pid"]}.json').read_text())
    if closed['reason'] != 'complete':
        raise RuntimeError('Panel did not close successfully')


def launch():
    if (ROOT / '.handover-model.lock').exists():
        raise RuntimeError('Another model owns the shared lock')
    if (HERE / 'STOP').exists() or (PARENT / 'STOP').exists():
        raise InterruptedError('Stop requested')
    if hashlib.sha256((HERE / 'plan.json').read_bytes()).hexdigest() != (HERE / 'plan.sha256').read_text().strip():
        raise RuntimeError('Chat plan changed')
    with (HERE / 'launch_started.json').open('x') as stream:
        json.dump(dict(started_utc=datetime.now(timezone.utc).isoformat()), stream)
    with (HERE / 'run.log').open('ab', buffering=0) as log:
        process = subprocess.Popen(['/usr/bin/sandbox-exec', '-p',
            '(version 1) (allow default) (deny network*)', sys.executable, '-u', str(HERE / 'chat.py')],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    for _ in range(100):
        if process.poll() is not None:
            raise RuntimeError('Chat exited during startup; see run.log')
        path = HERE / 'status.json'
        if path.exists() and json.loads(path.read_text()).get('pid') == process.pid:
            break
        time.sleep(.1)
    else:
        process.terminate()
        raise RuntimeError('Chat did not publish startup status')
    with (HERE / 'watchdog.log').open('ab', buffering=0) as log:
        guard = subprocess.Popen([sys.executable, '-u', str(HERE / 'watchdog.py')], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    awake = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(process.pid)],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True)
    record = dict(runner_pid=process.pid, watchdog_pid=guard.pid, caffeinate_pid=awake.pid,
        started_utc=datetime.now(timezone.utc).isoformat(), mode=PLAN['mode'],
        max_new_tokens=PLAN['decoding']['max_new_tokens'])
    atomic_json(HERE / 'launcher.json', record)
    return record


def main():
    deadline = time.monotonic() + 6 * 3600
    while time.monotonic() < deadline:
        if (HERE / 'STOP').exists() or (PARENT / 'STOP').exists():
            save('cancelled_before_launch')
            return
        state = json.loads((PARENT / 'status.json').read_text())
        save('waiting_for_panel', parent_state=state['state'], completed=state.get('completed', 0))
        if state['state'] in {'stopped', 'failed'}:
            save('blocked_parent_stopped', details=state)
            return
        if not parent_running(state['pid']):
            if state['state'] != 'complete':
                raise RuntimeError('Parent process exited before successful completion')
            if not (ROOT / '.handover-model.lock').exists():
                validate_parent(state)
                save('launched', **launch())
                return
        time.sleep(15)
    save('queue_timeout')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        save('queue_failed', error=type(exc).__name__ + ': ' + str(exc))
        raise
