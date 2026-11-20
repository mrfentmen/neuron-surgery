"""Run the priority L21 check, then resume the preserved L27 plan serially."""
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
L27 = ROOT / 'granite_L27_support_followups_20261114'


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def status(state, **extra):
    write(HERE / 'workflow_status.json', dict(state=state, updated_utc=datetime.now(timezone.utc).isoformat(), **extra))


def alive(pid, script):
    try:
        p = psutil.Process(pid)
        return p.status() != psutil.STATUS_ZOMBIE and str(script) in ' '.join(p.cmdline())
    except psutil.NoSuchProcess:
        return False


def wait_exit(folder, pid, allowed, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not alive(pid, folder / 'run_survey.py'):
            closed = folder / f'closed_{pid}.json'
            if not closed.exists():
                raise RuntimeError(f'{folder.name} exited without closure record')
            end = json.loads(closed.read_text())
            if end['reason'] not in allowed:
                raise RuntimeError(f'{folder.name} stopped unexpectedly: {end["reason"]}')
            return end
        time.sleep(5)
    raise TimeoutError(f'{folder.name} did not exit within its deadline')


def launch(folder, resume=False):
    assert not (ROOT / '.handover-model.lock').exists(), 'Another local model owns the lock'
    assert not (folder / 'STOP').exists(), 'Stop requested'
    config = json.loads((folder / 'plan.json').read_text())
    assert hashlib.sha256((folder / 'plan.json').read_bytes()).hexdigest() == (folder / 'plan.sha256').read_text().strip()
    if resume:
        previous = json.loads((folder / 'launcher.json').read_text())
        write(folder / f'launcher_before_priority_{previous["runner_pid"]}.json', previous)
        marker = folder / 'STOP_AFTER_REPLY'
        if marker.exists():
            marker.rename(folder / 'STOP_AFTER_REPLY_before_L21')
    else:
        assert not (folder / 'launcher.json').exists()
    with (folder / 'run.log').open('ab', buffering=0) as log:
        p = subprocess.Popen(['/usr/bin/sandbox-exec', '-p', '(version 1) (allow default) (deny network*)',
                              sys.executable, '-u', str(folder / 'run_survey.py')], cwd=ROOT,
                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(100):
        path = folder / 'status.json'
        if path.exists() and json.loads(path.read_text()).get('pid') == p.pid:
            break
        if p.poll() is not None:
            raise RuntimeError(f'Runner startup failed: {folder.name}')
        time.sleep(.1)
    else:
        p.terminate()
        raise RuntimeError('Runner did not publish its own startup status')
    children = {}
    for key, script, logfile in [('watchdog_pid', 'watchdog.py', 'watchdog.log'),
                                 ('completion_monitor_pid', 'monitor_completion.py', 'completion_monitor.log')]:
        with (folder / logfile).open('ab', buffering=0) as log:
            child = subprocess.Popen([sys.executable, '-u', str(folder / script)], cwd=ROOT,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        children[key] = child.pid
    awake = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(p.pid)], stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    record = dict(runner_pid=p.pid, **children, caffeinate_pid=awake.pid,
                  started_utc=datetime.now(timezone.utc).isoformat(), jobs=config['total_jobs'],
                  max_new_tokens=config['decoding']['max_new_tokens'], resumed=resume)
    write(folder / 'launcher.json', record)
    return record


def main():
    request = json.loads((HERE / 'workflow_request.json').read_text())
    status('waiting_for_L27_reply_boundary', parent_pid=request['paused_L27_pid'])
    prior = wait_exit(L27, request['paused_L27_pid'], {'user_stop_after_reply', 'complete'}, 600)
    if (HERE / 'STOP').exists():
        status('cancelled_before_L21')
        return
    assert not (ROOT / '.handover-model.lock').exists()
    status('launching_L21', preserved_L27_replies=prior['completed'])
    run = launch(HERE)
    status('L21_running', **run)
    end = wait_exit(HERE, run['runner_pid'], {'complete'}, 7500)
    plan = json.loads((HERE / 'plan.json').read_text())
    assert end['completed'] == plan['total_jobs']
    rows = [json.loads(s) for s in (HERE / 'responses.jsonl').read_text().splitlines()]
    assert len(rows) == plan['total_jobs']
    if (L27 / 'STOP').exists():
        status('L21_complete_L27_stop_requested')
        return
    l27_plan = json.loads((L27 / 'plan.json').read_text())
    assert hashlib.sha256((L27 / 'plan.json').read_bytes()).hexdigest() == request['L27_plan_sha256']
    if prior['completed'] == l27_plan['total_jobs']:
        status('both_complete')
        return
    resumed = launch(L27, resume=True)
    write(L27 / 'queue_status.json', dict(state='resumed_after_L21', **resumed))
    status('L21_complete_L27_resumed', **resumed)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('workflow_stopped_for_review', error=type(exc).__name__ + ': ' + str(exc))
        raise
