#!/usr/bin/env python3
"""Verify the stopped predecessor, cool, and run only this 15-reply job."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def status(state, **extra):
    record = {'state': state, 'updated_utc': datetime.now(timezone.utc).isoformat(), **extra}
    temp = HERE / 'status.tmp'
    temp.write_text(json.dumps(record, indent=2) + '\n')
    temp.replace(HERE / 'status.json')
    print(json.dumps(record), flush=True)


def main():
    assert not (HERE / 'status.json').exists()
    plan = load(HERE / 'plan.json')
    parent = ROOT / plan['stopped_parent']
    partial = load(parent / 'partial_completion.json')
    assert digest(parent / 'partial_completion.json') == plan['parent_partial_audit_sha256']
    assert digest(parent / 'responses.jsonl') == partial['response_sha256']
    assert partial['no_response_cut_short_by_user_stop']
    assert load(parent / 'status.json')['state'] == 'stopped_by_user'
    assert not (ROOT / '.handover-model.lock').exists()
    launch = load(parent / 'launch.json')
    try:
        os.kill(launch['launcher_pid'], 0)
    except ProcessLookupError:
        pass
    else:
        raise RuntimeError('Previous launcher still alive; not loading another model.')
    elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(partial['stopped_utc'])).total_seconds()
    pause = max(0, plan['cooling']['initial_cooldown_seconds'] - elapsed)
    status('cooling_before_start', seconds=round(pause, 1), generations=15)
    until = time.monotonic() + pause
    while time.monotonic() < until:
        if (HERE / 'STOP').exists() or (HERE / 'STOP_AFTER_REPLY').exists():
            status('cancelled'); return
        time.sleep(min(1, max(0, until - time.monotonic())))
    if (HERE / 'STOP').exists() or (HERE / 'STOP_AFTER_REPLY').exists():
        status('cancelled'); return
    status('running', generations=15)
    prefix = ['/usr/bin/sandbox-exec', '-p', '(version 1) (allow default) (deny network*)', sys.executable, '-u']
    with (HERE / 'run.log').open('x') as log:
        result = subprocess.run(prefix + [str(HERE / 'run_test.py')], cwd=ROOT,
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        failure = load(HERE / 'failure.json') if (HERE / 'failure.json').exists() else {}
        status('stopped_by_user' if failure.get('type') == 'InterruptedError' else 'runner_failed',
               exit_code=result.returncode, saved_replies=failure.get('completed'))
        return
    status('verifying')
    result = subprocess.run(prefix + [str(HERE / 'finish.py')], cwd=ROOT, stdin=subprocess.DEVNULL)
    if result.returncode:
        status('postprocessing_failed', exit_code=result.returncode); return
    status('complete', generations=15, transcript='TRANSCRIPT.txt', replies='FOLLOWUPS.txt', manual_review='pending')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('failed', error=type(exc).__name__ + ': ' + str(exc))
        raise
