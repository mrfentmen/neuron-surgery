#!/usr/bin/env python3
"""Detached sequential launch of the fixed remaining identity questions."""
from datetime import datetime, timezone
from pathlib import Path
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
    value = {'state': state, 'updated_utc': datetime.now(timezone.utc).isoformat(), **extra}
    tmp = HERE / 'status.tmp'
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(HERE / 'status.json')
    print(json.dumps(value), flush=True)


def stopped():
    return any((HERE / name).exists() for name in ['STOP', 'STOP_AFTER_REPLY'])


def main():
    assert not (HERE / 'status.json').exists()
    plan = load(HERE / 'plan.json')
    plan_hash = digest(HERE / 'plan.json')
    parent = ROOT / plan['wait_for_directory']
    assert load(parent / 'status.json')['state'] == 'complete'
    done = load(parent / 'completion.json')
    verified = load(parent / 'verification.json')
    assert done['generations'] == verified['generations_verified'] == plan['wait_for_generations']
    assert digest(parent / 'plan.json') == done['plan_sha256'] == plan['wait_for_plan_sha256']
    assert digest(parent / 'responses.jsonl') == done['response_sha256']
    for source, expected in plan['resume_sources'].items():
        assert digest(ROOT / source) == expected
    assert (ROOT / 'sentience_collection_20261113/verification.json').exists(), 'Finish CPU collection before model launch.'
    elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(done['completed_utc'])).total_seconds()
    pause = max(0, plan['cooling']['initial_cooldown_seconds'] - elapsed)
    status('cooling_before_start', seconds=round(pause, 1), generations=plan['total_generations'])
    until = time.monotonic() + pause
    while time.monotonic() < until:
        if stopped():
            status('cancelled')
            return
        time.sleep(min(1, max(0, until-time.monotonic())))
    assert not (ROOT / '.handover-model.lock').exists(), 'Another model runner owns the lock.'
    assert digest(HERE / 'plan.json') == plan_hash
    if stopped():
        status('cancelled')
        return
    prefix = ['/usr/bin/sandbox-exec', '-p', '(version 1) (allow default) (deny network*)', sys.executable, '-u']
    with (HERE / 'run.log').open('x') as log:
        child = subprocess.Popen(prefix + [str(HERE / 'run_test.py')], cwd=ROOT,
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        status('running', generations=plan['total_generations'], model_pid=child.pid)
        code = child.wait()
    if code:
        f = HERE / 'failure.json'
        failure = load(f) if f.exists() else {}
        state = 'stopped_by_user' if failure.get('type') == 'InterruptedError' else 'runner_failed'
        status(state, exit_code=code, saved_replies=failure.get('completed'))
        return
    status('verifying')
    result = subprocess.run(prefix + [str(HERE / 'finish.py')], cwd=ROOT, stdin=subprocess.DEVNULL)
    if result.returncode:
        status('postprocessing_failed', exit_code=result.returncode)
        return
    status('complete', generations=plan['total_generations'], transcript='TRANSCRIPT.txt', manual_review='pending')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('failed', error=type(exc).__name__ + ': ' + str(exc))
        raise
