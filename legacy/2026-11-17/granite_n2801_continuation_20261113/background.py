#!/usr/bin/env python3
"""Run one bounded continuation after the completed five-channel batch."""
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
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def status(state, **extra):
    value = {'state': state, 'updated_utc': datetime.now(timezone.utc).isoformat(), **extra}
    temp = HERE / 'status.tmp'
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(HERE / 'status.json')
    print(json.dumps(value), flush=True)


def main():
    assert not (HERE / 'status.json').exists()
    plan = load(HERE / 'plan.json')
    plan_hash = digest(HERE / 'plan.json')
    parent = ROOT / plan['wait_for_directory']
    assert digest(parent / 'plan.json') == plan['wait_for_plan_sha256']
    status('waiting', dependency=parent.name, generations=plan['total_generations'])
    deadline = time.monotonic() + 8 * 60 * 60
    while True:
        if (HERE / 'STOP').exists() or (HERE / 'STOP_AFTER_REPLY').exists():
            status('cancelled')
            return
        state = load(parent / 'status.json') or {}
        if (parent / 'STOP').exists() or (parent / 'failure.json').exists():
            status('dependency_stopped')
            return
        if state.get('state') == 'complete':
            break
        if state.get('state') not in ['waiting', 'cooling_before_start', 'running', 'verifying']:
            status('dependency_failed', parent_status=state)
            return
        launch = load(parent / 'launch.json')
        try:
            assert launch and launch.get('launcher_pid')
            os.kill(launch['launcher_pid'], 0)
        except (AssertionError, ProcessLookupError):
            status('dependency_missing')
            return
        if time.monotonic() >= deadline:
            status('timeout')
            return
        time.sleep(15)
    done = load(parent / 'completion.json')
    checked = load(parent / 'verification.json')
    assert done['generations'] == checked['generations_verified'] == plan['wait_for_generations']
    assert done['plan_sha256'] == plan['wait_for_plan_sha256']
    assert digest(parent / 'responses.jsonl') == done['response_sha256']
    elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(done['completed_utc'])).total_seconds()
    pause = max(0, plan['cooling']['initial_cooldown_seconds'] - elapsed)
    status('cooling_before_start', seconds=round(pause, 1))
    until = time.monotonic() + pause
    while time.monotonic() < until:
        if (HERE / 'STOP').exists() or (HERE / 'STOP_AFTER_REPLY').exists():
            status('cancelled')
            return
        time.sleep(min(1, max(0, until - time.monotonic())))
    assert not (ROOT / '.handover-model.lock').exists()
    assert digest(HERE / 'plan.json') == plan_hash
    if (HERE / 'STOP').exists() or (HERE / 'STOP_AFTER_REPLY').exists():
        status('cancelled')
        return
    status('running', generations=plan['total_generations'])
    prefix = ['/usr/bin/sandbox-exec', '-p', '(version 1) (allow default) (deny network*)', sys.executable, '-u']
    with (HERE / 'run.log').open('x') as log:
        result = subprocess.run(prefix + [str(HERE / 'run_test.py')], cwd=ROOT,
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        failure = load(HERE / 'failure.json') or {}
        state = 'stopped_by_user' if failure.get('type') == 'InterruptedError' else 'runner_failed'
        status(state, exit_code=result.returncode, saved_replies=failure.get('completed'))
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
