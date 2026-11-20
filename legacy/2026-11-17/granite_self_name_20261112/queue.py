#!/usr/bin/env python3
"""Start four self-name samples only after the 144-reply job exits successfully."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def status(state, **extra):
    record = {'state':state, 'updated_utc':datetime.now(timezone.utc).isoformat(), **extra}
    temp = HERE / 'status.tmp'
    temp.write_text(json.dumps(record,indent=2) + '\n')
    temp.replace(HERE / 'status.json')
    print(json.dumps(record),flush=True)


def load(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):
        return None


def dependency_state(parent):
    if (parent / 'STOP').exists() or (parent / 'failure.json').exists():
        return 'stopped'
    finished = load(parent / 'background_status.json')
    if finished is not None:
        if finished.get('runner_exit_code') != 0 or finished.get('export_exit_code') != 0:
            return 'failed'
        if not (parent / 'FINISHED.txt').exists() or not (parent / 'completion.json').exists():
            return 'failed'
        return 'ready'
    launched = load(parent / 'launch.json')
    if launched is None:
        return 'missing'
    try:
        os.kill(launched['launcher_pid'],0)
    except ProcessLookupError:
        return 'missing'
    return 'waiting'


def execute():
    if (HERE / 'status.json').exists():
        raise SystemExit('Refusing to launch a second queued run.')
    plan = load(HERE / 'plan.json')
    plan_hash = digest(HERE / 'plan.json')
    parent = ROOT / plan['wait_for_directory']
    assert digest(parent / 'plan.json') == plan['wait_for_plan_sha256']
    status('waiting', dependency=plan['wait_for_directory'], generations=plan['total_generations'])
    deadline = time.monotonic() + 8 * 60 * 60
    while True:
        if (HERE / 'STOP').exists():
            status('cancelled',reason='Self-name queue STOP file; main batch unchanged.')
            return
        state = dependency_state(parent)
        if state == 'ready':
            break
        if state != 'waiting':
            status('dependency_' + state,reason='No self-name generation started.')
            return
        if time.monotonic() >= deadline:
            status('timeout',reason='Main batch has not completed successfully within eight hours.')
            return
        time.sleep(15)
    completion = load(parent / 'completion.json')
    assert completion['generations'] == 144
    assert completion['plan_sha256'] == plan['wait_for_plan_sha256']
    assert digest(parent / 'responses.jsonl') == completion['response_sha256']
    status('cooling_before_start',seconds=90)
    for _ in range(90):
        if (HERE / 'STOP').exists():
            status('cancelled',reason='Stopped before model loading.')
            return
        time.sleep(1)
    assert not (ROOT / '.handover-model.lock').exists(), 'Another model runner holds the shared lock.'
    assert digest(HERE / 'plan.json') == plan_hash
    status('running',generations=4,question=plan['question'])
    command = ['/usr/bin/sandbox-exec','-p','(version 1) (allow default) (deny network*)',
               sys.executable,'-u',str(HERE / 'run_test.py')]
    with (HERE / 'run.log').open('x') as log:
        result = subprocess.run(command,cwd=ROOT,stdin=subprocess.DEVNULL,
                                stdout=log,stderr=subprocess.STDOUT)
    if result.returncode != 0:
        status('runner_failed',exit_code=result.returncode,reason='Partial outputs retained; no automatic retry.')
        return
    status('verifying')
    for name in ['verify.py','finish.py']:
        result = subprocess.run([sys.executable,'-u',str(HERE/name)],cwd=ROOT)
        if result.returncode != 0:
            status('postprocessing_failed',script=name,exit_code=result.returncode)
            return
    status('complete',generations=4,transcript='TRANSCRIPT.txt',manual_review='pending')


if __name__ == '__main__':
    try:
        execute()
    except Exception as exc:
        status('failed',error=type(exc).__name__ + ': ' + str(exc))
        raise
