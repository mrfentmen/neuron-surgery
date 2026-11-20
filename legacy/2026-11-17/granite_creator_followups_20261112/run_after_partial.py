#!/usr/bin/env python3
"""Run the same three frozen follow-ups after the user-approved partial stop."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def read(path):
    return json.loads(path.read_text())


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
    plan = read(HERE / 'plan.json')
    amendment = read(HERE / 'dependency_amendment.json')
    parent = ROOT / plan['wait_for_directory']
    partial = read(parent / 'partial_completion.json')
    assert digest(HERE / 'plan.json') == amendment['unchanged_plan_sha256']
    assert digest(parent / 'partial_completion.json') == amendment['partial_audit_sha256']
    assert partial['state'] == 'stopped_at_user_request_after_completed_reply'
    assert partial['no_response_cut_short_by_user_stop']
    assert digest(parent / 'responses.jsonl') == partial['response_sha256']
    assert partial['plan_sha256'] == plan['wait_for_plan_sha256']
    assert digest(parent / 'plan.json') == plan['wait_for_plan_sha256']
    source = read(ROOT / plan['source_file'])
    rows = [json.loads(line) for line in (parent / 'responses.jsonl').read_text().splitlines()]
    assert next(r for r in rows if r['sequence'] == plan['source_sequence']) == source
    elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(partial['stopped_utc'])).total_seconds()
    pause = max(0, 90 - elapsed)
    status('cooling_before_start', seconds=round(pause, 1), partial_parent_replies=len(rows))
    until = time.monotonic() + pause
    while time.monotonic() < until:
        if (HERE / 'STOP').exists():
            status('cancelled')
            return
        time.sleep(min(1, max(0, until - time.monotonic())))
    if (HERE / 'STOP').exists():
        status('cancelled')
        return
    assert not (ROOT / '.handover-model.lock').exists()
    status('running', generations=3, dependency_amendment='dependency_amendment.json')
    prefix = ['/usr/bin/sandbox-exec', '-p', '(version 1) (allow default) (deny network*)', sys.executable, '-u']
    with (HERE / 'run.log').open('x') as log:
        result = subprocess.run(prefix + [str(HERE / 'run_test.py')], cwd=ROOT,
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        status('runner_failed', exit_code=result.returncode)
        return
    status('verifying')
    for name in ['verify.py', 'finish.py']:
        result = subprocess.run(prefix + [str(HERE / name)], cwd=ROOT, stdin=subprocess.DEVNULL)
        if result.returncode:
            status('postprocessing_failed', script=name, exit_code=result.returncode)
            return
    status('complete', generations=3, transcript='TRANSCRIPT.txt', replies='FOLLOWUPS.txt')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('failed', error=type(exc).__name__ + ': ' + str(exc))
        raise
