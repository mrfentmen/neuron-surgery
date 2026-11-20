"""Detached queue for fixed clarification round two after the current L24 batch."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PLAN = json.loads((HERE / 'plan.json').read_text())
PARENT = ROOT / PLAN['parent_directory']


def save(state, **extra):
    value = dict(state=state, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
    tmp = HERE / 'queue_status.tmp'
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(HERE / 'queue_status.json')


def parent_running(pid):
    try:
        p = psutil.Process(pid)
        return p.status() != psutil.STATUS_ZOMBIE and str(PARENT / 'run_survey.py') in ' '.join(p.cmdline())
    except psutil.NoSuchProcess:
        return False


def main():
    deadline = time.monotonic() + 3 * 60 * 60
    while time.monotonic() < deadline:
        if (HERE / 'STOP').exists() or (PARENT / 'STOP').exists():
            save('cancelled_before_launch')
            return
        path = PARENT / 'status.json'
        if not path.exists():
            path = PARENT / 'queue_status.json'
        state = json.loads(path.read_text())
        save('waiting_for_L24_round1', parent_state=state['state'], completed=state.get('completed', 0))
        if state['state'] in {'stopped', 'failed', 'blocked_parent_failed', 'blocked_parent_stopped',
                               'blocked_priority_workflow', 'cancelled_before_launch', 'queue_failed', 'queue_timeout'}:
            save('blocked_parent_stopped', details=state)
            return
        if state['state'] == 'complete' and not parent_running(state['pid']) and not (ROOT / '.handover-model.lock').exists():
            break
        time.sleep(15)
    else:
        save('queue_timeout')
        return
    assert hashlib.sha256((PARENT / 'plan.json').read_bytes()).hexdigest() == PLAN['parent_plan_sha256']
    parent_plan = json.loads((PARENT / 'plan.json').read_text())
    records = [json.loads(s) for s in (PARENT / 'responses.jsonl').read_text().splitlines()]
    assert len(records) == parent_plan['total_jobs']
    assert all(r['job'] == parent_plan['jobs'][i] for i, r in enumerate(records))
    closed = json.loads((PARENT / f'closed_{state["pid"]}.json').read_text())
    assert closed['reason'] == 'complete'
    with (HERE / 'launch_started.json').open('x') as stream:
        json.dump({'started_utc': datetime.now(timezone.utc).isoformat()}, stream)
    helper_path = ROOT / 'granite_L21_support_clarification_20261114/run_priority.py'
    spec = importlib.util.spec_from_file_location('serial_launch_helper', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    record = helper.launch(HERE)
    save('launched', **record)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        save('queue_failed', error=type(exc).__name__ + ': ' + str(exc))
        raise
