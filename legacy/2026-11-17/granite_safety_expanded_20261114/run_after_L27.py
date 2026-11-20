"""Queue the fixed extension after the priority workflow and L27 finish."""
from datetime import datetime, timezone
import hashlib
import importlib.util
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
PRIORITY = ROOT / 'granite_L21_support_clarification_20261114'


def save(state, **extra):
    temporary = HERE / 'queue_status.tmp'
    temporary.write_text(json.dumps(dict(state=state, updated_utc=datetime.now(timezone.utc).isoformat(), **extra), indent=2) + '\n')
    temporary.replace(HERE / 'queue_status.json')


def running(pid):
    try:
        process = psutil.Process(pid)
        return process.status() != psutil.STATUS_ZOMBIE and str(PARENT / 'run_survey.py') in ' '.join(process.cmdline())
    except psutil.NoSuchProcess:
        return False


def main():
    deadline = time.monotonic() + 6 * 60 * 60
    while time.monotonic() < deadline:
        if (HERE / 'STOP').exists() or (PARENT / 'STOP').exists():
            save('cancelled_before_launch')
            return
        priority = json.loads((PRIORITY / 'workflow_status.json').read_text())
        state = json.loads((PARENT / 'status.json').read_text())
        save('waiting_for_L27', priority_state=priority['state'], parent_state=state['state'],
             parent_completed=state.get('completed', 0))
        if priority['state'] in {'workflow_stopped_for_review', 'cancelled_before_L21', 'L21_complete_L27_stop_requested'}:
            save('blocked_priority_workflow', details=priority)
            return
        if state['state'] == 'failed':
            save('blocked_parent_failed', details=state)
            return
        if state['state'] == 'stopped':
            expected_pause = state.get('reason') == 'user_stop_after_reply' and priority['state'] in {
                'waiting_for_L27_reply_boundary', 'launching_L21', 'L21_running'}
            if not expected_pause:
                save('blocked_parent_stopped', details=state)
                return
        if state['state'] == 'complete' and not running(state['pid']) and not (ROOT / '.handover-model.lock').exists():
            break
        time.sleep(15)
    else:
        save('queue_timeout')
        return
    assert hashlib.sha256((PARENT / 'plan.json').read_bytes()).hexdigest() == PLAN['parent_plan_sha256']
    parent_plan = json.loads((PARENT / 'plan.json').read_text())
    rows = [json.loads(s) for s in (PARENT / 'responses.jsonl').read_text().splitlines()]
    assert len(rows) == parent_plan['total_jobs']
    assert all(r['job'] == parent_plan['jobs'][i] for i, r in enumerate(rows))
    closed = json.loads((PARENT / f'closed_{state["pid"]}.json').read_text())
    assert closed['reason'] == 'complete'
    with (HERE / 'launch_started.json').open('x') as marker:
        json.dump({'started_utc': datetime.now(timezone.utc).isoformat()}, marker)
    # Reuse the already verified launch helper; it does not execute on import.
    spec = importlib.util.spec_from_file_location('priority_launch_helper', PRIORITY / 'run_priority.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    record = module.launch(HERE)
    save('launched', **record)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        save('queue_failed', error=type(exc).__name__ + ': ' + str(exc))
        raise
