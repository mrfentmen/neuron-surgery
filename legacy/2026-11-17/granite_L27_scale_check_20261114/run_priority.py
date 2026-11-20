"""Serial priority diagnostic, then resume the preserved safety batch and queue."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PARENT = ROOT / 'granite_safety_expanded_20261114'
DIALOGUE = ROOT / 'granite_L27_dialogue_20261114'


def save(state, **extra):
    tmp = HERE / 'workflow_status.tmp'
    tmp.write_text(json.dumps(dict(state=state, updated_utc=datetime.now(timezone.utc).isoformat(), **extra), indent=2) + '\n')
    tmp.replace(HERE / 'workflow_status.json')


def main():
    spec = importlib.util.spec_from_file_location('serial_helper', ROOT / 'granite_L21_support_clarification_20261114/run_priority.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    request = json.loads((HERE / 'workflow_request.json').read_text())
    plan = json.loads((HERE / 'plan.json').read_text())
    save('waiting_for_reply_boundary', parent_pid=request['paused_parent_pid'])
    prior = helper.wait_exit(PARENT, request['paused_parent_pid'], {'user_stop_after_reply', 'complete'}, 600)
    if (HERE / 'STOP').exists():
        save('cancelled_before_diagnostic')
        return
    assert hashlib.sha256((PARENT / 'plan.json').read_bytes()).hexdigest() == plan['parent_plan_sha256']
    record = helper.launch(HERE)
    save('diagnostic_running', **record)
    end = helper.wait_exit(HERE, record['runner_pid'], {'complete'}, plan['limits']['wall_seconds'] + 300)
    assert end['completed'] == plan['total_jobs']
    assert (HERE / 'baseline_summary.json').exists()
    if (PARENT / 'STOP').exists():
        save('diagnostic_complete_parent_stop_requested')
        return
    parent_plan = json.loads((PARENT / 'plan.json').read_text())
    if prior['completed'] < parent_plan['total_jobs']:
        record = helper.launch(PARENT, resume=True)
    if not (DIALOGUE / 'STOP').exists():
        with (DIALOGUE / 'queue.log').open('ab', buffering=0) as log:
            process = subprocess.Popen([sys.executable, '-u', str(DIALOGUE / 'run_after_safety.py')],
                                       cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        (DIALOGUE / 'queue_launcher.json').write_text(json.dumps(dict(pid=process.pid, detached=True,
            restarted_after='L27_scale_check'), indent=2) + '\n')
    save('diagnostic_complete_prior_batch_resumed', **record)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        save('stopped_for_review', error=type(exc).__name__ + ': ' + str(exc))
        raise
