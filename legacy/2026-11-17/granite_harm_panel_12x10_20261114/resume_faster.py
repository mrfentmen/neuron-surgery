"""Resume the identical saved experiment once the old runner closes cleanly."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def save(state, **extra):
    tmp = HERE / 'speedup_status.tmp'
    tmp.write_text(json.dumps(dict(state=state, updated_utc=datetime.now(timezone.utc).isoformat(), **extra), indent=2) + '\n')
    tmp.replace(HERE / 'speedup_status.json')


def main():
    request = json.loads((HERE / 'speedup_request.json').read_text())
    old = request['original_launcher']
    spec = importlib.util.spec_from_file_location('serial_helper', ROOT / 'granite_L21_support_clarification_20261114/run_priority.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    save('waiting_for_saved_reply_boundary', old_pid=old['runner_pid'])
    end = helper.wait_exit(HERE, old['runner_pid'], {'user_stop_after_reply', 'complete'}, 600)
    if (HERE / 'STOP').exists():
        save('cancelled_before_resume')
        return
    if end['reason'] == 'complete':
        save('already_complete')
        return
    # Retire only this run's old read-only monitors before starting replacements.
    for key, script in [('watchdog_pid','watchdog.py'), ('completion_monitor_pid','monitor_completion.py')]:
        try:
            process = psutil.Process(old[key])
            if process.status() != psutil.STATUS_ZOMBIE and str(HERE / script) in ' '.join(process.cmdline()):
                process.terminate()
                process.wait(timeout=5)
        except psutil.NoSuchProcess:
            pass
    records = [json.loads(s) for s in (HERE / 'responses.jsonl').read_text().splitlines()]
    plan = json.loads((HERE / 'plan.json').read_text())
    plan_hash = hashlib.sha256((HERE / 'plan.json').read_bytes()).hexdigest()
    assert plan_hash == (HERE / 'plan.sha256').read_text().strip()
    assert len(records) == end['completed']
    assert all(r['job'] == plan['jobs'][i] and r['plan_sha256'] == plan_hash for i,r in enumerate(records))
    preserved = hashlib.sha256((HERE / 'responses.jsonl').read_bytes()).hexdigest()
    marker = HERE / 'STOP_AFTER_REPLY'
    if marker.exists():
        marker.rename(HERE / 'STOP_AFTER_REPLY_before_speedup')
    record = helper.launch(HERE, resume=True)
    save('resumed_faster', preserved_replies=len(records), preserved_prefix_sha256=preserved, **record)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        save('failed', error=type(exc).__name__ + ': ' + str(exc))
        raise
