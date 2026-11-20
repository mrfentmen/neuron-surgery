"""Launch the finite local check detached, with lightweight safety monitors."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PYTHON = ROOT / '.venv/bin/python'

if (HERE / 'launcher.json').exists() or (ROOT / '.handover-model.lock').exists():
    sys.exit('Existing launch or model lock; inspect before starting another runner.')

def start(command, logfile):
    with (HERE / logfile).open('ab', buffering=0) as stream:
        return subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=True)

sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}
(HERE / 'source_hashes.json').write_text(json.dumps(sources, indent=2) + '\n')
runner = start(['/usr/bin/sandbox-exec', '-p', '(version 1) (allow default) (deny network*)',
                str(PYTHON), '-u', str(HERE / 'run_survey.py')], 'runner.log')
# Write the initial state only if the runner has not already written its own.
try:
    with (HERE / 'status.json').open('x') as stream:
        json.dump({'state': 'launching', 'pid': runner.pid, 'total_jobs': 16,
                   'completed': 0, 'updated_utc': datetime.now(timezone.utc).isoformat()}, stream)
except FileExistsError:
    pass
watchdog = start([str(PYTHON), '-u', str(HERE / 'watchdog.py')], 'watchdog.log')
monitor = start([str(PYTHON), '-u', str(HERE / 'monitor_completion.py')], 'monitor.log')
caffeinate = start(['/usr/bin/caffeinate', '-i', '-w', str(runner.pid)], 'caffeinate.log')
record = {'runner_pid': runner.pid, 'watchdog_pid': watchdog.pid,
          'completion_monitor_pid': monitor.pid, 'caffeinate_pid': caffeinate.pid,
          'started_utc': datetime.now(timezone.utc).isoformat(), 'jobs': 16,
          'detached': True, 'model_network_denied': True, 'memory_start_minimum_GiB': 10}
(HERE / 'launcher.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
