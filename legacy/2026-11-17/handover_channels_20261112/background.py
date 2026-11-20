#!/usr/bin/env python3
"""Detached supervisor: run the frozen job once and export text if it succeeds."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    runner = ['/usr/bin/sandbox-exec', '-p', '(version 1) (allow default) (deny network*)',
              sys.executable, '-u', str(HERE / 'run_test.py')]
    result = subprocess.run(runner, cwd=ROOT)
    status = {'runner_exit_code': result.returncode,
              'updated_utc': datetime.now(timezone.utc).isoformat(),
              'manual_review': 'pending'}
    if result.returncode == 0:
        exported = subprocess.run([sys.executable, '-u', str(HERE / 'finish.py')], cwd=ROOT)
        status['export_exit_code'] = exported.returncode
    else:
        print('Runner stopped; partial responses retained. No automatic retry.', flush=True)
    with (HERE / 'background_status.json').open('x') as stream:
        json.dump(status, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
