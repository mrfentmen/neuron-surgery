#!/usr/bin/env python3
"""Wait without a model, then resume the exact frozen survey suffix after chat."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
CHAT=ROOT/json.loads((HERE/'active_chat.json').read_text())['directory']


def status(state,**extra):
    obj={'state':state,'updated_utc':datetime.now(timezone.utc).isoformat(),**extra}
    tmp=HERE/'status.tmp';tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(HERE/'status.json')
    print(json.dumps(obj),flush=True)


def stop():
    return (HERE/'STOP').exists() or (HERE/'STOP_AFTER_REPLY').exists()


def main():
    assert not (HERE/'status.json').exists()
    status('waiting_for_terminal_chat',remaining=54)
    while not (CHAT/'closed.json').exists():
        if stop():status('cancelled');return
        time.sleep(2)
    closed=json.loads((CHAT/'closed.json').read_text())
    if closed['reason'] in ['error','generation_guard_stop']:
        status('held_after_chat_error',chat_reason=closed['reason']);return
    while True:
        try:os.kill(closed['pid'],0)
        except ProcessLookupError:break
        if stop():status('cancelled');return
        time.sleep(1)
    pause=max(0,180-(datetime.now(timezone.utc)-datetime.fromisoformat(closed['closed_utc'])).total_seconds())
    status('cooling_before_start',seconds=round(pause),remaining=54)
    end=time.monotonic()+pause
    while time.monotonic()<end:
        if stop():status('cancelled');return
        time.sleep(min(1,max(0,end-time.monotonic())))
    assert not (ROOT/'.handover-model.lock').exists(),'Another model owns the shared lock.'
    prefix=['/usr/bin/sandbox-exec','-p','(version 1) (allow default) (deny network*)',sys.executable,'-u']
    with (HERE/'run.log').open('x') as log:
        child=subprocess.Popen(prefix+[str(HERE/'run_test.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        status('running',generations=54,model_pid=child.pid);code=child.wait()
    if code:status('runner_failed_or_stopped',exit_code=code);return
    status('verifying')
    code=subprocess.run(prefix+[str(HERE/'finish.py')],cwd=ROOT,stdin=subprocess.DEVNULL).returncode
    if code:status('postprocessing_failed',exit_code=code);return
    status('complete',generations=54,transcript='TRANSCRIPT.txt')


if __name__=='__main__':
    try:main()
    except Exception as exc:
        status('failed',error=type(exc).__name__+': '+str(exc));raise
