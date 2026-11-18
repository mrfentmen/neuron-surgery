"""Detached API/log monitor and result downloader. Never launches or reruns a kernel."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from worker import atomic_json


def audit_output(output):
    episodes=0;steps=0;problems=[];status=[]
    for shard in (0,1):
        databases=list(output.rglob(f'shard_{shard}.sqlite'))
        if len(databases)!=1:problems.append(f'worker {shard}: missing/ambiguous database');continue
        db=sqlite3.connect(databases[0].resolve().as_uri()+'?mode=ro',uri=True)
        try:
            if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok':problems.append(f'worker {shard}: database integrity')
            episodes+=db.execute('SELECT COUNT(*) FROM episodes').fetchone()[0]
            steps+=db.execute('SELECT COUNT(*) FROM steps').fetchone()[0]
        finally:db.close()
        files=list(output.rglob(f'status_{shard}.json'))
        if len(files)==1:
            value=json.loads(files[0].read_text());status.append(value)
            if value.get('reason') not in {'complete','session_budget','user_stop'}:
                problems.append(f'worker {shard}: '+str(value.get('state')))
    gates=list(output.rglob('startup_validation.json'))
    startup_passed=len(gates)==1 and json.loads(gates[0].read_text()).get('passed') is True
    if not startup_passed:problems.append('Startup validation did not pass')
    return {'saved_episodes':episodes,'saved_replies':steps,'startup_passed':startup_passed,
            'worker_statuses':status,'problems':problems}


def main():
    p=argparse.ArgumentParser();p.add_argument('--launch',type=Path,required=True);p.add_argument('--version',type=int,required=True)
    args=p.parse_args();launch=args.launch.resolve()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate()
    metadata=json.loads((launch/'kernel-metadata.json').read_text())
    kernel=f"{metadata['id']}/{args.version}"
    status={'kernel':kernel,'pid':os.getpid()};stream=None
    def save(**kwargs):
        status.update(kwargs,updated_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(launch/'monitor_status.json',status)
    try:
        with (launch/'live.log').open('a',buffering=1) as log:
            stream=subprocess.Popen([str(Path(sys.executable).parent/'kaggle'),'kernels','logs',kernel,'--follow'],
                                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            deadline=time.monotonic()+5*3600;save(state='monitoring',log_pid=stream.pid)
            while time.monotonic()<deadline:
                try:
                    result=api.kernels_status(kernel);remote=str(result.status).split('.')[-1].upper()
                    save(remote_state=remote,last_error=None)
                    if remote in {'COMPLETE','ERROR','FAILED','CANCEL_ACKNOWLEDGED','CANCELED','CANCELLED'}:break
                except Exception as exc:save(last_error=f'{type(exc).__name__}: {exc}')
                time.sleep(30)
            else:save(state='monitor_timeout');return
            save(state='downloading')
            for attempt in range(5):
                try:
                    files,_=api.kernels_output(kernel,str(launch/'output'),quiet=True)
                    if not files:raise RuntimeError('No output files exposed yet')
                    break
                except Exception as exc:
                    save(download_error=f'{type(exc).__name__}: {exc}')
                    if attempt==4:raise
                    time.sleep(30)
            audit=audit_output(launch/'output')
            save(state='downloaded',files=len(files),**audit)
            if audit['problems']:(launch/'ATTENTION_REQUIRED.txt').write_text('\n'.join(audit['problems'])+'\n')
    except Exception as exc:save(state='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        if stream and stream.poll() is None:
            stream.terminate()
            try:stream.wait(timeout=10)
            except subprocess.TimeoutExpired:stream.kill();stream.wait()


if __name__=='__main__':main()
