"""Run exactly two GPU workers, validate first episodes, then release the fixed queue."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from worker import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True)
    p.add_argument('--model-path',required=True);p.add_argument('--seconds',type=int,default=10800)
    args=p.parse_args();work=args.work.resolve();out=work/'results';out.mkdir(exist_ok=True)
    deadline=time.monotonic()+args.seconds
    workers=[];gate_passed=False
    def record(state,**details):
        obj={'state':state,'updated_utc':datetime.now(timezone.utc).isoformat(),**details}
        atomic_json(out/'supervisor_status.json',obj);print(json.dumps(obj),flush=True)
    try:
        record('starting',expected_gpus=2,worker_budget_seconds=args.seconds)
        for shard in (0,1):
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(shard),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            log=(out/f'worker_{shard}.log').open('ab',buffering=0)
            command=[sys.executable,'-u',str(work/'worker.py'),'--work',str(work),'--model-path',args.model_path,
                     '--shard',str(shard),'--seconds',str(args.seconds)]
            try: proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            except BaseException:log.close();raise
            workers.append((proc,log))
        while any(p.poll() is None for p,_ in workers):
            if any(p.poll() not in (None,0) for p,_ in workers):raise RuntimeError('Worker failed; stopping its partner')
            if time.monotonic()>deadline+90:raise TimeoutError('Supervisor budget exceeded')
            if not gate_passed and all((out/f'startup_{i}.json').exists() for i in (0,1)):
                checks=[json.loads((out/f'startup_{i}.json').read_text()) for i in (0,1)]
                if not all(c['passed'] for c in checks):raise RuntimeError('Baseline failed startup comprehension/action check')
                assert {c['runtime']['cuda_visible_devices'] for c in checks}=={'0','1'}
                atomic_json(out/'startup_validation.json',{'passed':True,'workers':checks})
                (out/'CONTINUE').write_text('Both GPUs and baseline first episodes verified.\n')
                gate_passed=True;record('startup_verified',gpu_count=2)
            statuses=[]
            for i in (0,1):
                path=out/f'status_{i}.json'
                if path.exists():statuses.append(json.loads(path.read_text()))
            print('PROGRESS '+json.dumps(statuses),flush=True)
            time.sleep(20)
        if not gate_passed:raise RuntimeError('Workers exited before startup validation')
        if any(p.returncode!=0 for p,_ in workers):raise RuntimeError('Worker failure')
        statuses=[json.loads((out/f'status_{i}.json').read_text()) for i in (0,1)]
        reasons={s.get('reason') for s in statuses}
        if not reasons<= {'complete','session_budget','user_stop'}:raise RuntimeError('Unexpected worker final status')
        record('complete' if reasons=={'complete'} else 'partial',workers=statuses,
               note='Partial means the fixed queue did not finish; never infer full coverage from kernel completion.')
    except BaseException as exc:
        (out/'STOP').write_text('Supervisor stopped both workers.\n')
        record('failed',error=type(exc).__name__+': '+str(exc));raise
    finally:
        for proc,log in workers:
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            log.close()


if __name__=='__main__':main()
