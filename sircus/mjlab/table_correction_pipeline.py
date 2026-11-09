"""Finite owned 4090 worker followed by independent CPU audit; no server restart."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .gpu_slot import GPU_UUID
from .sweep_worker import atomic


def main():
    p=argparse.ArgumentParser(description=__doc__)
    keys=('parent-study','parent-audit','reference','assets','rule','output','gpu-lock')
    for key in keys:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--wall-time-limit',type=float,default=3600)
    p.add_argument('--queue-timeout',type=float,default=1800);a=p.parse_args()
    if not 60<=a.wall_time_limit<=3600 or not 0<a.queue_timeout<=7200:raise ValueError('Finite caps required')
    if a.output.exists():raise ValueError('Fresh evidence namespace required')
    source=Path(__file__).resolve().parents[2]
    state=dict(status='starting',source=str(source),output=str(a.output),pipeline_pid=os.getpid(),started_unix_s=time.time())
    statepath=a.output.with_suffix('.pipeline.json');a.output.parent.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-u','-m','sircus.mjlab.table_correction_study']
    for key in keys:command+=['--'+key,str(getattr(a,key.replace('-','_')))]
    command+=['--wall-time-limit',str(a.wall_time_limit),'--queue-timeout',str(a.queue_timeout)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU_UUID,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
    with a.output.with_suffix('.pipeline.log').open('x') as log:
        child=subprocess.Popen(command,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        state.update(status='running',child_pid=child.pid);atomic(statepath,state)
        deadline=time.monotonic()+a.queue_timeout+a.wall_time_limit+120
        while child.poll() is None:
            budget=a.output.with_name(a.output.name+'-budget.json')
            expired=budget.exists() and time.time()>=json.loads(budget.read_text())['acquired_unix_s']+a.wall_time_limit
            if expired or time.monotonic()>deadline:
                child.terminate()
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                state.update(status='failed',error='Own finite worker deadline reached; partial evidence retained');break
            time.sleep(2)
        state['study_exit_code']=child.returncode
        if child.returncode==0:
            state.update(status='cpu_audit');atomic(statepath,state)
            command=[sys.executable,'-u','-m','sircus.mjlab.table_correction_verify','--study',str(a.output),
                '--source',str(source),'--output',str(a.output.with_name(a.output.name+'-audit.json'))]
            for key in ('parent-study','parent-audit','assets'):command+=['--'+key,str(getattr(a,key.replace('-','_')))]
            try:
                result=subprocess.run(command,cwd=source,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,timeout=1800)
                state.update(audit_exit_code=result.returncode,status='completed' if result.returncode==0 else 'audit_failed')
            except subprocess.TimeoutExpired:state.update(status='audit_failed',error='Own CPU audit timeout; evidence retained')
        else:state.update(status='failed',error=state.get('error','Native worker exited; partial evidence retained'))
    state['finished_unix_s']=time.time();atomic(statepath,state);print(json.dumps(state),flush=True)
    if state['status']!='completed':raise RuntimeError('Correction pipeline incomplete: '+state['status'])

if __name__=='__main__':main()
