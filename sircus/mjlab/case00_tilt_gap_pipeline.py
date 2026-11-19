"""CPU admission, bounded leased4090 native64, then independent CPU audit."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .gpu_slot import GPU_UUID
from .sweep_worker import atomic


def supervise(command,*,source,env,log,state,statepath,output,phase,limit,queue=0):
    """Only signal the direct child owned by this pipeline on its deadline."""
    child=subprocess.Popen(command,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    state.update(status=phase,child_pid=child.pid);atomic(statepath,state)
    absolute=time.monotonic()+limit+queue+120
    while child.poll() is None:
        budget=output.with_name(output.name+('-cpu-budget.json' if phase=='cpu_admission' else '-native-budget.json'))
        expired=False
        if budget.exists():
            bound=json.loads(budget.read_text());began=bound.get('started_unix_s',bound.get('acquired_unix_s'))
            expired=time.time()>=began+limit
        if expired or time.monotonic()>=absolute:
            child.terminate()
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            state.update(error='Own '+phase+' deadline reached; partial evidence retained');break
        time.sleep(2)
    return child.returncode


def main():
    p=argparse.ArgumentParser(description=__doc__)
    keys=('parent-study','parent-audit','reference','assets','output','gpu-lock')
    for key in keys:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--cpu-limit',type=float,default=3600);p.add_argument('--native-limit',type=float,default=3600)
    p.add_argument('--queue-timeout',type=float,default=7200);a=p.parse_args()
    if not 60<=a.cpu_limit<=3600 or not 60<=a.native_limit<=3600 or not 0<a.queue_timeout<=7200:
        raise ValueError('Finite CPU/native/queue caps required')
    if a.output.exists():raise ValueError('Fresh study namespace required')
    for suffix in ('-cpu-budget.json','-native-budget.json'):
        if a.output.with_name(a.output.name+suffix).exists():raise ValueError('Fresh phase budget namespace required')
    source=Path(__file__).resolve().parents[2];a.output.parent.mkdir(parents=True,exist_ok=True)
    state=dict(status='starting',source=str(source),output=str(a.output),pipeline_pid=os.getpid(),started_unix_s=time.time())
    statepath=a.output.with_suffix('.pipeline.json')
    base=[sys.executable,'-u','-m','sircus.mjlab.case00_tilt_gap_study']
    for key in keys:base+=['--'+key,str(getattr(a,key.replace('-','_')))]
    base+=['--cpu-limit',str(a.cpu_limit),'--native-limit',str(a.native_limit),'--queue-timeout',str(a.queue_timeout)]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
    with a.output.with_suffix('.pipeline.log').open('x') as log:
        code=supervise(base+['--phase','admit'],source=source,env=dict(env,CUDA_VISIBLE_DEVICES=''),log=log,
            state=state,statepath=statepath,output=a.output,phase='cpu_admission',limit=a.cpu_limit)
        state['admission_exit_code']=code
        if code==0:
            code=supervise(base+['--phase','native'],source=source,env=dict(env,CUDA_VISIBLE_DEVICES=GPU_UUID),log=log,
                state=state,statepath=statepath,output=a.output,phase='native',limit=a.native_limit,queue=a.queue_timeout)
            state['native_exit_code']=code
        if code==0:
            state.update(status='cpu_audit');atomic(statepath,state)
            command=[sys.executable,'-u','-m','sircus.mjlab.case00_tilt_gap_verify','--study',str(a.output),
                '--source',str(source),'--output',str(a.output.with_name(a.output.name+'-audit.json'))]
            for key in ('parent-study','parent-audit','assets'):command+=['--'+key,str(getattr(a,key.replace('-','_')))]
            try:
                audit=subprocess.run(command,cwd=source,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,timeout=1800)
                state.update(audit_exit_code=audit.returncode,status='completed' if audit.returncode==0 else 'audit_failed')
            except subprocess.TimeoutExpired:state.update(status='audit_failed',error='Own CPU audit deadline; evidence retained')
        else:state.update(status='incomplete',error=state.get('error','Owned phase exited before completion; no new phase launched'))
    state['finished_unix_s']=time.time();atomic(statepath,state);print(json.dumps(state),flush=True)
    if state['status']!='completed':raise RuntimeError('Bounded case00-tilt-gap pipeline incomplete: '+state['status'])

if __name__=='__main__':main()
