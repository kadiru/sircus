"""Owned finite4090 corrected-imitation worker followed by a bounded independent CPU audit."""
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
from .coverage_recovery_pipeline import supervise

from .corrected_imitation_contract import INPUT_KEYS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in (*INPUT_KEYS,'output','gpu-lock'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--wall-time-limit',type=float,default=3600)
    p.add_argument('--queue-timeout',type=float,default=900)
    p.add_argument('--global-limit',type=float,default=6300)
    p.add_argument('--absolute-deadline-unix',type=float,default=None);a=p.parse_args()
    if not (60<=a.wall_time_limit<=3600 and 0<a.queue_timeout<=900 and a.wall_time_limit<a.global_limit<=6300):
        raise ValueError('Approved finite worker/queue/pipeline caps required')
    if a.output.exists():raise ValueError('Fresh evidence namespace required')
    for path in (a.output.with_suffix('.pipeline.json'),a.output.with_suffix('.pipeline.log'),a.output.with_name(a.output.name+'-budget.json'),a.output.with_name(a.output.name+'-audit.json')):
        if path.exists():raise ValueError('Fresh owned pipeline namespace required')
    source=Path(__file__).resolve().parents[2]
    if not (source/'sircus/mjlab/corrected_imitation_verify.py').is_file():raise ValueError('Independent auditor must exist before launch')
    a.output.parent.mkdir(parents=True,exist_ok=True);started=time.monotonic();unix=time.time()
    deadline_unix=min(unix+a.global_limit,a.absolute_deadline_unix) if a.absolute_deadline_unix is not None else unix+a.global_limit
    if not 1860<deadline_unix-unix<=6300:raise ValueError('Finite original deadline and CPU audit reserve required')
    deadline=started+deadline_unix-unix
    state=dict(status='starting',source=str(source),output=str(a.output),pipeline_pid=os.getpid(),started_unix_s=unix,
        global_limit_s=deadline_unix-unix,global_deadline_unix_s=deadline_unix)
    statepath=a.output.with_suffix('.pipeline.json');atomic(statepath,state)
    command=[sys.executable,'-u','-m','sircus.mjlab.corrected_imitation_study','--global-deadline-unix',str(deadline_unix)]
    for key in (*INPUT_KEYS,'output','gpu-lock'):command+=['--'+key,str(getattr(a,key.replace('-','_')))]
    for key in ('wall-time-limit','queue-timeout'):command+=['--'+key,str(getattr(a,key.replace('-','_')))]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU_UUID,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
    with a.output.with_suffix('.pipeline.log').open('x') as log:
        code=supervise(command,source=source,env=env,log=log,state=state,statepath=statepath,output=a.output,absolute_deadline=deadline,worker_limit=a.wall_time_limit)
        state['study_exit_code']=code
        if code==0:
            remaining=deadline-time.monotonic()-5
            if remaining<=0:state.update(status='audit_failed',error='Original deadline left no audit time; unverified evidence retained')
            else:
                state.update(status='cpu_audit');atomic(statepath,state)
                command=[sys.executable,'-u','-m','sircus.mjlab.corrected_imitation_verify','--study',str(a.output),'--source',str(source),
                    '--output',str(a.output.with_name(a.output.name+'-audit.json'))]
                for key in INPUT_KEYS:command+=['--'+key,str(getattr(a,key.replace('-','_')))]
                try:
                    result=subprocess.run(command,cwd=source,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,timeout=min(1800,remaining))
                    state.update(audit_exit_code=result.returncode,status='completed' if result.returncode==0 else 'audit_failed')
                except subprocess.TimeoutExpired:state.update(status='audit_failed',error='Own CPU audit deadline; evidence retained')
        else:state.update(status='failed',error=state.get('error','Own worker exited; later phases skipped'))
    state.update(finished_unix_s=time.time(),elapsed_s=time.monotonic()-started);atomic(statepath,state);print(json.dumps(state),flush=True)
    if state['status']!='completed':raise RuntimeError('corrected-imitation pipeline incomplete: '+state['status'])

if __name__=='__main__':main()
