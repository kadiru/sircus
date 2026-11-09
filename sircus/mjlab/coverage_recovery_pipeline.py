"""Three-hour absolute supervisor: own worker, CPU audit; no service restart."""
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

INPUT_KEYS=('parent-study','parent-audit','history-root','reference','assets')


def stop_owned(child):
    """Only the subprocess created by this pipeline may be signalled."""
    child.terminate()
    try:child.wait(timeout=5)
    except subprocess.TimeoutExpired:child.kill();child.wait()


def supervise(command,*,source,env,log,state,statepath,output,absolute_deadline,worker_limit):
    child=subprocess.Popen(command,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    state.update(status='running',child_pid=child.pid);atomic(statepath,state)
    while child.poll() is None:
        budget=output.with_name(output.name+'-budget.json');expired=False
        if budget.exists():
            record=json.loads(budget.read_text());expired=time.time()>=record['acquired_unix_s']+min(worker_limit,record.get('effective_worker_limit_s',worker_limit))
        if expired or time.monotonic()>=absolute_deadline:
            stop_owned(child);state.update(status='failed',error='Own worker/global deadline reached; partial evidence retained');break
        time.sleep(2)
    return child.returncode


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in (*INPUT_KEYS,'output','gpu-lock'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--wall-time-limit',type=float,default=7200)
    parser.add_argument('--repeat-cut',type=float,default=4500)
    parser.add_argument('--search-limit',type=float,default=3600)
    parser.add_argument('--queue-timeout',type=float,default=900)
    parser.add_argument('--global-limit',type=float,default=10800)
    parser.add_argument('--absolute-deadline-unix',type=float,default=None);a=parser.parse_args()
    if not (60<=a.search_limit<=3600 and a.search_limit<a.repeat_cut<=4500
            and a.repeat_cut<a.wall_time_limit<=7200 and a.wall_time_limit<a.global_limit<=10800):
        raise ValueError('Finite ordered search/training/worker/global caps required')
    if not 0<a.queue_timeout<=900 or a.output.exists():raise ValueError('Fresh output and finite queue required')
    for suffix in ('.pipeline.json','.pipeline.log'):
        if a.output.with_suffix(suffix).exists():raise ValueError('Fresh pipeline namespace required')
    if a.output.with_name(a.output.name+'-budget.json').exists():raise ValueError('Fresh worker budget namespace required')
    source=Path(__file__).resolve().parents[2]
    if not (source/'sircus/mjlab/coverage_recovery_verify.py').is_file():raise ValueError('Independent recovery auditor required before launch')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();started_unix=time.time()
    deadline_unix=min(started_unix+a.global_limit,a.absolute_deadline_unix) if a.absolute_deadline_unix is not None else started_unix+a.global_limit
    if not started_unix<deadline_unix<=started_unix+10800:raise ValueError('Future bounded original absolute deadline required')
    actual_window=deadline_unix-started_unix;deadline=started+actual_window
    state=dict(status='starting',source=str(source),output=str(a.output),pipeline_pid=os.getpid(),
        started_unix_s=started_unix,global_limit_s=actual_window,global_deadline_unix_s=deadline_unix)
    statepath=a.output.with_suffix('.pipeline.json');atomic(statepath,state)
    command=[sys.executable,'-u','-m','sircus.mjlab.coverage_recovery_study',
             '--global-deadline-unix',str(state['global_deadline_unix_s'])]
    for key in (*INPUT_KEYS,'output','gpu-lock'):command+=['--'+key,str(getattr(a,key.replace('-','_')))]
    for key in ('wall-time-limit','repeat-cut','search-limit','queue-timeout'):
        command+=['--'+key,str(getattr(a,key.replace('-','_')))]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU_UUID,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
    with a.output.with_suffix('.pipeline.log').open('x') as log:
        code=supervise(command,source=source,env=env,log=log,state=state,statepath=statepath,output=a.output,
            absolute_deadline=deadline,worker_limit=a.wall_time_limit)
        state['study_exit_code']=code
        if code==0:
            remaining=deadline-time.monotonic()-5
            if remaining<=0:state.update(status='audit_deferred',error='Global deadline left no audit time; results unverified')
            else:
                state.update(status='cpu_audit');atomic(statepath,state)
                audit=[sys.executable,'-u','-m','sircus.mjlab.coverage_recovery_verify','--study',str(a.output),
                    '--source',str(source),'--output',str(a.output.with_name(a.output.name+'-audit.json'))]
                for key in INPUT_KEYS:audit+=['--'+key,str(getattr(a,key.replace('-','_')))]
                try:
                    result=subprocess.run(audit,cwd=source,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,timeout=min(1800,remaining))
                    state.update(audit_exit_code=result.returncode,status='completed' if result.returncode==0 else 'audit_failed')
                except subprocess.TimeoutExpired:state.update(status='audit_failed',error='Own CPU audit/global deadline; evidence retained')
        else:state.update(status='failed',error=state.get('error','Own worker exited; later phases skipped'))
        if state.get('status')=='completed':state['video_status']='not_requested_saved_traces_only'
    state.update(finished_unix_s=time.time(),elapsed_s=time.monotonic()-started);atomic(statepath,state);print(json.dumps(state),flush=True)
    if state['status']!='completed':raise RuntimeError('Coverage recovery pipeline incomplete: '+state['status'])

if __name__=='__main__':main()
