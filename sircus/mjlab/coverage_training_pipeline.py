"""Seven-hour absolute supervisor: own worker, CPU audit; no service restart."""
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

INPUT_KEYS=('current-study','current-audit','case00-study','case00-audit','history-root','reference','assets')


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
    parser.add_argument('--wall-time-limit',type=float,default=19800)
    parser.add_argument('--training-cut',type=float,default=16200)
    parser.add_argument('--search-limit',type=float,default=7200)
    parser.add_argument('--queue-timeout',type=float,default=1800)
    parser.add_argument('--global-limit',type=float,default=25200);a=parser.parse_args()
    if not (60<=a.search_limit<=7200 and a.search_limit<=a.training_cut<=16200
            and a.training_cut<a.wall_time_limit<=19800 and a.wall_time_limit<a.global_limit<=25200):
        raise ValueError('Finite ordered search/training/worker/global caps required')
    if not 0<a.queue_timeout<=1800 or a.output.exists():raise ValueError('Fresh output and finite queue required')
    for suffix in ('.pipeline.json','.pipeline.log'):
        if a.output.with_suffix(suffix).exists():raise ValueError('Fresh pipeline namespace required')
    if a.output.with_name(a.output.name+'-budget.json').exists():raise ValueError('Fresh worker budget namespace required')
    source=Path(__file__).resolve().parents[2];a.output.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();deadline=started+a.global_limit
    state=dict(status='starting',source=str(source),output=str(a.output),pipeline_pid=os.getpid(),
        started_unix_s=time.time(),global_limit_s=a.global_limit,global_deadline_unix_s=time.time()+a.global_limit)
    statepath=a.output.with_suffix('.pipeline.json');atomic(statepath,state)
    command=[sys.executable,'-u','-m','sircus.mjlab.coverage_training_study',
             '--global-deadline-unix',str(state['global_deadline_unix_s'])]
    for key in (*INPUT_KEYS,'output','gpu-lock'):command+=['--'+key,str(getattr(a,key.replace('-','_')))]
    for key in ('wall-time-limit','training-cut','search-limit','queue-timeout'):
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
                audit=[sys.executable,'-u','-m','sircus.mjlab.coverage_training_verify','--study',str(a.output),
                    '--source',str(source),'--output',str(a.output.with_name(a.output.name+'-audit.json'))]
                for key in INPUT_KEYS:audit+=['--'+key,str(getattr(a,key.replace('-','_')))]
                try:
                    result=subprocess.run(audit,cwd=source,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,timeout=min(3600,remaining))
                    state.update(audit_exit_code=result.returncode,status='completed' if result.returncode==0 else 'audit_failed')
                except subprocess.TimeoutExpired:state.update(status='audit_failed',error='Own CPU audit/global deadline; evidence retained')
        else:state.update(status='failed',error=state.get('error','Own worker exited; later phases skipped'))
        remaining=deadline-time.monotonic()
        if state.get('status')=='completed' and remaining>=120:
            video=a.output.with_name(a.output.name+'-video')/'replay.mp4'
            command=[sys.executable,'-u','-m','sircus.mjlab.coverage_training_video',
                '--study',str(a.output),'--audit',str(a.output.with_name(a.output.name+'-audit.json')),
                '--assets',str(a.assets),'--output',str(video)]
            state.update(status='cpu_video',video_status='rendering',video_path=str(video));atomic(statepath,state)
            try:
                result=subprocess.run(command,cwd=source,env=dict(env,CUDA_VISIBLE_DEVICES='',
                    MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',LIBGL_ALWAYS_SOFTWARE='1',MESA_LOADER_DRIVER_OVERRIDE='llvmpipe'),
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=min(900,remaining-10))
                state.update(video_status='completed' if result.returncode==0 else 'failed',video_exit_code=result.returncode)
                if result.returncode:state['video_error']='CPU renderer exited; audited numerical results remain valid'
            except subprocess.TimeoutExpired:
                state.update(video_status='failed',video_error='Own CPU renderer deadline; audited numerical results remain valid')
            state['status']='completed'
        elif state.get('status')=='completed':state['video_status']='deferred_global_deadline'
    state.update(finished_unix_s=time.time(),elapsed_s=time.monotonic()-started);atomic(statepath,state);print(json.dumps(state),flush=True)
    if state['status']!='completed':raise RuntimeError('Coverage pipeline incomplete: '+state['status'])

if __name__=='__main__':main()
