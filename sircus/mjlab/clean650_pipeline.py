"""Bounded dedicated4090 worker plus independent CPU audit and additive display."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .sweep_worker import atomic
from .gpu_slot import GPU_UUID


def publish(study,display):
    from .prepare_clean650_report import prepare
    if not (study/'report.json').exists():return
    for _ in range(3):
        original=display.read_bytes();result=prepare(display,study,display.with_name('clean650-derived-display.json'))
        if original!=display.read_bytes():continue
        atomic(display,result);return
    raise RuntimeError('Concurrent existing display writes preserved')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    keys=('old-study','positive-study','positive-audit','calibration-report','verification-report','replay-study','assets','output','gpu-lock','display')
    for k in keys:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--batch',type=int,default=32,choices=(16,32));p.add_argument('--queue-timeout',type=float,default=14400);p.add_argument('--wall-time-limit',type=float,default=7200)
    a=p.parse_args();source=Path(__file__).resolve().parents[2]
    if a.output.exists() or not a.display.is_file():raise ValueError('New study namespace and existing display required')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU_UUID,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    command=[sys.executable,'-u','-m','sircus.mjlab.clean650_study']
    for k in (*[k.replace('-','_') for k in keys if k!='display'],'batch','queue_timeout','wall_time_limit'):command+=['--'+k.replace('_','-'),str(getattr(a,k))]
    state=dict(status='starting',source=str(source),output=str(a.output),pipeline_pid=os.getpid(),started_unix_s=time.time());statepath=a.output.with_suffix('.pipeline.json')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.with_suffix('.log').open('x') as log:
        child=subprocess.Popen(command,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        state.update(status='running',child_pid=child.pid);atomic(statepath,state);deadline=time.monotonic()+a.queue_timeout+a.wall_time_limit+300
        while child.poll() is None:
            try:publish(a.output,a.display)
            except (OSError,ValueError,KeyError,RuntimeError) as error:state['display_error']=str(error)
            if time.monotonic()>deadline:
                child.terminate()
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                state.update(status='failed',error='Own finite worker deadline exceeded');atomic(statepath,state);raise TimeoutError(state['error'])
            time.sleep(10)
        state['study_exit_code']=child.returncode
        if child.returncode==0:
            state['status']='cpu_audit';atomic(statepath,state)
            command=[sys.executable,'-u','-m','sircus.mjlab.clean650_verify','--study',str(a.output),'--source',str(source),'--assets',str(a.assets),
                '--shallow-study',str(a.old_study),'--positive-study',str(a.positive_study),'--positive-audit',str(a.positive_audit),'--replay-study',str(a.replay_study),
                '--output',str(a.output/'audit-clean650.json')]
            try:
                result=subprocess.run(command,cwd=source,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,timeout=1800)
                state.update(audit_exit_code=result.returncode,status='completed' if result.returncode==0 else 'audit_failed')
            except subprocess.TimeoutExpired:state.update(status='audit_failed',error='Own CPU audit deadline exceeded')
        else:state.update(status='failed',error='Own finite worker exited; original evidence/log retained')
    state['finished_unix_s']=time.time();atomic(statepath,state)
    try:publish(a.output,a.display)
    except (OSError,ValueError,KeyError,RuntimeError) as error:
        state['display_error']=str(error);atomic(statepath,state)
    print(json.dumps(state),flush=True)
    if state['status']!='completed':raise RuntimeError('Study incomplete: '+state['status'])


if __name__=='__main__':main()
