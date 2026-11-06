"""Finite queued physics study, independent CPU audit and additive report publication."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .gpu_slot import GPU_UUID
from .sweep_worker import atomic

def publish(study,display):
    from .prepare_physics_ensemble_report import prepare
    if not (study/'report.json').exists():return
    for _ in range(3):
        before=display.read_bytes()
        derived=display.with_name('physics-ensemble-derived-display.json')
        result=prepare(display,study,derived,audit_path=study.with_name(study.name+'-audit.json'))
        if before==display.read_bytes():atomic(display,result);return
    raise RuntimeError('Concurrent display writer preserved')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('parent-study','parent-audit','native-source','assets','output','gpu-lock','display'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--queue-timeout',type=float,default=86400)
    p.add_argument('--wall-time-limit',type=float,default=7200)
    a=p.parse_args();source=Path(__file__).resolve().parents[2]
    if a.output.exists() or not a.display.is_file():raise ValueError('New study namespace and existing display required')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU_UUID,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',
             OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
    command=[sys.executable,'-u',str(source/'sircus/mjlab/physics_ensemble_study.py')]
    for key in ('parent_study','parent_audit','native_source','assets','output','gpu_lock','queue_timeout','wall_time_limit'):
        command+=['--'+key.replace('_','-'),str(getattr(a,key))]
    state=dict(status='starting',pipeline_pid=os.getpid(),started_unix_s=time.time(),source=str(source),output=str(a.output))
    statepath=a.output.with_name(a.output.name+'-pipeline.json');a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.with_suffix('.log').open('x') as log:
        child=subprocess.Popen(command,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        state.update(status='preparing_or_queued',child_pid=child.pid);atomic(statepath,state)
        deadline=time.monotonic()+a.queue_timeout+a.wall_time_limit+7200
        while child.poll() is None:
            if time.monotonic()>deadline:
                child.terminate()
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                state.update(status='failed',error='Own worker total deadline reached; no foreign process signalled')
                break
            try:publish(a.output,a.display)
            except (OSError,ValueError,KeyError,RuntimeError) as e:state['display_error']=str(e)
            time.sleep(5)
        state['study_exit_code']=child.returncode
        if child.returncode==0:
            state['status']='cpu_audit';atomic(statepath,state)
            command=[sys.executable,'-u','-m','sircus.mjlab.physics_ensemble_verify',
                     '--study',str(a.output),'--parent-study',str(a.parent_study),'--parent-audit',str(a.parent_audit),
                     '--native-source',str(a.native_source),'--source',str(source),'--assets',str(a.assets),
                     '--output',str(a.output.with_name(a.output.name+'-audit.json'))]
            try:
                result=subprocess.run(command,cwd=source,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,timeout=3600)
                state.update(audit_exit_code=result.returncode,status='completed' if result.returncode==0 else 'audit_failed')
            except subprocess.TimeoutExpired:state.update(status='audit_failed',error='Own CPU audit deadline reached')
        else:state.update(status='failed',error=state.get('error','Own study failed; evidence preserved'))
    state['finished_unix_s']=time.time();atomic(statepath,state)
    try:publish(a.output,a.display)
    except (OSError,ValueError,KeyError,RuntimeError) as e:state['display_error']=str(e);atomic(statepath,state)
    print(json.dumps(state),flush=True)
    if state['status']!='completed':raise RuntimeError('Physics ensemble pipeline incomplete: '+state['status'])

if __name__=='__main__':main()
