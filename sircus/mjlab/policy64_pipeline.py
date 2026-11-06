"""Bounded4090 worker, CPU-only independent audit, additive read-only publication."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .gpu_slot import GPU_UUID
from .sweep_worker import atomic

def publish(study,display,state):
    from .prepare_policy64_report import prepare
    if not (study/'report.json').exists(): return
    for _ in range(3):
        original=display.read_bytes()
        result=prepare(display,study,display.with_name('policy64-derived-display.json'),
                       audit_path=study.with_name(study.name+'-audit.json'))
        if original!=display.read_bytes(): continue
        if state.get('status') in ('failed','audit_failed'):
            panel=result.get('policy64_study',{})
            panel.update(status='failed',pipeline_status=state['status'])
        atomic(display,result);return
    raise RuntimeError('Concurrent display writer preserved')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('parent-study','parent-audit','reference','assets','output','gpu-lock','display'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--wall-time-limit',type=float,default=7200);p.add_argument('--queue-timeout',type=float,default=1800)
    a=p.parse_args();source=Path(__file__).resolve().parents[2]
    if a.output.exists() or not a.display.is_file():raise ValueError('New evidence namespace and existing display required')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU_UUID,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',
             OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
    command=[sys.executable,'-u','-m','sircus.mjlab.policy64_study']
    for key in ('parent_study','parent_audit','reference','assets','output','gpu_lock','wall_time_limit','queue_timeout'):
        command+=['--'+key.replace('_','-'),str(getattr(a,key))]
    state=dict(status='starting',source=str(source),output=str(a.output),pipeline_pid=os.getpid(),started_unix_s=time.time())
    statepath=a.output.with_suffix('.pipeline.json');a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.with_suffix('.log').open('x') as log:
        child=subprocess.Popen(command,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        state.update(status='running',child_pid=child.pid);atomic(statepath,state)
        queued_deadline=time.monotonic()+a.queue_timeout+a.wall_time_limit+120
        while child.poll() is None:
            bp=a.output.with_name(a.output.name+'-budget.json')
            expired=(bp.exists() and time.time()>=json.loads(bp.read_text())['acquired_unix_s']+a.wall_time_limit-5)
            if expired or time.monotonic()>queued_deadline:
                child.terminate()
                try:child.wait(timeout=3)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                state.update(status='failed',error='Own finite worker deadline reached; no foreign process signalled')
                break
            try:publish(a.output,a.display,state)
            except (OSError,ValueError,KeyError,RuntimeError) as error:state['display_error']=str(error)
            time.sleep(5)
        state['study_exit_code']=child.returncode
        if child.returncode==0:
            state['status']='cpu_audit';atomic(statepath,state)
            command=[sys.executable,'-u','-m','sircus.mjlab.policy64_verify']
            for key in ('parent_study','parent_audit','reference','assets'):
                command+=['--'+key.replace('_','-'),str(getattr(a,key))]
            command+=['--study',str(a.output),'--source',str(source),'--output',str(a.output.with_name(a.output.name+'-audit.json'))]
            try:
                result=subprocess.run(command,cwd=source,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,timeout=3600)
                state.update(audit_exit_code=result.returncode,status='completed' if result.returncode==0 else 'audit_failed')
            except subprocess.TimeoutExpired:state.update(status='audit_failed',error='Own CPU audit deadline reached')
        else:state.update(status='failed',error=state.get('error','Own native worker exited; evidence retained'))
    state['finished_unix_s']=time.time();atomic(statepath,state)
    try:publish(a.output,a.display,state)
    except (OSError,ValueError,KeyError,RuntimeError) as error:state['display_error']=str(error);atomic(statepath,state)
    print(json.dumps(state),flush=True)
    if state['status']!='completed':raise RuntimeError('Policy64 pipeline incomplete: '+state['status'])

if __name__=='__main__':main()
