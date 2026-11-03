"""Bounded native study, independent CPU audit and read-only live display."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .contact_study import atomic
from .gpu_slot import GPU_UUID


def publish(study,display):
    if not (study/'report.json').exists():return
    from .prepare_neighborhood_report import prepare
    derived=display.with_name('neighborhood-derived-display.json')
    for _ in range(3):
        original=display.read_bytes()
        value=prepare(display,study,derived)
        if display.read_bytes()!=original:continue
        atomic(display,value);return
    raise RuntimeError('Concurrent display writes; changed data preserved')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('comparison-study','confirmation-study','assets','output','gpu-lock','display'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--batch',type=int,default=64,choices=(32,64,128))
    parser.add_argument('--queue-timeout',type=float,default=7200)
    parser.add_argument('--wall-time-limit',type=float,default=14400)
    a=parser.parse_args();source=Path(__file__).resolve().parents[2]
    if a.output.exists() or not a.display.is_file():raise ValueError('New evidence and an existing viewer report required')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU_UUID,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    native=[sys.executable,'-u','-m','sircus.mjlab.neighborhood_study']
    for key in ('comparison_study','confirmation_study','assets','output','gpu_lock','batch','queue_timeout','wall_time_limit'):
        native+=['--'+key.replace('_','-'),str(getattr(a,key))]
    state=dict(status='starting',source=str(source),output=str(a.output),pipeline_pid=os.getpid(),started_unix_s=time.time())
    state_path=a.output.with_suffix('.pipeline.json');log_path=a.output.with_suffix('.log');a.output.parent.mkdir(parents=True,exist_ok=True)
    with log_path.open('x') as log:
        child=subprocess.Popen(native,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        state.update(status='running',child_pid=child.pid);atomic(state_path,state)
        deadline=time.monotonic()+a.queue_timeout+a.wall_time_limit+300
        while child.poll() is None:
            try:publish(a.output,a.display)
            except (OSError,ValueError,KeyError) as error:state['display_error']=str(error)
            if time.monotonic()>deadline:
                child.terminate()
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                state.update(status='failed',error='Own finite study deadline exceeded');atomic(state_path,state);raise TimeoutError(state['error'])
            time.sleep(10)
        state['study_exit_code']=child.returncode
        if child.returncode==0:
            state['status']='cpu_audit';atomic(state_path,state)
            command=[sys.executable,'-m','sircus.mjlab.neighborhood_verify','--study',str(a.output),'--source',str(source),
                '--assets',str(a.assets),'--comparison-study',str(a.comparison_study),'--confirmation-study',str(a.confirmation_study),
                '--output',str(a.output/'audit-neighborhood.json')]
            try:
                result=subprocess.run(command,cwd=source,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,timeout=1800)
                state.update(audit_exit_code=result.returncode,status='completed' if result.returncode==0 else 'audit_failed')
            except subprocess.TimeoutExpired:state.update(status='audit_failed',error='Own CPU audit deadline exceeded')
        else:state['status']='failed'
    publish(a.output,a.display);state['finished_unix_s']=time.time();atomic(state_path,state)
    print(json.dumps(state),flush=True)
    if state['status']!='completed':raise RuntimeError('Pipeline incomplete: '+state['status'])


if __name__=='__main__':main()
