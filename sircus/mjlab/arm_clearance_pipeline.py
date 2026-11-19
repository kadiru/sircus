"""Finite isolated queue, fixed64 evaluation, CPU audit and additive display."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .arm_clearance_study import GPU_UUID,atomic


def publish(study,display,pipeline=None):
    from .prepare_arm_clearance_report import publish as publish_results
    if not (study/'report.json').is_file():return
    return publish_results(display,study,audit_path=study.with_name(study.name+'-audit.json'),pipeline=pipeline)


INPUT_KEYS=('parent_study','parent_audit','ensemble_study','ensemble_audit','consistent_study',
            'consistent_audit','gap_study','gap_audit','native_source','assets')


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('parent-study','parent-audit','ensemble-study','ensemble-audit','consistent-study','consistent-audit','gap-study','gap-audit','native-source','assets','output','gpu-lock','display'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--queue-timeout',type=float,default=86400);p.add_argument('--wall-time-limit',type=float,default=3600)
    return p


def worker_command(a,source):
    command=[sys.executable,'-u',str(source/'sircus/mjlab/arm_clearance_study.py')]
    for key in (*INPUT_KEYS,'output','gpu_lock','queue_timeout','wall_time_limit'):
        command+=['--'+key.replace('_','-'),str(getattr(a,key))]
    return command


def audit_command(a,source):
    command=[sys.executable,'-u','-m','sircus.mjlab.arm_clearance_verify','--study',str(a.output),'--source',str(source),
             '--output',str(a.output.with_name(a.output.name+'-audit.json'))]
    for key in INPUT_KEYS:command+=['--'+key.replace('_','-'),str(getattr(a,key))]
    return command


def main():
    a=parser().parse_args();source=Path(__file__).resolve().parents[2]
    if a.output.exists() or not a.display.is_file():raise ValueError('New study and existing viewer required')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU_UUID,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    command=worker_command(a,source)
    state=dict(status='starting',pipeline_pid=os.getpid(),started_unix_s=time.time(),source=str(source),output=str(a.output))
    statepath=a.output.with_name(a.output.name+'-pipeline.json');a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.with_suffix('.log').open('x') as log:
        child=subprocess.Popen(command,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        state.update(status='preparing_or_queued',child_pid=child.pid);atomic(statepath,state)
        deadline=time.monotonic()+a.queue_timeout+a.wall_time_limit+7200
        try:
            while child.poll() is None:
                if time.monotonic()>deadline:raise TimeoutError('Own bounded worker deadline')
                try:publish(a.output,a.display)
                except (OSError,ValueError,KeyError,RuntimeError) as e:state['display_error']=str(e)
                time.sleep(10)
        except BaseException:
            child.terminate()
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            state.update(status='failed',error='Own supervisor interrupted; evidence preserved and only own worker signalled')
            atomic(statepath,state);raise
        state['study_exit_code']=child.returncode
        if child.returncode==0:
            state.update(status='cpu_audit');atomic(statepath,state)
            cmd=audit_command(a,source)
            try:
                result=subprocess.run(cmd,cwd=source,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,timeout=3600)
                state.update(audit_exit_code=result.returncode,status='completed' if result.returncode==0 else 'audit_failed')
            except subprocess.TimeoutExpired:state.update(status='audit_failed',error='Own CPU audit timed out')
        else:state.update(status='failed',error='Own worker failed; evidence preserved')
    state['finished_unix_s']=time.time();atomic(statepath,state)
    try:publish(a.output,a.display,pipeline=state)
    except (OSError,ValueError,KeyError,RuntimeError) as e:state['display_error']=str(e);atomic(statepath,state)
    print(json.dumps(state),flush=True)
    if state['status']!='completed':raise RuntimeError('Bounded comparison incomplete: '+state['status'])

if __name__=='__main__':main()
