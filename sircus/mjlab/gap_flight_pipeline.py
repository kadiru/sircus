"""CPU diagnosis publication, one finite queued4090 test, audit, additive UI."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from .gap_flight_correction import atomic,sha,GPU_UUID
from .prepare_gap_flight_report import publish


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('parent-study','parent-audit','native-source','assets','diagnosis','output','gpu-lock','display'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--queue-timeout',type=float,default=86400);p.add_argument('--wall-time-limit',type=float,default=3600)
    a=p.parse_args();source=Path(__file__).resolve().parents[2]
    if a.output.exists() or not a.display.is_file():raise ValueError('New output and existing viewer required')
    if not 60<=a.wall_time_limit<=3600 or not 0<=a.queue_timeout<=86400:raise ValueError('Bounded test only')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    args=[]
    for key in ('parent_study','parent_audit','native_source','assets','output'):
        args+=['--'+key.replace('_','-'),str(getattr(a,key))]
    state=dict(status='preparing',pipeline_pid=os.getpid(),started_unix_s=time.time(),source=str(source),output=str(a.output))
    statepath=a.output.with_name(a.output.name+'-pipeline.json');a.output.parent.mkdir(parents=True,exist_ok=True);atomic(statepath,state)
    auditpath=a.output.with_name(a.output.name+'-audit.json')
    def update():
        try:publish(a.display,a.diagnosis,a.output,auditpath,pipeline=state);state.pop('display_error',None)
        except (OSError,ValueError,KeyError,RuntimeError) as e:state['display_error']=str(e)
        atomic(statepath,state)
    with a.output.with_suffix('.log').open('x') as log:
        prep=[sys.executable,'-u','-m','sircus.mjlab.gap_flight_correction',*args,'--prepare-only','--analysis',str(a.diagnosis/'diagnosis.json')]
        try:subprocess.run(prep,cwd=source,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
        except Exception as e:
            state.update(status='failed',error='CPU preparation: '+str(e));update();raise
        contract_sha=sha(a.output/'contract.json');state['contract_sha256']=contract_sha
        command=[sys.executable,'-u',str(source/'sircus/mjlab/gap_flight_correction.py'),*args,
                 '--contract-sha256',contract_sha,'--gpu-lock',str(a.gpu_lock),'--queue-timeout',str(a.queue_timeout),'--wall-time-limit',str(a.wall_time_limit)]
        child=subprocess.Popen(command,cwd=source,env=dict(env,CUDA_VISIBLE_DEVICES=GPU_UUID),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        state.update(status='queued_or_running',child_pid=child.pid);update();deadline=time.monotonic()+a.queue_timeout+a.wall_time_limit+1200
        try:
            while child.poll() is None:
                if time.monotonic()>deadline:raise TimeoutError('Own supervisor deadline')
                update();time.sleep(10)
        except BaseException:
            child.terminate()
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            state.update(status='failed',error='Own worker interrupted; no foreign process signalled');update();raise
        state['study_exit_code']=child.returncode
        if child.returncode==0:
            state.update(status='cpu_audit');update()
            command=[sys.executable,'-u','-m','sircus.mjlab.gap_flight_correction_verify','--study',str(a.output),
                '--source',str(source),'--parent',str(a.parent_study),'--parent-audit',str(a.parent_audit),'--native-source',str(a.native_source),'--assets',str(a.assets),'--output',str(auditpath)]
            try:
                result=subprocess.run(command,cwd=source,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,timeout=3600)
                state.update(status='completed' if result.returncode==0 else 'audit_failed',audit_exit_code=result.returncode)
            except subprocess.TimeoutExpired:state.update(status='audit_failed',error='Own CPU audit timeout')
        else:state.update(status='failed',error='Own worker failed; evidence preserved')
    state['finished_unix_s']=time.time();update()
    if state['status']!='completed' or state.get('display_error'):raise RuntimeError(str(state))

if __name__=='__main__':main()
