"""One bounded4090 development replay with CPU preparation/audit and additive UI."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from .clearance_correction_study import atomic,sha,GPU_UUID
from .prepare_clearance_diagnosis_report import publish

INPUT_KEYS=('parent_study','parent_audit','native_source','assets','output')


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for key in (*INPUT_KEYS,'diagnosis_report','diagnosis_audit','gpu_lock','display'):
        p.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    p.add_argument('--queue-timeout',type=float,default=86400)
    p.add_argument('--wall-time-limit',type=float,default=3600)
    return p


def input_args(a):
    return [v for key in INPUT_KEYS for v in ('--'+key.replace('_','-'),str(getattr(a,key)))]


def prepare_command(a,source):
    return [sys.executable,'-u','-m','sircus.mjlab.clearance_correction_study',*input_args(a),'--prepare-only']


def worker_command(a,source,contract_sha):
    return [sys.executable,'-u',str(source/'sircus/mjlab/clearance_correction_study.py'),*input_args(a),
        '--contract-sha256',contract_sha,'--gpu-lock',str(a.gpu_lock),'--queue-timeout',str(a.queue_timeout),
        '--wall-time-limit',str(a.wall_time_limit)]


def audit_command(a,source):
    return [sys.executable,'-u','-m','sircus.mjlab.clearance_correction_verify','--study',str(a.output),
        '--parent',str(a.parent_study),'--parent-audit',str(a.parent_audit),'--source',str(source),
        '--native-source',str(a.native_source),'--assets',str(a.assets),
        '--output',str(a.output.with_name(a.output.name+'-audit.json'))]


def environment(cuda):
    return dict(os.environ,CUDA_VISIBLE_DEVICES=cuda,PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')


def main():
    a=parser().parse_args();source=Path(__file__).resolve().parents[2]
    if a.output.exists() or not a.display.is_file():raise ValueError('New output and existing viewer required')
    if not a.diagnosis_report.is_file() or not a.diagnosis_audit.is_file():raise ValueError('Completed CPU diagnosis and proof required')
    if not 60<=a.wall_time_limit<=3600 or not 0<=a.queue_timeout<=86400:raise ValueError('Bounded test only')
    state=dict(status='preparing',pipeline_pid=os.getpid(),started_unix_s=time.time(),source=str(source),output=str(a.output))
    statepath=a.output.with_name(a.output.name+'-pipeline.json');a.output.parent.mkdir(parents=True,exist_ok=True)
    atomic(statepath,state);auditpath=a.output.with_name(a.output.name+'-audit.json')
    def update():
        try:
            publish(a.display,a.diagnosis_report,a.diagnosis_audit,a.output,auditpath,pipeline=state)
            state.pop('display_error',None)
        except (OSError,ValueError,KeyError,RuntimeError) as error:state['display_error']=str(error)
        atomic(statepath,state)
    update()
    with a.output.with_suffix('.log').open('x') as log:
        try:
            subprocess.run(prepare_command(a,source),cwd=source,env=environment(''),stdout=log,
                stderr=subprocess.STDOUT,check=True,timeout=900)
        except Exception as error:
            state.update(status='failed',error='CPU preparation: '+str(error));update();raise
        contract_sha=sha(a.output/'contract.json');state['contract_sha256']=contract_sha
        child=subprocess.Popen(worker_command(a,source,contract_sha),cwd=source,env=environment(GPU_UUID),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        state.update(status='queued_or_running',child_pid=child.pid);update()
        deadline=time.monotonic()+a.queue_timeout+a.wall_time_limit+1200
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
            try:
                result=subprocess.run(audit_command(a,source),cwd=source,env=environment(''),stdout=log,
                    stderr=subprocess.STDOUT,timeout=3600)
                state.update(status='completed' if result.returncode==0 else 'audit_failed',audit_exit_code=result.returncode)
            except subprocess.TimeoutExpired:state.update(status='audit_failed',error='Own CPU audit timeout')
        else:state.update(status='failed',error='Own worker failed; evidence preserved')
    state['finished_unix_s']=time.time();update()
    if state['status']!='completed' or state.get('display_error'):raise RuntimeError(str(state))

if __name__=='__main__':main()
