"""Bounded replay worker, independent CPU audit and additive live publication."""
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


def publish(study,display,queued):
    from .prepare_replay650_report import prepare,compact
    state=json.loads(study.with_suffix('.pipeline.json').read_text())
    for _ in range(3):
        original=display.read_bytes()
        if (study/'report.json').exists():
            value=prepare(display,study,display.with_name('replay650-derived-display.json'))
        else:
            raw=copy.deepcopy(queued)
            raw.update(status='failed' if state['status'] in ('failed','audit_failed') else 'waiting_for_4090',
                phase='device_queue',error=state.get('error'),queue=dict(wait_s=max(0,time.time()-state['started_unix_s']),
                    reason='Shared4090 allocation preserved; no calibrated650 physics has started'))
            value=json.loads(original);value['calibrated650_replay']=compact(raw)
        if display.read_bytes()!=original:continue
        atomic(display,value);return
    raise RuntimeError('Concurrent display writes preserved')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    names=('old-study','positive-study','positive-audit','calibration-report','verification-report','assets','output','gpu-lock','display')
    for name in names:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--batch',type=int,default=16,choices=(16,32))
    parser.add_argument('--queue-timeout',type=float,default=14400);parser.add_argument('--wall-time-limit',type=float,default=7200)
    a=parser.parse_args()
    if a.output.exists() or not a.display.is_file():raise ValueError('New evidence namespace and existing display required')
    from . import replay650_contract as p
    scene,sources,_,_=p.load_inputs(a.old_study,a.positive_study,a.positive_audit,a.calibration_report,a.verification_report)
    queued=dict(version=p.VERSION,scope=p.SCOPE,settings=p.settings(),groups=[p.metadata(g,scene) for g in sources],counts={},
        gate=dict(status='pending',window_opened=False),trace_examples=[])
    source=Path(__file__).resolve().parents[2]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU_UUID,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    command=[sys.executable,'-u','-m','sircus.mjlab.replay650_study']
    for key in ('old_study','positive_study','positive_audit','calibration_report','verification_report','assets','output','gpu_lock','batch','queue_timeout','wall_time_limit'):
        command+=['--'+key.replace('_','-'),str(getattr(a,key))]
    state=dict(status='starting',source=str(source),output=str(a.output),pipeline_pid=os.getpid(),started_unix_s=time.time())
    state_path=a.output.with_suffix('.pipeline.json');a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.with_suffix('.log').open('x') as log:
        child=subprocess.Popen(command,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        state.update(status='running',child_pid=child.pid);atomic(state_path,state)
        deadline=time.monotonic()+a.queue_timeout+a.wall_time_limit+300
        while child.poll() is None:
            try:publish(a.output,a.display,queued)
            except (OSError,ValueError,KeyError) as error:state['display_error']=str(error)
            if time.monotonic()>deadline:
                child.terminate()
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                state.update(status='failed',error='Own finite replay deadline exceeded');atomic(state_path,state);raise TimeoutError(state['error'])
            time.sleep(10)
        state['study_exit_code']=child.returncode
        if child.returncode==0:
            state['status']='cpu_audit';atomic(state_path,state)
            command=[sys.executable,'-m','sircus.mjlab.replay650_verify','--study',str(a.output),'--source',str(source),'--assets',str(a.assets),
                '--shallow-study',str(a.old_study),'--positive-study',str(a.positive_study),'--positive-audit',str(a.positive_audit),
                '--output',str(a.output/'audit-replay650.json')]
            try:
                result=subprocess.run(command,cwd=source,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,timeout=1800)
                state.update(audit_exit_code=result.returncode,status='completed' if result.returncode==0 else 'audit_failed')
            except subprocess.TimeoutExpired:state.update(status='audit_failed',error='Own replay CPU audit deadline exceeded')
        else:state.update(status='failed',error='Own finite replay exited; evidence/log retained')
    state['finished_unix_s']=time.time();atomic(state_path,state);publish(a.output,a.display,queued)
    print(json.dumps(state),flush=True)
    if state['status']!='completed':raise RuntimeError('Replay pipeline incomplete: '+state['status'])


if __name__=='__main__':main()
