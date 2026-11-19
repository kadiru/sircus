"""Bounded native lower-board study, independent CPU audit and read-only publishing."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .contact_study import atomic
from .gpu_slot import GPU_UUID


def publish(study,display,queued):
    from .prepare_lowboard_report import prepare,compact
    state=json.loads(study.with_suffix('.pipeline.json').read_text())
    for _ in range(3):
        original=display.read_bytes()
        if (study/'report.json').exists():
            value=prepare(display,study,display.with_name('lowboard-derived-display.json'))
        else:
            raw=copy.deepcopy(queued);failed=state.get('status') in ('failed','audit_failed')
            raw.update(status='failed' if failed else 'waiting_for_4090',phase='device_queue',error=state.get('error'),
                queue=dict(wait_s=max(0,time.time()-state['started_unix_s']),reason='Shared4090 allocation preserved; no lower-board physics has started'))
            value=json.loads(original);value['lowboard_evaluation']=compact(raw)
        if display.read_bytes()!=original:continue
        atomic(display,value);return
    raise RuntimeError('Concurrent display writes preserved')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('parent-study','assets','output','gpu-lock','display'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--batch',type=int,default=64,choices=(32,64,128));parser.add_argument('--queue-timeout',type=float,default=14400)
    parser.add_argument('--wall-time-limit',type=float,default=14400);a=parser.parse_args()
    if a.output.exists() or not a.display.is_file():raise ValueError('New output and an existing viewer required')
    from .lowboard_inputs import prepare_inputs
    from . import lowboard_contract as protocol
    scene,seeds,_,_,_=prepare_inputs(a.parent_study);settings=protocol.settings(scene,seeds)
    queued=dict(version=protocol.VERSION,scope=protocol.SCOPE,settings=settings,targets=settings['targets'],anchors=seeds,
        max_requests=protocol.MAX_REQUESTS,contexts=[dict(**c,board_admitted=None,reason='waiting_for_native_study',geometry_requests=0,eligible=0,
            search_attempts=0,search_native_releases=0,search_captures=0,clean=0,best_gap_m=None) for c in protocol.contexts(seeds,settings['targets'])],
        selected_recipes=[],trace_examples=[],counts=dict(requests=0,native_releases=0,captures=0,geometry_rejections=0))
    source=Path(__file__).resolve().parents[2];env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU_UUID,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    command=[sys.executable,'-u','-m','sircus.mjlab.lowboard_study']
    for key in ('parent_study','assets','output','gpu_lock','batch','queue_timeout','wall_time_limit'):
        command+=['--'+key.replace('_','-'),str(getattr(a,key))]
    state=dict(status='starting',source=str(source),output=str(a.output),pipeline_pid=os.getpid(),started_unix_s=time.time())
    state_path=a.output.with_suffix('.pipeline.json');a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.with_suffix('.log').open('x') as log:
        child=subprocess.Popen(command,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        state.update(status='running',child_pid=child.pid);atomic(state_path,state);deadline=time.monotonic()+a.queue_timeout+a.wall_time_limit+300
        while child.poll() is None:
            try:publish(a.output,a.display,queued)
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
            audit=[sys.executable,'-m','sircus.mjlab.lowboard_verify','--study',str(a.output),'--source',str(source),'--assets',str(a.assets),
                '--parent-study',str(a.parent_study),'--output',str(a.output/'audit-lowboard.json')]
            try:
                result=subprocess.run(audit,cwd=source,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,timeout=1800)
                state.update(audit_exit_code=result.returncode,status='completed' if result.returncode==0 else 'audit_failed')
            except subprocess.TimeoutExpired:state.update(status='audit_failed',error='Own CPU audit deadline exceeded')
        else:state.update(status='failed',error='Own lower-board child exited; original log/evidence retained')
    state['finished_unix_s']=time.time();atomic(state_path,state);publish(a.output,a.display,queued);print(json.dumps(state),flush=True)
    if state['status']!='completed':raise RuntimeError('Pipeline incomplete: '+state['status'])


if __name__=='__main__':main()
