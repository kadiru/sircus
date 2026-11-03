"""Finite study -> CPU audit -> existing viewer updates, without server restart.

Only the spawned study may use4090. Existing simulator/Isaac/4070 jobs are never
signalled. The display file is a derived viewer document, not original evidence.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .contact_study import atomic
from .gpu_slot import GPU_UUID
from .prepare_window_report import compact, KEY


def publish(study,display):
    path=study/'report.json'
    if not path.is_file():return
    raw=json.loads(path.read_text());value=compact(raw)
    audit=study/'audit-window.json'
    if audit.is_file():
        result=json.loads(audit.read_text())
        value['audit']={k:result.get(k) for k in ('status','contract_fingerprint','checks_count','limitations')}
        if value['audit']['contract_fingerprint']!=raw['contract_fingerprint']:
            value['audit'].update(status='failed',reason='Audit contract mismatch')
    # Preserve every other study and active-solver setting in the live document.
    # If another writer changed it during preparation, reread before replacing.
    for _ in range(3):
        original=display.read_bytes();base=json.loads(original);base[KEY]=value
        if display.read_bytes()!=original:continue
        atomic(display,base);return
    raise RuntimeError('Concurrent viewer writes; refusing to replace changed display')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('assets','scene','anchor','output','gpu-lock','display'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--batch',type=int,default=64,choices=(32,64,128))
    parser.add_argument('--queue-timeout',type=float,default=7200)
    parser.add_argument('--wall-time-limit',type=float,default=14400)
    a=parser.parse_args();source=Path(__file__).resolve().parents[2]
    if a.output.exists():raise ValueError('A new evidence directory is required')
    if not a.display.is_file():raise ValueError('Existing derived viewer report required')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU_UUID,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    command=[sys.executable,'-u','-m','sircus.mjlab.window_study']
    for key in ('assets','scene','anchor','output','gpu_lock','batch','queue_timeout','wall_time_limit'):
        command += ['--'+key.replace('_','-'),str(getattr(a,key))]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    state_path=a.output.with_suffix('.pipeline.json');log_path=a.output.with_suffix('.log')
    state=dict(status='starting',source=str(source),output=str(a.output),started_unix_s=time.time(),pipeline_pid=os.getpid())
    with log_path.open('x') as log:
        process=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
        state.update(status='running',child_pid=process.pid);atomic(state_path,state)
        deadline=time.monotonic()+a.queue_timeout+a.wall_time_limit+300
        while process.poll() is None:
            try:publish(a.output,a.display)
            except (OSError,ValueError,KeyError) as error:state['display_error']=str(error)
            if time.monotonic()>deadline:
                process.terminate()
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                state.update(status='failed',error='Own finite study exceeded queue+execution deadline');atomic(state_path,state)
                raise TimeoutError(state['error'])
            time.sleep(10)
        state['study_exit_code']=process.returncode
        if process.returncode==0:
            state['status']='cpu_audit';atomic(state_path,state)
            cpu_env=dict(env,CUDA_VISIBLE_DEVICES='')
            checked=subprocess.run([sys.executable,'-m','sircus.mjlab.window_verify','--study',str(a.output),
                '--source',str(source),'--assets',str(a.assets),'--output',str(a.output/'audit-window.json')],
                cwd=source,env=cpu_env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
            state['audit_exit_code']=checked.returncode
            state['status']='completed' if checked.returncode==0 else 'audit_failed'
        else:state['status']='failed'
    publish(a.output,a.display);state['finished_unix_s']=time.time();atomic(state_path,state)
    print(json.dumps(state),flush=True)
    if state['status']!='completed':raise RuntimeError('Study pipeline did not complete: '+state['status'])


if __name__=='__main__':main()
