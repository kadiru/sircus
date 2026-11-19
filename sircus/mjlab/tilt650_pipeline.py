"""One bounded 4090 verification worker, independent CPU audit and additive UI."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .gpu_slot import GPU_UUID
from .sweep_worker import atomic


def publish(study, display):
    from .prepare_tilt650_report import prepare
    if not (study/'report.json').is_file():
        return
    for _ in range(3):
        original = display.read_bytes()
        value = prepare(display, study, display.with_name('tilt650-derived-display.json'),
                        audit_path=study.with_name(study.name+'-audit.json'))
        if original != display.read_bytes():
            continue
        atomic(display, value)
        return
    raise RuntimeError('Concurrent display writer preserved')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    keys = ('parent-study','parent-audit','assets','output','gpu-lock','display')
    for key in keys:
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--batch', type=int, default=32, choices=(16,32))
    p.add_argument('--queue-timeout', type=float, default=1800)
    p.add_argument('--wall-time-limit', type=float, default=1800)
    a = p.parse_args()
    source = Path(__file__).resolve().parents[2]
    if a.output.exists() or not a.display.is_file():
        raise ValueError('New evidence namespace and existing display required')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU_UUID, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    command = [sys.executable,'-u','-m','sircus.mjlab.tilt650_study']
    for key in (*[k.replace('-','_') for k in keys if k != 'display'], 'batch','queue_timeout','wall_time_limit'):
        command += ['--'+key.replace('_','-'), str(getattr(a,key))]
    state = dict(status='starting', source=str(source), output=str(a.output), pipeline_pid=os.getpid(), started_unix_s=time.time())
    statepath = a.output.with_suffix('.pipeline.json')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.with_suffix('.log').open('x') as log:
        child = subprocess.Popen(command, cwd=source, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        state.update(status='running', child_pid=child.pid)
        atomic(statepath, state)
        deadline = time.monotonic()+a.queue_timeout+a.wall_time_limit+300
        while child.poll() is None:
            try:
                publish(a.output, a.display)
            except (OSError,ValueError,KeyError,RuntimeError) as error:
                state['display_error'] = str(error)
            if time.monotonic() > deadline:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill();child.wait()
                state.update(status='failed', error='Own finite worker deadline exceeded')
                atomic(statepath,state)
                raise TimeoutError(state['error'])
            time.sleep(5)
        state['study_exit_code'] = child.returncode
        if child.returncode == 0:
            state['status'] = 'cpu_audit';atomic(statepath,state)
            command = [sys.executable,'-u','-m','sircus.mjlab.tilt650_verify', '--study',str(a.output),
                       '--source',str(source),'--assets',str(a.assets),'--parent-study',str(a.parent_study),
                       '--parent-audit',str(a.parent_audit),'--output',str(a.output.with_name(a.output.name+'-audit.json'))]
            try:
                result = subprocess.run(command,cwd=source,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=log,
                                        stderr=subprocess.STDOUT,timeout=1800)
                state.update(audit_exit_code=result.returncode,status='completed' if result.returncode == 0 else 'audit_failed')
            except subprocess.TimeoutExpired:
                state.update(status='audit_failed',error='Own CPU audit deadline exceeded')
        else:
            state.update(status='failed',error='Finite worker exited; original evidence retained')
    state['finished_unix_s'] = time.time();atomic(statepath,state)
    try:
        publish(a.output,a.display)
    except (OSError,ValueError,KeyError,RuntimeError) as error:
        state['display_error'] = str(error);atomic(statepath,state)
    print(json.dumps(state),flush=True)
    if state['status'] != 'completed':
        raise RuntimeError('Study incomplete: '+state['status'])


if __name__ == '__main__':
    main()
