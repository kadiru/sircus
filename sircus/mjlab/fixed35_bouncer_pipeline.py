"""Bounded isolated bouncer-branch worker and independent audit; no UI publishing."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .gpu_slot import GPU_UUID
from .sweep_worker import atomic
from .coverage_recovery_pipeline import supervise

INPUT_KEYS = ('parent-study', 'parent-audit', 'original-study', 'original-audit', 'recovery-study', 'recovery-audit', 'reference', 'assets')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in (*INPUT_KEYS, 'output', 'gpu-lock'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--wall-time-limit', type=float, default=3600)
    parser.add_argument('--queue-timeout', type=float, default=600)
    parser.add_argument('--global-limit', type=float, default=5400)
    args = parser.parse_args()
    if not (60 <= args.wall_time_limit <= 3600 and 0 < args.queue_timeout <= 600
            and args.wall_time_limit < args.global_limit <= 5400):
        raise ValueError('Finite worker, queue and pipeline caps required')
    if args.output.exists():
        raise ValueError('Fresh evidence directory required')
    for path in (args.output.with_suffix('.pipeline.json'), args.output.with_suffix('.pipeline.log'),
                 args.output.with_name(args.output.name + '-budget.json'),
                 args.output.with_name(args.output.name + '-audit.json')):
        if path.exists():
            raise ValueError('Fresh owned pipeline namespace required')
    source = Path(__file__).resolve().parents[2]
    if not (source / 'sircus/mjlab/fixed35_bouncer_verify.py').is_file():
        raise ValueError('Independent auditor must exist before execution')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started, unix = time.monotonic(), time.time()
    deadline_unix = unix + args.global_limit
    state = dict(status='starting', source=str(source), output=str(args.output),
                 pipeline_pid=os.getpid(), started_unix_s=unix, global_limit_s=args.global_limit,
                 global_deadline_unix_s=deadline_unix, ui_publication=False, training=False)
    statepath = args.output.with_suffix('.pipeline.json')
    atomic(statepath, state)
    command = [sys.executable, '-u', '-m', 'sircus.mjlab.fixed35_bouncer_study',
               '--global-deadline-unix', str(deadline_unix)]
    for key in (*INPUT_KEYS, 'output', 'gpu-lock', 'wall-time-limit', 'queue-timeout'):
        command += ['--' + key, str(getattr(args, key.replace('-', '_')))]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU_UUID, PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2')
    with args.output.with_suffix('.pipeline.log').open('x') as log:
        code = supervise(command, source=source, env=env, log=log, state=state,
                         statepath=statepath, output=args.output,
                         absolute_deadline=started + args.global_limit,
                         worker_limit=args.wall_time_limit)
        state['study_exit_code'] = code
        if code == 0:
            remaining = started + args.global_limit - time.monotonic() - 5
            if remaining <= 0:
                state.update(status='audit_failed', error='No original audit time remains')
            else:
                state.update(status='cpu_audit')
                atomic(statepath, state)
                audit_command = [sys.executable, '-u', '-m', 'sircus.mjlab.fixed35_bouncer_verify',
                                 '--study', str(args.output), '--source', str(source),
                                 '--output', str(args.output.with_name(args.output.name + '-audit.json'))]
                for key in INPUT_KEYS:
                    audit_command += ['--' + key, str(getattr(args, key.replace('-', '_')))]
                try:
                    result = subprocess.run(audit_command, cwd=source,
                                            env=dict(env, CUDA_VISIBLE_DEVICES=''),
                                            stdin=subprocess.DEVNULL, stdout=log,
                                            stderr=subprocess.STDOUT, timeout=min(900, remaining))
                    state.update(audit_exit_code=result.returncode,
                                 status='completed' if result.returncode == 0 else 'audit_failed')
                except subprocess.TimeoutExpired:
                    state.update(status='audit_failed', error='CPU audit deadline; evidence retained')
        else:
            state.update(status='failed', error=state.get('error', 'Worker exited; later phases skipped'))
    state.update(finished_unix_s=time.time(), elapsed_s=time.monotonic() - started)
    atomic(statepath, state)
    print(json.dumps(state), flush=True)
    if state['status'] != 'completed':
        raise RuntimeError('Bouncer-branch pipeline incomplete: ' + state['status'])


if __name__ == '__main__':
    main()
