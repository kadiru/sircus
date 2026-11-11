"""Bounded 4090 broad search and CPU audit; preserve incomplete evidence honestly."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .coverage_recovery_pipeline import supervise
from .gpu_slot import GPU_UUID
from .sweep_worker import atomic

INPUT_KEYS = ('parent-study', 'parent-audit', 'original-study', 'original-audit',
              'recovery-study', 'recovery-audit', 'reference', 'assets')
WORKER_LIMIT_S = 43200
SEARCH_LIMIT_S = 36000
QUEUE_LIMIT_S = 900
AUDIT_LIMIT_S = 7200
GLOBAL_LIMIT_S = 51300


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def terminal_status(output, audit_path):
    """An exit code alone is not evidence of completion or an audited shortfall."""
    report_path = output / 'report.json'
    report = json.loads(report_path.read_text())
    audit = json.loads(audit_path.read_text())
    if (audit.get('status') != 'passed' or audit.get('failures') != []
            or audit.get('report_sha256') != sha(report_path)):
        raise ValueError('Independent audit does not bind the passed worker report')
    if report.get('status') not in ('completed', 'incomplete'):
        raise ValueError('Worker report lacks a completed or explicitly incomplete status')
    return ('completed' if report['status'] == 'completed' else 'incomplete_audited'), report, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in (*INPUT_KEYS, 'output', 'gpu-lock'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--wall-time-limit', type=float, default=WORKER_LIMIT_S)
    parser.add_argument('--search-limit', type=float, default=SEARCH_LIMIT_S)
    parser.add_argument('--queue-timeout', type=float, default=QUEUE_LIMIT_S)
    parser.add_argument('--global-limit', type=float, default=GLOBAL_LIMIT_S)
    args = parser.parse_args()
    if not (60 < args.wall_time_limit <= WORKER_LIMIT_S
            and 0 < args.search_limit <= SEARCH_LIMIT_S
            and args.search_limit < args.wall_time_limit
            and 0 < args.queue_timeout <= QUEUE_LIMIT_S
            and args.wall_time_limit < args.global_limit <= GLOBAL_LIMIT_S):
        raise ValueError('Finite ordered search, worker, queue and global limits required')
    audit_path = args.output.with_name(args.output.name + '-audit.json')
    statepath = args.output.with_suffix('.pipeline.json')
    logpath = args.output.with_suffix('.pipeline.log')
    budgetpath = args.output.with_name(args.output.name + '-budget.json')
    if args.output.exists() or any(p.exists() for p in (statepath, logpath, budgetpath, audit_path)):
        raise ValueError('Fresh owned output and pipeline namespace required; no in-place resume')
    source = Path(__file__).resolve().parents[2]
    if not (source / 'sircus/mjlab/fixed35_broad_verify.py').is_file():
        raise ValueError('Independent broad-search auditor must exist before execution')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started, unix = time.monotonic(), time.time()
    deadline_unix = unix + args.global_limit
    state = dict(status='starting', source=str(source), output=str(args.output),
                 pipeline_pid=os.getpid(), started_unix_s=unix,
                 global_limit_s=args.global_limit, global_deadline_unix_s=deadline_unix,
                 worker_limit_s=args.wall_time_limit, search_limit_s=args.search_limit,
                 queue_limit_s=args.queue_timeout, audit_limit_s=AUDIT_LIMIT_S,
                 target_search_releases=1000, ui_publication=False, training=False,
                 resume=False)
    atomic(statepath, state)
    command = [sys.executable, '-u', '-m', 'sircus.mjlab.fixed35_broad_study',
               '--global-deadline-unix', str(deadline_unix)]
    for key in (*INPUT_KEYS, 'output', 'gpu-lock', 'wall-time-limit', 'search-limit', 'queue-timeout'):
        command += ['--' + key, str(getattr(args, key.replace('-', '_')))]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU_UUID, PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2')
    try:
        with logpath.open('x') as log:
            code = supervise(command, source=source, env=env, log=log, state=state,
                             statepath=statepath, output=args.output,
                             absolute_deadline=started + args.global_limit,
                             worker_limit=args.wall_time_limit)
            state['study_exit_code'] = code
            if code != 0:
                state.update(status='failed', error=state.get('error', 'Worker failed; partial evidence retained'))
            else:
                remaining = started + args.global_limit - time.monotonic() - 5
                if remaining <= 0:
                    state.update(status='audit_failed', error='Original global deadline leaves no audit time; evidence retained')
                else:
                    state.update(status='cpu_audit')
                    atomic(statepath, state)
                    audit_command = [sys.executable, '-u', '-m', 'sircus.mjlab.fixed35_broad_verify',
                                     '--study', str(args.output), '--source', str(source),
                                     '--output', str(audit_path)]
                    for key in INPUT_KEYS:
                        audit_command += ['--' + key, str(getattr(args, key.replace('-', '_')))]
                    try:
                        result = subprocess.run(audit_command, cwd=source,
                            env=dict(env, CUDA_VISIBLE_DEVICES=''), stdin=subprocess.DEVNULL,
                            stdout=log, stderr=subprocess.STDOUT, timeout=min(AUDIT_LIMIT_S, remaining))
                        state['audit_exit_code'] = result.returncode
                        if result.returncode:
                            state.update(status='audit_failed', error='Independent audit failed; frozen evidence retained')
                        else:
                            status, report, audit = terminal_status(args.output, audit_path)
                            state.update(status=status, study_status=report['status'],
                                         report_sha256=sha(args.output / 'report.json'),
                                         audit_sha256=sha(audit_path), audit_checks=audit.get('checks_count'))
                    except subprocess.TimeoutExpired:
                        state.update(status='audit_failed', error='CPU audit deadline reached; frozen evidence retained')
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        state.update(status='audit_failed', error=type(error).__name__ + ': ' + str(error))
    except Exception as error:
        state.update(status='failed', error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        state.update(finished_unix_s=time.time(), elapsed_s=time.monotonic() - started)
        atomic(statepath, state)
    print(json.dumps(state), flush=True)
    if state['status'] not in ('completed', 'incomplete_audited'):
        raise RuntimeError('Broad-search pipeline not audited: ' + state['status'])
    return state


if __name__ == '__main__':
    main()
