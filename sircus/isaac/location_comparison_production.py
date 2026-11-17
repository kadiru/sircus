"""Explicit, one-start location comparison entry. Import is inert.

No default action, remote dispatch, container start, retry or recovery route.
Mock tests inject fake services and cannot produce a native-complete claim.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time

from candidate_continuation_journal import write_new_json
from candidate_continuation_ledger import canonical_hash as process_digest
from location_comparison import digest, require
from location_comparison_native import (SOURCE, PREPARED_FINGERPRINT,
    validate_native_inputs, validate_source_imports)

SCHEMA = 'sircus-location-comparison-production-inputs-v1'
RUNTIME = dict(python='3.11.13', torch='2.7.0+cu128', numpy='1.26.0', scipy='1.15.3')
ROLES = {'parent', 'experiment', 'scene', 'original_splits', 'parent_progress',
    'parent_audit', 'original_film_source', 'original_settled_source', 'source_contract',
    'source_result', 'source_menu', 'camera', 'dropper_urdf', 'board_urdf', 'poses',
    'comparison', 'python_wrapper'}
WORKSPACE = Path('/workspace')


def canonical(path):
    path = Path(path)
    require(path.is_absolute() and path == path.resolve()
            and not any(p.is_symlink() for p in (path, *path.parents)), 'Noncanonical or linked path')
    return path


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON field')
            result[key] = value
        return result
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def asset_inventory(assets):
    descendants = list(assets.rglob('*'))
    require(all(not p.is_symlink() and (p.is_file() or p.is_dir()) for p in descendants),
        'Linked or special asset member')
    return {str(p.relative_to(assets)) for p in descendants if p.is_file()}


def validate_inputs(manifest, *, expected_output=None):
    """Actual source/asset/parent byte checks only: no model or simulator load."""
    require(manifest.get('schema') == SCHEMA and manifest.get('fingerprint') == digest(
        {k:v for k,v in manifest.items() if k != 'fingerprint'}), 'Changed production manifest')
    require(manifest.get('runtime') == RUNTIME, 'Unreviewed production dependency versions')
    files = manifest['files']
    require(ROLES <= set(files), 'Missing independent input roles')
    for role, item in files.items():
        p = canonical(item['path'])
        require(p.is_file() and file_sha(p) == item['sha256'], 'Input bytes changed: '+role)
    experiment, plan = read(files['experiment']['path']), read(files['comparison']['path'])
    scene = read(files['scene']['path'])
    require(plan.get('fingerprint') == PREPARED_FINGERPRINT, 'Wrong prepared comparison')
    validate_native_inputs(manifest, experiment, plan, env_cfg=scene,
        env_geometry_fingerprint=experiment['source']['geometry_fingerprint'])
    validate_source_imports(manifest)
    require(all(canonical(p) == p for p in SOURCE.glob('*.py')), 'Linked source member')
    from location_comparison import SPLIT_SHA
    require(files['original_splits']['sha256'] == SPLIT_SHA, 'Original split changed')
    source = canonical(manifest['source_run'])
    assets = canonical(manifest['assets'])
    require(source == WORKSPACE/'output/expanded-training-20261102-v4'
        and assets == WORKSPACE/'assets', 'Original scene source changed')
    for role, filename in (('source_contract','contract.json'), ('source_result','result.json'),
                          ('source_menu','expanded-menu-qualified.json'), ('camera','selected-camera.json')):
        require(Path(files[role]['path']) == source/filename, 'Source role path changed: '+role)
    require(Path(files['scene']['path']) == SOURCE/'scene.json', 'Source scene path changed')
    require(Path(files['python_wrapper']['path']) == SOURCE/'python.sh', 'Wrong verified producer wrapper')
    for role, name in (('dropper_urdf','dropper.urdf'), ('board_urdf','board_arm.urdf'), ('poses','poses.json')):
        require(Path(files[role]['path']) == assets/name, 'Asset role path changed: '+role)
    actual = asset_inventory(assets)
    require(actual == {k[6:] for k in files if k.startswith('asset:')}
        and all(Path(files['asset:'+name]['path']) == assets/name for name in actual),
        'Asset inventory or paths changed')
    contract = read(files['source_contract']['path'])
    require(contract['scene'] == scene and contract['fingerprint'] == experiment['source']['geometry_fingerprint'],
        'Original scene contract changed')
    from shot_benchmark_contract import make_protocol
    require(experiment['protocol'] == make_protocol(), 'Original learning/physics protocol changed')
    output, audit_output = canonical(manifest['output']), canonical(manifest['audit_output'])
    require(output.parent == WORKSPACE/'output' and output.name.startswith('seed22-location-comparison-')
        and audit_output == output.with_name(output.name+'-audit'), 'Unexpected output pair')
    if expected_output is not None:
        require(output == canonical(expected_output), 'Launch output differs from manifest')
    protected = {canonical(p) for p in manifest['protected_roots']}
    require({source, assets, SOURCE.parent.parent, WORKSPACE/'output/sweep-ppo-10k-20261106-v2'} <= protected,
        'Missing original evidence protection')
    for target in (output, audit_output):
        require(all(not target.is_relative_to(p) and not p.is_relative_to(target) for p in protected),
            'Output overlaps original evidence')
        require(all(not Path(v['path']).is_relative_to(target) for v in files.values()),
            'Inputs cannot live inside outputs')
    require(manifest.get('shared_lock') == '/workspace/scene.lock', 'Wrong shared scene lock')
    return experiment, plan


def load_inputs(path, expected_sha256):
    require(re.fullmatch('[0-9a-f]{64}', expected_sha256 or '') is not None, 'Expected manifest SHA256')
    path = canonical(path)
    require(file_sha(path) == expected_sha256, 'Manifest bytes changed')
    manifest = read(path)
    experiment, plan = validate_inputs(manifest)
    require(all(not path.is_relative_to(canonical(manifest[key])) for key in ('output','audit_output')),
        'Manifest cannot live inside either output')
    return manifest, experiment, plan


def fresh_outputs(manifest):
    for role in ('output','audit_output'):
        path = canonical(manifest[role])
        require(path.parent.is_dir() and not path.exists(), 'Both native and audit outputs must be untouched')
    return True


def resources(output):
    return dict(free_storage_bytes=shutil.disk_usage(output).free)


def progress_identity(snapshot):
    """Publish transitions immediately; throttle only unchanged tick heartbeats."""
    return digest({key:snapshot[key] for key in ('phase_index','active_phase','train_requested',
        'train_completed','assessment_requested','assessment_completed','pending','stopped')})


def await_owner(path, *, clock=time.monotonic, sleep=time.sleep, parent=os.getppid):
    """The supervisor must spawn before it can pin/write the child identity."""
    until = clock()+3.
    while not path.exists() and clock() < until:
        require(parent() > 1, 'Watchdog exited before ownership handshake')
        sleep(.01)
    require(path.is_file(), 'Watchdog ownership handshake timed out')


def validate_owner(output, session, runtime_fingerprint, plan_fingerprint, *, clock=time.monotonic):
    """Bind child to the watchdog's pinned process identity and shared lock."""
    from candidate_continuation_watchdog import GPU_UUID
    from seed22_continuation import validate_owner as ancestry
    env = os.environ
    owner_path = canonical(output)/'comparison-watchdog-owner.json'
    require(env.get('SIRCUS_COMPARISON_SESSION') == session
        and env.get('SIRCUS_COMPARISON_RUNTIME_FINGERPRINT') == runtime_fingerprint
        and env.get('SIRCUS_COMPARISON_FINGERPRINT') == plan_fingerprint
        and env.get('SIRCUS_COMPARISON_OWNER_FILE') == str(owner_path)
        and env.get('CUDA_VISIBLE_DEVICES') == GPU_UUID, 'Unowned native child environment')
    await_owner(owner_path,clock=clock)
    owner = read(owner_path)
    require(owner.get('schema') == 'sircus-location-comparison-watchdog-v1'
        and owner.get('fingerprint') == process_digest({k:v for k,v in owner.items() if k != 'fingerprint'})
        and owner['session_id'] == session and owner['runtime_fingerprint'] == runtime_fingerprint
        and owner['comparison_fingerprint'] == plan_fingerprint, 'Changed watchdog owner receipt')
    started = float(env['SIRCUS_COMPARISON_STARTED_MONOTONIC'])
    require(math.isfinite(started) and owner['start_monotonic'] == started
        and started <= clock() < owner['absolute_deadline_monotonic']
        and owner['absolute_deadline_monotonic'] == float(env['SIRCUS_COMPARISON_DEADLINE_MONOTONIC'])
        and owner['absolute_deadline_monotonic']-started == 17*3600
        and owner['phase_limits'] == dict(shared=3600,original_usable=28800,broader=28800), 'Changed native wall ceiling')
    require(owner['gpu_uuid'] == GPU_UUID and owner['shared_lock'] == '/workspace/scene.lock',
        'Changed GPU or lock ownership')
    stat = canonical(owner['shared_lock']).stat()
    require(owner['shared_lock_identity'] == [stat.st_dev,stat.st_ino], 'Scene lock inode changed')
    require(ancestry(owner,session) is True, 'Native process ownership lost')
    return True


def validate_audit_owner(manifest, plan, session, *, clock=time.monotonic):
    from seed22_continuation import validate_owner as ancestry
    from location_comparison_watchdog import AUDIT_SCHEMA
    env = os.environ
    output = canonical(manifest['audit_output'])
    owner_path = output/'comparison-audit-owner.json'
    require(env.get('CUDA_VISIBLE_DEVICES') == '' and env.get('SIRCUS_COMPARISON_SESSION') == session
        and env.get('SIRCUS_COMPARISON_RUNTIME_FINGERPRINT') == manifest['fingerprint']
        and env.get('SIRCUS_COMPARISON_FINGERPRINT') == plan['fingerprint']
        and env.get('SIRCUS_COMPARISON_OWNER_FILE') == str(owner_path), 'Unowned CPU audit child')
    await_owner(owner_path,clock=clock)
    owner = read(owner_path)
    native_exit = read(canonical(manifest['output'])/'comparison-watchdog-exit.json')
    require(owner.get('schema') == AUDIT_SCHEMA and owner.get('fingerprint') == process_digest(
        {k:v for k,v in owner.items() if k != 'fingerprint'}), 'Changed audit owner receipt')
    require(native_exit.get('fingerprint') == process_digest({k:v for k,v in native_exit.items() if k != 'fingerprint'})
        and native_exit.get('status') == 'native_exit_zero_independent_audit_required'
        and native_exit.get('lock_release_safe') is True and type(native_exit.get('exit_code')) is int
        and native_exit['exit_code'] == 0 and native_exit.get('native_application_starts') == 1,
        'Audit cannot follow failed or unbound native work')
    for key, wanted in (('session_id',session),('runtime_fingerprint',manifest['fingerprint']),
                        ('comparison_fingerprint',plan['fingerprint'])):
        require(owner.get(key) == native_exit.get(key) == wanted, 'Audit input/session mismatch')
    require(owner['native_exit_fingerprint'] == native_exit['fingerprint']
        == env.get('SIRCUS_COMPARISON_NATIVE_EXIT_FINGERPRINT'), 'Native exit binding changed')
    start,deadline = float(env['SIRCUS_COMPARISON_STARTED_MONOTONIC']),float(env['SIRCUS_COMPARISON_DEADLINE_MONOTONIC'])
    require(math.isfinite(start) and math.isfinite(deadline) and start <= clock() < deadline
        and deadline-start == 7200 and owner['start_monotonic'] == start
        and owner['deadline_monotonic'] == deadline and owner['cuda_visible_devices'] == '', 'Audit deadline changed')
    require(ancestry(owner,session) is True, 'Audit process owner changed')
    return native_exit


def audit_envelope(manifest, plan, session, native_exit, report):
    output = canonical(manifest['output'])
    trace = read(output/'comparison-trace.json')
    child = read(output/'child-exit.json')
    require(child.get('status') == 'native_complete_pending_audit' and child.get('synthetic') is False
        and child.get('app_closed') is True and child.get('session_id') == session
        and child.get('runtime_fingerprint') == manifest['fingerprint']
        and child.get('comparison_fingerprint') == plan['fingerprint']
        and child.get('trace_sha256') == file_sha(output/'comparison-trace.json'), 'Child completion differs')
    require(report.get('status') == 'native_verified' and report.get('synthetic') is False
        and report.get('fingerprint') == digest({k:v for k,v in report.items() if k != 'fingerprint'})
        and report.get('manifest_fingerprint') == manifest['fingerprint']
        and report.get('comparison_fingerprint') == plan['fingerprint']
        and report.get('trace_canonical_sha256') == digest(trace)
        and all(report.get(key) is True for key in ('checkpoint_bytes_independently_verified',
            'model_likelihoods_independently_verified','native_physics_verified')), 'Full audit binding failed')
    value = dict(schema='sircus-location-comparison-audit-envelope-v1',status='native_verified',
        synthetic=False,session_id=session,manifest_fingerprint=manifest['fingerprint'],
        comparison_fingerprint=plan['fingerprint'],native_exit_fingerprint=native_exit['fingerprint'],
        trace_sha256=file_sha(output/'comparison-trace.json'),native_output=str(output),report=report)
    return dict(value,fingerprint=digest(value))


def verify_report(directory, native_exit_fingerprint, manifest, plan, session):
    native_exit = read(Path(manifest['output'])/'comparison-watchdog-exit.json')
    require(native_exit.get('fingerprint') == native_exit_fingerprint == process_digest(
        {k:v for k,v in native_exit.items() if k != 'fingerprint'}), 'Native exit receipt changed after audit')
    saved = read(canonical(directory)/'audit.json')
    require(saved == audit_envelope(manifest,plan,session,native_exit,saved['report']),
        'Published audit envelope changed')
    return True


class NativeServices:
    """Lazy real dependencies. Construction itself performs no work."""
    synthetic = False

    def preflight(self, manifest, experiment):
        from location_comparison_verify import verify_inputs
        verify_inputs(manifest,read(manifest['files']['comparison']['path']))
        import numpy, scipy, torch
        require(dict(python=sys.version.split()[0],torch=torch.__version__,numpy=numpy.__version__,
            scipy=scipy.__version__) == manifest['runtime'], 'Actual runtime versions changed')
        from seed22_continuation import restore_parent, check_parent_predictions
        policy, optimizer, saved = restore_parent(manifest,experiment,device='cpu')
        error = check_parent_predictions(policy, manifest, read(manifest['files']['original_splits']['path']))
        del policy, optimizer
        return dict(status='passed', parent_sha256=manifest['files']['parent']['sha256'],
            inherited_attempts=saved['attempted_shots'], inherited_updates=saved['update_count'],
            max_train_prediction_error=error, simulator_starts=0, optimizer_updates=0)

    def launch_app(self):
        import torch
        require(torch.cuda.device_count() == 1 and '4090' in torch.cuda.get_device_name(0), 'Wrong GPU')
        from isaaclab.app import AppLauncher
        return AppLauncher(headless=True,enable_cameras=True,device='cuda:0').app

    def create_scene(self, manifest, output):
        from candidate_continuation_scene import construct_scene
        env, bundle = construct_scene(output=output/'native',assets=manifest['assets'],
            source_run=manifest['source_run'],camera=manifest['files']['camera']['path'],device='cuda:0')
        return env

    def startup(self, env, output, guard):
        from candidate_continuation_scene import StartupRecorder, audit_startup
        original_tick = env.tick
        def tick():
            require(guard() is True, 'Startup guard failed')
            return original_tick()
        env.tick = tick
        try:
            result = StartupRecorder(env,output/'native/startup').run()
            require(result['status'] == 'completed', 'Startup checks incomplete')
            checked = audit_startup(output/'native/startup',scene=env.cfg,cells=env.spec['cells'],origins=env.origins)
            require(checked['status'] == 'passed', 'Independent startup verification failed')
            return checked
        finally:
            env.tick = original_tick

    def make_engine(self, env, manifest, experiment, output, plan, guard):
        from location_comparison_native import NativeEngine
        return NativeEngine(env,manifest,experiment,output,plan,guard=guard,execution_authorized=True)

    def check_resources(self):
        memory = {line.split(':')[0]: int(line.split()[1])*1024
            for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')}
        require(memory['MemAvailable'] >= 1024**3, 'Available host RAM below 1GiB')
        import torch
        free,_ = torch.cuda.mem_get_info()
        require(free >= 512*1024**2, 'Free GPU memory below 512MiB')
        return True


def execute_child(manifest, experiment, plan, output, session, *, started_monotonic,
                  services=None, owner_check=None, clock=time.monotonic,
                  resource_check=None, integrity_check=None, synthetic=False,
                  execute_approved=False):
    """One application lifetime. Full post-run audit belongs to a CPU child."""
    require(type(synthetic) is bool and type(execute_approved) is bool, 'Explicit execution mode required')
    if synthetic:
        require(getattr(services,'synthetic',False) is True and callable(owner_check)
            and callable(integrity_check) and callable(resource_check), 'Mocks require explicit fake boundaries')
    else:
        require(execute_approved is True and services is None and owner_check is None
            and resource_check is None and integrity_check is None, 'Native dependencies cannot be substituted')
        services = NativeServices()
        owner_check = lambda: validate_owner(output,session,manifest['fingerprint'],plan['fingerprint'],clock=clock)
        integrity_check = lambda: validate_inputs(manifest,expected_output=output) and True
        resource_check = lambda: resources(output)
    from location_comparison_bounds import Bounds
    from location_comparison_runtime import ComparisonRunner, Journal
    from location_comparison_watchdog import ProgressWriter
    output = canonical(output)
    require(output.is_dir(), 'Watchdog must create fresh output')
    require(not any((output/name).exists() for name in ('child-start.json','comparison-trace.json','child-exit.json','events','native')),
        'One start only; prior child evidence cannot be reused')
    bounds = Bounds(plan, clock, resource_check, started_monotonic=started_monotonic)
    writer = ProgressWriter(output,session_id=session,runtime_fingerprint=manifest['fingerprint'],
        comparison_fingerprint=plan['fingerprint'])
    app, engine, trace = None, None, None
    last_integrity, last_publish, last_progress = -math.inf, -math.inf, None
    def guard(snapshot=None, force=False):
        nonlocal last_integrity,last_publish,last_progress
        require(owner_check() is True, 'External owner identity failed')
        bounds.check()
        if clock()-last_integrity >= 60:
            require(integrity_check() is True, 'Actual source/input integrity failed')
            last_integrity = clock()
        if app is not None:
            require(services.check_resources() is True, 'Native memory/resource check failed')
        snapshot = bounds.snapshot() if snapshot is None else snapshot
        changed = progress_identity(snapshot)
        if force or changed != last_progress or clock()-last_publish >= .5:
            writer.publish(snapshot)
            last_publish = clock()
            last_progress = changed
        return True
    write_new_json(output/'child-start.json',dict(synthetic=synthetic,session_id=session,
        runtime_fingerprint=manifest['fingerprint'],comparison_fingerprint=plan['fingerprint'],
        started_monotonic=started_monotonic))
    error,closed = None,False
    try:
        guard(force=True)
        preflight = services.preflight(manifest,experiment)
        require(preflight.get('status') == 'passed', 'Parent preflight failed')
        write_new_json(output/'parent-preflight.json',preflight)
        guard(force=True)
        app = services.launch_app()
        guard(force=True)
        env = services.create_scene(manifest,output)
        startup = services.startup(env,output,guard)
        require(startup.get('status') == 'passed', 'Independent startup audit failed')
        write_new_json(output/'startup-audit.json',startup)
        guard(force=True)
        engine = services.make_engine(env,manifest,experiment,output,plan,guard)
        require(bool(engine.synthetic) == synthetic, 'Engine evidence mode changed')
        runner = ComparisonRunner(plan,engine,bounds,Journal(output/'events'),clock=clock,
            execution_authorized=not synthetic,external_supervisor=guard)
        trace = runner.run()
        write_new_json(output/'comparison-trace.json',trace)
        require(trace['events'][-1]['payload']['status'] == 'complete_audit_pending', 'Comparison incomplete; no retry')
    except BaseException as exception:
        error = type(exception).__name__+': '+str(exception)
    finally:
        try:
            if engine is not None:
                engine.close()
        except BaseException as exception:
            error = error or ('Engine close failed: '+type(exception).__name__+': '+str(exception))
        try:
            if app is not None:
                app.close()  # blocked native close is independently bounded by the OS watchdog
                app = None
            closed = True
        except BaseException as exception:
            error = error or ('Close failed: '+type(exception).__name__+': '+str(exception))
        if error is None:
            try:
                guard(force=True)
                writer.publish(bounds.snapshot(),running=False,status='runner_complete_pending_audit')
            except BaseException as exception:
                error = type(exception).__name__+': '+str(exception)
        result = dict(status='synthetic_complete_pending_audit' if synthetic and error is None else
            'native_complete_pending_audit' if error is None else 'incomplete_no_retry',
            synthetic=synthetic,error=error,app_closed=closed,session_id=session,
            runtime_fingerprint=manifest['fingerprint'],comparison_fingerprint=plan['fingerprint'],
            trace_sha256=file_sha(output/'comparison-trace.json') if trace else None)
        write_new_json(output/'child-exit.json',result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--expected-manifest-sha256',required=True)
    p.add_argument('--session-id',required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight-only',action='store_true')
    mode.add_argument('--native-child',action='store_true')
    mode.add_argument('--audit-child',action='store_true')
    mode.add_argument('--supervise',action='store_true')
    p.add_argument('--execute-approved',action='store_true')
    args = p.parse_args(argv)
    require(re.fullmatch('[0-9a-f]{32}',args.session_id) is not None,'Invalid session')
    require(args.preflight_only or args.execute_approved,'Execution is off without explicit authorization')
    manifest, experiment, plan = load_inputs(args.manifest,args.expected_manifest_sha256)
    output = Path(manifest['output'])
    if args.preflight_only:
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Preflight must hide CUDA')
        print(json.dumps(NativeServices().preflight(manifest,experiment)))
        return 0
    if args.native_child:
        validate_owner(output,args.session_id,manifest['fingerprint'],plan['fingerprint'])
        result = execute_child(manifest,experiment,plan,output,args.session_id,
            started_monotonic=float(os.environ['SIRCUS_COMPARISON_STARTED_MONOTONIC']),execute_approved=True)
        return 0 if result['status'] == 'native_complete_pending_audit' else 2
    if args.audit_child:
        native_exit = validate_audit_owner(manifest,plan,args.session_id)
        write_new_json(Path(manifest['audit_output'])/'audit-start.json',dict(
            session_id=args.session_id,native_exit_fingerprint=native_exit['fingerprint']))
        from location_comparison_verify import audit
        trace = read(output/'comparison-trace.json')
        report = audit(output,manifest,plan,trace)
        validate_audit_owner(manifest,plan,args.session_id)
        write_new_json(Path(manifest['audit_output'])/'audit.json',
            audit_envelope(manifest,plan,args.session_id,native_exit,report))
        return 0
    from location_comparison_watchdog import supervise, supervise_audit
    from candidate_continuation_watchdog import COOKIE
    require(os.environ.get(COOKIE) == args.session_id,
        'Supervisor must be started with its session cookie already in the process environment')
    fresh_outputs(manifest)
    command = [manifest['files']['python_wrapper']['path'],str(Path(__file__).resolve()),
        '--manifest',str(args.manifest),'--expected-manifest-sha256',args.expected_manifest_sha256,
        '--session-id',args.session_id,'--execute-approved']
    last_check = [-math.inf]
    def check():
        now = time.monotonic()
        require(now >= last_check[0], 'Source-check clock moved backwards')
        if now-last_check[0] >= 60:
            load_inputs(args.manifest,args.expected_manifest_sha256)
            last_check[0] = now
        return True
    result = supervise(command+['--native-child'],plan=plan,output=output,lock=manifest['shared_lock'],
        session_id=args.session_id,runtime_fingerprint=manifest['fingerprint'],integrity_check=check,
        execute_approved=True)
    # CPU audit is separately bounded and never turns a failed/partial run into a retry.
    if result['status'] != 'native_exit_zero_independent_audit_required':
        return 2
    def checked_report(directory, native_exit_fingerprint):
        return verify_report(directory,native_exit_fingerprint,manifest,plan,args.session_id)
    audited = supervise_audit(command+['--audit-child'],output=manifest['audit_output'],native_exit=result,
        session_id=args.session_id,runtime_fingerprint=manifest['fingerprint'],comparison_fingerprint=plan['fingerprint'],
        integrity_check=check,verify_report=checked_report,execute_approved=True)
    print(json.dumps(audited))
    return 0 if audited.get('status') == 'audit_exit_zero_report_verified' else 2


if __name__ == '__main__':
    raise SystemExit(main())
