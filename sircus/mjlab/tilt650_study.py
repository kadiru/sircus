"""Finite 54-request replay of the five previously unverified 30-degree recipes."""
import argparse
from collections import Counter
import copy
import datetime
import importlib.metadata
from pathlib import Path
import shutil
import time

from . import tilt650_contract as protocol
from . import xy650_contract as xy
from .benchmark_adapter import jsonable
from .corrected_contract import sha
from .corrected_grid import source_manifest
from .learning_contract import digest
from .replay650_contract import sensitivity
from .replay650_study import storage_check
from .sweep_worker import atomic
from .xy650_study import metadata


def run(a):
    from .clean650_geometry import CleanGeometry
    from .xy650_native import NativeExecutor
    from .learning_solve import required_asset_paths
    parent, sources, controls, bindings = protocol.load_parent(a.parent_study, a.parent_audit)
    scene, calibration = copy.deepcopy(parent['scene']), copy.deepcopy(parent['calibration'])
    packages = {n: importlib.metadata.version(n) for n in parent['packages']}
    assets = {n: sha(a.assets.parent/n) for n in required_asset_paths(a.assets)}
    if packages != parent['packages'] or assets != parent['asset_sha256']:
        raise ValueError('Exact pinned parent environment and assets required')
    a.output.mkdir(parents=True, exist_ok=False)
    storage_check(a.output)
    started = time.monotonic()
    scene_fp = xy.scene_fingerprint(scene, calibration, assets)
    if scene_fp != parent['scene_fingerprint']:
        raise ValueError('Static parent scene fingerprint changed')
    for name in ('calibration.json', 'verification.json'):
        shutil.copyfile(a.parent_study/name, a.output/name)
    atomic(a.output/'source-selection.json', sources)
    geometry = CleanGeometry(scene, a.assets, sources[0])
    groups = geometry.check_many(protocol.groups(sources, controls, scene_fp))
    control_groups, verification_groups = groups[:2], groups[2:]
    plan = dict(controls_pre=xy.plan(control_groups, 'controls_pre', protocol.STEPS, 3),
                verification=xy.plan(verification_groups, 'verification', protocol.STEPS, 3),
                controls_post=xy.plan(control_groups, 'controls_post', protocol.STEPS, 3))
    atomic(a.output/'initial-groups.json', groups)
    atomic(a.output/'initial-geometry.json', jsonable(geometry.proofs))
    atomic(a.output/'initial-plan.json', plan)
    source = Path(__file__).resolve().parents[2]
    spec = dict(version=protocol.VERSION, scene=scene, scene_fingerprint=scene_fp,
                settings=protocol.settings(), calibration=calibration, input_bindings=bindings,
                parent_contract_fingerprint=parent['fingerprint'], source_sha256=source_manifest(source),
                packages=packages, asset_sha256=assets,
                frozen_files_sha256={n: sha(a.output/n) for n in ('initial-groups.json','initial-geometry.json','initial-plan.json','source-selection.json','calibration.json','verification.json')},
                created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                native_environment_batch=a.batch, scope=protocol.SCOPE,
                evidence_kind='native_calibrated_tilt30_verification')
    spec['fingerprint'] = digest(spec)
    atomic(a.output/'contract.json', spec)
    report = dict(version=protocol.VERSION, status='running', phase='prepared', contract_fingerprint=spec['fingerprint'],
                  scene_fingerprint=scene_fp, parent_contract_fingerprint=parent['fingerprint'], scope=protocol.SCOPE,
                  settings=spec['settings'], counts={}, groups=[], controls=[], top_settings=[], verification=[], gate={}, trace_examples=[],
                  calibration={k: calibration[k] for k in ('calibration_sha256','verification_sha256','score_config_sha256')})

    def publish(phase):
        rows = executor.rows
        report.update(phase=phase, elapsed_s=time.monotonic()-started)
        report['counts'] = dict(**protocol.statistics(rows), phase_counts=dict(Counter(r['phase'] for r in rows)),
                                substep_counts=dict(Counter(str(r['substeps']) for r in rows)))
        pre = protocol.control_gate([r for r in rows if r['phase'] == 'controls_pre'])
        post = protocol.control_gate([r for r in rows if r['phase'] == 'controls_post'])
        if report['status'] == 'completed' and pre['status'] != 'passed':
            post['status'] = 'not_run'
        report['control_gates'] = dict(controls_pre=pre, controls_post=post)
        report['groups'] = [dict(**metadata(g, scene), by_substeps={str(s): protocol.statistics([r for r in rows if r['group_id'] == g['id'] and r['substeps'] == s]) for s in protocol.STEPS}) for g in groups]
        report['controls'] = [dict(phase=p, group_id=g['id'], role=g['role'], substeps=s,
                                  stats=protocol.statistics([r for r in rows if r['phase'] == p and r['group_id'] == g['id'] and r['substeps'] == s]))
                              for p in ('controls_pre','controls_post') for g in control_groups for s in protocol.STEPS]
        report['verification'] = []
        for group, src in zip(verification_groups, sources):
            selected = [r for r in rows if r['group_id'] == group['id']]
            report['verification'].append(dict(**metadata(group, scene), parent_route_quality=src['route_quality'],
                by_substeps={str(s): protocol.statistics([r for r in selected if r['substeps'] == s]) for s in protocol.STEPS},
                sensitivity=sensitivity(selected, executor.positions),
                qualification=protocol.qualification(selected, executor.positions,
                    controls_passed=pre['status'] == post['status'] == 'passed', parent_source_capture=protocol.captured(src))))
        count = sum(r['qualification']['clean_qualified'] for r in report['verification'])
        capture_count = sum(r['qualification']['capture_qualified'] for r in report['verification'])
        status = 'pending' if report['status'] != 'completed' else 'blocked_controls' if pre['status'] != post['status'] or pre['status'] != 'passed' else 'passed' if count else 'capture_only' if capture_count else 'failed'
        report['gate'] = dict(status=status, controls_pre_passed=pre['status']=='passed', controls_post_passed=post['status']=='passed',
                              qualified_settings=count, capture_qualified_settings=capture_count, more_ppo_started=False)
        atomic(a.output/'groups.json', groups)
        atomic(a.output/'geometry.json', jsonable(geometry.proofs))
        report['storage'] = storage_check(a.output)
        atomic(a.output/'report.json', report)
        print(__import__('json').dumps(dict(phase=phase, elapsed_s=round(report['elapsed_s'],1), counts=report['counts'], gate=report['gate'])), flush=True)
        if len(rows) > 54 or report['elapsed_s'] > a.wall_time_limit:
            raise RuntimeError('Finite verification budget exceeded')

    executor = NativeExecutor(scene, a.assets, calibration, a.output, a.batch, publish)
    publish('prepared')
    executor.execute(plan['controls_pre'])
    pre = protocol.control_gate(executor.rows)
    atomic(a.output/'initial-control-gate.json', pre)
    if pre['status'] == 'passed':
        executor.execute(plan['verification'])
        executor.execute(plan['controls_post'])
    post = protocol.control_gate([r for r in executor.rows if r['phase'] == 'controls_post'])
    if pre['status'] != 'passed':
        post['status'] = 'not_run'
    atomic(a.output/'final-control-gate.json', post)
    for row in (r for r in executor.rows if r['repeat_index'] == 0 and r['phase'] != 'controls_post' and r['native_release']):
        group = next(g for g in groups if g['id'] == row['group_id'])
        report['trace_examples'].append(dict(**metadata(group, scene), record_id=row['id'], group_id=group['id'], substeps=row['substeps'],
            position_m=executor.positions[row['id']][::row['substeps']*4].tolist(), sample_dt_s=1/60,
            first_flight=copy.deepcopy(row.get('first_flight')), execution_fingerprint=row.get('execution_fingerprint'),
            **{k: row[k] for k in ('route_success','route_quality','rim_band_contact','whole_ball_entry','max_retention_s','invalid','invalid_physics','scalar_score_audit_passed','external_contact_steps')},
            selection_scope='First nominal controls and each verification target/resolution; actual saved observations, not independent noise tests'))
    report['status'] = 'completed'
    manifest = {str(p.relative_to(a.output)): sha(p) for p in sorted(a.output.rglob('*')) if p.is_file()
                and p.name not in ('report.json','evidence-manifest.json') and not p.name.startswith('audit')}
    report['evidence_sha256'] = manifest
    atomic(a.output/'evidence-manifest.json', manifest)
    publish('completed')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('parent-study','parent-audit','assets','output','gpu-lock'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--batch', type=int, default=32, choices=(16,32))
    p.add_argument('--queue-timeout', type=float, default=1800)
    p.add_argument('--wall-time-limit', type=float, default=1800)
    a = p.parse_args()
    from .gpu_slot import gpu_slot
    try:
        with gpu_slot(a.gpu_lock, timeout_s=a.queue_timeout, notify=lambda r: print(__import__('json').dumps(r), flush=True)):
            run(a)
    except Exception as error:
        if a.output.exists():
            path = a.output/'report.json'
            value = protocol.read(path) if path.exists() else {}
            value.update(status='failed', error=type(error).__name__+': '+str(error))
            atomic(path, value)
        raise


if __name__ == '__main__':
    main()
