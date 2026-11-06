"""Bounded 189-request impact-offset study with a hard acquired-slot time budget."""
import argparse
from collections import Counter
import copy
import datetime
import importlib.metadata
from pathlib import Path
import shutil
import signal
import time

from . import impact650_contract as protocol
from .benchmark_adapter import jsonable
from .corrected_contract import sha
from .corrected_grid import source_manifest
from .learning_contract import digest
from .replay650_contract import metadata as prior_metadata,sensitivity
from .replay650_study import storage_check
from .sweep_worker import atomic


def metadata(group,scene):
    result=prior_metadata(group,scene)
    result.update({k:copy.deepcopy(group[k]) for k in ('phase','eligible','family_id','seed_id','impact_offset_m',
        'impact_point_world_m','impact_to_bucket_horizontal_m','release_center_m','release_center_height_m',
        'query_kind','query_fingerprint')})
    for key in ('translation_m','source_query_fingerprint','source_route_quality'):
        if key in group:result[key]=copy.deepcopy(group[key])
    return result


def run(a):
    from .impact650_geometry import ImpactGeometry
    from .impact650_native import NativeExecutor
    from .learning_solve import required_asset_paths
    parent,sources,controls,bindings=protocol.load_inputs(a.parent_study,a.parent_audit)
    scene,calibration=copy.deepcopy(parent['scene']),copy.deepcopy(parent['calibration'])
    packages={n:importlib.metadata.version(n) for n in parent['packages']}
    assets={n:sha(a.assets.parent/n) for n in required_asset_paths(a.assets)}
    if packages!=parent['packages'] or assets!=parent['asset_sha256']:
        raise ValueError('Exact pinned parent packages and assets required')
    a.output.mkdir(parents=True,exist_ok=False);storage_check(a.output)
    started=time.monotonic();scene_fp=protocol.scene_fingerprint(parent)
    for name in ('calibration.json','verification.json'):shutil.copyfile(a.parent_study/name,a.output/name)
    atomic(a.output/'source-selection.json',sources)
    geometry=ImpactGeometry(scene,a.assets)
    groups=geometry.check_many(protocol.bind(protocol.controls(controls)+protocol.coarse(sources),scene_fp))
    control_groups=groups[:2]
    initial=dict(controls_pre=protocol.plan(control_groups,'controls_pre',protocol.STEPS,3),
                 coarse=protocol.plan(groups[2:],'coarse'),controls_post=protocol.plan(control_groups,'controls_post',protocol.STEPS,3))
    atomic(a.output/'initial-groups.json',groups);atomic(a.output/'initial-geometry.json',jsonable(geometry.proofs));atomic(a.output/'initial-plan.json',initial)
    source=Path(__file__).resolve().parents[2]
    spec=dict(version=protocol.VERSION,scene=scene,physical_scene_fingerprint=parent['scene_fingerprint'],scene_fingerprint=scene_fp,
              settings=protocol.settings(),calibration=calibration,input_bindings=bindings,parent_contract_fingerprint=parent['fingerprint'],
              source_sha256=source_manifest(source),packages=packages,asset_sha256=assets,
              frozen_files_sha256={n:sha(a.output/n) for n in ('initial-groups.json','initial-geometry.json','initial-plan.json','source-selection.json','calibration.json','verification.json')},
              native_environment_batch=a.batch,scope=protocol.SCOPE,evidence_kind='native_calibrated_impact650_development',
              created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    spec['fingerprint']=digest(spec);atomic(a.output/'contract.json',spec)
    report=dict(version=protocol.VERSION,status='running',phase='prepared',scope=protocol.SCOPE,settings=spec['settings'],
                contract_fingerprint=spec['fingerprint'],scene_fingerprint=scene_fp,physical_scene_fingerprint=parent['scene_fingerprint'],
                parent_contract_fingerprint=parent['fingerprint'],counts={},groups=[],controls=[],top_settings=[],verification=[],gate={},trace_examples=[],
                calibration={k:calibration[k] for k in ('calibration_sha256','verification_sha256','score_config_sha256')})
    selected_sources=[];verification_groups=[]

    def publish(phase):
        rows=executor.rows;report.update(phase=phase,elapsed_s=time.monotonic()-started)
        report['counts']=dict(**protocol.statistics(rows),phase_counts=dict(Counter(r['phase'] for r in rows)),
                              substep_counts=dict(Counter(str(r['substeps']) for r in rows)))
        pre=protocol.control_gate([r for r in rows if r['phase']=='controls_pre'])
        post=protocol.control_gate([r for r in rows if r['phase']=='controls_post'])
        if report['status']=='completed' and pre['status']!='passed':post['status']='not_run'
        report['control_gates']=dict(controls_pre=pre,controls_post=post)
        report['groups']=[dict(**metadata(g,scene),by_substeps={str(s):protocol.statistics([r for r in rows if r['group_id']==g['id'] and r['substeps']==s]) for s in protocol.STEPS}) for g in groups]
        report['controls']=[dict(phase=p,group_id=g['id'],role=g['role'],substeps=s,stats=protocol.statistics([r for r in rows if r['phase']==p and r['group_id']==g['id'] and r['substeps']==s]))
                            for p in ('controls_pre','controls_post') for g in control_groups for s in protocol.STEPS]
        lookup={g['id']:g for g in groups};coarse=[r for r in rows if r['phase']=='coarse']
        report['top_settings']=[dict(metadata(lookup[r['group_id']],scene),record_id=r['id'],group_id=r['group_id'],substeps=r['substeps'],
            stats=protocol.statistics([r]),first_flight=copy.deepcopy(r.get('first_flight')),execution_fingerprint=r.get('execution_fingerprint'),
            minimum_mouth_clearance_mm=r.get('minimum_mouth_clearance_mm')) for r in protocol.xy.ranked(coarse,10)]
        report['verification']=[]
        for group,src in zip(verification_groups,selected_sources):
            cohort=[r for r in rows if r['group_id']==group['id']]
            report['verification'].append(dict(metadata(group,scene),source_route_quality=src.get('route_quality',0),
                by_substeps={str(s):protocol.statistics([r for r in cohort if r['substeps']==s]) for s in protocol.STEPS},
                sensitivity=sensitivity(cohort,executor.positions),qualification=protocol.qualification(cohort,executor.positions,
                    controls_passed=pre['status']==post['status']=='passed',source_capture=protocol.captured(src))))
        nclean=sum(v['qualification']['clean_qualified'] for v in report['verification'])
        ncap=sum(v['qualification']['capture_qualified'] for v in report['verification'])
        status='pending' if report['status']!='completed' else 'blocked_controls' if pre['status']!='passed' or post['status']!='passed' else 'passed' if nclean else 'capture_only' if ncap else 'failed'
        report['gate']=dict(status=status,controls_pre_passed=pre['status']=='passed',controls_post_passed=post['status']=='passed',
                            qualified_settings=nclean,capture_qualified_settings=ncap,more_ppo_started=False)
        atomic(a.output/'groups.json',groups);atomic(a.output/'geometry.json',jsonable(geometry.proofs))
        report['storage']=storage_check(a.output);atomic(a.output/'report.json',report)
        print(__import__('json').dumps(dict(phase=phase,elapsed_s=round(report['elapsed_s'],1),counts=report['counts'],gate=report['gate'])),flush=True)
        if len(rows)>protocol.MAX_REQUESTS or report['elapsed_s']>a.wall_time_limit:raise RuntimeError('Finite impact study budget exceeded')

    executor=NativeExecutor(scene,a.assets,calibration,a.output,a.batch,publish);publish('prepared')
    executor.execute(initial['controls_pre']);pre=protocol.control_gate(executor.rows)
    atomic(a.output/'initial-control-gate.json',pre)
    if pre['status']=='passed':
        executor.execute(initial['coarse']);selected_sources=protocol.select([r for r in executor.rows if r['phase']=='coarse'])
        verification_groups=geometry.check_many(protocol.bind(protocol.verification(selected_sources),scene_fp));groups.extend(verification_groups)
        plan=protocol.plan(verification_groups,'verification',protocol.STEPS,3)
        atomic(a.output/'verification-selection.json',selected_sources);atomic(a.output/'verification-groups.json',verification_groups);atomic(a.output/'verification-plan.json',plan)
        stage=dict(prior_records_sha256=sha(a.output/'records.jsonl'),geometry_sha256=digest(geometry.proofs),
                   files_sha256={n:sha(a.output/n) for n in ('verification-selection.json','verification-groups.json','verification-plan.json')})
        atomic(a.output/'verification-stage.json',stage);publish('verification-prepared')
        executor.execute(plan);executor.execute(initial['controls_post'])
    else:
        for name in ('verification-selection.json','verification-groups.json','verification-plan.json'):atomic(a.output/name,[])
    post=protocol.control_gate([r for r in executor.rows if r['phase']=='controls_post'])
    if pre['status']!='passed':post['status']='not_run'
    atomic(a.output/'final-control-gate.json',post)
    trace_rows=[r for r in executor.rows if r['phase']=='controls_pre' and r['repeat_index']==0 and r['native_release']]
    trace_rows+=protocol.xy.ranked([r for r in executor.rows if r['phase']=='coarse'],3)
    trace_rows+=[r for r in executor.rows if r['phase']=='verification' and r['repeat_index']==0 and r['native_release']]
    for row in trace_rows:
        group=next(g for g in groups if g['id']==row['group_id'])
        report['trace_examples'].append(dict(**metadata(group,scene),record_id=row['id'],group_id=group['id'],substeps=row['substeps'],
            position_m=executor.positions[row['id']][::row['substeps']*4].tolist(),sample_dt_s=1/60,
            execution_fingerprint=row.get('execution_fingerprint'),first_flight=copy.deepcopy(row.get('first_flight')),
            **{k:row[k] for k in ('route_success','route_quality','rim_band_contact','whole_ball_entry','max_retention_s','invalid','invalid_physics','scalar_score_audit_passed','external_contact_steps')},
            selection_scope='First nominal controls/verification and three ranked development traces; actual saved observations, not aggregate or held-out performance'))
    report['status']='completed'
    manifest={str(p.relative_to(a.output)):sha(p) for p in sorted(a.output.rglob('*')) if p.is_file() and p.name not in ('report.json','evidence-manifest.json') and not p.name.startswith('audit')}
    report['evidence_sha256']=manifest;atomic(a.output/'evidence-manifest.json',manifest);publish('completed')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('parent-study','parent-audit','assets','output','gpu-lock'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--batch',type=int,default=32,choices=(16,32));p.add_argument('--queue-timeout',type=float,default=1800)
    p.add_argument('--wall-time-limit',type=float,default=3600);a=p.parse_args()
    if not 30<=a.wall_time_limit<=3600:raise ValueError('Acquired-slot worker budget must be 30..3600 seconds')
    from .gpu_slot import gpu_slot
    def deadline(signum,frame):raise TimeoutError('Own acquired-slot worker budget reached; no further requests permitted')
    try:
        with gpu_slot(a.gpu_lock,timeout_s=a.queue_timeout,notify=lambda r:print(__import__('json').dumps(r),flush=True)):
            atomic(a.output.with_name(a.output.name+'-budget.json'),dict(acquired_unix_s=time.time(),limit_s=a.wall_time_limit))
            signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,a.wall_time_limit-10)
            try:run(a)
            finally:signal.setitimer(signal.ITIMER_REAL,0)
    except Exception as error:
        if a.output.exists():
            path=a.output/'report.json';r=protocol.read(path) if path.exists() else {};r.update(status='failed',error=type(error).__name__+': '+str(error));atomic(path,r)
        raise


if __name__=='__main__':main()
