"""Finite4090 translatedXY650 sweep with first-flight ranking and nominal checks."""
import argparse
from collections import Counter
import copy
import datetime
import importlib.metadata
from pathlib import Path
import shutil
import time

from . import xy650_contract as protocol
from .benchmark_adapter import jsonable
from .corrected_contract import PINNED_PACKAGES,sha
from .corrected_grid import source_manifest
from .learning_contract import digest
from .replay650_contract import REPLAY_EXTRA_PACKAGES,validate_replay_packages,metadata as old_metadata,sensitivity
from .replay650_study import storage_check
from .sweep_worker import atomic


def metadata(group,scene):
    return dict(**old_metadata(group,scene),phase=group['phase'],eligible=group['eligible'],
        **{k:copy.deepcopy(group[k]) for k in ('translation_m','query_kind','query_fingerprint','source_query_fingerprint') if k in group})


def run(a):
    from .clean650_geometry import CleanGeometry
    from .xy650_native import NativeExecutor
    from .learning_solve import required_asset_paths
    scene,sources,calibration,bindings=protocol.load_inputs(a)
    version=protocol.VERSION;settings=protocol.settings()
    old=protocol.read(a.old_study/'contract.json')
    packages=validate_replay_packages({n:importlib.metadata.version(n) for n in set(PINNED_PACKAGES)|set(REPLAY_EXTRA_PACKAGES)})
    assets={n:sha(a.assets.parent/n) for n in required_asset_paths(a.assets)}
    if packages!=old['packages'] or assets!=old['asset_sha256']:raise ValueError('Exact pinned ancestry mismatch')
    a.output.mkdir(parents=True,exist_ok=False);storage_check(a.output)
    for src,name in [(a.calibration_report,'calibration.json'),(a.verification_report,'verification.json')]:shutil.copyfile(src,a.output/name)
    started=time.monotonic();source=Path(__file__).resolve().parents[2];seed=next(g for g in sources if g['role']=='candidate')
    scene_fp=protocol.scene_fingerprint(scene,calibration,assets)
    geometry=CleanGeometry(scene,a.assets,seed)
    controls=protocol.controls(sources)
    groups=geometry.check_many(protocol.bind(controls+protocol.coarse(seed),scene_fp));control_groups=groups[:2]
    atomic(a.output/'initial-groups.json',groups);atomic(a.output/'initial-geometry.json',jsonable(geometry.proofs))
    initial=dict(controls_pre=protocol.plan(control_groups,'controls_pre',protocol.STEPS,3),
        coarse=protocol.plan(groups[2:],'coarse'),controls_post=protocol.plan(control_groups,'controls_post',protocol.STEPS,3))
    atomic(a.output/'initial-plan.json',initial)
    spec=dict(version=version,scene=scene,scene_fingerprint=scene_fp,settings=settings,calibration=calibration,input_bindings=bindings,
        source_sha256=source_manifest(source),asset_sha256=assets,packages=packages,
        frozen_files_sha256={n:sha(a.output/n) for n in ('initial-groups.json','initial-geometry.json','initial-plan.json','calibration.json','verification.json')},
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),native_environment_batch=a.batch,scope=protocol.SCOPE,
        evidence_kind='native_calibrated_xy650_sweep')
    spec['fingerprint']=digest(spec);atomic(a.output/'contract.json',spec)
    report=dict(version=version,status='running',phase='prepared',contract_fingerprint=spec['fingerprint'],scene_fingerprint=scene_fp,scope=protocol.SCOPE,
        settings=spec['settings'],calibration={k:calibration[k] for k in ('calibration_sha256','verification_sha256','score_config_sha256')},
        counts={},groups=[],controls=[],top_settings=[],verification=[],gate={},trace_examples=[])
    stages=[];verification_groups=[];verification_sources=[]
    def stage(name,new_groups,requests,selection):
        atomic(a.output/(name+'-selection.json'),dict(source_ids=[r['id'] for r in selection]))
        atomic(a.output/(name+'-groups.json'),new_groups);atomic(a.output/(name+'-plan.json'),requests)
        stages.append(dict(phase=name,prior_records_sha256=sha(a.output/'records.jsonl'),
            files_sha256={n:sha(a.output/n) for n in (name+'-selection.json',name+'-groups.json',name+'-plan.json')},
            geometry_sha256=digest(geometry.proofs)))
        atomic(a.output/'adaptive-stages.json',stages)
    def publish(phase):
        rows=executor.rows;report.update(phase=phase,elapsed_s=time.monotonic()-started)
        report['counts']=dict(**protocol.statistics(rows),phase_counts=dict(Counter(r['phase'] for r in rows)),substep_counts=dict(Counter(str(r['substeps']) for r in rows)))
        pre=protocol.control_gate([r for r in rows if r['phase']=='controls_pre']);post=protocol.control_gate([r for r in rows if r['phase']=='controls_post'])
        if report['status']=='completed' and pre['status']!='passed':post['status']='not_run'
        report['control_gates']=dict(controls_pre=pre,controls_post=post)
        report['groups']=[]
        for group in groups:
            selected=[r for r in rows if r['group_id']==group['id']]
            report['groups'].append(dict(**metadata(group,scene),by_substeps={str(s):protocol.statistics([r for r in selected if r['substeps']==s]) for s in protocol.STEPS}))
        report['controls']=[dict(phase=phase,group_id=g['id'],role=g['role'],substeps=s,stats=protocol.statistics([r for r in rows if r['phase']==phase and r['group_id']==g['id'] and r['substeps']==s]))
            for phase in ('controls_pre','controls_post') for g in control_groups for s in protocol.STEPS]
        lookup={g['id']:g for g in groups};discovery=[r for r in rows if r['phase'] in ('coarse','refinement')]
        report['top_settings']=[dict(metadata(lookup[r['group_id']],scene),id=r['id'],group_id=r['group_id'],substeps=r['substeps'],stats=protocol.statistics([r]),
            minimum_mouth_clearance_mm=r.get('minimum_mouth_clearance_mm'),closest_opening_distance_m=r.get('closest_opening_distance_m'),
            first_flight=copy.deepcopy(r.get('first_flight')),execution_fingerprint=r.get('execution_fingerprint')) for r in protocol.ranked(discovery,10)]
        report['verification']=[]
        for group,src in zip(verification_groups,verification_sources):
            selected=[r for r in rows if r['group_id']==group['id']]
            report['verification'].append(dict(**metadata(group,scene),by_substeps={str(s):protocol.statistics([r for r in selected if r['substeps']==s]) for s in protocol.STEPS},
                sensitivity=sensitivity(selected,executor.positions),qualification=protocol.qualification(selected,executor.positions,
                    controls_passed=pre['status']==post['status']=='passed',source_clean=protocol.clean(src))))
        qualified=sum(r['qualification']['qualified'] for r in report['verification'])
        status='pending'
        if report['status']=='completed':status='blocked_controls' if pre['status']!='passed' or post['status']!='passed' else 'passed' if qualified else 'failed'
        report['gate']=dict(status=status,controls_pre_passed=pre['status']=='passed',controls_post_passed=post['status']=='passed',qualified_settings=qualified,more_ppo_started=False)
        atomic(a.output/'groups.json',groups);atomic(a.output/'geometry.json',jsonable(geometry.proofs));report['storage']=storage_check(a.output)
        atomic(a.output/'report.json',report)
        print(__import__('json').dumps(dict(phase=phase,elapsed_s=round(report['elapsed_s'],1),counts=report['counts'],gate=report['gate'])),flush=True)
        if len(rows)>settings['max_requests'] or report['elapsed_s']>a.wall_time_limit:raise RuntimeError('Finite sweep budget exceeded')
    executor=NativeExecutor(scene,a.assets,calibration,a.output,a.batch,publish);publish('prepared')
    executor.execute(initial['controls_pre']);pre=protocol.control_gate(executor.rows)
    atomic(a.output/'initial-control-gate.json',pre)
    if pre['status']=='passed':
        executor.execute(initial['coarse'])
        seeds=protocol.select_unique([r for r in executor.rows if r['phase']=='coarse'],3)
        refined=geometry.check_many(protocol.bind(protocol.refinement(seeds),scene_fp));groups.extend(refined)
        requests=protocol.plan(refined,'refinement');stage('refinement',refined,requests,seeds);publish('refinement-prepared');executor.execute(requests)
        verification_sources=protocol.select_unique([r for r in executor.rows if r['phase'] in ('coarse','refinement')],5)
        verification_groups=geometry.check_many(protocol.bind(protocol.verification(verification_sources),scene_fp));groups.extend(verification_groups)
        requests=protocol.plan(verification_groups,'verification',protocol.STEPS,3);stage('verification',verification_groups,requests,verification_sources)
        publish('verification-prepared');executor.execute(requests);executor.execute(initial['controls_post'])
    else:
        stage('refinement',[],[],[]);stage('verification',[],[],[])
    post=protocol.control_gate([r for r in executor.rows if r['phase']=='controls_post'])
    if pre['status']!='passed':post['status']='not_run'
    atomic(a.output/'final-control-gate.json',post)
    rows=executor.rows;trace_rows=[r for r in rows if r['phase']=='controls_pre' and r['repeat_index']==0 and r['native_release']]
    trace_rows+=protocol.ranked([r for r in rows if r['phase'] in ('coarse','refinement')],3)
    trace_rows+=[r for r in rows if r['phase']=='verification' and r['repeat_index']==0 and r['native_release']]
    for row in trace_rows:
        group=next(g for g in groups if g['id']==row['group_id'])
        report['trace_examples'].append(dict(**metadata(group,scene),record_id=row['id'],group_id=group['id'],substeps=row['substeps'],
            position_m=executor.positions[row['id']][::row['substeps']*4].tolist(),sample_dt_s=1/60,
            first_flight=copy.deepcopy(row.get('first_flight')),execution_fingerprint=row.get('execution_fingerprint'),
            **{k:row[k] for k in ('route_success','route_quality','rim_band_contact','whole_ball_entry','max_retention_s','invalid','invalid_physics','scalar_score_audit_passed','external_contact_steps')},
            selection_scope='First nominal controls/verification and top3 ranked development records; illustrative actual saved observations, not aggregate performance'))
    report['status']='completed'
    manifest={str(p.relative_to(a.output)):sha(p) for p in sorted(a.output.rglob('*')) if p.is_file() and p.name not in ('report.json','evidence-manifest.json') and not p.name.startswith('audit')}
    report['evidence_sha256']=manifest;atomic(a.output/'evidence-manifest.json',manifest);publish('completed')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('old-study','positive-study','positive-audit','calibration-report','verification-report','replay-study','coarse-study','raised-study','assets','output','gpu-lock'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--batch',type=int,default=32,choices=(16,32));p.add_argument('--queue-timeout',type=float,default=14400);p.add_argument('--wall-time-limit',type=float,default=7200)
    a=p.parse_args()
    from .gpu_slot import gpu_slot
    try:
        with gpu_slot(a.gpu_lock,timeout_s=a.queue_timeout,notify=lambda r:print(__import__('json').dumps(r),flush=True)):run(a)
    except Exception as error:
        if a.output.exists():
            path=a.output/'report.json';r=protocol.read(path) if path.exists() else {};r.update(status='failed',error=type(error).__name__+': '+str(error));atomic(path,r)
        raise


if __name__=='__main__':main()
