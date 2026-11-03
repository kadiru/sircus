"""Finite fixed-bucket, matched-budget adaptive-surrogate versus terminal SAC study.

No hardware, active inference deployment or service management is implemented.
Every requested action, including rejected geometry, consumes its declared budget.
"""
import argparse
from collections import Counter
import copy
import datetime
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np

from . import window_contract as protocol
from .contact_study import atomic
from .corrected_contract import PHYSICS, geometry_hint, sha, validate_packages
from .corrected_grid import source_manifest, storage_check
from .learning_contract import digest, reward


def vector(contexts):
    return np.asarray([[c['board_height_m'],c['tilt_deg']] for c in contexts])


def train_rows(rows,method):
    return [r for r in rows if r['role']=='train' and r['method'] in ('shared',method)]


def checkpoint_choice(checkpoints,rows,method):
    lookup={r['id']:r for r in rows}
    candidates=[c for c in checkpoints if c['method']==method]
    def rank(c):
        selected=[lookup[i] for i in c['validation_record_ids']]
        return (sum(bool(r['route_success']) for r in selected),
                float(np.mean([r['learning_reward'] for r in selected])),-c['milestone'])
    return copy.deepcopy(max(candidates,key=rank))


def evaluation_requests(contexts,gaps,*,role,method,milestone,checkpoint_hash,seed):
    """Freeze base actions and paired perturbations without reading outcomes."""
    if role not in ('validation','test'):raise ValueError('Evaluation role required')
    gaps=np.asarray(gaps,float)
    if gaps.shape!=(len(contexts),) or not np.isfinite(gaps).all():raise ValueError('One finite action/context')
    rng=np.random.default_rng(seed);requests=[]
    nominal=1 if role=='validation' else 3
    fresh=2 if role=='validation' else 20
    for context,gap in zip(contexts,gaps):
        for repeat,noise in enumerate([0.]*nominal+rng.normal(0,protocol.NOISE_GAP_M,fresh).tolist()):
            requests.append(dict(role=role,method=method,phase='nominal' if repeat<nominal else 'fresh',
                context_id=context['id'],gap_m=float(gap+noise),requested_gap_m=float(gap+noise),
                nominal_gap_m=float(gap),gap_noise_m=float(noise),noise_seed=seed,repeat_index=repeat,
                milestone=milestone,source_checkpoint_sha256=checkpoint_hash))
    return requests


def _run(a):
    # All native imports are behind the explicit runner. Unit tests stay CPU/lightweight.
    from .admission import VERIFIED_PARK_Q10
    from .bank_study import run_bank_rows
    from .contact_env import ContactEnvironment
    from .corrected_admission import TransitAdmission
    from .learning_solve import required_asset_paths
    from .window_geometry import WindowGeometry, VERSION as GEOMETRY_VERSION
    from .window_learning import SACAgent, SuccessModel

    scene=json.loads(a.scene.read_text());scene=copy.deepcopy(scene)
    scene['bucket']['center_xy_m']=list(protocol.settings()['bucket_xy_m'])
    raw_anchor=json.loads(a.anchor.read_text())
    anchor=geometry_hint(raw_anchor,'window-reference-geometry')
    if anchor is None:raise ValueError('Complete immutable reference joint geometry required')
    if not np.array_equal(anchor['values'],__import__('sircus.mjlab.admission',fromlist=['REFERENCE_VALUES']).REFERENCE_VALUES):
        raise ValueError('Unexpected reference geometry')
    packages=validate_packages({name:importlib.metadata.version(name) for name in
        ('mjlab','mujoco','mujoco-warp','torch','warp-lang','numpy','scipy','ml-dtypes')})
    a.output.mkdir(parents=True,exist_ok=False)
    groups=protocol.contexts();all_contexts=sum(groups.values(),[])
    spec=dict(version=protocol.VERSION,settings=protocol.settings(),scene=scene,physics=PHYSICS,
        geometry_version=GEOMETRY_VERSION,source_sha256=source_manifest(Path(__file__).resolve().parents[2]),
        asset_sha256={p:sha(a.assets.parent/p) for p in required_asset_paths(a.assets)},packages=packages,
        anchor_source_sha256=sha(a.anchor),anchor_geometry=anchor,
        parked_dropper_q10=list(VERIFIED_PARK_Q10),batch=a.batch,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        wall_time_limit_s=a.wall_time_limit,scope=protocol.settings()['scope'])
    spec['fingerprint']=digest(spec);atomic(a.output/'contract.json',spec)
    atomic(a.output/'scene.json',scene)
    report=dict(version=protocol.VERSION,status='running',phase='prepare_contexts',contract_fingerprint=spec['fingerprint'],
        settings=spec['settings'],physics=PHYSICS,scope=spec['scope'],counts={},learning_curve=[],test={},test_contexts=[],
        observed_windows=[],trace_examples=[],selected_checkpoints={},prepared_contexts=[])
    rows=[];batches=[];checkpoints=[];sequence=0;started=time.monotonic()
    def publish(phase,**extra):
        report.update(extra);report['phase']=phase;report['elapsed_s']=time.monotonic()-started
        report['counts']=dict(requests=len(rows),native_releases=sum(bool(r.get('native_release')) for r in rows),
            captures=sum(bool(r['route_success']) for r in rows),geometry_rejections=sum(bool(r.get('invalid')) for r in rows),
            role_counts=dict(Counter(r['role'] for r in rows)),method_counts=dict(Counter(r['method'] for r in rows)),
            charged_training={m:len(train_rows(rows,m)) for m in ('adaptive_surrogate','sac')})
        report['storage']=storage_check(a.output);atomic(a.output/'report.json',report)
        print(json.dumps(dict(phase=phase,elapsed_s=round(report['elapsed_s'],1),counts=report['counts'])),flush=True)
        if report['elapsed_s']>a.wall_time_limit:raise TimeoutError('Finite study time budget exceeded; no automatic restart')

    geometry=WindowGeometry(scene,a.assets,anchor,bounds=protocol.BOUNDS)
    prepared=geometry.prepare_contexts(all_contexts,progress=lambda c:publish('prepare_contexts',preparing_context=c['id']))
    atomic(a.output/'prepared-contexts.json',prepared);report['prepared_contexts']=prepared
    context_map={c['id']:c for c in all_contexts}
    publish('contexts_prepared')
    env=ContactEnvironment(scene,a.assets,n=a.batch,bounds=protocol.BOUNDS,substeps=64,
        board_mode='articulated',gravity_mode='passive_only')
    # mjlab applies MujocoCfg to its simulation model, separate from env.m's
    # CPU geometry template. Record/check the model actually integrated.
    option=env.sim.mj_model.opt
    if not np.isclose(option.timestep,1/15360) or option.iterations!=100 or not np.isclose(option.tolerance,1e-8,rtol=0,atol=1e-12):
        raise RuntimeError('Actual native options differ from frozen corrected-contact protocol')
    import mujoco
    effective=dict(timestep_s=float(option.timestep),solver=int(option.solver),solver_name=mujoco.mjtSolver(int(option.solver)).name,
        integrator=int(option.integrator),integrator_name=mujoco.mjtIntegrator(int(option.integrator)).name,
        cone=int(option.cone),cone_name=mujoco.mjtCone(int(option.cone)).name,iterations=int(option.iterations),
        tolerance=float(option.tolerance),gravity=option.gravity.tolist(),substeps=env.substeps,
        board_mode=env.board_mode,gravity_mode=env.gravity_mode)
    atomic(a.output/'effective-options.json',effective);report['effective_native_options']=effective
    def execute(requests):
        nonlocal sequence
        completed=[]
        for offset in range(0,len(requests),a.batch):
            chunk=copy.deepcopy(requests[offset:offset+a.batch]);packet=len(batches)
            for r in chunk:
                r['id']=f'window-shot-{sequence:06d}';sequence+=1
                r['requested_gap_m']=r['gap_m']
            # Preserve the proposed sequence before any geometry or native outcomes.
            atomic(a.output/f'plan-{packet:04d}.json',chunk)
            gates=geometry.audit([r['context_id'] for r in chunk],[r['gap_m'] for r in chunk])
            atomic(a.output/f'geometry-{packet:04d}.json',dict(record_ids=[r['id'] for r in chunk],rows=gates))
            eligible=np.asarray([g['eligible'] for g in gates],bool)
            measured=run_bank_rows(env,np.asarray([g['bucket_xy_m'] for g in gates]),np.asarray([g['values'] for g in gates]),
                np.asarray([g['q10'] for g in gates]),np.asarray([g['q14'] for g in gates]),eligible,record=True)
            if any(bool(r.get('native_release'))!=bool(ok) for r,ok in zip(measured,eligible)):
                raise RuntimeError('Native release changed exact CPU admission; refusing to hide incompatible endpoints')
            indices=np.flatnonzero(eligible);cache=None
            if len(indices):
                cache=f'native-{packet:04d}.npz';arrays=env.last_arrays
                np.savez_compressed(a.output/cache,**arrays,
                    record_ids=np.asarray([chunk[i]['id'] for i in indices]))
            native_index={int(i):j for j,i in enumerate(indices)}
            for i,(request,gate,result) in enumerate(zip(chunk,gates,measured)):
                row=dict(result);row.update(request)
                row.update(values=gate['values'],bucket_xy_m=gate['bucket_xy_m'],q10=gate['q10'],q14=gate['q14'],
                    eligible=gate['eligible'],geometry=gate,board_branch_sha256=gate['board_branch_sha256'],
                    geometry_eligible=gate['eligible'],minimum_clearance_lower_bound_m=gate['minimum_clearance_lower_bound_m'],
                    geometry_cache_file=f'geometry-{packet:04d}.json',geometry_cache_row=i,
                    native_cache_file=cache if eligible[i] else None,native_cache_row=native_index.get(i))
                row['learning_reward']=reward(row)
                args=tuple(arrays[k][:,native_index[i]] for k in ('position_m','velocity_m_s','forces_N')) if eligible[i] else (None,None,None)
                row['diagnostic']=protocol.outcome_diagnostic(row,*args,scene=scene)
                if eligible[i] and len(report['trace_examples'])<10 and (row['route_success'] or len(report['trace_examples'])<2):
                    report['trace_examples'].append(dict(context_id=row['context_id'],gap_m=row['gap_m'],
                        route_success=row['route_success'],record_id=row['id'],phase=row['phase'],role=row['role'],
                        position_m=args[0][::3].tolist()))
                rows.append(row);completed.append(row)
                with (a.output/'records.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
            batches.append(dict(packet=packet,record_ids=[r['id'] for r in chunk],native_cache_file=cache,
                native_record_ids=[chunk[i]['id'] for i in indices],plan_file=f'plan-{packet:04d}.json',
                geometry_cache_file=f'geometry-{packet:04d}.json'))
            atomic(a.output/'batches.json',batches);publish(chunk[0]['role']+'_'+chunk[0]['phase'])
        return completed

    anchor_context=next(c for c in groups['train'] if np.array_equal(vector([c])[0],anchor['values'][2:4]))
    controls=execute([dict(role='control',method='control',phase='quarantined_anchor',
        context_id=anchor_context['id'],gap_m=float(anchor['values'][4]),repeat_index=i) for i in range(3)])
    if sum(bool(r['route_success']) for r in controls)<2:
        raise RuntimeError('Native anchor control failed; stop to diagnose, do not train around an incompatible baseline')
    seed=execute([dict(role='train',method='shared',phase='coarse_seed',context_id=c['id'],gap_m=float(g),milestone=504)
        for c in groups['train'] for g in protocol.seed_gaps()])
    sac=SACAgent(seed=protocol.SEED,alpha=.02,device='cpu')
    def add_sac(experience):
        sac.add(vector([context_map[r['context_id']] for r in experience]),[r['gap_m'] for r in experience],
            [r['learning_reward'] for r in experience],roles=[r['role'] for r in experience],
            provenance=dict(record_ids=[r['id'] for r in experience],contract_fingerprint=spec['fingerprint']))
    add_sac(seed);updates=[sac.update(steps=400,batch_size=64)]
    atomic(a.output/'sac-updates.json',updates)
    for stage,milestone in enumerate(protocol.MILESTONES):
        if stage:
            proposals=[]
            for c in groups['train']:
                history=[r for r in train_rows(rows,'adaptive_surrogate') if r['context_id']==c['id']]
                proposals += [dict(role='train',method='adaptive_surrogate',phase='adaptive_refinement',
                    context_id=c['id'],gap_m=g,round=stage,milestone=milestone) for g in protocol.adaptive_gaps(history)]
            execute(proposals)
            for round_index in range(7):
                gaps=sac.propose(vector(groups['train']),stochastic=True)
                fresh=execute([dict(role='train',method='sac',phase='sac_exploration',context_id=c['id'],
                    gap_m=float(g),round=(stage-1)*7+round_index,milestone=milestone) for c,g in zip(groups['train'],gaps)])
                add_sac(fresh);updates.append(sac.update(steps=32,batch_size=64));atomic(a.output/'sac-updates.json',updates)
        surrogate=SuccessModel();surrogate.fit(train_rows(rows,'adaptive_surrogate'))
        for method,model in (('adaptive_surrogate',surrogate),('sac',sac)):
            extension='json' if method=='adaptive_surrogate' else 'pt';name=f'{method}-{milestone}.{extension}'
            save_metadata=model.save(a.output/name,spec['fingerprint'],include_replay=True) if method=='sac' else model.save(a.output/name,spec['fingerprint'])
            checkpoint=dict(method=method,milestone=milestone,path=name,sha256=sha(a.output/name),save_metadata=save_metadata,
                training_record_ids=[r['id'] for r in train_rows(rows,method)],validation_record_ids=[])
            # This list freezes inference before any validation observation.
            requests=evaluation_requests(groups['validation'],model.propose(vector(groups['validation'])),role='validation',
                method=method,milestone=milestone,checkpoint_hash=checkpoint['sha256'],seed=protocol.SEED+100+stage)
            validation=execute(requests);checkpoint['validation_record_ids']=[r['id'] for r in validation]
            checkpoints.append(checkpoint);atomic(a.output/'checkpoints.json',checkpoints)
            report['learning_curve'].append(dict(method=method,milestone=milestone,validation=protocol.summary(validation),
                checkpoint_sha256=checkpoint['sha256']))
        report['observed_windows']=protocol.windows(train_rows(rows,'adaptive_surrogate'));publish('milestone_complete')
    selected={m:checkpoint_choice(checkpoints,rows,m) for m in ('adaptive_surrogate','sac')}
    atomic(a.output/'selected-checkpoints.json',selected);report['selected_checkpoints']=selected
    selected_models=dict(adaptive_surrogate=SuccessModel.load(a.output/selected['adaptive_surrogate']['path'],spec['fingerprint']),
        sac=SACAgent.load(a.output/selected['sac']['path'],spec['fingerprint']))
    inference={m:model.propose(vector(groups['test'])).tolist() for m,model in selected_models.items()}
    inference['uniform_bounded']=np.random.default_rng(protocol.SEED+300).uniform(*protocol.GAP_BOUNDS,len(groups['test'])).tolist()
    frozen_test=[]
    for method,gaps in inference.items():
        chosen=selected.get(method,{})
        frozen_test += evaluation_requests(groups['test'],gaps,role='test',method=method,
            milestone=chosen.get('milestone',0),checkpoint_hash=chosen.get('sha256'),seed=protocol.SEED+200)
    atomic(a.output/'frozen-test-plan.json',frozen_test)
    execute(frozen_test)
    for method in inference:
        report['test'][method]=protocol.summary([r for r in rows if r['role']=='test' and r['method']==method])
    for c in groups['test']:
        methods={}
        for method,gaps in inference.items():
            selected_rows=[r for r in rows if r['role']=='test' and r['method']==method and r['context_id']==c['id']]
            methods[method]=dict(gap_m=gaps[groups['test'].index(c)],total=protocol.summary(selected_rows),
                nominal=protocol.summary([r for r in selected_rows if r['phase']=='nominal']),
                fresh=protocol.summary([r for r in selected_rows if r['phase']=='fresh']))
        report['test_contexts'].append(dict(context_id=c['id'],**{k:v for k,v in c.items() if k!='id'},methods=methods))
    # Separate geometric path evidence: never reinterpret held-posture capture as motion validation.
    admission=TransitAdmission(scene,a.assets);atomic(a.output/'common-home.json',admission.home)
    path_rows=[]
    for method in selected_models:
        candidates=[r for r in rows if r['role']=='test' and r['method']==method and r['phase']=='nominal' and r['repeat_index']==0 and r['route_success']]
        for row in candidates[:4]:
            if not (.12<=row['values'][2]<=.60 and 8<=row['values'][3]<=60 and .05<=row['values'][4]<=.65):
                path_rows.append(dict(record_id=row['id'],method=method,passed=False,reason='outside_legacy_transit_bounds',scope='Path unverified; native flight remains separate'));continue
            proof=admission.admit(row);name='path-'+row['id']+'.json';atomic(a.output/name,proof)
            path_rows.append(dict(record_id=row['id'],method=method,passed=bool(proof.get('passed')),file=name,
                reason=proof.get('reason'),scope=proof.get('scope')))
    atomic(a.output/'selected-paths.json',path_rows);report['selected_paths']=path_rows
    expected=3+1512+144+552
    if len(rows)!=expected or any(len(train_rows(rows,m))!=1008 for m in selected_models):raise RuntimeError('Changed matched study budget')
    report['evidence_sha256']={str(p.relative_to(a.output)):sha(p) for p in sorted(a.output.rglob('*'))
        if p.is_file() and p.name not in ('report.json','evidence-manifest.json','audit.json')}
    atomic(a.output/'evidence-manifest.json',report['evidence_sha256'])
    report['status']='completed';publish('completed')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('assets','scene','anchor','output','gpu-lock'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--batch',type=int,default=64,choices=(32,64,128))
    parser.add_argument('--queue-timeout',type=float,default=7200)
    parser.add_argument('--wall-time-limit',type=float,default=14400)
    a=parser.parse_args()
    from .gpu_slot import gpu_slot
    try:
        with gpu_slot(a.gpu_lock,timeout_s=a.queue_timeout,notify=lambda r:print(json.dumps(r),flush=True)):_run(a)
    except Exception as error:
        if a.output.is_dir():
            path=a.output/'report.json';current=json.loads(path.read_text()) if path.exists() else {}
            current.update(status='failed',error=type(error).__name__+': '+str(error));atomic(path,current)
        raise


if __name__=='__main__':main()
