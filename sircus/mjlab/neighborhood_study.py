"""Finite 4090-only local height/tilt search and paired contact timestep study."""
import argparse
from collections import Counter
import copy
import datetime
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np

from . import neighborhood_contract as protocol
from .contact_study import atomic
from .corrected_contract import sha, validate_packages
from .corrected_grid import source_manifest, storage_check
from .learning_contract import digest, reward


def read(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def parent_input(path,kind):
    path=Path(path);spec=read(path/'contract.json');report=read(path/'report.json')
    audit_name='audit-window.json' if kind=='comparison' else 'audit-window-confirmation.json'
    audit=read(path/audit_name);payload=dict(spec);fp=payload.pop('fingerprint',None)
    if digest(payload)!=fp or fp!=report.get('contract_fingerprint') or fp!=audit.get('contract_fingerprint'):
        raise ValueError('Parent fingerprint mismatch: '+kind)
    if report.get('status')!='completed' or audit.get('status')!='passed' or audit.get('failures'):
        raise ValueError('Completed audited parent required: '+kind)
    manifest=read(path/'evidence-manifest.json')
    if manifest!=report.get('evidence_sha256') or any(sha(path/name)!=value for name,value in manifest.items()):
        raise ValueError('Parent evidence changed: '+kind)
    files=('contract.json','report.json',audit_name,'evidence-manifest.json','records.jsonl')
    return spec,lines(path/'records.jsonl'),dict(kind=kind,contract_fingerprint=fp,
        files_sha256={n:sha(path/n) for n in files})


def prepare_inputs(comparison,confirmation):
    parents={k:parent_input(p,k) for k,p in (('comparison',comparison),('confirmation',confirmation))}
    scene=parents['comparison'][0]['scene']
    if parents['confirmation'][0]['scene']!=scene:raise ValueError('Parent scenes differ')
    anchors=[]
    for definition in protocol.ANCHORS:
        source=next(r for r in parents[definition['study']][1] if r['id']==definition['record_id'])
        if not source.get('native_release') or not source['route_success'] or source.get('invalid'):
            raise ValueError('Known reference source was not a native capture')
        expected=[definition['height'],definition['tilt'],definition['gap']]
        if not np.array_equal(source['values'][2:],expected):raise ValueError('Reference parameters differ')
        anchor={k:copy.deepcopy(source[k]) for k in ('values','bucket_xy_m','q10','q14')}
        anchor['id']=definition['id'];anchors.append(anchor)
    if any(not np.array_equal(a['values'][:2],anchors[0]['values'][:2]) or a['bucket_xy_m']!=anchors[0]['bucket_xy_m'] for a in anchors):
        raise ValueError('Fixed board/bucket XY required')
    return copy.deepcopy(scene),anchors,[v[2] for v in parents.values()],parents


def _run(a):
    from .bank_study import run_bank_rows
    from .learning_solve import required_asset_paths
    from .neighborhood_geometry import NeighborhoodGeometry,audit_local_variations
    from .timestep_env import TimestepEnvironment,physics_settings

    scene,anchors,parent_studies,parents=prepare_inputs(a.comparison_study,a.confirmation_study)
    repository=Path(__file__).resolve().parents[2];source=source_manifest(repository)
    for parent,_,_ in parents.values():
        for name,expected in parent['source_sha256'].items():
            if name!='sircus/mjlab/prepare_window_report.py' and source.get(name)!=expected:
                raise ValueError('Parent core source changed: '+name)
    packages=validate_packages({n:importlib.metadata.version(n) for n in parents['comparison'][0]['packages']})
    assets={n:sha(a.assets.parent/n) for n in required_asset_paths(a.assets)}
    if any(p[0]['packages']!=packages or p[0]['asset_sha256']!=assets for p in parents.values()):
        raise ValueError('Parent native packages/assets differ')
    a.output.mkdir(parents=True,exist_ok=False);storage_check(a.output)
    spec=dict(version=protocol.VERSION,settings=protocol.settings(),scene=scene,anchors=anchors,parent_studies=parent_studies,
        source_sha256=source,asset_sha256=assets,packages=packages,batch=a.batch,
        physics_by_substeps={str(s):physics_settings(s) for s in protocol.SUBSTEPS},
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_time_limit_s=a.wall_time_limit,
        scope=protocol.SCOPE)
    spec['fingerprint']=digest(spec);atomic(a.output/'contract.json',spec);atomic(a.output/'scene.json',scene)
    contexts=protocol.contexts();lookup={c['id']:c for c in contexts}
    report=dict(version=protocol.VERSION,status='running',phase='prepare_contexts',contract_fingerprint=spec['fingerprint'],
        settings=spec['settings'],anchors=anchors,scope=protocol.SCOPE,max_requests=protocol.MAX_REQUESTS,counts={},
        contexts=[dict(**c,board_admitted=None,geometry_requests=0,eligible=0,search_attempts=0,search_native_releases=0,
            search_captures=0,clean=0,reason='pending_geometry',best_gap_m=None) for c in contexts],
        selected_recipes=[],trace_examples=[])
    rows=[];batches=[];environments={};options={};sequence=0;started=time.monotonic();positions={}
    def publish(phase,**extra):
        report.update(extra);report['phase']=phase;report['elapsed_s']=time.monotonic()-started
        report['counts']=dict(requests=len(rows),native_releases=sum(bool(r.get('native_release')) for r in rows),
            captures=sum(bool(r['route_success']) for r in rows),clean=sum(r['route_quality']==2 for r in rows),
            geometry_rejections=sum(bool(r.get('invalid')) for r in rows),phase_counts=dict(Counter(r['phase'] for r in rows)),
            substep_counts=dict(Counter(str(r['substeps']) for r in rows)))
        for context in report['contexts']:
            rr=[r for r in rows if r['context_id']==context['id'] and r['role']=='development']
            context.update(geometry_requests=len(rr),eligible=sum(r['eligible'] for r in rr),search_attempts=len(rr),
                search_native_releases=sum(bool(r['native_release']) for r in rr),search_captures=sum(bool(r['route_success']) for r in rr),
                clean=sum(r['route_quality']==2 for r in rr))
            hits=[r for r in rr if r['route_success']]
            context['best_gap_m']=max(hits,key=lambda r:(r['route_quality'],reward(r)))['gap_m'] if hits else None
        report['storage']=storage_check(a.output);atomic(a.output/'report.json',report)
        print(json.dumps(dict(phase=phase,elapsed_s=round(report['elapsed_s'],1),counts=report['counts'])),flush=True)
        if len(rows)>protocol.MAX_REQUESTS or report['elapsed_s']>a.wall_time_limit:
            raise RuntimeError('Finite request/time budget exceeded; no automatic restart')

    geometry=NeighborhoodGeometry(scene,a.assets,anchors,bounds=protocol.BOUNDS)
    prepared=geometry.prepare_contexts(contexts)
    atomic(a.output/'prepared-contexts.json',prepared)
    for c,p in zip(report['contexts'],prepared):
        c.update(board_admitted=p['board_admitted'],reason=p['reason'],source_anchor_id=p['source_anchor_id'])
    publish('contexts_prepared')
    def environment(step):
        if step not in environments:
            environments[step]=TimestepEnvironment(scene,a.assets,n=a.batch,bounds=protocol.BOUNDS,substeps=step)
            options[str(step)]=environments[step].effective_options
            atomic(a.output/'effective-options.json',options);report['effective_native_options']=options
        return environments[step]

    def execute(requests,step,*,audited=None):
        nonlocal sequence
        completed=[]
        for offset in range(0,len(requests),a.batch):
            chunk=copy.deepcopy(requests[offset:offset+a.batch]);packet=len(batches)
            for r in chunk:
                r.update(id=f'neighborhood-shot-{sequence:06d}',substeps=step);sequence+=1
            plan_file=f'plan-{packet:04d}.json';geometry_file=f'geometry-{packet:04d}.json'
            atomic(a.output/plan_file,chunk)
            gates=copy.deepcopy(audited[offset:offset+a.batch]) if audited is not None else geometry.audit([r['context_id'] for r in chunk],[r['gap_m'] for r in chunk])
            atomic(a.output/geometry_file,dict(record_ids=[r['id'] for r in chunk],rows=gates))
            eligible=np.asarray([g['eligible'] for g in gates],bool);indices=np.flatnonzero(eligible);arrays=None;cache_file=None
            # Avoid integrating padded worlds for wholly rejected request batches.
            if len(indices):
                env=environment(step)
                measured=run_bank_rows(env,np.asarray([g['bucket_xy_m'] for g in gates]),np.asarray([g['values'] for g in gates]),
                    np.asarray([g['q10'] for g in gates]),np.asarray([g['q14'] for g in gates]),eligible,record=True)
                if any(bool(r.get('native_release'))!=bool(ok) for r,ok in zip(measured,eligible)):
                    raise RuntimeError('Native release differs from prior strict endpoint gate')
                arrays=env.last_arrays;cache_file=f'native-{packet:04d}.npz'
                np.savez_compressed(a.output/cache_file,**arrays,record_ids=np.asarray([chunk[i]['id'] for i in indices]))
            else:
                from .bank_study import unresolved_row
                measured=[unresolved_row(g['bucket_xy_m']) for g in gates]
            mapping={int(i):j for j,i in enumerate(indices)}
            for i,(request,g,result) in enumerate(zip(chunk,gates,measured)):
                row=dict(result);row.pop('trajectory_m',None);row.update(request)
                row.update({k:g[k] for k in ('values','bucket_xy_m','q10','q14','eligible','board_branch_sha256','minimum_clearance_lower_bound_m')})
                row.update(geometry=g,geometry_file=geometry_file,geometry_row=i,
                    native_cache_file=cache_file if eligible[i] else None,native_cache_row=mapping.get(i))
                if not eligible[i]:row['invalid']=g['reason']
                row['learning_reward']=reward(row)
                if row['native_release'] and row['role']=='engineering_repeat':positions[row['id']]=arrays['position_m'][:,mapping[i]].copy()
                if row['native_release'] and len(report['trace_examples'])<8 and (row['route_success'] or len(report['trace_examples'])<2):
                    report['trace_examples'].append(dict(record_id=row['id'],context_id=row['context_id'],method=row['phase'],
                        substeps=step,gap_m=row['gap_m'],route_success=row['route_success'],position_m=arrays['position_m'][:,mapping[i]][::3].tolist()))
                rows.append(row);completed.append(row)
                with (a.output/'records.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
            batches.append(dict(packet=packet,substeps=step,record_ids=[r['id'] for r in chunk],
                native_record_ids=[chunk[i]['id'] for i in indices],cache_file=cache_file,plan_file=plan_file,geometry_file=geometry_file))
            atomic(a.output/'batches.json',batches);publish(chunk[0]['phase']+'-'+str(step))
        return completed

    controls=[]
    for anchor in anchors:
        c=next(c for c in contexts if np.array_equal([c['board_height_m'],c['tilt_deg']],anchor['values'][2:4]))
        for repeat in range(3):controls.append(dict(role='control',phase='controls',anchor_id=anchor['id'],
            context_id=c['id'],source_family=c['source_family'],gap_m=anchor['values'][4],repeat_index=repeat))
    atomic(a.output/'control-plan.json',controls)
    for step in protocol.SUBSTEPS:
        observed=execute(controls,step)
        if step==64 and any(sum(bool(r['route_success']) for r in observed if r['anchor_id']==anchor['id'])!=3 for anchor in anchors):
            raise RuntimeError('Corrected64 reference did not reproduce3/3; stop to diagnose')
    coarse=protocol.coarse_plan();atomic(a.output/'coarse-plan.json',coarse);execute(coarse,64)
    refinement=[]
    for context in contexts:
        history=[r for r in rows if r['role']=='development' and r['context_id']==context['id']]
        refinement += [dict(role='development',phase='refinement',context_id=context['id'],source_family=context['source_family'],gap_m=gap)
            for gap in protocol.refine_gaps(history)]
    atomic(a.output/'refinement-plan.json',refinement);execute(refinement,64)
    recipes=protocol.select_recipes(rows);atomic(a.output/'recipes.json',recipes)
    repeats=protocol.repeat_plan(recipes);atomic(a.output/'frozen-repeat-plan.json',repeats)
    repeat_gates=[]
    for recipe in recipes:
        requested=[r for r in repeats if r['recipe_id']==recipe['id']]
        repeat_gates += audit_local_variations(scene,a.assets,recipe['source'],requested,bounds=protocol.BOUNDS)
    atomic(a.output/'repeat-geometry.json',dict(requests=repeats,rows=repeat_gates))
    publish('repeat_plan_frozen')
    for step in protocol.SUBSTEPS:execute(repeats,step,audited=repeat_gates)
    for recipe in recipes:
        rr=[r for r in rows if r.get('recipe_id')==recipe['id']]
        result={k:recipe[k] for k in ('id','context_id','board_height_m','tilt_deg','gap_m','selection','sampled_run_width_m')}
        result.update(source_record_id=recipe['source']['id'],release_center_height_m=recipe['board_height_m']+recipe['gap_m']+scene['ball']['radius_m'],
            distance_m=float(np.linalg.norm(np.asarray(recipe['source']['values'][:2])-recipe['source']['bucket_xy_m'])),
            by_substeps={str(s):protocol.summary([r for r in rr if r['substeps']==s]) for s in protocol.SUBSTEPS},
            nominal_by_substeps={str(s):protocol.summary([r for r in rr if r['substeps']==s and r['phase']=='nominal']) for s in protocol.SUBSTEPS},
            fresh_by_substeps={str(s):protocol.summary([r for r in rr if r['substeps']==s and r['phase']=='fresh']) for s in protocol.SUBSTEPS},
            sensitivity=protocol.sensitivity(rr,positions))
        report['selected_recipes'].append(result)
    # Include informative, real repeat trajectories selected for illustration only.
    report['trace_examples']=[]
    for step in protocol.SUBSTEPS:
        for recipe in recipes[:3]:
            row=next(r for r in rows if r.get('recipe_id')==recipe['id'] and r['substeps']==step and r['repeat_index']==0)
            if row['native_release']:report['trace_examples'].append(dict(record_id=row['id'],context_id=recipe['context_id'],method=recipe['id'],
                substeps=step,gap_m=row['gap_m'],route_success=row['route_success'],position_m=positions[row['id']][::3].tolist(),
                scope='Saved nominal anchor-repeat trajectory; illustrative, not aggregate performance'))
    report['status']='completed'
    report['evidence_sha256']={str(p.relative_to(a.output)):sha(p) for p in sorted(a.output.rglob('*'))
        if p.is_file() and p.name not in ('report.json','evidence-manifest.json') and not p.name.startswith('audit')}
    atomic(a.output/'evidence-manifest.json',report['evidence_sha256']);publish('completed')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('comparison-study','confirmation-study','assets','output','gpu-lock'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--batch',type=int,default=64,choices=(32,64,128))
    parser.add_argument('--queue-timeout',type=float,default=7200)
    parser.add_argument('--wall-time-limit',type=float,default=14400)
    a=parser.parse_args();from .gpu_slot import gpu_slot
    try:
        with gpu_slot(a.gpu_lock,timeout_s=a.queue_timeout,notify=lambda r:print(json.dumps(r),flush=True)):_run(a)
    except Exception as error:
        if a.output.is_dir():
            path=a.output/'report.json';report=read(path) if path.exists() else {}
            report.update(status='failed',error=type(error).__name__+': '+str(error));atomic(path,report)
        raise


if __name__=='__main__':main()
