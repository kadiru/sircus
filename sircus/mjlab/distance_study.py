"""Finite seeded distance search and paired repeatability on the authorized4090."""
import argparse
from collections import Counter
import copy
import datetime
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import numpy as np

from . import distance_contract as protocol
from .contact_study import atomic
from .corrected_contract import sha, validate_packages
from .corrected_grid import source_manifest
from .learning_contract import digest, reward


def read(path):return json.loads(Path(path).read_text())


def prepare_inputs(parent):
    parent=Path(parent);spec=read(parent/'contract.json');report=read(parent/'report.json');audit=read(parent/'audit-neighborhood.json')
    payload=dict(spec);fp=payload.pop('fingerprint',None)
    if digest(payload)!=fp or fp!=report.get('contract_fingerprint') or fp!=audit.get('contract_fingerprint'):
        raise ValueError('Parent fingerprint mismatch')
    if report.get('status')!='completed' or audit.get('status')!='passed' or audit.get('failures') or not all(c['passed'] for c in audit['checks']):
        raise ValueError('Completed independently audited parent required')
    manifest=read(parent/'evidence-manifest.json')
    if manifest!=report['evidence_sha256'] or any(sha(parent/name)!=value for name,value in manifest.items()):raise ValueError('Parent evidence changed')
    summaries={r['id']:r for r in report['selected_recipes']};originals=read(parent/'recipes.json');seeds=[];provenance=[]
    for name in protocol.SEED_IDS:
        summary=summaries[name];source=next(r['source'] for r in originals if r['id']==name)
        if not all(summary['by_substeps'][str(s)]['attempts']==23 and summary['by_substeps'][str(s)]['captures']==23 for s in protocol.SUBSTEPS):
            raise ValueError('Every declared parent seed must have perfect paired repeats')
        if source['id']!=summary['source_record_id'] or not source['native_release'] or not source['route_success'] or source.get('invalid'):
            raise ValueError('Parent seed source identity/outcome changed')
        seeds.append(dict(id=name,**{k:copy.deepcopy(source[k]) for k in ('values','bucket_xy_m','q10','q14')}))
        provenance.append(dict(id=name,parent_source_record_id=source['id'],parent_context_id=source['context_id'],geometry_sha256=digest(seeds[-1])))
    files=('contract.json','report.json','audit-neighborhood.json','evidence-manifest.json','recipes.json')
    return copy.deepcopy(spec['scene']),seeds,provenance,dict(contract_fingerprint=fp,files_sha256={n:sha(parent/n) for n in files}),spec


def storage_check(path):
    free=shutil.disk_usage(path).free;used=sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())
    if free<512*1024**2 or used>256*1024**2:raise RuntimeError('Finite study storage budget exceeded; no files deleted')
    return dict(free_bytes=free,study_bytes=used,maximum_study_bytes=256*1024**2,minimum_free_bytes=512*1024**2)


def _run(a):
    from .bank_study import run_bank_rows,unresolved_row
    from .distance_geometry import DistanceGeometry,audit_distance_variations
    from .learning_solve import required_asset_paths
    from .timestep_env import TimestepEnvironment,physics_settings
    scene,seeds,provenance,parent,parent_spec=prepare_inputs(a.parent_study)
    repository=Path(__file__).resolve().parents[2];source=source_manifest(repository)
    if any(source.get(n)!=v for n,v in parent_spec['source_sha256'].items()):raise ValueError('Parent source changed')
    packages=validate_packages({n:importlib.metadata.version(n) for n in parent_spec['packages']})
    assets={n:sha(a.assets.parent/n) for n in required_asset_paths(a.assets)}
    if assets!=parent_spec['asset_sha256'] or packages!=parent_spec['packages']:raise ValueError('Native assets/packages differ from parent')
    a.output.mkdir(parents=True,exist_ok=False);storage_check(a.output)
    settings=protocol.settings(scene,seeds);rays,targets=protocol.targets(scene,seeds);declared=protocol.contexts(seeds,targets)
    spec=dict(version=protocol.VERSION,settings=settings,scene=scene,seeds=seeds,seed_provenance=provenance,parent_study=parent,
        source_sha256=source,asset_sha256=assets,packages=packages,batch=a.batch,physics_by_substeps={str(s):physics_settings(s) for s in protocol.SUBSTEPS},
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_time_limit_s=a.wall_time_limit,scope=protocol.SCOPE)
    spec['fingerprint']=digest(spec);atomic(a.output/'contract.json',spec);atomic(a.output/'scene.json',scene)
    report=dict(version=protocol.VERSION,status='running',phase='prepare_contexts',contract_fingerprint=spec['fingerprint'],settings=settings,
        targets=copy.deepcopy(targets),contexts=[],selected_recipes=[],trace_examples=[],scope=protocol.SCOPE,max_requests=protocol.MAX_REQUESTS,counts={})
    rows=[];batches=[];environments={};options={};sequence=0;positions={};started=time.monotonic()
    def publish(phase,**extra):
        report.update(extra);report.update(phase=phase,elapsed_s=time.monotonic()-started)
        report['counts']=dict(requests=len(rows),native_releases=sum(bool(r.get('native_release')) for r in rows),captures=sum(bool(r['route_success']) for r in rows),
            clean=sum(r['route_quality']==2 for r in rows),geometry_rejections=sum(bool(r.get('invalid')) for r in rows),
            phase_counts=dict(Counter(r['phase'] for r in rows)),substep_counts=dict(Counter(str(r['substeps']) for r in rows)))
        for c in report['contexts']:
            rr=[r for r in rows if r['context_id']==c['id'] and r['role']=='development'];hits=[r for r in rr if r['route_success']]
            c.update(geometry_requests=len(rr),eligible=sum(r['eligible'] for r in rr),search_attempts=len(rr),search_native_releases=sum(bool(r['native_release']) for r in rr),
                search_captures=len(hits),clean=sum(r['route_quality']==2 for r in rr),best_gap_m=max(hits,key=lambda r:(r['route_quality'],reward(r)))['gap_m'] if hits else None)
        for t in report['targets']:
            rr=[r for r in rows if r['distance_id']==t['id']];search=[r for r in rr if r['role']=='development']
            t['counts']=dict(requests=len(rr),native_releases=sum(bool(r['native_release']) for r in rr),captures=sum(bool(r['route_success']) for r in rr),
                geometry_rejections=sum(bool(r.get('invalid')) for r in rr),search_attempts=len(search),search_native_releases=sum(bool(r['native_release']) for r in search),search_captures=sum(bool(r['route_success']) for r in search))
        report['storage']=storage_check(a.output);atomic(a.output/'report.json',report)
        print(json.dumps(dict(phase=phase,elapsed_s=round(report['elapsed_s'],1),counts=report['counts'])),flush=True)
        if len(rows)>protocol.MAX_REQUESTS or report['elapsed_s']>a.wall_time_limit:raise RuntimeError('Finite request/time budget exceeded')
    geometry=DistanceGeometry(scene,a.assets,seeds,bounds=protocol.BOUNDS);prepared=[]
    def prepare_contexts(contexts):
        prepared.extend(geometry.prepare_contexts(contexts));atomic(a.output/'declared-contexts.json',declared)
        atomic(a.output/'prepared-contexts.json',prepared)
        for c,p in zip(contexts,prepared[-len(contexts):]):
            report['contexts'].append(dict(**c,board_admitted=p['board_admitted'],reason=p.get('reason'),board_branch_sha256=p.get('board_branch_sha256'),
                geometry_requests=0,eligible=0,search_attempts=0,search_native_releases=0,search_captures=0,clean=0,best_gap_m=None))
    prepare_contexts(declared);publish('contexts_prepared')
    def environment(step):
        if step not in environments:
            environments[step]=TimestepEnvironment(scene,a.assets,n=a.batch,bounds=protocol.BOUNDS,substeps=step)
            options[str(step)]=environments[step].effective_options;atomic(a.output/'effective-options.json',options);report['effective_native_options']=options
        return environments[step]
    def execute(requests,step,*,audited=None):
        nonlocal sequence
        completed=[]
        for offset in range(0,len(requests),a.batch):
            chunk=copy.deepcopy(requests[offset:offset+a.batch]);packet=len(batches)
            for r in chunk:r.update(id=f'distance-shot-{sequence:06d}',substeps=step);sequence+=1
            plan_file=f'plan-{packet:04d}.json';geo_file=f'geometry-{packet:04d}.json';atomic(a.output/plan_file,chunk)
            gates=copy.deepcopy(audited[offset:offset+a.batch]) if audited is not None else geometry.audit([r['context_id'] for r in chunk],[r['gap_m'] for r in chunk])
            atomic(a.output/geo_file,dict(record_ids=[r['id'] for r in chunk],rows=gates));eligible=np.asarray([g['eligible'] for g in gates],bool)
            indices=np.flatnonzero(eligible);arrays=None;cache=None
            if len(indices):
                env=environment(step);measured=run_bank_rows(env,np.asarray([g['bucket_xy_m'] for g in gates]),np.asarray([g['values'] for g in gates]),
                    np.asarray([g['q10'] for g in gates]),np.asarray([g['q14'] for g in gates]),eligible,record=True)
                if any(bool(r.get('native_release'))!=bool(ok) for r,ok in zip(measured,eligible)):raise RuntimeError('Native admission differs from strict geometry')
                arrays=env.last_arrays;cache=f'native-{packet:04d}.npz';np.savez_compressed(a.output/cache,**arrays,record_ids=np.asarray([chunk[i]['id'] for i in indices]))
            else:measured=[unresolved_row(g['bucket_xy_m']) for g in gates]
            mapping={int(i):j for j,i in enumerate(indices)}
            for i,(request,g,result) in enumerate(zip(chunk,gates,measured)):
                row=dict(result);row.pop('trajectory_m',None);row.update(request)
                row.update({k:g[k] for k in ('values','bucket_xy_m','q10','q14','eligible','board_branch_sha256','minimum_clearance_lower_bound_m')})
                row.update(geometry=g,geometry_file=geo_file,geometry_row=i,native_cache_file=cache if eligible[i] else None,native_cache_row=mapping.get(i))
                if not eligible[i]:row['invalid']=g['reason']
                row['learning_reward']=reward(row)
                if row['native_release'] and row['role']=='engineering_repeat':positions[row['id']]=arrays['position_m'][:,mapping[i]].copy()
                if row['native_release'] and len(report['trace_examples'])<8:
                    report['trace_examples'].append(dict(record_id=row['id'],context_id=row['context_id'],distance_id=row['distance_id'],ray_id=row['ray_id'],distance_m=row['distance_m'],
                        board_xy_m=row['values'][:2],bucket_xy_m=row['bucket_xy_m'],method=row['phase'],substeps=step,gap_m=row['gap_m'],route_success=row['route_success'],position_m=arrays['position_m'][:,mapping[i]][::3].tolist(),scope='Saved native illustration; aggregate metrics authoritative'))
                rows.append(row);completed.append(row)
                with (a.output/'records.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
            batches.append(dict(packet=packet,substeps=step,record_ids=[r['id'] for r in chunk],native_record_ids=[chunk[i]['id'] for i in indices],cache_file=cache,plan_file=plan_file,geometry_file=geo_file))
            atomic(a.output/'batches.json',batches);publish(chunk[0]['phase']+'-'+str(step))
        return completed
    controls=protocol.control_plan(seeds,targets);atomic(a.output/'control-plan.json',controls);observed=execute(controls,64)
    for seed in seeds:
        rr=[r for r in observed if r['source_anchor_id']==seed['id']]
        if len(rr)!=3 or not all(r['route_success'] and all(r[k]==seed[k] for k in ('values','bucket_xy_m','q10','q14')) for r in rr):raise RuntimeError('Exact source control did not reproduce3/3')
    coarse=protocol.coarse_plan(declared);atomic(a.output/'coarse-plan.json',coarse);coarse_rows=execute(coarse,64)
    extension=protocol.extension_contexts(seeds,targets,coarse_rows);atomic(a.output/'extension-contexts.json',extension)
    if extension:declared.extend(extension);prepare_contexts(extension)
    extra=protocol.extension_plan(extension);atomic(a.output/'extension-plan.json',extra);execute(extra,64)
    initial=[r for r in rows if r['role']=='development'];refinement=protocol.refinement_plan(declared,initial)
    atomic(a.output/'refinement-plan.json',refinement);execute(refinement,64)
    recipes=protocol.select_recipes([r for r in rows if r['role']=='development'],targets);atomic(a.output/'recipes.json',recipes)
    repeats=protocol.repeat_plan(recipes);atomic(a.output/'frozen-repeat-plan.json',repeats);repeat_gates=[]
    for recipe in recipes:
        repeat_gates.extend(audit_distance_variations(scene,a.assets,recipe['source'],[r for r in repeats if r['recipe_id']==recipe['id']],bounds=protocol.BOUNDS))
    atomic(a.output/'repeat-geometry.json',dict(requests=repeats,rows=repeat_gates));publish('repeat_plan_frozen')
    # Retain effective options at each declared resolution even if a finite
    # search finds no repeatable recipe; no flight is inferred from allocation.
    for step in protocol.SUBSTEPS:
        environment(step);execute(repeats,step,audited=repeat_gates)
    for recipe in recipes:
        rr=[r for r in rows if r.get('recipe_id')==recipe['id']]
        result={k:recipe[k] for k in ('id','context_id','distance_id','ray_id','distance_m','bucket_xy_m','board_xy_m','board_height_m','tilt_deg','gap_m','selection','sampled_run_width_m')}
        result.update(source_record_id=recipe['source']['id'],release_center_height_m=recipe['board_height_m']+recipe['gap_m']+scene['ball']['radius_m'],
            by_substeps={str(s):protocol.summary([r for r in rr if r['substeps']==s]) for s in protocol.SUBSTEPS},
            nominal_by_substeps={str(s):protocol.summary([r for r in rr if r['substeps']==s and r['phase']=='nominal']) for s in protocol.SUBSTEPS},
            fresh_by_substeps={str(s):protocol.summary([r for r in rr if r['substeps']==s and r['phase']=='fresh']) for s in protocol.SUBSTEPS},sensitivity=protocol.sensitivity(rr,positions))
        report['selected_recipes'].append(result)
    report['trace_examples']=[]
    for recipe in recipes:
        for step in (64,128):
            row=next(r for r in rows if r.get('recipe_id')==recipe['id'] and r['substeps']==step and r['repeat_index']==0)
            if row['native_release']:report['trace_examples'].append(dict(record_id=row['id'],context_id=row['context_id'],source_context_id=recipe['context_id'],distance_id=recipe['distance_id'],ray_id=recipe['ray_id'],distance_m=recipe['distance_m'],
                board_xy_m=recipe['board_xy_m'],bucket_xy_m=recipe['bucket_xy_m'],method=recipe['id'],substeps=step,gap_m=row['gap_m'],route_success=row['route_success'],position_m=positions[row['id']][::3].tolist(),scope='Recorded nominal selected-recipe illustration; not aggregate performance'))
    report['status']='completed';report['evidence_sha256']={str(p.relative_to(a.output)):sha(p) for p in sorted(a.output.rglob('*')) if p.is_file() and p.name not in ('report.json','evidence-manifest.json') and not p.name.startswith('audit')}
    atomic(a.output/'evidence-manifest.json',report['evidence_sha256']);publish('completed');return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('parent-study','assets','output','gpu-lock'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--batch',type=int,default=64,choices=(32,64,128));parser.add_argument('--queue-timeout',type=float,default=14400)
    parser.add_argument('--wall-time-limit',type=float,default=14400);a=parser.parse_args()
    from .gpu_slot import gpu_slot
    try:
        with gpu_slot(a.gpu_lock,timeout_s=a.queue_timeout,notify=lambda r:print(json.dumps(r),flush=True)):_run(a)
    except Exception as error:
        if a.output.is_dir():
            path=a.output/'report.json';report=read(path) if path.exists() else {};report.update(status='failed',error=type(error).__name__+': '+str(error));atomic(path,report)
        raise


if __name__=='__main__':main()
