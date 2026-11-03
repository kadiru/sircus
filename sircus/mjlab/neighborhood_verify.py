"""Independent CPU audit of local height/tilt search and paired timestep evidence.

Saved traces are rescored without advancing dynamics. Original parents, request
denominators and frozen joint arrays are checked separately from capture outcomes.
"""
import argparse
from collections import Counter, defaultdict
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

from . import neighborhood_contract as protocol
from .verify_expansion import Audit, digest, sha256
from .verify_contact import SCORER_KEYS, NUMERIC_SCORER_KEYS
from .window_verify import row_reward

VERSION = 'sircus-local-height-tilt-evidence-audit-v1'


def read(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def inside(root,name):
    root=Path(root).resolve();path=(root/name).resolve()
    if not path.is_relative_to(root):raise ValueError('Evidence path escapes its declared root')
    return path


def statistics(rows):
    return dict(attempts=len(rows),captures=sum(bool(r['route_success']) and not r.get('invalid') for r in rows),
        clean=sum(r['route_quality']==2 and not r.get('invalid') for r in rows),
        invalid=sum(bool(r.get('invalid')) for r in rows),board_bounces=sum(bool(r['board_bounce']) and not r.get('invalid') for r in rows),
        mean_reward=float(np.mean([row_reward(r) for r in rows])) if rows else 0.,
        native_releases=sum(bool(r.get('native_release')) for r in rows))


def audit_native(a,scene,rows,arrays,substeps,label):
    """Full scalar-score reconstruction at the unchanged240Hz observation rate."""
    from .contract import FORCE_NAMES, ScoreStream
    actual_ids=list(map(str,arrays.get('record_ids',[])));expected_ids=[r['id'] for r in rows]
    a.check(label+':record_ids',actual_ids==expected_ids and len(actual_ids)==len(set(actual_ids)))
    good=all(k in arrays and arrays[k].shape==(960,len(rows),width) and np.isfinite(arrays[k]).all()
        for k,width in (('position_m',3),('velocity_m_s',3),('forces_N',6)))
    a.check(label+':finite_960_sample_arrays',good)
    if not good or actual_ids!=expected_ids:return
    a.check(label+':paired_substep_protocol',substeps in (32,64,128) and all(r.get('native_release') is True and not r.get('invalid')
        and r.get('substeps')==substeps and r.get('contact_protocol')==dict(substeps=substeps,board_mode='articulated',gravity_mode='passive_only') for r in rows))
    a.check(label+':nonnegative_force_categories',bool((arrays['forces_N']>=0).all()))
    bad=[]
    for i,row in enumerate(rows):
        scorer=ScoreStream(scene,row['bucket_xy_m'],row['board_normal'])
        for p,v,f in zip(arrays['position_m'][:,i],arrays['velocity_m_s'][:,i],arrays['forces_N'][:,i]):
            scorer.step(p,v,dict(zip(FORCE_NAMES,f)))
            if scorer.finished:break
        actual=scorer.result()
        discrete=all(actual[k]==row.get(k) for k in SCORER_KEYS)
        numeric=all((actual[k] is None and row.get(k) is None) or (actual[k] is not None and row.get(k) is not None
            and np.isclose(actual[k],row[k],atol=1e-3,rtol=1e-5)) for k in NUMERIC_SCORER_KEYS)
        if not discrete or not numeric:bad.append(row['id'])
    a.check(label+':independent_scalar_rescore',not bad,bad[:10])


def audit_contexts(a,spec,prepared):
    expected=protocol.contexts();lookup={c['id']:c for c in expected};anchors=spec['anchors']
    a.check('all_75_local_contexts_including_unsupported',len(prepared)==75 and Counter(c['id'] for c in prepared)==Counter(lookup.keys()))
    bad=[]
    from .admission import VERIFIED_PARK_Q10
    for row in prepared:
        c=lookup.get(row.get('id'))
        if c is None or any(row.get(k)!=c[k] for k in ('board_height_m','tilt_deg')):bad.append(row.get('id'));continue
        nearest=min(anchors,key=lambda anchor:(float(np.sum(((np.asarray([c['board_height_m'],c['tilt_deg']])-np.asarray(anchor['values'][2:4]))/[.04,4.])**2)),anchor['id']))
        if row.get('source_anchor_id')!=nearest['id'] or row.get('source_anchor_geometry_sha256')!=digest(nearest):bad.append(c['id'])
        if row.get('board_admitted'):
            q=np.asarray(row.get('q14',[]),float)
            proof=dict(version='sircus-fixed-board-gap-window-geometry-v1',scene_sha256=digest(spec['scene']),
                bucket_xy_m=spec['settings']['bucket_xy_m'],board_world=row.get('board_world'),q14=row.get('q14'),park_q10=list(VERIFIED_PARK_Q10))
            if q.shape!=(6,) or not np.isfinite(q).all() or digest(proof)!=row.get('board_branch_sha256'):bad.append(c['id'])
            if [c['board_height_m'],c['tilt_deg']]==nearest['values'][2:4] and row['q14']!=nearest['q14']:bad.append(c['id'])
        elif row.get('q14') is not None or row.get('board_branch_sha256') is not None:bad.append(c['id'])
    a.check('geometry_only_nearest_anchor_and_frozen_branch',not bad,bad[:10])


def audit_rows(a,spec,prepared,rows,recipes,repeat_plan,repeat_geometry):
    contexts={r['id']:r for r in prepared};selected={r['id']:r for r in recipes}
    repeat_lookup={r['repeat_id']:(r,g) for r,g in zip(repeat_geometry['requests'],repeat_geometry['rows'])}
    a.check('all_rows_have_unique_sequential_ids',[r['id'] for r in rows]==[f'neighborhood-shot-{i:06d}' for i in range(len(rows))])
    a.check('frozen_repeat_geometry_request_identity',repeat_geometry['requests']==repeat_plan and len(repeat_lookup)==len(repeat_plan)==len(repeat_geometry['rows']))
    bad=[];bad_reward=[];bad_release=[];paired=defaultdict(dict)
    for r in rows:
        values=np.asarray(r.get('values',[]),float);g=r.get('geometry',{})
        if values.shape!=(5,) or not np.isfinite(values).all() or any(r.get(k)!=g.get(k) for k in ('values','bucket_xy_m','q10','q14','eligible','board_branch_sha256','minimum_clearance_lower_bound_m')):
            bad.append(r['id']);continue
        if not np.array_equal(values[:2],spec['settings']['board_xy_m']) or r['bucket_xy_m']!=spec['settings']['bucket_xy_m'] or values[4]!=r['gap_m']:bad.append(r['id'])
        if r['role'] in ('control','development'):
            c=contexts.get(r['context_id'])
            if c is None or values[2]!=c['board_height_m'] or values[3]!=c['tilt_deg'] or r['board_branch_sha256']!=c['board_branch_sha256'] or (c['board_admitted'] and r['q14']!=c['q14']):bad.append(r['id'])
            if c is not None and (g.get('wrapper_version')!='sircus-multi-anchor-height-tilt-geometry-v1' or g.get('source_anchor_id')!=c.get('source_anchor_id')
                    or g.get('source_anchor_geometry_sha256')!=c.get('source_anchor_geometry_sha256')):bad.append(r['id'])
        elif r['role']=='engineering_repeat':
            matched=repeat_lookup.get(r.get('repeat_id'));recipe=selected.get(r.get('recipe_id'))
            if matched is None or recipe is None or any(r.get(k)!=v for k,v in matched[0].items()) or g!=matched[1]:bad.append(r['id']);continue
            nominal_geometry={k:recipe['source'][k] for k in ('values','bucket_xy_m','q10','q14')}
            if g.get('wrapper_version')!='sircus-nominal-branch-local-variation-v1' or g.get('nominal_geometry_sha256')!=digest(nominal_geometry):bad.append(r['id'])
            if r['eligible'] and any(np.max(np.abs(np.asarray(r[k])-recipe['source'][k]))>.10 for k in ('q10','q14')):bad.append(r['id'])
            if r['phase']=='nominal' and any(r[k]!=recipe['source'][k] for k in ('q10','q14','values')):bad.append(r['id'])
            paired[r['repeat_id']][r['substeps']]=digest(dict(values=r['values'],q10=r['q10'],q14=r['q14'],geometry=g))
        else:bad.append(r['id'])
        if r.get('native_release')!=bool(r.get('eligible')) or (not r.get('eligible') and (not r.get('invalid') or r['route_success'])):bad_release.append(r['id'])
        if r.get('eligible') and (r.get('minimum_clearance_lower_bound_m') is None or r['minimum_clearance_lower_bound_m']<.005 or r.get('invalid')
                or not ((values>=spec['settings']['bounds']['low']).all() and (values<=spec['settings']['bounds']['high']).all())):bad_release.append(r['id'])
        if r['route_success'] and (not r['board_bounce'] or not r['whole_ball_entry'] or r['max_retention_s']<1-1e-9 or r['external_contact_steps']):bad_release.append(r['id'])
        if not np.isclose(r.get('learning_reward',np.nan),row_reward(r),rtol=0,atol=1e-9):bad_reward.append(r['id'])
    a.check('exact_fixed_xy_request_geometry_and_local_branch_identity',not bad,bad[:10])
    a.check('every_rejected_request_remains_failed_and_every_release_is_admitted',not bad_release,bad_release[:10])
    a.check('unchanged_terminal_reward',not bad_reward,bad_reward[:10])
    a.check('same_joint_geometry_at_all_three_timesteps',len(paired)==len(repeat_plan) and all(set(v)=={32,64,128} and len(set(v.values()))==1 for v in paired.values()))


def audit_plans(a,study,spec,rows,recipes):
    controls=[r for r in rows if r['role']=='control'];coarse=[r for r in rows if r['phase']=='coarse'];refined=[r for r in rows if r['phase']=='refinement']
    expected_controls=[]
    for anchor in spec['anchors']:
        context=next(c for c in protocol.contexts() if [c['board_height_m'],c['tilt_deg']]==anchor['values'][2:4])
        for index in range(3):expected_controls.append(dict(role='control',phase='controls',anchor_id=anchor['id'],context_id=context['id'],source_family=context['source_family'],gap_m=anchor['values'][4],repeat_index=index))
    a.check('control_plan_frozen_exact_nine_requests',read(study/'control-plan.json')==expected_controls)
    anchors={r['id']:r for r in spec['anchors']}
    a.check('all_27_paired_anchor_controls_with_exact_source_joints',Counter((r['anchor_id'],r['substeps'],r['repeat_index']) for r in controls)==Counter((anchor,step,i) for anchor in anchors for step in (32,64,128) for i in range(3))
        and all(all(r[k]==anchors[r['anchor_id']][k] for k in ('values','bucket_xy_m','q10','q14')) for r in controls))
    a.check('trusted64_controls_reproduce_three_captures_each',all(sum(bool(r['route_success']) for r in controls if r['anchor_id']==anchor and r['substeps']==64)==3 for anchor in anchors))
    expected_coarse=protocol.coarse_plan()
    a.check('complete_1275_coarse_requests',read(study/'coarse-plan.json')==expected_coarse and len(coarse)==1275
        and all(all(r.get(k)==v for k,v in p.items()) and r['substeps']==64 for r,p in zip(coarse,expected_coarse)))
    expected_refinement=[]
    for context in protocol.contexts():
        history=[r for r in coarse if r['context_id']==context['id']]
        for gap in protocol.refine_gaps(history):expected_refinement.append(dict(role='development',phase='refinement',context_id=context['id'],source_family=context['source_family'],gap_m=gap))
    a.check('all_and_only_six_refinements_per_captured_coarse_context',read(study/'refinement-plan.json')==expected_refinement and len(refined)==len(expected_refinement)<=450
        and all(all(r.get(k)==v for k,v in p.items()) and r['substeps']==64 for r,p in zip(refined,expected_refinement)))
    candidates=[r for r in rows if r['role'] in ('control','development')]
    expected_recipes=protocol.select_recipes(candidates)
    a.check('recipes_selected_only_from_completed_development_and_controls',recipes==expected_recipes and 3<=len(recipes)<=12
        and len({r['context_id'] for r in recipes})==len(recipes) and all(r['source']['route_success'] and r['source']['native_release'] for r in recipes))
    expected_repeat=protocol.repeat_plan(recipes)
    a.check('all_seeded_joint_noise_requests_frozen_before_outcomes',read(study/'frozen-repeat-plan.json')==expected_repeat)
    repeat_rows=[r for r in rows if r['role']=='engineering_repeat']
    a.check('no_retry_resampling_or_silent_budget_reduction',len(rows)==27+1275+len(expected_refinement)+len(expected_repeat)*3<=2580
        and Counter((r['repeat_id'],r['substeps']) for r in repeat_rows)==Counter((r['repeat_id'],step) for r in expected_repeat for step in (32,64,128))
        and set(r['phase'] for r in rows)<=set(('controls','coarse','refinement','nominal','fresh')))


def audit_batches(a,study,spec,rows):
    lookup={r['id']:r for r in rows};ordered=[];released=[];positions={}
    for batch in read(study/'batches.json'):
        label='batch-'+str(batch['packet']);ids=batch['record_ids'];ordered.extend(ids)
        plan=read(inside(study,batch['plan_file']));geometry=read(inside(study,batch['geometry_file']))
        a.check(label+':outcome_free_plan_and_geometry_ids',[p['id'] for p in plan]==ids==geometry['record_ids']
            and len(geometry['rows'])==len(ids) and all(i in lookup for i in ids)
            and all(all(lookup[p['id']].get(k)==v for k,v in p.items()) for p in plan))
        for index,name in enumerate(ids):
            row=lookup[name]
            a.check(label+':geometry-'+str(index),row['geometry']==geometry['rows'][index] and row['geometry_file']==batch['geometry_file'] and row['geometry_row']==index)
        native=[lookup[i] for i in ids if lookup[i].get('native_release')]
        a.check(label+':released_only_cache_membership',[r['id'] for r in native]==batch['native_record_ids'])
        a.check(label+':cache_pointers',all(r['native_cache_file']==batch['cache_file'] and r['native_cache_row']==i for i,r in enumerate(native))
            and all(lookup[i]['native_cache_file'] is None and lookup[i]['native_cache_row'] is None for i in ids if not lookup[i].get('native_release')))
        if native:
            with np.load(inside(study,batch['cache_file']),allow_pickle=False) as arrays:
                audit_native(a,spec['scene'],native,arrays,batch['substeps'],label)
                for i,row in enumerate(native):positions[row['id']]=arrays['position_m'][:,i].copy()
            released.extend(r['id'] for r in native)
        else:a.check(label+':no_empty_native_cache',batch['cache_file'] is None)
    a.check('all_attempts_and_native_releases_accounted_once',ordered==[r['id'] for r in rows]
        and Counter(released)==Counter(r['id'] for r in rows if r.get('native_release')))
    return positions,len(released)


def audit_options(a,study,report):
    from .timestep_env import validate_options
    options=read(study/'effective-options.json')
    a.check('all_effective_options_retained',set(options)=={'32','64','128'} and options==report.get('effective_native_options'))
    for step in (32,64,128):
        try:
            claim=options[str(step)];w=claim['warp']
            native=NS(timestep=claim['timestep_s'],tolerance=claim['tolerance'],gravity=claim['gravity'],
                solver=claim['solver'],integrator=claim['integrator'],cone=claim['cone'],iterations=claim['iterations'])
            warp=NS(timestep=np.asarray(w['timestep_s'],np.float32),tolerance=np.asarray(w['tolerance'],np.float32),gravity=np.asarray(w['gravity'],np.float32),
                solver=w['solver'],integrator=w['integrator'],cone=w['cone'],iterations=w['iterations'])
            actual=validate_options(native,warp,step)
            passed=actual==claim
        except (KeyError,ValueError,TypeError,RuntimeError) as error:passed=False;claim=str(error)
        a.check('actual_native_and_warp_options_'+str(step),passed,claim if not passed else None)


def audit_sources(a,study,source,assets,spec,report,comparison_study,confirmation_study):
    source=Path(source);assets=Path(assets)
    for name,expected in spec['source_sha256'].items():a.check('source:'+name,sha256(inside(source,name))==expected)
    for name,expected in spec['asset_sha256'].items():a.check('asset:'+name,sha256(inside(assets.parent,name))==expected)
    own=Path(__file__).resolve().parents[2]
    for name in ('sircus/mjlab/neighborhood_contract.py','sircus/mjlab/neighborhood_geometry.py','sircus/mjlab/timestep_env.py',
            'sircus/mjlab/contact_env.py','sircus/mjlab/contact_model.py','sircus/mjlab/contract.py','sircus/mjlab/learning_contract.py',
            'sircus/isaac/geometry.py','sircus/isaac/rl_contract.py','sircus/isaac/entry_quality.py','sircus/isaac/board_route.py'):
        a.check('matching_independent_logic:'+name,sha256(own/name)==spec['source_sha256'][name])
    parent_paths=dict(comparison=Path(comparison_study),confirmation=Path(confirmation_study));parent_rows={};parent_specs={}
    a.check('both_original_parent_namespaces',Counter(p['kind'] for p in spec['parent_studies'])==dict(comparison=1,confirmation=1))
    for parent in spec['parent_studies']:
        root=parent_paths[parent['kind']];ps=read(root/'contract.json');pr=read(root/'report.json')
        audit_name='audit-window.json' if parent['kind']=='comparison' else 'audit-window-confirmation.json';pa=read(root/audit_name)
        payload=dict(ps);fp=payload.pop('fingerprint',None)
        a.check('parent:'+parent['kind'],digest(payload)==fp==parent['contract_fingerprint']==pr['contract_fingerprint']==pa['contract_fingerprint']
            and pr['status']=='completed' and pa['status']=='passed' and not pa.get('failures')
            and all(sha256(inside(root,n))==h for n,h in parent['files_sha256'].items()))
        manifest=read(root/'evidence-manifest.json')
        a.check('parent_immutable_evidence:'+parent['kind'],manifest==pr['evidence_sha256'] and all(sha256(inside(root,n))==h for n,h in manifest.items()))
        a.check('parent_core_source_compatibility:'+parent['kind'],all(name=='sircus/mjlab/prepare_window_report.py' or spec['source_sha256'].get(name)==expected
            for name,expected in ps['source_sha256'].items()))
        parent_rows[parent['kind']]={r['id']:r for r in lines(root/'records.jsonl')};parent_specs[parent['kind']]=ps
    expected=[]
    for definition in protocol.ANCHORS:
        row=parent_rows[definition['study']][definition['record_id']]
        expected.append(dict(id=definition['id'],**{k:row[k] for k in ('values','bucket_xy_m','q10','q14')}))
        a.check('known_native_capture_geometry:'+definition['id'],row['native_release'] and row['route_success'] and not row.get('invalid')
            and row['values'][2:]==[definition['height'],definition['tilt'],definition['gap']])
    a.check('geometry_only_anchors_from_exact_preserved_parent_records',expected==spec['anchors']
        and all(set(r)=={'id','values','bucket_xy_m','q10','q14'} for r in spec['anchors'])
        and all(spec['scene']==p['scene'] and spec['asset_sha256']==p['asset_sha256'] and spec['packages']==p['packages'] for p in parent_specs.values()))
    for name,version in spec['packages'].items():
        try:actual=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:actual=None
        a.check('package:'+name,actual==version)
    manifest=read(study/'evidence-manifest.json');files={str(p.relative_to(study)) for p in study.rglob('*') if p.is_file()
        and p.name not in ('report.json','evidence-manifest.json') and not p.name.startswith('audit')}
    a.check('complete_saved_evidence_manifest',set(manifest)==files and manifest==report.get('evidence_sha256'))
    for name,expected in manifest.items():a.check('evidence:'+name,sha256(inside(study,name))==expected)


def audit_report(a,spec,report,rows,recipes,positions):
    counts=report['counts'];expected=dict(requests=len(rows),native_releases=sum(bool(r['native_release']) for r in rows),
        captures=sum(bool(r['route_success']) for r in rows),clean=sum(r['route_quality']==2 for r in rows),
        geometry_rejections=sum(bool(r.get('invalid')) for r in rows),phase_counts=dict(Counter(r['phase'] for r in rows)),substep_counts=dict(Counter(str(r['substeps']) for r in rows)))
    a.check('reported_complete_request_and_release_denominators',counts==expected)
    a.check('all_75_reported_contexts',Counter(c['id'] for c in report['contexts'])==Counter(c['id'] for c in protocol.contexts()))
    for c in report['contexts']:
        rr=[r for r in rows if r['context_id']==c['id'] and r['role']=='development']
        a.check('context_counts:'+c['id'],all(c[k]==v for k,v in dict(geometry_requests=len(rr),eligible=sum(r['eligible'] for r in rr),search_attempts=len(rr),
            search_native_releases=sum(bool(r['native_release']) for r in rr),search_captures=sum(bool(r['route_success']) for r in rr),clean=sum(r['route_quality']==2 for r in rr)).items()))
    a.check('selected_recipe_report_identity',[r['id'] for r in report['selected_recipes']]==[r['id'] for r in recipes])
    for claim,recipe in zip(report['selected_recipes'],recipes):
        rr=[r for r in rows if r.get('recipe_id')==recipe['id']]
        a.check(recipe['id']+':source_identity',claim['source_record_id']==recipe['source']['id'] and all(claim[k]==recipe[k] for k in ('context_id','board_height_m','tilt_deg','gap_m','selection','sampled_run_width_m')))
        for step in (32,64,128):
            for key,phase in (('by_substeps',None),('nominal_by_substeps','nominal'),('fresh_by_substeps','fresh')):
                selected=[r for r in rr if r['substeps']==step and (phase is None or r['phase']==phase)]
                a.stats(recipe['id']+':'+str(step)+':'+key,claim[key][str(step)],statistics(selected))
        paired=defaultdict(dict)
        for r in rr:paired[r['repeat_id']][r['substeps']]=r
        agreements=sum(bool(pair[64]['route_success'])==bool(pair[128]['route_success']) for pair in paired.values())
        deltas=[float(np.linalg.norm(positions[pair[64]['id']][83]-positions[pair[128]['id']][83])) for pair in paired.values()
            if pair[64]['phase']=='nominal' and pair[64]['native_release'] and pair[128]['native_release']]
        passed=(len(paired)==23 and agreements>=21 and len(deltas)==3 and max(deltas)<=.005 and all(
            sum(bool(r['route_success']) for r in rr if r['substeps']==step and r['phase']=='nominal')==3
            and sum(bool(r['route_success']) for r in rr if r['substeps']==step and r['phase']=='fresh')>=18 for step in (64,128)))
        expected=dict(paired_attempts=len(paired),capture_agreement_64_128=agreements,nominal_position_delta_350ms_max_m=max(deltas) if deltas else None,paired_engineering_gate=passed)
        a.check(recipe['id']+':independent_paired_sensitivity',all(claim['sensitivity'].get(k)==v for k,v in expected.items()))


def audit(study,source,assets,comparison_study,confirmation_study):
    study=Path(study);spec=read(study/'contract.json');report=read(study/'report.json');rows=lines(study/'records.jsonl')
    a=Audit();payload=dict(spec);fp=payload.pop('fingerprint',None)
    a.check('contract_identity',digest(payload)==fp==report.get('contract_fingerprint'))
    a.check('completed_version',spec.get('version')==report.get('version')==protocol.VERSION and report.get('status')==report.get('phase')=='completed')
    a.check('prespecified_settings_and_scope',spec['settings']==json.loads(json.dumps(protocol.settings()))==report['settings'] and spec['scope']==report['scope']==protocol.SCOPE)
    from .timestep_env import physics_settings
    a.check('distinct_declared_timestep_contracts',spec['physics_by_substeps']=={str(s):physics_settings(s) for s in (32,64,128)})
    prepared=read(study/'prepared-contexts.json');recipes=read(study/'recipes.json');plan=read(study/'frozen-repeat-plan.json');gates=read(study/'repeat-geometry.json')
    audit_sources(a,study,source,assets,spec,report,comparison_study,confirmation_study)
    audit_contexts(a,spec,prepared);audit_rows(a,spec,prepared,rows,recipes,plan,gates);audit_plans(a,study,spec,rows,recipes)
    audit_options(a,study,report);positions,rescored=audit_batches(a,study,spec,rows);audit_report(a,spec,report,rows,recipes,positions)
    failures=[r for r in a.checks if not r['passed']]
    return dict(version=VERSION,status='failed' if failures else 'passed',contract_fingerprint=fp,checks_count=len(a.checks),checks=a.checks,
        failures=failures,records=len(rows),independently_rescored_throws=rescored,auditor_source_sha256=sha256(__file__),
        scope='Saved evidence and CPU score/provenance audit; no new dynamics or physical validation')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('study','source','assets','comparison-study','confirmation-study','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    try:result=audit(args.study,args.source,args.assets,args.comparison_study,args.confirmation_study)
    except (KeyError,ValueError,TypeError,OSError,IndexError,AttributeError) as error:
        result=dict(version=VERSION,status='failed',checks_count=0,checks=[],failures=[dict(name='malformed_or_missing_evidence',passed=False,detail=str(error))])
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:result.get(k) for k in ('status','checks_count','independently_rescored_throws')}))
    if result['status']!='passed':raise RuntimeError('Neighborhood evidence audit failed')


if __name__=='__main__':main()
