"""Independent CPU proof of the literal36-request drop-gap restoration experiment.

No producer plan builder, summary function, or scorer is imported. Original
proofs, all requested outcomes, strict setup geometry, native reset and dense
v3 scoring are checked independently; a failed audit cannot verify the report.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from . import replay650_verify as replay
from . import table_training_verify as native
from . import coverage_training_verify as geometry
from . import table_assisted_score_oracle as oracle
from .benchmark_verify import EvidenceError, contained, digest, read, sha

VERSION='sircus-regression-drop-gap-correction-audit-v1'
STUDY_VERSION='sircus-regression-drop-gap-correction-v1'
CASES=tuple('recovery-test-'+x for x in ('r0-c5-n1','r0-c6-n1'))
VARIANTS=('parent_exact','recovery_exact','corrected')
ROLES=('direct_control','table_control','negative_control')
META=('source_hint','effective_source_hint','source_teacher_id','context_resolution_file',
    'context_resolution_sha256','context_home_sha256','context_layout_eligible','context_anchor_admitted','context_hard_reject','observation')
PARENT_REPORT='6a255223820656059fe16a5de2944ba2593f7ac3603b287f4955979cd4677504'
PARENT_AUDIT='e2ba653792b37d302425ba4af6cb22460c3d5b426274b3c1b3cbce270a857a9b'
PARENT_FP='f54c341eba5258b427e4e1eb05a1b2e93bf76518d9f94a2fe8cf6bad744f6082'


def statistics(rows):
    result=native.statistics(rows)
    result['robot_contact_shots']=sum(r.get('diagnostic_robot_contact_steps',0)>0 for r in rows)
    result['cached_refusal_requests']=sum(r.get('admission_mode')=='cached_parent_refusal' for r in rows)
    return result


def audit_plan(plan,sources,controls,rule,checks):
    checks.equal((rule['parameter_index'],rule['parameter'],rule['mode']),
        (4,'drop_gap_m','restore_matched_parent_coordinate'),'One literal drop-gap restoration')
    checks.equal(rule['contexts'],list(CASES),'Two exact formerly held-out development contexts')
    checks.equal(rule['diagnosis_sha256'],'511e9f82004af0d603628e25adf8fab71284faa8f11de9307b4c8a611d049ab7','Literal reviewed CPU diagnosis bytes')
    checks.require(isinstance(rule.get('description'),str) and bool(rule['description'].strip()),'Evidence-based correction rationale')
    checks.equal(len(sources),4,'Exactly two original methods at two contexts')
    checks.equal([r['role'] for r in controls],list(ROLES),'Three original17 direct/table/miss controls')
    expected=[(r,None,'control_exact','controls_pre',i) for r in controls for i in range(3)]
    for context in CASES:
        pair={}
        for method in ('parent_bc','recovery_bc'):
            group=[r for r in sources if r['context_id']==context and r['method']==method and r['phase']=='test']
            checks.equal(len(group),1,context+'/one exact historical '+method);pair[method]=group[0]
        parent,recovery=pair['parent_bc'],pair['recovery_bc']
        checks.require(native.captured(parent) and not native.captured(recovery) and not recovery.get('invalid_physics'),context+'/original valid capture loss')
        checks.require(parent.get('native_release') is True and recovery.get('native_release') is True,context+'/both original recipes are released flights')
        checks.equal(parent['bucket_xy_m'],recovery['bucket_xy_m'],context+'/matched original bucket')
        for key in META:checks.equal(parent[key],recovery[key],context+'/same original geometric context/'+key)
        for variant in VARIANTS:
            source=parent if variant=='parent_exact' else recovery
            expected.extend((source,parent if variant=='corrected' else None,variant,'diagnostic',i) for i in range(3))
    expected.extend((r,None,'control_exact','controls_post',i) for r in controls for i in range(3))
    checks.equal(len(plan),36,'All36 requested slots including refusals')
    checks.equal(len({r['id'] for r in plan}),36,'No duplicated or hidden request identity')
    for request,(source,donor,variant,phase,repeat) in zip(plan,expected):
        values=copy.deepcopy(source['values'])
        if variant=='corrected':
            values[4]=donor['values'][4]
            checks.require(values!=source['values'],'Drop-gap restoration actually changes one scalar')
        wanted={k:copy.deepcopy(source[k]) for k in META}
        wanted.update(id=f'drop-gap-{phase}-{source["context_id"]}-{variant}-r{repeat}',phase=phase,
            role=source['role'] if phase.startswith('controls_') else 'diagnostic',
            method='control' if phase.startswith('controls_') else variant,variant=variant,repeat=repeat,
            context_id=source['context_id'],bucket_xy_m=source['bucket_xy_m'],values=values,command_values=values,
            source_record_id=source['id'],source_record_sha256=digest(source),
            donor_record_id=donor['id'] if donor else None,donor_record_sha256=digest(donor) if donor else None,
            adapter_directory='native-controls' if phase.startswith('controls_') else 'native',substeps=64,
            perturbation_mixed_units=[0.]*5,selection_or_learning=False,original_role=source['role'],posthoc_development=True,
            admission_mode='new_action_strict_proof' if variant=='corrected' else 'exact_parent_branch' if source['native_release'] else 'cached_parent_refusal')
        checks.equal(request,wanted,request['id']+'/independently reconstructed exact intervention')


def realized_comparison(row,source,checks):
    """Report IK realization changes independently; never force another branch."""
    if not row['native_release']:
        return None
    values={}
    for key in ('q10','q14'):
        a,b=(np.asarray(item[key],dtype=float) for item in (row,source))
        checks.require(a.shape==b.shape==(6,) and np.isfinite(a).all() and np.isfinite(b).all(),row['id']+'/finite '+key+' branch comparison')
        values[key+'_equal_to_source']=bool(np.array_equal(a,b))
        values[key+'_max_abs_delta_rad']=float(np.max(np.abs(a-b)))
    a,b=(np.asarray(item['board_center_m'],dtype=float) for item in (row,source))
    checks.require(a.shape==b.shape==(3,) and np.isfinite(a).all() and np.isfinite(b).all(),row['id']+'/finite realized board centers')
    values['board_center_delta_m']=(a-b).tolist()
    a,b=(np.asarray(item['board_normal'],dtype=float) for item in (row,source))
    checks.require(a.shape==b.shape==(3,) and np.isfinite(a).all() and np.isfinite(b).all() and np.linalg.norm(a)>0 and np.linalg.norm(b)>0,row['id']+'/finite nonzero realized board normals')
    values['board_normal_angle_deg']=float(np.degrees(np.arccos(np.clip(np.dot(a/np.linalg.norm(a),b/np.linalg.norm(b)),-1.,1.))))
    return dict(source_record_id=source['id'],**values)


def audit_observations(rows,plan,sources,controls,checks):
    checks.equal(len(rows),36,'Complete requested denominator including cached refusals')
    lookup={r['id']:r for r in sources+controls}
    for row,request in zip(rows,plan):
        for key,value in request.items():checks.equal(row.get(key),value,row['id']+'/frozen request/'+key)
        original=lookup[request['source_record_id']]
        expected_realization=realized_comparison(row,original,checks)
        actual_realization=row.get('realization_comparison')
        if expected_realization is None:
            checks.equal(actual_realization,None,row['id']+'/no fabricated realization for declined setup')
        else:
            checks.equal(set(actual_realization or {}),set(expected_realization),row['id']+'/complete realization fields')
            for key,value in expected_realization.items():
                if key in ('source_record_id','q10_equal_to_source','q14_equal_to_source'):
                    checks.equal(actual_realization[key],value,row['id']+'/independent realization/'+key)
                else:
                    checks.numeric(actual_realization[key],value,row['id']+'/independent realization/'+key,2e-6 if key=='board_normal_angle_deg' else 1e-10)
        if request['variant']!='corrected':
            for key in ('connection_file','connection_sha256','native_release','q10','q14'):
                checks.equal(row.get(key),original.get(key),row['id']+'/exact original admission/branch/'+key)
            if not original['native_release']:
                checks.equal(row['invalid'],original['invalid'],row['id']+'/cached original refusal reason')
                checks.require(not row.get('native_cache_file') and not row.get('native_initial_q10') and not row.get('native_initial_q14'),row['id']+'/no placeholder-joint physics')
        else:
            checks.require(bool(row.get('connection_sha256')) and row['connection_sha256']!=original['connection_sha256'],row['id']+'/changed action has fresh strict proof')
    for context in CASES:
        changed=[r for r in rows if r['context_id']==context and r['variant']=='corrected']
        checks.equal(len({r['connection_sha256'] for r in changed}),1,context+'/one immutable corrected-action proof across repeats')


def audit_parent(study,parent_study,parent_audit,spec,checks):
    checks.equal(sha(parent_study/'report.json'),PARENT_REPORT,'Exact original recovery report')
    checks.equal(sha(parent_audit),PARENT_AUDIT,'Exact complete independent recovery audit')
    report,proof,parent,manifest=(read(p) for p in (parent_study/'report.json',parent_audit,parent_study/'contract.json',parent_study/'evidence-manifest.json'))
    unsigned=dict(parent);fp=unsigned.pop('fingerprint')
    checks.equal((digest(unsigned),fp,report['contract_fingerprint'],proof['contract_fingerprint']),(PARENT_FP,)*4,'Immutable original recovery contract')
    checks.equal((proof['status'],proof['full_checks_verified'],proof['native_checks_verified'],proof['failures']),('passed',True,True,[]),'Full native parent proof passed')
    checks.equal((proof['checks_count'],len(proof['checks'])),(283251,283251),'Every original audit check retained')
    checks.require(all(c['passed'] is True for c in proof['checks']),'All original audit checks passed')
    checks.equal((report['status'],report['phase']),('completed','completed'),'Original study finished')
    checks.equal(report['evidence_sha256'],manifest,'Parent report binds original manifest')
    checks.equal(proof['report_sha256'],PARENT_REPORT,'Original audit authenticates original report')
    checks.equal(proof['evidence_manifest_sha256'],sha(parent_study/'evidence-manifest.json'),'Original manifest authenticated')
    bindings=spec['parent_bindings'];files=bindings['files_sha256']
    checks.equal({k:v for k,v in bindings.items() if k!='files_sha256'},dict(report_sha256=PARENT_REPORT,audit_sha256=PARENT_AUDIT,
        contract_fingerprint=PARENT_FP,manifest_sha256=sha(parent_study/'evidence-manifest.json')),'Published ancestry bindings')
    for name,expected in files.items():
        checks.equal(expected,manifest[name],'Consumed parent file in original manifest/'+name)
        checks.equal(sha(contained(parent_study,name)),expected,'Consumed parent bytes unchanged/'+name)
    rows=replay.read_records(parent_study/'rows.jsonl');checks.equal((len(rows),len({r['id'] for r in rows})),(970,970),'Complete original970 ledger')
    sources=read(study/'parent-rows.json');controls=read(study/'control-replays.json')
    expected=[]
    for context in CASES:
        for method in ('parent_bc','recovery_bc'):
            group=[r for r in rows if r.get('phase')=='test' and r.get('context_id')==context and r.get('method')==method]
            checks.equal(len(group),1,'Original unique source/'+context+'/'+method);expected.append(group[0])
    checks.equal(sources,expected,'Full original source rows copied without relabeling')
    checks.equal(controls,read(parent_study/'control-replays.json'),'Original17 controls copied exactly')
    required={'rows.jsonl','control-replays.json','contract.json','prior.json','old-prior.json','calibration.json','verification.json','setup-home.json'}
    for namespace,group in (('native',sources),('control-parent/native',controls)):
        required.update(namespace+'/'+name for name in ('effective-options.json','native-compiled-geometry.json'))
        for row in group:
            for key,hkey in (('context_resolution_file','context_resolution_sha256'),('connection_file','connection_sha256')):
                name=namespace+'/'+row[key];required.add(name);checks.equal(files.get(name),row[hkey],row['id']+'/exact original proof binding/'+key)
    checks.equal(set(files),required,'Every consumed original proof accounted once')
    for name in ('prior.json','old-prior.json','calibration.json','verification.json','setup-home.json'):
        checks.equal(sha(study/name),sha(parent_study/name),'Inherited configuration byte-identical/'+name)
    for key in ('scene','identity','packages','runtime_reference'):checks.equal(spec[key],parent[key],'Unchanged native identity/'+key)
    return sources,controls,parent


def audit_diagnosis(study,rule,spec,checks):
    diagnosis=read(study/'diagnosis.json');unsigned=dict(diagnosis);fp=unsigned.pop('diagnosis_fingerprint')
    checks.equal(digest(unsigned),fp,'Complete saved CPU diagnosis fingerprint')
    checks.equal(diagnosis['version'],'sircus-coverage-regression-diagnosis-v1','Named saved-evidence diagnosis')
    checks.equal((sha(study/'diagnosis.json'),spec['diagnosis_sha256']),(rule['diagnosis_sha256'],)*2,'Reviewed diagnosis frozen before experiment')
    for key,value in dict(report_sha256=PARENT_REPORT,audit_sha256=PARENT_AUDIT,contract_fingerprint=PARENT_FP,
        rows_sha256=spec['parent_bindings']['files_sha256']['rows.jsonl'],full_checks_verified=True,native_checks_verified=True,checks_count=283251).items():
        checks.equal(diagnosis['source'][key],value,'CPU diagnosis original evidence binding/'+key)
    checks.equal([r['context_id'] for r in diagnosis['losses']],
        ['recovery-test-'+x for x in ('r0-c5-n1','r0-c6-n1','r1-c1-n1','r2-c4-n0','r3-c1-n0')],
        'All five original diagnosis cases retained without relabeling')


def audit_timing(study,rows,report,checks):
    boundaries=replay.read_records(study/'phase-boundaries.jsonl')
    checks.equal([(r['phase'],r['rows_count']) for r in boundaries],[('plan-frozen',0),('controls_pre-complete',9),('diagnostic-complete',27),('controls_post-complete',36)],'One frozen plan and complete bracketing phases')
    raw=(study/'rows.jsonl').read_bytes().splitlines(keepends=True);previous=0
    for boundary in boundaries:
        n=boundary['rows_count'];expected=hashlib.sha256(b''.join(raw[:n])).hexdigest() if n else None
        checks.equal(boundary['rows_sha256'],expected,boundary['phase']+'/immutable observed prefix')
        checks.equal(boundary['plan_sha256'],sha(study/'plan.json'),boundary['phase']+'/same pre-outcome plan')
        checks.require(previous<=boundary['acquired_elapsed_s']<3600,'Monotone bounded own worker times');previous=boundary['acquired_elapsed_s']
    packets=geometry.audit_request_packets(study,rows,checks)
    timing=replay.read_records(study/'timing.jsonl')
    checks.equal([t['phase'] for t in timing],['controls_pre','diagnostic','controls_post'],'Exactly three prescribed batches, no retries')
    checks.require(boundaries[0]['acquired_elapsed_s']<=timing[0]['acquired_start_s'],'All36 decisions frozen before first outcome')
    checks.require(all(0<=t['acquired_start_s']<=t['acquired_end_s']<3600 for t in timing),'All native batches within one-hour limit')
    budget=read(study.with_name(study.name+'-budget.json'));pipeline=read(study.with_suffix('.pipeline.json'))
    checks.equal(budget['gpu_uuid'],'GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4','Only owned authorized4090 UUID')
    checks.equal(budget['audit_reserve_s'],1800,'Half-hour CPU audit reserve')
    checks.require(60<=budget['limit_s']<=3600 and 0<pipeline['global_limit_s']<=6300,'Finite worker and total pipeline caps')
    checks.numeric(pipeline['global_deadline_unix_s']-pipeline['started_unix_s'],pipeline['global_limit_s'],'Original absolute deadline',.05)
    checks.equal(budget['global_deadline_unix_s'],pipeline['global_deadline_unix_s'],'Worker shares original pipeline deadline')
    checks.require(pipeline['started_unix_s']<=budget['acquired_unix_s'],'Own lease follows supervisor start')
    checks.numeric(budget['effective_worker_limit_s'],min(budget['limit_s'],budget['global_deadline_unix_s']-budget['acquired_unix_s']-1800),'Postqueue effective worker budget',.1)
    checks.equal(report['deadlines'],dict(worker_limit_s=budget['limit_s'],effective_worker_limit_s=budget['effective_worker_limit_s'],global_deadline_unix_s=budget['global_deadline_unix_s']),'Report actual immutable budget')
    checks.require(0<report['elapsed_s']<budget['effective_worker_limit_s']<=3600,'Worker completes within acquired allowance')
    return packets,dict(worker_receipt_sha256=sha(study.with_name(study.name+'-budget.json')),effective_worker_limit_s=budget['effective_worker_limit_s'],global_limit_s=pipeline['global_limit_s'])


def independent_cases(plan,rows):
    result=[]
    for context in CASES:
        planned=[r for r in plan if r['context_id']==context];group=[r for r in rows if r['context_id']==context];outcomes=[]
        for row in group:
            item={k:copy.deepcopy(row.get(k)) for k in ('id','variant','repeat','values','native_release','invalid',
                'invalid_physics','connection_sha256','native_cache_file','native_cache_row','native_cache_sha256','admission_mode',
                'board_center_m','board_normal','pose_gate','realization_comparison')}
            item.update(capture=native.captured(row),table_contact_steps=row.get('table_contact_steps',0),robot_contact_steps=row.get('diagnostic_robot_contact_steps',0));outcomes.append(item)
        values={v:next(r['values'] for r in planned if r['variant']==v) for v in VARIANTS}
        result.append(dict(context_id=context,bucket_xy_m=planned[0]['bucket_xy_m'],parent_values=values['parent_exact'],
            recovery_values=values['recovery_exact'],corrected_values=values['corrected'],
            variants={v:statistics([r for r in group if r['variant']==v]) for v in VARIANTS},outcomes=outcomes))
    return result


def audit(study,source,assets,*,parent_study,parent_audit):
    started=time.monotonic();checks=replay.AuditChecks()
    study,source,assets,parent_study,parent_audit=map(Path,(study,source,assets,parent_study,parent_audit))
    manifest=replay.check_manifest(study,checks);spec=read(study/'contract.json');report=read(study/'report.json')
    unsigned=dict(spec);fp=unsigned.pop('fingerprint')
    checks.equal((digest(unsigned),report['contract_fingerprint']),(fp,fp),'Complete experiment contract fingerprint')
    checks.equal((spec['version'],report['version'],report['status'],report['phase']),(STUDY_VERSION,STUDY_VERSION,'completed','completed'),'Completed literal regression experiment')
    checks.equal(report['evidence_sha256'],manifest,'Final report binds every original artifact')
    checks.equal(spec['protocol'],report['protocol'],'Protocol unchanged after outcomes')
    for key,value in dict(substeps=64,maximum_requests=36,repeats=3,diagnostic_requests=18,control_requests=18,
        maximum_acquired_s=3600,maximum_queue_s=900,maximum_pipeline_s=6300,audit_reserve_s=1800,storage_reserve_bytes=8*1024**3,
        training=False,noise=False,posthoc_development=True,variants=list(VARIANTS),contexts=list(CASES),changed_parameter='drop_gap_m',parameter_index=4,mode='restore_matched_parent_coordinate').items():
        checks.equal(spec['protocol'][key],value,'Literal frozen design/'+key)
    replay.check_source_files(source,spec['source_sha256'],checks,'Executed frozen source snapshot')
    for module in (Path(__file__),Path(oracle.__file__)):
        checks.equal(sha(module),spec['source_sha256']['sircus/mjlab/'+module.name],'Actual independent checker frozen/'+module.name)
    replay.audit_packages(spec['packages'],checks)
    sources,controls,parent=audit_parent(study,parent_study,parent_audit,spec,checks)
    for name,expected in spec['identity']['asset_sha256'].items():checks.equal(sha(contained(assets.parent,name)),expected,'Original robot asset/'+name)
    for name,key in (('plan.json','plan_sha256'),('rule.json','rule_sha256'),('parent-rows.json','parent_rows_sha256'),('control-replays.json','control_replays_sha256')):
        checks.equal(sha(study/name),spec[key],'Frozen experiment input/'+name)
    plan=read(study/'plan.json');rule=read(study/'rule.json');checks.equal(rule,spec['rule'],'Reviewed rule unchanged')
    checks.equal(report['rule'],rule,'Report discloses actual intervention')
    audit_diagnosis(study,rule,spec,checks);audit_plan(plan,sources,controls,rule,checks)
    rows=replay.read_records(study/'rows.jsonl');audit_observations(rows,plan,sources,controls,checks);native.check_rows(rows,checks)
    calibration=replay.check_calibration(study,spec['identity']['calibration'],checks)
    setup,native_count=geometry.audit_native_namespaces(study,rows,read(study/'prior.json'),read(study/'old-prior.json'),read(study/'setup-home.json'),spec,calibration,assets,checks)
    for namespace,parent_namespace in (('native','native'),('native-controls','control-parent/native')):
        for name in ('effective-options.json','native-compiled-geometry.json'):
            checks.equal(read(study/namespace/name),read(parent_study/parent_namespace/name),'Actual compiled geometry/options match parent/'+namespace+'/'+name)
        files={}
        for row in [r for r in rows if r['adapter_directory']==namespace]:
            if row['native_release']:files.setdefault(row['native_cache_file'],[]).append(row)
            else:checks.equal(row['diagnostic_robot_contact_steps'],0,row['id']+'/no flight means no measured contact')
        for name,members in files.items():
            with np.load(contained(study/namespace,name),allow_pickle=False) as data:
                for row in members:
                    n=int(np.count_nonzero(data['forces_N'][:int(row['physics_steps']),int(row['native_cache_row']),4]>.01))
                    checks.equal(row['diagnostic_robot_contact_steps'],n,row['id']+'/independent dense robot-force count')
    packets,budget=audit_timing(study,rows,report,checks)
    gates={}
    for phase,key in (('controls_pre','pre'),('controls_post','post')):
        gates[key]=native.control_gate([r for r in rows if r['phase']==phase]);checks.equal(gates[key]['status'],'passed',phase+'/only formal direct/table/miss hard gate')
    checks.equal(report['control_gates'],gates,'All18 original17 control outcomes')
    counts={v:statistics([r for r in rows if r['variant']==v]) for v in VARIANTS};cases=independent_cases(plan,rows)
    checks.equal(report['counts'],statistics(rows),'Full36 denominator includes all refusals')
    checks.equal(report['counts_by_variant'],counts,'All6 requested outcomes per diagnostic variant')
    checks.equal(report['cases'],cases,'Exact matched cases and every outcome retained')
    checks.equal((report['expected_requests'],report['training'],report['posthoc_development']),(36,False,True),'No training or new held-out evaluation claim')
    checks.equal(report['gate'],dict(status='passed',substeps=64,controls_passed=True,complete_plan=True,no_policy_promotion=True),'Complete execution without deployment')
    return dict(version=VERSION,status='passed',full_checks_verified=True,native_checks_verified=True,checks_count=checks.count,checks=checks.entries,failures=[],
        contract_fingerprint=fp,report_sha256=sha(study/'report.json'),evidence_manifest_sha256=sha(study/'evidence-manifest.json'),auditor_sha256=sha(Path(__file__)),
        requested_trials=len(rows),native_releases=native_count,counts_by_variant=counts,cases=cases,control_gates=gates,
        setup_evidence=setup,request_packets=packets,budget=budget,elapsed_s=time.monotonic()-started,maximum_numeric_error=checks.numeric_error,
        limitations=['Two inspected formerTEST cases are development data; no new policy, generalization or promotion.',
            'Only original-recovery drop gap is restored; board pose stays fixed and absolute release height changes by the gap delta.',
            'Original refusal requests reuse authenticated decisions, not fresh IK attempts; no placeholder joints released.',
            'Three nominal repeats per arm at64, without noise,128 qualification, driven transit or hardware.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('study','source','assets','parent-study','parent-audit','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or a.output.resolve().is_relative_to(a.study.resolve()):raise ValueError('Separate fresh audit output required')
    try:result=audit(a.study,a.source,a.assets,parent_study=a.parent_study,parent_audit=a.parent_audit)
    except Exception as error:
        a.output.write_text(json.dumps(dict(version=VERSION,status='failed',failures=[type(error).__name__+': '+str(error)]),indent=2)+'\n');raise
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','checks_count','requested_trials','native_releases','counts_by_variant')}))

if __name__=='__main__':main()
