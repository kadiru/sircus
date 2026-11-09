"""Independent CPU proof of the literal72-request imitation regression experiment.

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

VERSION='sircus-imitation-regression-audit-v1'
STUDY_VERSION='sircus-imitation-regression-v1'
CASES=tuple('corrected-imitation-test-'+x for x in ('r0-c0-n0','r0-c1-n0','r0-c6-n1','r1-c1-n0','r1-c2-n0','r3-c0-n1'))
IK_CASES=(CASES[0],CASES[1],CASES[5])
VARIANTS=('current_exact','corrected_exact','blend50')
MODE='matched_current_action_midpoint'
DIAGNOSIS_SHA256='e16f551802c6c378e7911a0bca2470e4302babb075f8ef570df189839a7bd428'
ROLES=('direct_control','table_control','negative_control')
META=('source_hint','effective_source_hint','source_teacher_id','context_resolution_file',
    'context_resolution_sha256','context_home_sha256','context_layout_eligible','context_anchor_admitted','context_hard_reject','observation')
PARENT_REPORT='1fde3f7de5b5c2912a133c858105388a704a80ab0e2cb39e8d79c0c7a189bf14'
PARENT_AUDIT='ea4992828178d9e62ba41c63579ecc3876cb5bffaacf6b590c828d45876a7f63'
PARENT_FP='297b367fa2702d688084bc2e74dd30a83c6cf4312ba078938def5f9b223adce4'
PARENT_ROWS='f9d48dd328e203b76d7f6be41c01e436ea938701821e902bbbd7945df7248843'
PARENT_MANIFEST='e65451629a36e3c486c3c7dbedd5f2de67de97fac46f6d4631eb3c38c68f3957'


def statistics(rows):
    result=native.statistics(rows)
    result['robot_contact_shots']=sum(r.get('diagnostic_robot_contact_steps',0)>0 for r in rows)
    result['cached_refusal_requests']=sum(r.get('admission_mode')=='cached_parent_refusal' for r in rows)
    return result


def audit_plan(plan,sources,controls,rule,checks):
    checks.equal(set(rule),{'mode','contexts','interpolation_alpha','parameter_indices','diagnosis_sha256','description'},'Literal reviewed rule schema')
    for key,value in dict(mode=MODE,contexts=list(CASES),interpolation_alpha=.5,parameter_indices=list(range(5)),diagnosis_sha256=DIAGNOSIS_SHA256).items():
        checks.equal(rule[key],value,'Single frozen midpoint correction/'+key)
    checks.require(isinstance(rule['description'],str) and bool(rule['description'].strip()),'Evidence-based correction rationale')
    checks.equal(len(sources),12,'Both original methods at all six regressed contexts')
    checks.equal([r['role'] for r in controls],list(ROLES),'Three original17 direct/table/miss controls')
    expected=[(r,None,r,'control_exact','controls_pre',i) for r in controls for i in range(3)]
    for context in CASES:
        pair={}
        for method in ('current_bc','corrected_bc'):
            group=[r for r in sources if r['context_id']==context and r['method']==method and r['phase']=='test']
            checks.equal(len(group),1,context+'/one exact historical '+method);pair[method]=group[0]
        current,corrected=pair['current_bc'],pair['corrected_bc']
        checks.require(native.captured(current) and not native.captured(corrected) and not corrected.get('invalid_physics'),context+'/original valid capture loss')
        checks.equal(corrected.get('native_release'),context not in IK_CASES,context+'/three original refusals and three genuine flights')
        checks.equal(corrected.get('invalid'),'finite_ik_unresolved' if context in IK_CASES else None,context+'/original failure type unchanged')
        checks.equal(current['bucket_xy_m'],corrected['bucket_xy_m'],context+'/matched original bucket')
        for key in META:checks.equal(current[key],corrected[key],context+'/same original geometry/'+key)
        for variant in VARIANTS:
            source=current if variant=='current_exact' else corrected
            donor=current if variant=='blend50' else None
            expected.extend((source,donor,current,variant,'diagnostic',i) for i in range(3))
    expected.extend((r,None,r,'control_exact','controls_post',i) for r in controls for i in range(3))
    checks.equal(len(plan),72,'All72 requested slots including refusals')
    checks.equal(len({r['id'] for r in plan}),72,'No duplicated or hidden request identity')
    for request,(source,donor,reference,variant,phase,repeat) in zip(plan,expected):
        values=copy.deepcopy(source['values']);indices=[];alpha=None
        if donor is not None:
            indices=list(range(5));alpha=.5
            values=[old+.5*(target-old) for old,target in zip(source['values'],donor['values'])]
            checks.require(values!=source['values'] and values!=donor['values'],'Midpoint differs from both preserved endpoints')
        wanted={k:copy.deepcopy(source[k]) for k in META}
        wanted.update(id=f'imitation-regression-{phase}-{source["context_id"]}-{variant}-r{repeat}',phase=phase,
            role=source['role'] if phase.startswith('controls_') else 'diagnostic',
            method='control' if phase.startswith('controls_') else variant,variant=variant,repeat=repeat,
            context_id=source['context_id'],bucket_xy_m=source['bucket_xy_m'],values=values,command_values=values,
            source_record_id=source['id'],source_record_sha256=digest(source),
            donor_record_id=donor['id'] if donor else None,donor_record_sha256=digest(donor) if donor else None,
            realization_reference_record_id=reference['id'],realization_reference_record_sha256=digest(reference),
            parameter_indices=indices,interpolation_alpha=alpha,
            adapter_directory='native-controls' if phase.startswith('controls_') else 'native',substeps=64,
            perturbation_mixed_units=[0.]*5,selection_or_learning=False,original_role=source['role'],posthoc_development=True,
            admission_mode='new_action_strict_proof' if donor is not None else 'exact_parent_branch' if source['native_release'] else 'cached_parent_refusal')
        checks.equal(request,wanted,request['id']+'/independently reconstructed exact midpoint intervention')


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
    return dict(source_record_id=source['id'],basis='exact_control_source' if row['phase'].startswith('controls_') else 'matched_successful_current',**values)


def audit_observations(rows,plan,sources,controls,checks):
    checks.equal(len(rows),72,'Complete requested denominator including cached refusals')
    lookup={r['id']:r for r in sources+controls}
    for row,request in zip(rows,plan):
        for key,value in request.items():checks.equal(row.get(key),value,row['id']+'/frozen request/'+key)
        original=lookup[request['source_record_id']]
        reference=lookup[request['realization_reference_record_id']]
        checks.require(reference.get('native_release') is True,row['id']+'/pose comparison reference was actually released')
        checks.equal(digest(reference),request['realization_reference_record_sha256'],row['id']+'/immutable realization reference')
        expected_realization=realized_comparison(row,reference,checks)
        actual_realization=row.get('realization_comparison')
        if expected_realization is None:
            checks.equal(actual_realization,None,row['id']+'/no fabricated realization for declined setup')
        else:
            checks.equal(set(actual_realization or {}),set(expected_realization),row['id']+'/complete realization fields')
            for key,value in expected_realization.items():
                if key in ('source_record_id','basis','q10_equal_to_source','q14_equal_to_source'):
                    checks.equal(actual_realization[key],value,row['id']+'/independent realization/'+key)
                else:
                    checks.numeric(actual_realization[key],value,row['id']+'/independent realization/'+key,2e-6 if key=='board_normal_angle_deg' else 1e-10)
        if request['variant'] in ('current_exact','corrected_exact','control_exact'):
            for key in ('connection_file','connection_sha256','native_release','q10','q14'):
                checks.equal(row.get(key),original.get(key),row['id']+'/exact original admission/branch/'+key)
            if not original['native_release']:
                checks.equal(row['invalid'],original['invalid'],row['id']+'/cached original refusal reason')
                checks.require(not row.get('native_cache_file') and not row.get('native_initial_q10') and not row.get('native_initial_q14'),row['id']+'/no placeholder-joint physics')
        else:
            checks.require(bool(row.get('connection_sha256')) and row['connection_sha256']!=original['connection_sha256'],row['id']+'/changed action has fresh strict proof')
            checks.require(row['connection_sha256']!=reference['connection_sha256'],row['id']+'/new action cannot reuse exact parent proof')
    for context in CASES:
        for variant in VARIANTS[2:]:
            changed=[r for r in rows if r['context_id']==context and r['variant']==variant]
            checks.equal(len(changed),3,context+'/'+variant+'/three requested nominal repeats')
            checks.equal(len({r['connection_sha256'] for r in changed}),1,context+'/'+variant+'/one immutable changed-action proof across repeats')


def audit_setup_summaries(study,rows,checks):
    """Reconstruct display admission evidence from hash-bound proof records."""
    for row in rows:
        root=Path(study)/row['adapter_directory']
        path=contained(root,row['connection_file']);resolution_path=contained(root,row['context_resolution_file'])
        checks.equal(sha(path),row['connection_sha256'],row['id']+'/setup summary connection bytes')
        checks.equal(sha(resolution_path),row['context_resolution_sha256'],row['id']+'/setup summary resolution bytes')
        connection=read(path);resolution=read(resolution_path)
        endpoints=connection.get('endpoint_attempts') or {};examples=[]
        for pair in endpoints.get('pair_attempts',[]):
            checked=pair.get('checked') or {};strict=checked.get('strict_geometry') or {}
            examples.append(dict(branch10=pair.get('branch10'),branch14=pair.get('branch14'),
                reason=checked.get('reason'),board_position_error_m=checked.get('board_position_error_m'),
                dropper_position_error_m=checked.get('dropper_position_error_m'),board_normal_error_deg=checked.get('board_normal_error_deg'),
                collision_pairs=checked.get('pairs',[]),independent_reject_arms=checked.get('independent_reject_arms',[]),
                pose_rejected=pair.get('pose_rejected',False),native_pose_errors=pair.get('native_pose_errors'),
                minimum_clearance_m=strict.get('minimum_clearance_m'),closest_pair=strict.get('closest_pair'),
                margin_contacts=strict.get('margin_contacts',[])))
        branches=endpoints.get('valid_ik_branches');group=connection.get('reason')
        if group=='finite_ik_unresolved' and isinstance(branches,list) and len(branches)==2:
            group='finite_ik_'+('both_arms' if branches==[0,0] else 'dropper' if branches[0]==0 else 'board' if branches[1]==0 else 'unknown')
        home=resolution.get('home_geometry') or {};proof=connection.get('proof') or {}
        wanted=dict(reason=connection.get('reason'),group=group or 'admitted',passed=connection['passed'],
            connection_sha256=row['connection_sha256'],resolution_sha256=row['context_resolution_sha256'],
            action_substituted=connection['action_substituted'],valid_ik_branches=branches,
            endpoint_examples=examples,endpoint_pair_attempts=len(examples),
            admitted_branches=[dict(branch10=c.get('branch10'),branch14=c.get('branch14')) for c in endpoints.get('candidates',[]) if c.get('q10')==row.get('q10') and c.get('q14')==row.get('q14')],
            layout_eligible=resolution.get('layout_eligible'),anchor_admitted=resolution.get('anchor_admitted'),
            proof_minimum_clearance_m=proof.get('minimum_clearance_m'),home_minimum_clearance_m=home.get('minimum_clearance_m'),
            home_closest_pair=home.get('closest_pair'),fixed_base_blockers=home.get('fixed_base_blockers',[]),required_clearance_m=.005,
            scope='Recorded finite admission proof; rejection does not prove unreachability, and a conservative zero bound does not prove penetration.')
        checks.equal(row.get('setup_proof'),wanted,row['id']+'/independent full admission summary')
        checks.equal(wanted['action_substituted'],False,row['id']+'/no silent target substitution')
        if row['native_release']:
            checks.equal(wanted['passed'],True,row['id']+'/actual release has admitted setup')
        else:
            checks.equal(wanted['passed'],False,row['id']+'/no flight fabricated for rejected setup')


def audit_parent(study,parent_study,parent_audit,spec,checks):
    checks.equal(sha(parent_study/'report.json'),PARENT_REPORT,'Exact original corrected-imitation report')
    checks.equal(sha(parent_audit),PARENT_AUDIT,'Exact complete independent corrected-imitation audit')
    report,proof,parent,manifest=(read(p) for p in (parent_study/'report.json',parent_audit,parent_study/'contract.json',parent_study/'evidence-manifest.json'))
    unsigned=dict(parent);fp=unsigned.pop('fingerprint')
    checks.equal((digest(unsigned),fp,report['contract_fingerprint'],proof['contract_fingerprint']),(PARENT_FP,)*4,'Immutable original corrected-imitation contract')
    checks.equal((proof['status'],proof['full_checks_verified'],proof['native_checks_verified'],proof['failures']),('passed',True,True,[]),'Full native parent proof passed')
    checks.equal((proof['checks_count'],len(proof['checks'])),(55633,55633),'Every original audit check retained')
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
    checks.equal(sha(parent_study/'rows.jsonl'),PARENT_ROWS,'Literal original158-row ledger bytes')
    checks.equal(sha(parent_study/'evidence-manifest.json'),PARENT_MANIFEST,'Literal original evidence manifest bytes')
    rows=replay.read_records(parent_study/'rows.jsonl');checks.equal((len(rows),len({r['id'] for r in rows})),(158,158),'Complete original158 ledger')
    sources=read(study/'parent-rows.json');controls=read(study/'control-replays.json')
    paired={}
    for row in rows:
        if row.get('phase')=='test':paired.setdefault(row['context_id'],{})[row['method']]=row
    checks.equal(len(paired),70,'All70 original paired positions retained')
    checks.require(all(set(p)=={'current_bc','corrected_bc'} for p in paired.values()),'Exactly both original methods per position')
    losses=sorted(context for context,p in paired.items() if native.captured(p['current_bc']) and not native.captured(p['corrected_bc']))
    checks.equal(losses,list(CASES),'Literal cases are all six original regressions, without outcome filtering')
    expected=[]
    for context in CASES:
        for method in ('current_bc','corrected_bc'):
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
    for row in sources:
        if row.get('native_release'):
            name='native/'+row['native_cache_file'];required.add(name)
            checks.equal(files.get(name),row['native_cache_sha256'],row['id']+'/original dense trace hash bound')
    checks.equal(set(files),required,'Every consumed original proof and dense trace accounted once')
    for name in ('prior.json','old-prior.json','calibration.json','verification.json','setup-home.json'):
        checks.equal(sha(study/name),sha(parent_study/name),'Inherited configuration byte-identical/'+name)
    for key in ('scene','identity','packages','runtime_reference'):checks.equal(spec[key],parent[key],'Unchanged native identity/'+key)
    return sources,controls,parent


def audit_diagnosis(study,rule,spec,checks):
    diagnosis=read(study/'diagnosis.json');unsigned=dict(diagnosis);fp=unsigned.pop('diagnosis_fingerprint')
    checks.equal((digest(unsigned),fp),('eb7a03972c4b6ef731a9eb3a477f670ccdcfe022b5c80cc4a20d7323b9258d59',)*2,'Complete reviewed CPU diagnosis fingerprint')
    checks.equal(diagnosis['version'],'sircus-imitation-regression-diagnosis-v1','Named saved-evidence diagnosis')
    checks.equal((sha(study/'diagnosis.json'),spec['diagnosis_sha256'],rule['diagnosis_sha256']),(DIAGNOSIS_SHA256,)*3,'Reviewed diagnosis frozen before experiment')
    for key,value in dict(report_sha256=PARENT_REPORT,audit_sha256=PARENT_AUDIT,contract_fingerprint=PARENT_FP,
        rows_sha256=spec['parent_bindings']['files_sha256']['rows.jsonl'],full_checks_verified=True,native_checks_verified=True,checks_count=55633).items():
        checks.equal(diagnosis['source'][key],value,'CPU diagnosis original evidence binding/'+key)
    checks.equal([r['context_id'] for r in diagnosis['losses']],list(CASES),'All six original regressions retained without relabeling')


def audit_timing(study,rows,report,checks):
    boundaries=replay.read_records(study/'phase-boundaries.jsonl')
    checks.equal([(r['phase'],r['rows_count']) for r in boundaries],[('plan-frozen',0),('controls_pre-complete',9),('diagnostic-complete',63),('controls_post-complete',72)],'One frozen plan and complete bracketing phases')
    raw=(study/'rows.jsonl').read_bytes().splitlines(keepends=True);previous=0
    for boundary in boundaries:
        n=boundary['rows_count'];expected=hashlib.sha256(b''.join(raw[:n])).hexdigest() if n else None
        checks.equal(boundary['rows_sha256'],expected,boundary['phase']+'/immutable observed prefix')
        checks.equal(boundary['plan_sha256'],sha(study/'plan.json'),boundary['phase']+'/same pre-outcome plan')
        checks.require(previous<=boundary['acquired_elapsed_s']<3600,'Monotone bounded own worker times');previous=boundary['acquired_elapsed_s']
    packets=geometry.audit_request_packets(study,rows,checks)
    timing=replay.read_records(study/'timing.jsonl')
    checks.equal([t['phase'] for t in timing],['controls_pre','diagnostic','diagnostic','controls_post'],'Exactly four prescribed batches, no retries')
    checks.require(boundaries[0]['acquired_elapsed_s']<=timing[0]['acquired_start_s'],'All72 decisions frozen before first outcome')
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
                'board_center_m','board_normal','pose_gate','realization_comparison','setup_proof')}
            item.update(capture=native.captured(row),table_contact_steps=row.get('table_contact_steps',0),robot_contact_steps=row.get('diagnostic_robot_contact_steps',0));outcomes.append(item)
        values={v:next(r['values'] for r in planned if r['variant']==v) for v in VARIANTS}
        result.append(dict(context_id=context,bucket_xy_m=planned[0]['bucket_xy_m'],current_values=values['current_exact'],
            corrected_values=values['corrected_exact'],blend_values=values['blend50'],variant_values=values,
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
    for key,value in dict(substeps=64,maximum_requests=72,repeats=3,diagnostic_requests=54,control_requests=18,
        maximum_acquired_s=3600,maximum_queue_s=900,maximum_pipeline_s=6300,audit_reserve_s=1800,storage_reserve_bytes=8*1024**3,
        training=False,noise=False,posthoc_development=True,variants=list(VARIANTS),contexts=list(CASES),mode=MODE,parameter_indices=list(range(5)),interpolation_alpha=.5).items():
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
    rows=replay.read_records(study/'rows.jsonl');audit_observations(rows,plan,sources,controls,checks);audit_setup_summaries(study,rows,checks);native.check_rows(rows,checks)
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
    checks.equal(report['counts'],statistics(rows),'Full72 denominator includes all refusals')
    checks.equal(report['counts_by_variant'],counts,'All18 requested outcomes per diagnostic variant')
    checks.equal(report['cases'],cases,'Exact matched cases and every outcome retained')
    checks.equal((report['expected_requests'],report['training'],report['posthoc_development']),(72,False,True),'No training or new held-out evaluation claim')
    checks.equal(report['gate'],dict(status='passed',substeps=64,controls_passed=True,complete_plan=True,no_policy_promotion=True),'Complete execution without deployment')
    return dict(version=VERSION,status='passed',full_checks_verified=True,native_checks_verified=True,checks_count=checks.count,checks=checks.entries,failures=[],
        contract_fingerprint=fp,report_sha256=sha(study/'report.json'),evidence_manifest_sha256=sha(study/'evidence-manifest.json'),auditor_sha256=sha(Path(__file__)),
        requested_trials=len(rows),native_releases=native_count,counts_by_variant=counts,cases=cases,control_gates=gates,
        setup_evidence=setup,request_packets=packets,budget=budget,elapsed_s=time.monotonic()-started,maximum_numeric_error=checks.numeric_error,
        limitations=['Six inspected formerTEST regressions are posthoc development data; original38/70 versus40/70 remains unchanged; no new policy, generalization or promotion.',
            'One componentwise midpoint intervention changes all five action parameters; it does not isolate any single parameter cause. Finite admission failures do not establish unreachability.',
            'Original refusal requests reuse authenticated decisions, not fresh IK attempts; no placeholder joints released.',
            'Three nominal requests per variant at64, sharing one admission proof per action; no repeated independent IK searches, noise,128 qualification, driven transit or hardware.'])


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
