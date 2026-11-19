"""Independent CPU audit of the frozen 42-request tilt-only correction.

No producer request builder or scorer is used. The inherited independent dense
checker verifies physics/geometry and independently evaluates the v3 route.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import time
import numpy as np
from . import replay650_verify as replay
from . import table_training_verify as v3
from . import calibrated_benchmark_verify as calibrated
from . import policy64_verify as old
from .benchmark_verify import EvidenceError, contained, digest, read, sha

VERSION='sircus-table-training-correction-audit-v1'
STUDY_VERSION='sircus-table-training-correction-v1'
CASES=tuple(f'table-training-test-{i:02d}' for i in (0,1,18,26))
ROLES=('direct_control','table_control','negative_control')
PARENT_REPORT='4605bcb801351ae91ad16b71a033ceb3866027db621f9d84227f23b1ef257e95'
PARENT_AUDIT='6e50d87ddb7b2d4b37debbc2a5ac96507ff40bc236223277121acd385b8ec91b'
META=('source_hint','effective_source_hint','source_teacher_id','context_resolution_file',
      'context_resolution_sha256','context_home_sha256','context_layout_eligible',
      'context_anchor_admitted','context_hard_reject','observation')


def statistics(rows):
    result=v3.statistics(rows)
    result['robot_contact_shots']=sum(r.get('diagnostic_robot_contact_steps',0)>0 for r in rows)
    return result


def audit_plan(plan, sources, rule, checks):
    checks.equal((rule['parameter_index'],rule['parameter'],rule['target_value']),
                 (3,'tilt_deg',32),'One literal tilt-only correction')
    checks.equal(rule['contexts'],list(CASES),'Four frozen post-hoc contexts')
    checks.require(isinstance(rule.get('description'),str) and bool(rule['description'].strip()),'Correction rationale disclosed')
    controls=[];cases=[]
    for role in ROLES:
        group=[r for r in sources if r.get('phase')=='controls_pre' and r.get('role')==role and r.get('repeat')==0]
        checks.equal(len(group),1,'One exact parent control/'+role);controls.append(group[0])
    for context in CASES:
        group=[r for r in sources if r.get('phase')=='test' and r.get('method')=='ppo' and r.get('context_id')==context]
        checks.equal(len(group),1,'One observed parent PPO case/'+context);row=group[0]
        checks.require(row.get('native_release') is True and not row.get('route_success') and not row.get('invalid_physics'),context+'/admitted failure, not new selection')
        checks.require(row['values'][3]<22 and math.dist(row['values'][:2],row['bucket_xy_m'])>=.45,context+'/frozen shallow-far eligibility')
        cases.append(row)
    checks.equal(len(sources),7,'Only three controls and four diagnostic inputs')
    expected=[(r,'control_exact','controls_pre',i) for r in controls for i in range(3)]
    expected += [(r,v,'diagnostic',i) for r in cases for v in ('baseline','corrected') for i in range(3)]
    expected += [(r,'control_exact','controls_post',i) for r in controls for i in range(3)]
    checks.equal(len(plan),42,'All42 planned requests including refusals')
    checks.equal(len({r['id'] for r in plan}),42,'Unique request IDs')
    for request,(source,variant,phase,repeat) in zip(plan,expected):
        context=source['context_id'];label=f'{phase}/{context}/{variant}/r{repeat}'
        values=copy.deepcopy(source['values'])
        if variant=='corrected':values[3]=32.
        wanted={key:copy.deepcopy(source[key]) for key in META}
        wanted.update(id=f'correction-{phase}-{context}-{variant}-r{repeat}',phase=phase,
            role=source['role'] if phase.startswith('controls_') else 'diagnostic',
            method='control' if phase.startswith('controls_') else variant,variant=variant,repeat=repeat,
            context_id=context,bucket_xy_m=source['bucket_xy_m'],values=values,command_values=values,
            source_record_id=source['id'],source_record_sha256=digest(source),substeps=64,
            perturbation_mixed_units=[0.]*5,selection_or_learning=False,posthoc_development=True)
        checks.equal(request,wanted,label+'/independently reconstructed exact request')


def audit_observations(rows,plan,sources,checks):
    checks.equal(len(rows),42,'Every frozen request has an observed outcome')
    lookup={r['id']:r for r in sources}
    for row,request in zip(rows,plan):
        for key,value in request.items():checks.equal(row.get(key),value,row['id']+'/request preserved/'+key)
        source=lookup[request['source_record_id']]
        if request['variant']!='corrected':
            for key in ('connection_file','connection_sha256','native_release','q10','q14'):
                checks.equal(row.get(key),source.get(key),row['id']+'/original admitted joint branch/'+key)
        else:
            checks.require(row.get('connection_sha256')!=source.get('connection_sha256'),row['id']+'/fresh corrected-action admission proof')


def audit(study,source,assets,*,parent_study,parent_audit):
    started=time.monotonic();checks=replay.AuditChecks()
    study,source,assets,parent_study,parent_audit=map(Path,(study,source,assets,parent_study,parent_audit))
    manifest=replay.check_manifest(study,checks)
    spec,report=read(study/'contract.json'),read(study/'report.json')
    unsigned=dict(spec);fp=unsigned.pop('fingerprint')
    checks.equal(digest(unsigned),fp,'Whole correction contract fingerprint')
    checks.equal((spec['version'],report['version']),(STUDY_VERSION,STUDY_VERSION),'Explicit correction namespace')
    checks.equal((report['status'],report['phase']),('completed','completed'),'All planned phases completed')
    checks.equal(report['contract_fingerprint'],fp,'Report binds actual correction contract')
    checks.equal(report['evidence_sha256'],manifest,'Report binds all original artifacts')
    for key,value in dict(substeps=64,maximum_requests=42,actual_planned_requests=42,maximum_acquired_s=3600,repeats=3,
                          training=False,noise=False,posthoc_development=True).items():
        checks.equal(spec[key],value,'Literal correction protocol/'+key)
    checks.require(report['elapsed_s']<=3600,'Own acquired execution stays within one hour')
    replay.check_source_files(source,spec['source_sha256'],checks,'Executed source snapshot')
    checks.equal(sha(Path(__file__)),spec['source_sha256']['sircus/mjlab/table_correction_verify.py'],'Independent auditor source frozen')
    replay.audit_packages(spec['packages'],checks)
    checks.equal(sha(parent_study/'report.json'),PARENT_REPORT,'Exact audited parent report')
    checks.equal(sha(parent_audit),PARENT_AUDIT,'Exact completed independent parent audit')
    proof,parent_spec=read(parent_audit),read(parent_study/'contract.json')
    checks.equal((proof['status'],proof['full_checks_verified'],proof['failures']),('passed',True,[]),'Parent proof completed without failures')
    checks.equal((proof['checks_count'],len(proof['checks'])),(823442,823442),'Complete original audit accounting')
    checks.require(all(c['passed'] is True for c in proof['checks']),'All original parent checks passed')
    checks.equal(proof['report_sha256'],PARENT_REPORT,'Parent proof authenticates report')
    checks.equal(proof['contract_fingerprint'],parent_spec['fingerprint'],'Parent proof authenticates contract')
    parent_unsigned=dict(parent_spec);parent_fp=parent_unsigned.pop('fingerprint')
    checks.equal(digest(parent_unsigned),parent_fp,'Parent contract fingerprint intact')
    bindings=dict(report_sha256=PARENT_REPORT,audit_sha256=PARENT_AUDIT,
        manifest_sha256=sha(parent_study/'evidence-manifest.json'),rows_sha256=sha(parent_study/'rows.jsonl'),
        contract_sha256=sha(parent_study/'contract.json'),contract_fingerprint=parent_fp)
    checks.equal(spec['parent_bindings'],bindings,'Exact immutable source provenance')
    checks.equal(proof['evidence_manifest_sha256'],bindings['manifest_sha256'],'Parent evidence manifest bound')
    parent_manifest=read(parent_study/'evidence-manifest.json')
    for name in ('rows.jsonl','contract.json','calibration.json','verification.json','setup-home.json','prior.json',
                 'native/effective-options.json','native/native-compiled-geometry.json'):
        checks.equal(sha(parent_study/name),parent_manifest[name],'Original parent evidence unchanged/'+name)
    for key in ('scene','identity','packages'):checks.equal(spec[key],parent_spec[key],'Unchanged native/scoring identity/'+key)
    for name,expected in spec['identity']['asset_sha256'].items():checks.equal(sha(assets.parent/name),expected,'Unchanged robot asset/'+name)
    plan,sources,rule=(read(study/name) for name in ('plan.json','parent-rows.json','rule.json'))
    for name,key in (('plan.json','plan_sha256'),('parent-rows.json','parent_rows_sha256'),('rule.json','rule_sha256')):
        checks.equal(sha(study/name),spec[key],'Frozen request input/'+name)
    checks.equal(rule,spec['rule'],'Frozen rule before execution')
    source_ids={r['id'] for r in sources}
    checks.equal(sources,[r for r in replay.read_records(parent_study/'rows.jsonl') if r['id'] in source_ids],'Full parent rows copied without rewriting')
    audit_plan(plan,sources,rule,checks)
    prior=read(study/'prior.json');calibration=replay.check_calibration(study,spec['identity']['calibration'],checks)
    for name in ('calibration.json','verification.json','setup-home.json','prior.json'):
        checks.equal(sha(study/name),sha(parent_study/name),'Unchanged inherited configuration/'+name)
    rows=replay.read_records(study/'rows.jsonl');audit_observations(rows,plan,sources,checks);v3.check_rows(rows,checks)
    setup=calibrated.check_setup_evidence(study,rows,prior,spec['identity']['setup']['home'],checks)
    old.audit_native_options(study/'native',spec['scene'],calibration,assets,checks)
    for name in ('effective-options.json','native-compiled-geometry.json'):
        checks.equal(read(study/'native'/name),read(parent_study/'native'/name),'Actual native options/geometry identical/'+name)
    for row in rows:
        if row.get('native_release'):
            checks.equal((row['effective_options_file'],row['effective_options_sha256']),
                         ('effective-options.json',sha(study/'native/effective-options.json')),row['id']+'/allocated options bytes')
    traces=v3.rescore_rows(study/'native',rows,spec['scene'],calibration,spec['identity']['trace_schema'],assets,checks)
    files={}
    for row in rows:
        if row.get('native_release'):files.setdefault(row['native_cache_file'],[]).append(row)
        else:checks.equal(row['diagnostic_robot_contact_steps'],0,row['id']+'/no flight means no measured robot contacts')
    for name,members in files.items():
        with np.load(contained(study/'native',name),allow_pickle=False) as data:
            forces=data['forces_N']
            for row in members:
                n=int(np.count_nonzero(forces[:int(row['physics_steps']),int(row['native_cache_row']),4]>.01))
                checks.equal(row['diagnostic_robot_contact_steps'],n,row['id']+'/separate dense robot-force count')
    gates={}
    for phase,key in (('controls_pre','pre'),('controls_post','post')):
        gates[key]=v3.control_gate([r for r in rows if r['phase']==phase])
        checks.equal(gates[key]['status'],'passed',phase+'/unchanged direct/table/miss controls')
        # Historical producer control counts omit only this explicit zero field.
        actual=copy.deepcopy(report['control_gates'][key]);actual['counts'].setdefault('invalid_physics',0)
        checks.equal(actual,gates[key],phase+'/all control accounting independently reconstructed')
    counts={v:statistics([r for r in rows if r['variant']==v]) for v in ('baseline','corrected')}
    checks.equal(report['counts'],statistics(rows),'All requests retained in headline denominator')
    checks.equal(report['expected_requests'],42,'Finite requested denominator shown')
    checks.equal(report['counts_by_variant'],counts,'Paired variant results independently reconstructed')
    expected_cases=[]
    for context in CASES:
        inputs=[r for r in plan if r['context_id']==context]
        expected_cases.append(dict(context_id=context,source_record_id=inputs[0]['source_record_id'],
            bucket_xy_m=inputs[0]['bucket_xy_m'],baseline_values=inputs[0]['values'],
            corrected_values=next(r['values'] for r in inputs if r['variant']=='corrected'),
            variants={v:statistics([r for r in rows if r['context_id']==context and r['variant']==v]) for v in ('baseline','corrected')}))
    checks.equal(report['cases'],expected_cases,'All four cases and every refusal exposed')
    checks.equal((report['description'],report['changed_parameter'],report['target_value']),
                 (rule['description'],'tilt_deg',32),'Actual correction shown without invented policy change')
    checks.equal((report['training'],report['posthoc_development']),(False,True),'No new training/generalization claim')
    return dict(version=VERSION,status='passed',full_checks_verified=True,checks_count=checks.count,checks=checks.entries,failures=[],
        contract_fingerprint=fp,report_sha256=sha(study/'report.json'),evidence_manifest_sha256=sha(study/'evidence-manifest.json'),
        auditor_sha256=sha(Path(__file__)),requested_trials=len(rows),native_releases=len(traces),counts_by_variant=counts,
        cases=expected_cases,control_gates=gates,setup_evidence=setup,elapsed_s=time.monotonic()-started,
        maximum_numeric_error=checks.numeric_error,limitations=['Post-hoc four-case diagnosis, not a new policy or fresh holdout evaluation.',
        'Nominal repeats at64; no noise,128 qualification or driven arm transit.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('study','source','assets','parent-study','parent-audit','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or a.output.resolve().is_relative_to(a.study.resolve()):raise ValueError('Separate fresh immutable audit output required')
    try:result=audit(a.study,a.source,a.assets,parent_study=a.parent_study,parent_audit=a.parent_audit)
    except Exception as error:
        a.output.write_text(json.dumps(dict(version=VERSION,status='failed',failures=[type(error).__name__+': '+str(error)]),indent=2)+'\n');raise
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','checks_count','requested_trials','native_releases','counts_by_variant')}))

if __name__=='__main__':main()
