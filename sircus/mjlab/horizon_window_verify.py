"""Independent evidence audit for the case01 horizon-only development test.

Recompute the longer time boundary by bisection, geometric certificates from
vertex support, outcomes from native traces, and the fixed request denominator.
No simulator stepping or new scoring definition is introduced by this audit.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import time
import numpy as np

from . import clearance_correction_verify as previous
from . import calibrated_benchmark_verify as calibrated
from . import policy64_verify as policy
from . import replay650_verify as replay
from .benchmark_verify import contained, digest, read, sha

VERSION='sircus-clearance-horizon-audit-v1'
STUDY_VERSION='sircus-clearance-horizon-study-v1'
CONTEXT='arm-clearance-test-01'
METHODS=('short_window','long_window')
counts, control_gate, same = previous.counts, previous.control_gate, previous.same


def bisected_horizon(origin_z,velocity_z,gravity_z,target_z,short_horizon):
    """Find the descending boundary independently of the producer's formula."""
    if not all(type(x) in (int,float) and math.isfinite(x) for x in
               (origin_z,velocity_z,gravity_z,target_z,short_horizon)):
        raise ValueError('Finite trajectory scalars required')
    if gravity_z>=0 or not 0<short_horizon<=1.5 or origin_z<target_z:
        raise ValueError('Downward gravity and valid original horizon required')
    height=lambda t:origin_z+velocity_z*t+.5*gravity_z*t*t-target_z
    if height(short_horizon)<-1e-12 or height(1.5)>0:
        raise ValueError('Missing, shortened or overlong extension')
    lo,hi=short_horizon,1.5
    for _ in range(80):
        middle=(lo+hi)/2
        if height(middle)>0:lo=middle
        else:hi=middle
    result=(lo+hi)/2
    if velocity_z+gravity_z*result>=0:raise ValueError('Crossing must descend')
    return result


def budget_checks(plan,checks):
    checks.equal(len(plan),18,'Exactly18 requested slots including abstentions')
    checks.equal(len({r['id'] for r in plan}),18,'Unique requested identities')
    trials=[r for r in plan if r['role']=='diagnostic']
    checks.equal(Counter((r['context_id'],r['method'],r['repeat']) for r in trials),
        Counter((CONTEXT,m,i) for m in METHODS for i in range(3)),
        'One fixed layout and three matched repeats per horizon')
    controls=[r for r in plan if r['role']!='diagnostic']
    checks.equal(Counter((r['phase'],r['role'],r['repeat']) for r in controls),
        Counter((p,r,i) for p in ('controls_pre','controls_post')
                for r in ('positive_control','negative_control') for i in range(3)),
        'Twelve controls with original source identities')
    checks.require(all(r['phase']=='controls_pre' for r in plan[:6])
        and all(r['role']=='diagnostic' for r in plan[6:12])
        and all(r['phase']=='controls_post' for r in plan[12:]),'Controls surround exactly six requested comparisons')


def extended_certificate(gate,connection,bucket,values,diagnostic,gravity,short,long,checks):
    """Audit the one-factor window extension, using saved directions as witnesses."""
    from .arm_clearance_distance import RULE as SHORT_RULE
    from .clearance_horizon_selector import RULE as LONG_RULE
    from .arm_clearance_selector import fingerprint
    tag='Longer conditional free-flight certificate'
    checks.equal(short['rule_sha256'],fingerprint(SHORT_RULE),tag+'/unchanged short rule')
    checks.equal(long['rule_sha256'],fingerprint(LONG_RULE),tag+'/one frozen horizon rule')
    checks.equal(long['original_rule_sha256'],short['rule_sha256'],tag+'/exact baseline rule')
    checks.equal(long['original_passed'],short['passed'],tag+'/original decision retained')
    checks.equal(long['original_minimum_clearance_lower_bound_m'],short['minimum_clearance_lower_bound_m'],tag+'/original bound retained')
    for proof in (short,long):
        for key in ('model_sha256','geometry_sha256','vertex_geometry_sha256'):
            checks.equal(proof[key],getattr(gate,key),tag+'/actual compiled '+key)
        checks.equal(proof['endpoint_sha256'],fingerprint(dict(q10=connection['q10'],q14=connection['q14'],
            bucket_xy_m=bucket,values=values)),tag+'/same certified joint branch')
        checks.equal(proof['excluded_collision_geometry'],gate.excluded,tag+'/only impact board excluded')
    checks.numeric(long['tabletop_z_m'],gate.tabletop_z,tag+'/actual table elevation',0)
    checks.numeric(long['target_ball_center_z_m'],gate.tabletop_z+gate.radius,tag+'/actual tangent-center elevation',0)
    checks.equal(long['original_horizon_end'],SHORT_RULE['end'],tag+'/original endpoint meaning')
    checks.equal(long['horizon_end'],LONG_RULE['end'],tag+'/new endpoint meaning')
    checks.equal(len(long['horizon_metadata']),5,tag+'/all five boundaries recorded')
    extended=copy.deepcopy(diagnostic)
    for i,(member,metadata,proof) in enumerate(zip(extended['members'],long['horizon_metadata'],long['members'])):
        checks.equal(member['origin_m'],diagnostic['launch']['origin_m'],tag+'/same member origin')
        before=member['rim_band_start_s']
        after=bisected_horizon(float(member['origin_m'][2]),float(member['velocity_world_m_s'][2]),
            float(gravity[2]),gate.tabletop_z+gate.radius,before)
        checks.equal(metadata['member'],i,tag+'/ordered metadata identity')
        checks.numeric(metadata['original_horizon_s'],before,tag+'/original time',0)
        checks.numeric(metadata['new_horizon_s'],after,tag+'/independent descending time',1e-12)
        checks.numeric(metadata['maximum_horizon_s'],1.5,tag+'/cap unchanged',0)
        checks.numeric(proof['original_horizon_s'],before,tag+'/original certificate time',0)
        checks.numeric(proof['horizon_s'],after,tag+'/extended certificate time',1e-12)
        member['rim_band_start_s']=after
    # The reused independent certificate auditor reads the supplied time; its
    # acceptance math does not call production path/certificate routines.
    check_long=copy.deepcopy(long);check_long['passed']=long['extended_window_passed']
    previous.audit_support_certificate(gate,connection['q10'],connection['q14'],bucket,extended,
        gravity,check_long,checks,tag)
    checks.require(long['passed'] is bool(short['passed'] and long['extended_window_passed']),
        tag+'/both old and new windows must pass')


def audit_plan(study,parent,logs,parent_packet,spec,packet,assets,checks):
    from .arm_clearance_distance import PreciseArmClearanceGate
    from .clearance_horizon_selector import HorizonArmClearanceGate
    certificates=read(study/'window-certificates.json')
    checks.equal(sha(study/'window-certificates.json'),spec['window_certificates_sha256'],'Frozen complete window certificates')
    checks.equal(certificates['context_id'],CONTEXT,'One fixed context')
    checks.equal(certificates['candidate_index'],14,'Same prespecified candidate14; no alternate selection')
    log=next(r for r in logs if r['context_id']==CONTEXT);bank=log['banks']['arm_clearance']
    original=next(c for c in log['selection']['arm_clearance']['checks'] if c['candidate_index']==14)
    checks.require(original['passed'] is True,'Original endpoint independently admitted')
    for key in ('connection_file','connection_sha256'):
        checks.equal(certificates[key],original[key],'Identical frozen branch/'+key)
    connection=read(contained(parent/'native',original['connection_file']))
    checks.equal(sha(contained(parent/'native',original['connection_file'])),original['connection_sha256'],'Bound exact endpoint bytes')
    diagnostic=bank['ranking']['diagnostics'][14]
    same(certificates['diagnostic'],diagnostic,checks,'Forecasts unchanged',0)
    proofs=certificates['certificates'];checks.equal(set(proofs),set(METHODS),'Exactly two compared horizons')
    checks.equal(sha(study/'selection-plan.json'),spec['selection_plan_sha256'],'Frozen exact selection bytes')
    selections=read(study/'selection-plan.json');checks.equal(set(selections),{CONTEXT},'No extra selection context')
    checks.equal(set(selections[CONTEXT]),set(METHODS),'No extra selection method')
    for method,selected in selections[CONTEXT].items():
        checks.equal(selected['selected_index'],14,'No alternate selected candidate')
        checks.require(selected['policy_abstain'] is (not proofs[method]['passed']),'Exact window refusal decision')
        checks.equal(len(selected['checks']),1,'Only original candidate checked')
        expected=copy.deepcopy(original);expected['clearance']=proofs[method]
        same(selected['checks'][0],expected,checks,'Exact saved endpoint and window proof',0)
    gravity=read(parent/'contract.json')['native_reference']['allocated_options']['native']['gravity']
    short=proofs['short_window'];long=proofs['long_window'];bucket=log['bucket_xy_m'];values=bank['candidates'][14]['values']
    short_gate=PreciseArmClearanceGate(spec['scene'],assets)
    previous.audit_support_certificate(short_gate,connection['q10'],connection['q14'],bucket,diagnostic,gravity,short,checks,'Original refined window')
    extended_certificate(HorizonArmClearanceGate(spec['scene'],assets),connection,bucket,values,diagnostic,gravity,short,long,checks)
    windows=[dict(method=m,passed=proofs[m]['passed'],horizon_min_s=min(p['horizon_s'] for p in proofs[m]['members']),
        horizon_max_s=max(p['horizon_s'] for p in proofs[m]['members']),
        minimum_clearance_lower_bound_m=proofs[m]['minimum_clearance_lower_bound_m'],required_clearance_m=.002) for m in METHODS]
    old_requests={r['id']:r for r in parent_packet['requests']}
    old_rows={r['id']:r for r in replay.read_records(parent/'records.jsonl')}
    for request in packet['requests']:
        source=old_requests[request['parent_request_id']];source_row=old_rows[request['parent_record_id']];tag=request['id']
        checks.equal(source['id'],source_row['id'],tag+'/one source identity')
        checks.equal(request['parent_request_sha256'],digest(source),tag+'/exact source request')
        checks.equal(request['parent_record_sha256'],digest(source_row),tag+'/exact source record')
        # Only bookkeeping, horizon decision and previously measured joint/proof
        # fields may be added or changed. The physical action is identical.
        for key in ('values','command_values','bucket_xy_m','observation','source_hint','effective_source_hint',
                    'context_resolution_file','context_resolution_sha256','context_home_sha256',
                    'frozen_artifact_sha256','prediction_model_sha256'):
            if key in source:same(request[key],source[key],checks,tag+'/unchanged '+key,0)
        if request['role']=='diagnostic':
            checks.equal((source['context_id'],source['method'],source['candidate_index']),(CONTEXT,'arm_clearance',14),tag+'/exact known shot')
            checks.equal(request['candidate_index'],14,tag+'/no new candidate')
            checks.equal(request['candidate_id'],bank['candidates'][14]['id'],tag+'/exact candidate label')
            same(request['command_values'],request['values'],checks,tag+'/unchanged physical command',0)
            same(request['model_prediction'],bank['predictions'][14],checks,tag+'/unchanged model prediction',0)
            proof=proofs[request['method']];checks.require(request['policy_abstain'] is (not proof['passed']),tag+'/admission only change')
            checks.equal(request['clearance_rule_sha256'],proof['rule_sha256'],tag+'/correct rule')
            checks.equal(request['clearance_minimum_m'],proof['minimum_clearance_lower_bound_m'],tag+'/correct bound')
            expected=original
        else:
            checks.equal((source['phase'],source['role'],source['repeat']),('controls_pre',request['role'],0),tag+'/exact matched control')
            checks.require(request['policy_abstain'] is False,tag+'/control cannot abstain')
            expected=source_row
        for key in ('connection_file','connection_sha256'):checks.equal(request[key],expected[key],tag+'/correct branch proof/'+key)
        branch=read(contained(parent/'native',expected['connection_file']))
        for key in ('q10','q14'):same(request[key],branch[key],checks,tag+'/exact branch/'+key,0)
        checks.require(request['selection_or_learning'] is False,tag+'/no fitting or alternative search')
    return windows


def enrich_contacts(study,rows,spec,calibration,checks):
    enriched=previous.contact_rows(study,rows,spec,calibration,checks)
    by_id={r['id']:r for r in rows}
    for record in enriched:
        row=by_id[record['record_id']]
        record.update(bucket_retained=None,table_contact_before_terminal=None,table_contact_after_terminal=None)
        if row['native_release']:
            # success is the existing bucket-retention result; route_success
            # remains the stricter board-route score. Do not conflate them.
            record['bucket_retained']=bool(row['success'])
            with np.load(contained(study/'native',row['native_cache_file']),allow_pickle=False) as cache:
                f=cache['forces_N'][:,row['native_cache_row'],2]
                stop=row['physics_steps']
                record['table_contact_before_terminal']=bool(np.any(f[:stop]>.01))
                record['table_contact_after_terminal']=bool(np.any(f[stop:]>.01))
            checks.require(type(row['success']) is bool,record['record_id']+'/existing retention label explicit')
    return enriched


def audit(study,parent,parent_audit,source,native_source,assets):
    started=time.monotonic();checks=replay.AuditChecks()
    study,parent,parent_audit,source,native_source,assets=map(lambda p:Path(p).resolve(),(study,parent,parent_audit,source,native_source,assets))
    parent_spec,_,logs,parent_packet,parent_manifest=previous.audit_parent(parent,parent_audit,checks)
    spec,report,packet=read(study/'contract.json'),read(study/'report.json'),read(study/'prediction-plan.json')
    manifest=replay.check_manifest(study,checks);checks.equal(report['evidence_sha256'],manifest,'Complete evidence binding')
    checks.equal(spec['version'],STUDY_VERSION,'Frozen horizon-only study version')
    unsigned=dict(spec);fp=unsigned.pop('fingerprint');checks.equal(digest(unsigned),fp,'Frozen contract fingerprint')
    checks.equal((report['status'],report['phase']),('completed','completed'),'Finite worker completed')
    checks.equal((packet['contract_fingerprint'],report['contract_fingerprint']),(fp,fp),'One exact contract')
    budget_checks(packet['requests'],checks)
    checks.equal(spec['protocol'],dict(methods=list(METHODS),substeps=64,new_training_requests=0,maximum_requests=18),'Fixed protocol')
    checks.equal(report['protocol'],spec['protocol'],'Report keeps frozen protocol')
    checks.equal((spec['substeps'],report['substeps'],spec['maximum_requests']),(64,64,18),'Fixed64 and18 cap')
    checks.equal((spec['context_id'],spec['methods'],spec['repeats']),(CONTEXT,list(METHODS),3),'No substituted case or repeats')
    checks.require(spec['training'] is False and spec['noise'] is False,'No fitting or noise')
    checks.equal(report['plan_sha256'],sha(study/'prediction-plan.json'),'Exact frozen requested plan')
    checks.require(packet['frozen_before_any_native_outcome'] is True and packet['native_planning_queries']==0
        and packet['frozen_unix_s']<=report['acquired_unix_s'],'Plan frozen before GPU outcomes')
    for key in ('scene','identity','native_source_sha256','packages'):same(spec[key],parent_spec[key],checks,'Unchanged native setup/'+key,0)
    bindings=spec['parent_bindings']
    for name,key in (('report.json','parent_report_sha256'),('contract.json','parent_contract_sha256'),
                     ('evidence-manifest.json','parent_manifest_sha256'),('prediction-plan.json','parent_plan_sha256')):
        checks.equal(bindings[key],sha(parent/name),'Bound original evidence/'+name)
    checks.equal(bindings['parent_audit_sha256'],sha(parent_audit),'Bound original audit')
    checks.equal(bindings['parent_evidence_sha256'],parent_manifest,'Complete parent manifest')
    for name,value in parent_spec['source_sha256'].items():checks.equal(spec['diagnostic_source_sha256'].get(name),value,'Original helper unchanged/'+name)
    required={str(p.relative_to(source)) for folder in ('sircus/mjlab','sircus/isaac') for p in (source/folder).glob('*.py')}
    checks.require(required.issubset(spec['diagnostic_source_sha256']),'Every current CPU/native helper bound before outcomes')
    for name in ('calibration.json','verification.json','setup-home.json','prior.json'):checks.equal(sha(study/name),parent_manifest[name],'Unchanged native input/'+name)
    windows=audit_plan(study,parent,logs,parent_packet,spec,packet,assets,checks)
    same(report['windows'],windows,checks,'Independent window summary')
    rows=replay.read_records(study/'records.jsonl');previous.audit_requests(study,parent,rows,packet['requests'],spec,checks)
    calibrated.check_rows(rows,checks);calibration=replay.check_calibration(study,spec['identity']['calibration'],checks)
    from .learning_solve import required_asset_paths
    checks.equal({name:sha(contained(assets.parent,name)) for name in required_asset_paths(assets)},spec['identity']['asset_sha256'],'Original collision assets')
    policy.audit_native_options(study/'native',spec['scene'],calibration,assets,checks)
    traces=calibrated.rescore_rows(study/'native',rows,spec['scene'],calibration,checks,'Horizon-only fixed64',spec['identity']['trace_schema'],assets=assets)
    same(report['counts'],counts(rows),checks,'All18 outcomes independently counted')
    gates={phase:control_gate([r for r in rows if r['phase']=='controls_'+phase]) for phase in ('pre','post')}
    same(report['control_gates'],gates,checks,'Matched control outcomes independently checked')
    checks.require(all(g['status']=='passed' for g in gates.values()),'Both control groups passed')
    comparisons=[dict(method=m,counts=counts([r for r in rows if r['role']=='diagnostic' and r['method']==m])) for m in METHODS]
    same(report['comparisons'],comparisons,checks,'Matched3-repeat comparisons')
    enriched=enrich_contacts(study,rows,spec,calibration,checks)
    basic=[{k:r[k] for k in ('record_id','context_id','method','repeat','bucket_xy_m','values','native_release','capture','clean','invalid','clearance_rejected')} for r in enriched]
    same(report['rows'],basic,checks,'Every basic UI outcome verified')
    previous.audit_sources(source,native_source,spec,checks)
    checks.equal(spec['diagnostic_source_sha256'].get('sircus/mjlab/horizon_window_verify.py'),sha(__file__),'Executing horizon auditor frozen')
    return dict(version=VERSION,status='passed',checks_count=checks.count,checks=checks.entries,failures=[],
        report_sha256=sha(study/'report.json'),plan_sha256=sha(study/'prediction-plan.json'),
        evidence_manifest_sha256=sha(study/'evidence-manifest.json'),contract_fingerprint=fp,auditor_sha256=sha(__file__),
        producer_source_sha256=spec['diagnostic_source_sha256'],native_source_sha256=spec['native_source_sha256'],
        requested_trials=len(rows),training_requests=0,native_releases=len(traces),counts=counts(rows),comparisons=comparisons,
        control_gates=gates,windows=windows,rows=enriched,elapsed_s=time.monotonic()-started,
        scope='One diagnosed layout, exact same shot, held joints and unchanged strict scorer. Refused shots have no observed flight. No training, generalization or table-assisted objective claimed.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('study','parent','parent-audit','source','native-source','assets','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if any(a.output.resolve().is_relative_to(x.resolve()) for x in (a.study,a.parent)):raise ValueError('Audit must not overwrite evidence')
    try:result=audit(a.study,a.parent,a.parent_audit,a.source,a.native_source,a.assets)
    except Exception as error:
        a.output.write_text(json.dumps(dict(version=VERSION,status='failed',failures=[type(error).__name__+': '+str(error)]),indent=2)+'\n');raise
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','checks_count','requested_trials','native_releases')}))

if __name__=='__main__':main()
