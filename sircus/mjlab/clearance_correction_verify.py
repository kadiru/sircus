"""Independent CPU audit of one bounded clearance-distance correction.

The six source layouts are development evidence. This auditor never launches
physics or chooses a new shot from its observed outcome.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import importlib.metadata
import json
import math
from pathlib import Path
import time

import numpy as np

from . import calibrated_benchmark_verify as calibrated
from . import policy64_verify as policy
from . import replay650_verify as replay
from . import gap_flight_correction_verify as common
from .benchmark_verify import contained, digest, read, sha

VERSION = 'sircus-clearance-correction-audit-v1'
STUDY_VERSION = 'sircus-clearance-correction-v1'
METHODS = ('original', 'corrected')
CONTEXTS = tuple('arm-clearance-test-'+x for x in ('00','01','04','07','39','44'))
REPEATS = 3
MAX_REQUESTS = 48
PARENT_REPORT_SHA = 'fbc071332ad74302f9c3438196daaf9da036b72c3ee28059539ca234d9e09f38'
PARENT_MANIFEST_SHA = '8ced80c8e20ade204baa0795a66e940987ee32989e1dfe29b0715a92e27e4602'
PARENT_AUDIT_SHA = 'd3f91a2243483c0aa5deede9e5bf1f5770646c3845fdcab7766730d13420d2bf'
same = common.same


def counts(rows):
    released = [r for r in rows if r.get('native_release') is True]
    return dict(requested=len(rows), native_releases=len(released),
        captures=sum(policy.captured(r) for r in released),
        clean=sum(policy.captured(r) and r.get('route_quality') == 2 for r in released),
        rejected=len(rows)-len(released),
        native_misses=sum(not policy.captured(r) and not r.get('invalid_physics') for r in released),
        invalid=sum(bool(r.get('invalid_physics')) for r in released))


def control_gate(rows):
    good = [r for r in rows if r['role'] == 'positive_control']
    bad = [r for r in rows if r['role'] == 'negative_control']
    passed = (len(rows) == 6 and len(good) == len(bad) == 3
        and all(policy.captured(r) for r in good)
        and all(r['native_release'] and not policy.captured(r) and not r['whole_ball_entry'] for r in bad)
        and all(r.get('invalid_physics') is False and not r.get('invalid')
                and r.get('scalar_score_audit_passed') is True for r in rows))
    return dict(status='passed' if passed else 'failed', counts=counts(rows))


def audit_parent(parent, proof_path, checks):
    for path, expected in ((parent/'report.json', PARENT_REPORT_SHA),
                          (parent/'evidence-manifest.json', PARENT_MANIFEST_SHA),
                          (proof_path, PARENT_AUDIT_SHA)):
        checks.equal(sha(path), expected, 'Exact completed clearance source/'+path.name)
    spec, report, proof = read(parent/'contract.json'), read(parent/'report.json'), read(proof_path)
    checks.equal(proof['version'], 'sircus-arm-clearance-audit-v1', 'Original audit version')
    checks.require(proof['status'] == 'passed' and not proof.get('failures')
        and proof['checks_count'] == len(proof['checks']) == 1461935
        and all(c.get('passed') is True for c in proof['checks']), 'All original independent checks passed')
    checks.equal(proof['report_sha256'], PARENT_REPORT_SHA, 'Original report bound by full audit')
    checks.equal(proof['evidence_manifest_sha256'], PARENT_MANIFEST_SHA, 'Original manifest bound by full audit')
    unsigned = dict(spec); fp = unsigned.pop('fingerprint')
    checks.equal(digest(unsigned), fp, 'Original complete contract fingerprint')
    checks.equal(proof['contract_fingerprint'], fp, 'Original full contract bound by audit')
    manifest = replay.check_manifest(parent, checks)
    checks.equal(report['evidence_sha256'], manifest, 'Original complete manifest')
    checks.equal((report['status'], report['phase']), ('completed','completed'), 'Original execution completed')
    rows = replay.read_records(parent/'records.jsonl')
    checks.equal(len(rows), 112, 'Original requested denominator retained')
    return spec, rows, read(parent/'rankings.json'), read(parent/'prediction-plan.json'), manifest


def budget_checks(plan, checks):
    checks.equal(len(plan), MAX_REQUESTS, 'Exactly48 prescribed slots')
    checks.equal(len({r['id'] for r in plan}), MAX_REQUESTS, 'Unique identity of all48 slots')
    diagnostics = [r for r in plan if r['role'] == 'diagnostic']
    expected = Counter((c,m,i) for c in CONTEXTS for m in METHODS for i in range(REPEATS))
    checks.equal(Counter((r['context_id'],r['method'],r['repeat']) for r in diagnostics), expected,
                 'Six prespecified cases paired by three repeats without replacements')
    controls = [r for r in plan if r['role'] != 'diagnostic']
    checks.equal(Counter((r['phase'],r['role'],r['repeat']) for r in controls),
        Counter((p,r,i) for p in ('controls_pre','controls_post')
                for r in ('positive_control','negative_control') for i in range(3)), 'Twelve exact matched controls')
    checks.require(all(r['phase']=='controls_pre' for r in plan[:6])
        and all(r['role']=='diagnostic' for r in plan[6:-6])
        and all(r['phase']=='controls_post' for r in plan[-6:]), 'Precontrols and postcontrols surround comparison')


def audit_support_certificate(gate,q10,q14,bucket,diagnostic,gravity,saved,checks,label):
    """Reconstruct valid support planes without the producer's bound routine.

    Compiled vertices/FK are shared. The producer's nearest-point algorithm is
    not trusted: any finite unit direction certifies a support plane when its
    offset is recomputed over every collider vertex. Reject/accept arithmetic,
    continuous sampling allowance and full-collider coverage are independent.
    """
    colliders=gate.colliders(q10,q14,bucket);g=np.asarray(gravity,float)
    checks.equal(len(saved['members']),5,label+'/five member certificates')
    checks.equal([p['member'] for p in saved['members']],list(range(5)),label+'/ordered certificate member identities')
    checks.equal([p['member'] for p in diagnostic['members']],list(range(5)),label+'/ordered frozen prediction identities')
    minima=[]
    for member,proof in zip(diagnostic['members'],saved['members']):
        tag=label+'/member'+str(member['member']);h=member['rim_band_start_s']
        checks.require(member['full_downward_rim_band_available'] is True and 0<h<=1.5,tag+'/unchanged horizon')
        n=math.ceil(h/.002);times=np.arange(n+1)*h/n
        p=np.asarray(member['origin_m'],float);v=np.asarray(member['velocity_world_m_s'],float)
        xyz=p+times[:,None]*v+.5*times[:,None]**2*g
        speed=max(float(np.linalg.norm(v)),float(np.linalg.norm(v+h*g)));allowance=speed*h/n/2
        refs=proof.get('refinements',[]);refined={r['geom']:r for r in refs}
        checks.equal(len(refined),len(refs),tag+'/unique refinement identities')
        oldbest=float('inf');best=float('inf');needed=set();byname={}
        for c in colliders:
            planes=np.asarray(c['planes'],float)
            faces=np.einsum('ti,fi->tf',xyz,planes[:,:3])+planes[:,3]
            face_max=faces.max(axis=1);old=face_max-gate.radius-allowance-1e-10
            oldbest=min(oldbest,float(old.min()));bound=old
            if float(old.min())<.002:
                needed.add(c['name']);checks.require(c['name'] in refined,tag+'/every uncertain collider refined')
                normals=np.asarray(refined[c['name']]['support_normals'],float)
                checks.require(normals.shape==xyz.shape and np.isfinite(normals).all(),tag+'/finite complete directions')
                norms=np.linalg.norm(normals,axis=1)
                checks.require(np.allclose(norms,1.,atol=1e-9,rtol=0),tag+'/unit separating directions')
                normals=normals/norms[:,None]
                vertices=np.asarray(c['vertices'],float)
                checks.require(vertices.ndim==2 and vertices.shape[1]==3 and len(vertices)>=4
                    and np.isfinite(vertices).all(),tag+'/complete compiled vertex set')
                offsets=np.einsum('ti,vi->tv',normals,vertices).max(axis=1)
                center_plane=np.sum(xyz*normals,axis=1)-offsets
                bound=np.maximum(face_max,center_plane)-gate.radius-allowance-1e-10
            byname[c['name']]=bound;best=min(best,float(bound.min()))
        checks.equal(set(refined),needed,tag+'/refine exactly original failing collider set')
        checks.numeric(proof['legacy_minimum_clearance_lower_bound_m'],oldbest,tag+'/original conservative bound',1e-10)
        checks.numeric(proof['minimum_clearance_lower_bound_m'],best,tag+'/independent refined bound',1e-10)
        checks.require(best>=oldbest-1e-10,tag+'/correction cannot loosen geometric certificate')
        for key,wanted in dict(horizon_s=h,samples=n+1,actual_spacing_s=h/n,
            maximum_speed_m_s=speed,between_sample_allowance_m=allowance,required_margin_m=.002,
            collision_geometry_count=len(colliders)).items():checks.numeric(proof[key],wanted,tag+'/'+key,1e-10)
        witness=proof['closest'];checks.require(witness['geom'] in byname,tag+'/known closest collider')
        index=int(round(witness['time_s']/(h/n)));checks.require(0<=index<=n,tag+'/closest sample index')
        checks.numeric(witness['time_s'],times[index],tag+'/witness time',1e-12)
        checks.numeric(witness['center_m'],xyz[index],tag+'/witness center',1e-10)
        checks.numeric(byname[witness['geom']][index],best,tag+'/witness equals global minimum',1e-10)
        checks.numeric(witness['surface_clearance_lower_bound_m'],best,tag+'/witness bound',1e-10)
        checks.equal(proof['passed'],best>=.002,tag+'/independent member decision')
        minima.append(best)
    checks.equal(saved['passed'],all(v>=.002 for v in minima),label+'/all five must clear')
    checks.numeric(saved['minimum_clearance_lower_bound_m'],min(minima),label+'/five-member minimum',1e-10)
    checks.require(saved['native_queries']==0 and saved['uses_observed_rebound'] is False,
                   label+'/model prediction only, no measured flight substituted')


def audit_selection(study,parent,logs,parent_packet,spec,packet,assets,checks):
    from .arm_clearance_selector import ArmClearanceGate,RULE as OLD_RULE,fingerprint
    from .arm_clearance_distance import PreciseArmClearanceGate,RULE as NEW_RULE
    from .arm_clearance_verify import independent_clearance_arithmetic
    checks.equal(sha(study/'selection-plan.json'),spec['selection_plan_sha256'],'Frozen selected-path ledger')
    selections=read(study/'selection-plan.json');bycontext={r['context_id']:r for r in logs}
    checks.equal(set(selections),set(CONTEXTS),'Exactly six predeclared development layouts')
    gravity=read(parent/'contract.json')['native_reference']['allocated_options']['native']['gravity']
    gates=dict(original=ArmClearanceGate(spec['scene'],assets),corrected=PreciseArmClearanceGate(spec['scene'],assets))
    old_requests={r['id']:r for r in parent_packet['requests']}
    old_rows={r['id']:r for r in replay.read_records(parent/'records.jsonl')}
    chosen={}
    for ctx in CONTEXTS:
        log=bycontext[ctx];bank=log['banks']['arm_clearance'];prior=log['selection']['arm_clearance']
        oldchecks={r['candidate_index']:r for r in prior['checks']}
        checks.equal(set(selections[ctx]),set(METHODS),ctx+'/exact paired methods')
        for method in METHODS:
            selection=selections[ctx][method];recorded=selection['checks'];first_geometry=None;first_pass=None
            checks.require(1<=len(recorded)<=4,ctx+'/same bounded geometry budget')
            for offset,entry in enumerate(recorded):
                index=bank['ranking']['order'][offset];tag=ctx+'/'+method+'/'+str(index)
                checks.equal(entry['candidate_index'],index,tag+'/original candidate rank unchanged')
                checks.require(index in oldchecks,tag+'/original frozen endpoint evidence available')
                original=oldchecks[index]
                for key,value in original.items():
                    if key!='clearance':same(entry[key],value,checks,tag+'/same endpoint proof/'+key,0)
                if entry['passed'] is True:
                    if first_geometry is None:first_geometry=index
                    c=read(contained(parent/'native',entry['connection_file']))
                    checks.equal(sha(contained(parent/'native',entry['connection_file'])),entry['connection_sha256'],tag+'/bound endpoint bytes')
                    proof=entry['clearance'];diagnostic=bank['ranking']['diagnostics'][index]
                    if method=='original':
                        same(proof,original['clearance'],checks,tag+'/unchanged original rule',0)
                        independent_clearance_arithmetic(gates[method],c['q10'],c['q14'],log['bucket_xy_m'],diagnostic,gravity,proof,checks,tag)
                    else:
                        checks.equal(proof['rule_sha256'],fingerprint(NEW_RULE),tag+'/one frozen correction rule')
                        gate=gates[method]
                        for key in ('model_sha256','geometry_sha256','vertex_geometry_sha256'):
                            checks.equal(proof[key],getattr(gate,key),tag+'/actual compiled '+key)
                        checks.equal(proof['endpoint_sha256'],fingerprint(dict(q10=c['q10'],q14=c['q14'],bucket_xy_m=log['bucket_xy_m'],values=bank['candidates'][index]['values'])),tag+'/exact certified endpoint')
                        checks.equal(proof['excluded_collision_geometry'],gate.excluded,tag+'/only intended board excluded')
                        for member in diagnostic['members']:
                            checks.equal(member['origin_m'],diagnostic['launch']['origin_m'],tag+'/identical frozen launch origin')
                        audit_support_certificate(gates[method],c['q10'],c['q14'],log['bucket_xy_m'],diagnostic,gravity,proof,checks,tag)
                    if proof['passed']:
                        first_pass=index;checks.equal(offset,len(recorded)-1,tag+'/stop at first admissible candidate')
                else:checks.require('clearance' not in entry,tag+'/no imagined path for rejected endpoint')
            if first_pass is None:checks.equal(len(recorded),4,ctx+'/refusal spends same four-check budget')
            checks.require(first_geometry is not None,ctx+'/chosen source cases have endpoint evidence')
            index=first_pass if first_pass is not None else first_geometry
            checks.equal(selection['selected_index'],index,ctx+'/'+method+'/first admissible choice')
            checks.require(selection['policy_abstain'] is (first_pass is None),ctx+'/'+method+'/correct refusal')
            checks.require(selection['all_geometry_checks_rejected'] is False,ctx+'/clearance versus setup failure')
            chosen[(ctx,method)]=(index,selection)
    common_fields=('bucket_xy_m','observation','source_hint','effective_source_hint','context_resolution_file',
        'context_resolution_sha256','context_home_sha256','context_layout_eligible','context_anchor_admitted','context_hard_reject',
        'source_teacher_id','frozen_artifact_sha256','prediction_model_sha256')
    for r in packet['requests']:
        source=old_requests[r['parent_request_id']];source_row=old_rows[r['parent_record_id']]
        checks.equal(r['parent_request_sha256'],digest(source),r['id']+'/exact source request')
        checks.equal(r['parent_record_sha256'],digest(source_row),r['id']+'/exact source result provenance')
        checks.equal(source['id'],source_row['id'],r['id']+'/one matched source request/result')
        checks.require(r['selection_or_learning'] is False,r['id']+'/no model fitting')
        for key in common_fields:
            if key in source: same(r[key],source[key],checks,r['id']+'/unchanged context/'+key,0)
        if r['role']=='diagnostic':
            checks.equal(source['method'],'arm_clearance',r['id']+'/baseline is original clearance gate')
            ctx=r['context_id'];index,selection=chosen[(ctx,r['method'])];bank=bycontext[ctx]['banks']['arm_clearance']
            checks.equal(source['context_id'],ctx,r['id']+'/exact development context')
            checks.equal(r['values'],bank['candidates'][index]['values'],r['id']+'/selected frozen proposal')
            checks.equal(r['candidate_index'],index,r['id']+'/selected candidate ID')
            checks.equal(r['candidate_id'],bank['candidates'][index]['id'],r['id']+'/selected candidate label')
            entry=next(e for e in selection['checks'] if e['candidate_index']==index)
            checks.equal(r['clearance_minimum_m'],entry['clearance']['minimum_clearance_lower_bound_m'],r['id']+'/selected certificate bound')
            exact_branch=entry
            same(r['model_prediction'],bank['predictions'][index],checks,r['id']+'/unchanged forecast',0)
            checks.require(r['policy_abstain'] is selection['policy_abstain'],r['id']+'/refusal propagated')
            checks.equal(r['geometry_checks'],len(selection['checks']),r['id']+'/geometry budget charged')
            expected_rule=fingerprint(OLD_RULE if r['method']=='original' else NEW_RULE)
            checks.equal(r['clearance_rule_sha256'],expected_rule,r['id']+'/expected single rule')
        else:
            checks.equal((source['phase'],source['role'],source['repeat']),('controls_pre',r['role'],0),r['id']+'/exact matched control source')
            checks.equal(r['values'],source['values'],r['id']+'/unchanged control action')
            checks.require(r['policy_abstain'] is False,r['id']+'/control executes')
            exact_branch=source_row
        for key in ('connection_file','connection_sha256'):
            checks.equal(r[key],exact_branch[key],r['id']+'/selected exact branch/'+key)
        endpoint=read(contained(parent/'native',exact_branch['connection_file']))
        for key in ('q10','q14'):same(r[key],endpoint[key],checks,r['id']+'/certified branch joints/'+key,0)
        checks.equal(r['command_values'],r['values'],r['id']+'/no hidden action replacement')


def audit_requests(study, parent, rows, plan, spec, checks):
    from .calibrated_admission import verify_resolution, verify_connection
    checks.equal([r['id'] for r in rows], [r['id'] for r in plan], 'Every ordered requested slot retained')
    refused = [r for r in rows if r.get('policy_abstain') is True]
    refusal_path = study/'policy-rejections.jsonl'
    saved_refused = replay.read_records(refusal_path) if refusal_path.is_file() else []
    checks.equal(saved_refused, refused, 'Separate exact policy-refusal ledger')
    native_rows = replay.read_records(study/'native/shots.jsonl')
    checks.equal(native_rows, [r for r in rows if r.get('policy_abstain') is not True],
                 'No policy-refused request entered native adapter')
    home = spec['identity']['setup']['home']; prior = read(parent/'prior.json')['records']
    for row, request in zip(rows, plan):
        label = row['id']
        for key, value in request.items(): same(row.get(key), value, checks, label+'/frozen request/'+key, 0)
        checks.require(type(row.get('native_release')) is bool and row.get('action_substituted') is False,
                       label+'/explicit outcome with no action substitution')
        path = contained(study/'native', row['context_resolution_file'])
        checks.equal(sha(path), row['context_resolution_sha256'], label+'/resolution bytes')
        checks.equal(sha(contained(parent/'native',row['context_resolution_file'])),sha(path),label+'/original resolution retained')
        resolution = dict(read(path),file=row['context_resolution_file'],sha256=sha(path),cache_hit=True)
        checks.require(verify_resolution(resolution,prior,home),label+'/verified context resolution')
        path = contained(study/'native',row['connection_file']); connection = read(path)
        checks.equal(sha(path),row['connection_sha256'],label+'/connection bytes')
        checks.equal(sha(contained(parent/'native',row['connection_file'])),sha(path),label+'/unchanged admitted branch')
        checks.require(verify_connection(connection,resolution,home,row['values']),label+'/exact endpoint proof')
        checks.require(connection['passed'] is True,label+'/these prescribed candidates have admitted endpoints')
        for key in ('q10','q14'): checks.numeric(row[key],connection[key],label+'/'+key,0)
        if row.get('policy_abstain') is True:
            checks.require(row['native_release'] is False and row['route_success'] is False,
                           label+'/refused slot cannot claim simulated capture')
            for key in ('invalid_physics','board_bounce','whole_ball_entry','max_retention_s','route_quality'):
                checks.require(row.get(key) is None,label+'/missing native observation/'+key)
            checks.require(not any(k in row for k in ('native_cache_file','native_cache_sha256','native_initial_q10','native_initial_q14')),
                           label+'/no fabricated native cache')
        else:
            checks.require(row['native_release'] is True,label+'/admitted requested shot actually released')
            checks.require(row['scalar_score_audit_passed'] is True,label+'/online independent scoring check')
            checks.equal(row['contact_protocol'],dict(substeps=64,board_mode='articulated',gravity_mode='passive_only'),label+'/unchanged physics')
            checks.equal(row['scoring_version'],'sircus-calibrated-native-capture-v2',label+'/unchanged scoring')
            for key in ('q10','q14'):
                checks.numeric(row['native_initial_'+key],np.asarray(row[key],dtype=np.float32).astype(float),label+'/observed '+key,0)


def contact_rows(study, rows, spec, calibration, checks):
    from .xy650_verify import first_flight
    from .gap_flight_results import scan_contact
    measured = {}; byfile = {}
    for row in rows:
        if row['native_release']: byfile.setdefault(row['native_cache_file'],[]).append(row)
    for name, group in byfile.items():
        path = contained(study/'native',name)
        with np.load(path,allow_pickle=False) as archive:
            arrays = {k:archive[k] for k in ('record_ids','position_m','velocity_m_s','forces_N','time_s')}
        for row in group:
            column = row['native_cache_row']
            checks.equal(str(arrays['record_ids'][column]),row['id'],row['id']+'/contact trace identity')
            checks.equal(sha(path),row['native_cache_sha256'],row['id']+'/contact trace bytes')
            trace = {k:v[:,column] if v.ndim>=2 else v for k,v in arrays.items() if k!='record_ids'}
            metric = first_flight(spec['scene'],row['bucket_xy_m'],row['board_normal'],calibration,trace,
                                 valid=row.get('invalid_physics') is False)
            measured[row['id']] = scan_contact(row,trace,metric,checks)
    result = []
    for row in rows:
        if row['role'] != 'diagnostic': continue
        record = measured.get(row['id']); known = record is not None and record['observation_status']=='observed'
        result.append(dict(record_id=row['id'],context_id=row['context_id'],repeat=row['repeat'],method=row['method'],
            bucket_xy_m=row['bucket_xy_m'],values=row['values'],native_release=row['native_release'],capture=policy.captured(row),
            clean=policy.captured(row) and row.get('route_quality')==2,invalid=bool(row['native_release'] and row.get('invalid_physics')),
            clearance_rejected=row.get('policy_abstain') is True,
            retention_s=row.get('max_retention_s'),
            first_contact_delay_ms=record['first_contact_delay_ms'] if known else None,
            first_post_bounce_contact_channels=record['first_post_bounce_contact_channels'] if known else None,
            **{'robot_contact_'+s:(record['robot_contact_intervals_'+s]>0 if known else None)
               for s in ('before_terminal','post_rebound','after_terminal')}))
    return result


def audit_sources(source,native_source,spec,checks):
    for root,key in ((source,'diagnostic_source_sha256'),(native_source,'native_source_sha256')):
        checks.require(bool(spec[key]),'Nonempty source binding/'+key)
        for name,expected in spec[key].items():checks.equal(sha(contained(root,name)),expected,'Frozen source/'+name)
    own='sircus/mjlab/clearance_correction_verify.py'
    checks.equal(spec['diagnostic_source_sha256'].get(own),sha(Path(__file__)),'Executing independent auditor frozen before outcomes')
    checks.equal({name:importlib.metadata.version(name) for name in spec['packages']},spec['packages'],'Pinned dependencies')


def audit(study,parent,parent_audit,source,native_source,assets):
    # Selection/certificate reconstruction is defined below, separately from
    # the producer and its outcome accounting.
    start=time.monotonic();checks=replay.AuditChecks()
    study,parent,parent_audit,source,native_source,assets=map(lambda p:Path(p).resolve(),
        (study,parent,parent_audit,source,native_source,assets))
    parent_spec,parent_rows,logs,parent_packet,parent_manifest=audit_parent(parent,parent_audit,checks)
    spec,report,packet=read(study/'contract.json'),read(study/'report.json'),read(study/'prediction-plan.json')
    manifest=replay.check_manifest(study,checks)
    checks.equal(report['evidence_sha256'],manifest,'Full completed native evidence')
    checks.equal(spec['version'],STUDY_VERSION,'Single correction study version')
    unsigned=dict(spec);fp=unsigned.pop('fingerprint');checks.equal(digest(unsigned),fp,'Frozen complete contract')
    checks.equal((report['status'],report['phase']),('completed','completed'),'Bounded native worker completed')
    checks.equal((report['contract_fingerprint'],packet['contract_fingerprint']),(fp,fp),'Report and requests bound to contract')
    budget_checks(packet['requests'],checks)
    checks.equal((spec['substeps'],spec['maximum_requests'],spec['repeats']),(64,48,3),'Fixed native budget')
    checks.equal(spec['contexts'],[c.rsplit('-',1)[1] for c in CONTEXTS],'Prespecified development cases')
    checks.equal(spec['methods'],list(METHODS),'Exactly paired original/corrected methods')
    checks.require(spec['training'] is False and spec['noise'] is False,'No training or noise change')
    checks.require(type(packet.get('native_planning_queries')) is int and packet['native_planning_queries']==0,'No native outcome search')
    checks.equal(report['plan_sha256'],sha(study/'prediction-plan.json'),'Completed report retains preoutcome plan')
    checks.require(packet['frozen_before_any_native_outcome'] is True
        and type(packet['frozen_unix_s']) in (int,float)
        and packet['frozen_unix_s']<=report['acquired_unix_s'],'All selection frozen before GPU acquisition')
    for key in ('scene','identity','native_source_sha256','packages'):
        same(spec[key],parent_spec[key],checks,'Unchanged native environment/'+key,0)
    bindings=spec['parent_bindings']
    for name,key in (('report.json','parent_report_sha256'),('contract.json','parent_contract_sha256'),
                     ('evidence-manifest.json','parent_manifest_sha256'),('prediction-plan.json','parent_plan_sha256')):
        checks.equal(bindings[key],sha(parent/name),'Frozen parent binding/'+name)
    checks.equal(bindings['parent_audit_sha256'],sha(parent_audit),'Frozen parent audit binding')
    checks.equal(bindings['parent_evidence_sha256'],parent_manifest,'Complete frozen parent evidence map')
    for name,value in parent_spec['source_sha256'].items():
        checks.equal(spec['diagnostic_source_sha256'].get(name),value,'Original helper code unchanged/'+name)
    for name in ('clearance_correction_study.py','clearance_correction_pipeline.py','clearance_correction_verify.py',
                 'arm_clearance_distance.py','clearance_prediction_diagnosis.py','clearance_horizon_diagnosis.py',
                 'prepare_clearance_diagnosis_report.py'):
        checks.require('sircus/mjlab/'+name in spec['diagnostic_source_sha256'],'Required new source frozen/'+name)
    for name in ('calibration.json','verification.json','setup-home.json','prior.json'):
        checks.equal(sha(study/name),parent_manifest[name],'Unchanged copied input/'+name)
    audit_selection(study,parent,logs,parent_packet,spec,packet,assets,checks)
    rows=replay.read_records(study/'records.jsonl');audit_requests(study,parent,rows,packet['requests'],spec,checks)
    calibrated.check_rows(rows,checks)
    calibration=replay.check_calibration(study,spec['identity']['calibration'],checks)
    from .learning_solve import required_asset_paths
    checks.equal({name:sha(contained(assets.parent,name)) for name in required_asset_paths(assets)},
                 spec['identity']['asset_sha256'],'Unchanged collision assets')
    policy.audit_native_options(study/'native',spec['scene'],calibration,assets,checks)
    traces=calibrated.rescore_rows(study/'native',rows,spec['scene'],calibration,checks,
        'One clearance-distance correction64',spec['identity']['trace_schema'],assets=assets)
    same(report['counts'],counts(rows),checks,'All48 requested outcomes recomputed')
    gates={key:control_gate([r for r in rows if r['phase']==phase])
           for key,phase in (('pre','controls_pre'),('post','controls_post'))}
    same(report['control_gates'],gates,checks,'Matched controls reconstructed')
    checks.require(all(g['status']=='passed' for g in gates.values()),'Both control gates passed')
    comparisons=[dict(method=m,counts=counts([r for r in rows if r['role']=='diagnostic' and r['method']==m])) for m in METHODS]
    same(report['comparisons'],comparisons,checks,'Both paired18 denominators reconstructed')
    enriched=contact_rows(study,rows,spec,calibration,checks)
    expected_ui=[{key:r[key] for key in ('record_id','context_id','method','repeat','bucket_xy_m','values',
        'native_release','capture','clean','invalid','clearance_rejected')} for r in enriched]
    same(report['rows'],expected_ui,checks,'Every reported UI outcome independently reconstructed')
    audit_sources(source,native_source,spec,checks)
    return dict(version=VERSION,status='passed',checks_count=checks.count,checks=checks.entries,failures=[],
        contract_fingerprint=fp,report_sha256=sha(study/'report.json'),plan_sha256=sha(study/'prediction-plan.json'),
        evidence_manifest_sha256=sha(study/'evidence-manifest.json'),auditor_sha256=sha(Path(__file__)),
        producer_source_sha256=spec['diagnostic_source_sha256'],native_source_sha256=spec['native_source_sha256'],
        requested_trials=len(rows),training_requests=0,native_releases=len(traces),counts=counts(rows),control_gates=gates,
        comparisons=comparisons,rows=enriched,source_parent=dict(report_sha256=PARENT_REPORT_SHA,audit_sha256=PARENT_AUDIT_SHA),
        elapsed_s=time.monotonic()-start,
        limitations=['Six post-hoc development cases with three exact repeats; not new held-out generalization.',
            'Only the distance certificate changes; margin, horizon, models, physics and scoring remain fixed.',
            'Held endpoints only; no training, dynamic arm transit, noise,128 validation or policy activation.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('study','parent','parent-audit','source','native-source','assets','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if any(a.output.resolve().is_relative_to(x.resolve()) for x in (a.study,a.parent)):raise ValueError('Audit must be outside immutable evidence')
    try:result=audit(a.study,a.parent,a.parent_audit,a.source,a.native_source,a.assets)
    except Exception as error:
        a.output.write_text(json.dumps(dict(version=VERSION,status='failed',failures=[type(error).__name__+': '+str(error)]),indent=2)+'\n');raise
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','checks_count','requested_trials','native_releases')}))


if __name__=='__main__':main()
