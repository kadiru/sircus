"""Independent fixed64 audit of the clear 135-degree branch's gap/tilt range study."""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
from pathlib import Path
import numpy as np
from .benchmark_verify import Checks, contained, digest, fingerprint, read, sha
from .board_pose_recovery_audit import (exact_repeat,
    verify_confirmations, verify_request_keys)
from .nearby_transfer_audit import load_source, check_source_trace, qualified
from .link_clearance_audit import verify_preservation_slots, release_accounting
from .sweep_worker import atomic
VERSION='sircus-clear-shot-range-independent-audit-v1'
PREPARATION_VERSION='sircus-clear-shot-range-preparation-audit-v1'
STUDY_VERSION='sircus-clear-shot-range-v1'
CASE='parallel-board-miss-09'
SOURCE_ID='parallel-board-miss-09-clearance-0016-b0'


def verify_sources(a,plan,bank,spec,checks):
    source=load_source(a.source_study,a.source_audit,'sircus-link-clearance-study-v1',
        ('preparation.json','rows.jsonl','source-bank.json','preserved-slots.json'),checks)
    checks.equal((spec['source_bindings'],plan['source']),(source['binding'],)*2,'Exact audited source')
    checks.equal((source['spec']['identity'],source['spec']['scene']),(spec['identity'],spec['scene']),'Unchanged calibrated scene')
    for name in ('calibration.json','training-prior.json','setup-home.json'):
        checks.equal((source['manifest'][name],sha(source['study']/name)),(sha(Path(a.output)/name),)*2,'Frozen '+name)
    previous=read(source['study']/'preparation.json')
    old=next(c for c in previous['cases'] if c['id']==CASE)
    original=next(r for r in old['candidates'] if r['id']==SOURCE_ID)
    found=[r for r in source['rows'] if r.get('variant')=='search' and r.get('candidate_id')==SOURCE_ID]
    checks.equal(len(found),1,'Exactly one original source release')
    row=found[0]
    checks.require(row['native_release'] and not row['invalid_physics'] and not qualified(row)
        and row['experimental_route']['robot_contact_steps']==0,'Source was an observed valid arm-free miss')
    checks.equal(original['board_twist_deg'],135.,'Previously clear135-degree board branch')
    checks.numeric(original['values'],[.6803429248048187,-.011044302443505356,.4618683735173024,25.4687518028256,.1094386406066069],
        'Exact source board center, tilt and gap',tolerance=1e-12)
    check_source_trace(source,original,row,checks)
    expected_bank=[dict(id=SOURCE_ID,source_request=original,case_id=CASE,source_directory=row['adapter_directory'],
        source_row_id=row['id'],source_row_sha256=digest(row),source_trace_file=row['native_cache_file'],
        source_trace_sha256=row['native_cache_sha256'],strength='observed_arm_free_non_capture',source_binding=source['binding'])]
    checks.equal(bank,expected_bank,'One exact observed arm-free source with native provenance')
    checks.equal([c['id'] for c in plan['cases']],[CASE],'Only authorized target09')
    expected_case=dict(id=CASE,original=original,source_directory=row['adapter_directory'],historical=row,sources=expected_bank)
    checks.equal({k:plan['cases'][0][k] for k in expected_case},expected_case,'Original source command and evidence unchanged')
    preserved=[]
    for parent in previous['preserved_cases']:
        command=parent['original'];found=[r for r in source['rows'] if r.get('variant')=='preserved' and r.get('context_id')==parent['id']]
        checks.equal(sorted(r['repeat'] for r in found),[0,1,2],'Three previous preservation releases')
        checks.require(all(qualified(r) for r in found),'Previously preserved settings all captured')
        for row in found:check_source_trace(source,command,row,checks)
        preserved.append(dict(id=parent['id'],original=command,source_directory=parent['directory'],directory=parent['directory'],
            source_row_ids=[r['id'] for r in found],source_row_sha256=[digest(r) for r in found]))
    checks.equal([c['id'] for c in preserved],['parallel-board-miss-'+i for i in ('03','13','15')],'All recovered positions preserved')
    checks.equal(plan['preserved_cases'],preserved,'Independent unchanged preservation commands')
    return source


def independent_grid(source):
    """Exactly the requested finite gap/tilt grid, excluding the original."""
    result=[]
    for tilt_delta in (0.,-3.,3.,-6.,6.):
        for gap_delta in (0.,.02,.04,.06,.08,.10,.13,.16,.20,.24):
            if tilt_delta==gap_delta==0:continue
            values=list(source['values']);values[3]+=tilt_delta;values[4]+=gap_delta
            result.append(dict(values=values,tilt_delta_deg=tilt_delta,gap_delta_m=gap_delta))
    return result


def board_joint_difference(request,source):
    return max(abs(math.remainder(x-y,2*math.pi)) for x,y in zip(request['selected_q14'],source['selected_q14'],strict=True))


def verify_grid(case,checks):
    source=case['original'];expected=independent_grid(source)
    checks.equal(len(case['proposals']),49,'All49 prespecified combinations retained')
    checks.equal(len({p['id'] for p in case['proposals']}),49,'Unique proposal identities')
    for index,(proposal,values) in enumerate(zip(case['proposals'],expected,strict=True)):
        checks.equal(proposal['id'],f'{CASE}-range-{index:02d}','Independent grid identity')
        checks.equal(proposal['values'],values['values'],'Exact tilt/gap offset and fixed board center')
        checks.equal((proposal['tilt_delta_deg'],proposal['gap_delta_m']),
            (values['tilt_delta_deg'],values['gap_delta_m']),'Offset labels match actual commands')
        checks.equal((proposal['source_request_id'],proposal['proposal_family'],proposal['method'],proposal['role'],proposal['intended_route']),
            (SOURCE_ID,'gap_only' if values['tilt_delta_deg']==0 else 'tilt_and_gap','clear_shot_range','diagnostic',source['intended_route']),
            'Exact source and prespecified command family')
        for key in ('bucket_xy_m','board_heading_offset_deg','board_twist_deg'):
            checks.equal(proposal[key],source[key],'Unchanged '+key)
        checks.equal((proposal['seed_q10'],proposal['seed_q14']),(source['selected_q10'],source['selected_q14']),'Original joint hints preserved')
        checks.equal((proposal['context_id'],proposal['phase']),(CASE,'search'),'Fixed target and study role')
        checks.require(all(math.isfinite(float(v)) for v in proposal['values']),'Finite requested shot parameters')
        checks.require(all(lo<=v<=hi for lo,v,hi in zip(case['bounds']['low'],proposal['values'],case['bounds']['high'],strict=True)),'Shot remains within configured action limits')


def verify_records(case,checks):
    proposals={p['id']:p for p in case['proposals']};records=case['records']
    checks.equal([r['proposal_id'] for r in records],list(proposals),'Every requested setting accounted in grid order')
    skipped=False
    for record in records:
        checks.equal(record['proposal_sha256'],digest(proposals[record['proposal_id']]),'Exact checked/skipped proposal')
        if record['reason']=='case_budget_not_attempted':
            skipped=True
            checks.equal(record['attempted'],False,'Case-budget skips are not attempted checks')
            checks.equal((record['admitted'],record['admitted_request_ids']),(0,[]),'No candidate from skipped check')
        else:
            checks.require(not skipped,'No preparation resumes after exhausted case budget')
            checks.equal(record['attempted'],True,'Attempted check explicitly identified')
        if record['reason'] in ('proposal_preparation_time_limit','case_preparation_time_limit'):
            checks.equal((record['admitted'],record['admitted_request_ids']),(0,[]),'Interrupted check cannot produce accepted proof')
            checks.require(0<record['timing']['budget_s']<=30,'Bounded proposal preparation time')
    checks.equal(case['preparation_timed_out'],skipped or any(r['reason']=='case_preparation_time_limit' for r in records),'Explicit case budget outcome')
    timeouts=[r for r in records if r['reason'] in ('proposal_preparation_time_limit','case_preparation_time_limit')]
    checks.equal(case['proposal_timeouts'],len(timeouts),'All timeout events accounted')
    checks.equal(case['engine_rebuilds'],sum(r['reason']=='proposal_preparation_time_limit' for r in records),'Fresh verified engine after each nonfinal interruption')


def verify_admissions(case,directory,prior,home,resolutions,checks):
    from .direct_path_first_audit import _proof,verify_seed
    proposals={p['id']:p for p in case['proposals']};admitted={r['id']:r for r in case['admitted']}
    rejected={r['request']['id']:r for r in case['rejected_branches']};seen=[];seen_rejected=[]
    checks.equal(len(admitted),len(case['admitted']),'Unique admitted candidates')
    checks.equal(len(rejected),len(case['rejected_branches']),'Unique rejected board branches')
    checks.require(not(set(admitted)&set(rejected)),'Rejected branches cannot enter admitted candidates')
    for record in case['records']:
        proposal=proposals[record['proposal_id']];ids=record['admitted_request_ids'];rejected_ids=record.get('rejected_request_ids',[])
        checks.equal(record['admitted'],len(ids),'Complete accepted record count')
        checks.require(len(ids)+len(rejected_ids)<=1,'At most one known-seed branch per proposed command')
        for ident in ids+rejected_ids:
            row=admitted[ident] if ident in admitted else rejected[ident]['request']
            checks.equal(ident,proposal['id']+'-b0','Single prepared branch identity')
            checks.equal({k:row[k] for k in proposal if k!='id'},{k:v for k,v in proposal.items() if k!='id'},'No command substitution at admission')
            checks.equal((row['proposal_id'],row['native_prepared'],row['command_values']),(proposal['id'],True,proposal['values']),'Exact prepared request')
            proof=_proof(row,dict(id=CASE,bucket_xy_m=case['original']['bucket_xy_m']),directory,prior,home,resolutions,checks)
            verify_seed(proof,proposal,checks)
            difference=board_joint_difference(row,case['original']);same_tilt=proposal['values'][3]==case['original']['values'][3]
            if ident in admitted:
                checks.require(not same_tilt or difference<=1e-4,'Gap-only candidate preserves the clear board joint branch')
                seen.append(ident)
            else:
                checks.require(same_tilt and difference>1e-4,'Only changed branches at fixed tilt are rejected by this guard')
                checks.equal(rejected[ident]['reason'],'fixed_board_joint_branch_changed','Explicit rejection cause')
                checks.numeric(rejected[ident]['board_joint_change_rad'],difference,'Measured wrapped board joint change')
                checks.equal(record['reason'],'fixed_board_joint_branch_changed','Record reports branch rejection')
                seen_rejected.append(ident)
    checks.equal(seen,list(admitted),'Every and only admitted branch accounted')
    checks.equal(seen_rejected,list(rejected),'Every rejected branch proof accounted')
    checks.equal(case['candidates'],case['admitted'],'Test every admitted command, without post-hoc selection')


def verify_preparation(a,checks):
    from . import replay650_verify as replay
    from .cell1050_rebound_study import parent_inputs
    from .direct_path_first_audit import _proof
    from .intergrid_route_labels import LABEL_VERSION
    from .clear_shot_range_plan import SEARCH_RULE
    study=Path(a.output);spec,plan,bank=(read(study/name) for name in ('contract.json','preparation.json','source-bank.json'))
    fingerprint(spec,checks,'Frozen single-target contract')
    checks.equal((spec['version'],plan['version'],spec['physics_substeps'],spec['repeats'],spec['controls'],spec['preservation_requests'],spec['gpu_cap_s']),
        (STUDY_VERSION,STUDY_VERSION,64,3,18,9,3600),'Bounded native64 protocol')
    checks.equal((plan['seed'],plan['repeats']),(2026112309,3),'Frozen seed and repeats')
    checks.equal((spec['label_version'],plan['label_version']),(LABEL_VERSION,)*2,'Unchanged route scoring')
    checks.equal((spec['search_rule'],plan['search_rule']),(SEARCH_RULE,)*2,'Frozen finite search rule')
    checks.equal((SEARCH_RULE['version'],SEARCH_RULE['proposal_limit_s'],SEARCH_RULE['cpu_case_limit_s'],SEARCH_RULE['one_per_setting'],SEARCH_RULE['board_branch_tolerance_rad']),
        ('sircus-clear-shot-range-menu-v1',30,600,True,1e-4),'Prespecified CPU and board-branch limits')
    checks.equal(spec['plan_sha256'],sha(study/'preparation.json'),'Frozen preparation bytes')
    checks.equal(plan['source_bank_sha256'],sha(study/'source-bank.json'),'Frozen source bank')
    paths={key:str(Path(getattr(a,key)).resolve()) for key in ('source_study','source_audit','parent_study','parent_audit')}
    checks.equal((spec['source_paths'],plan['source_paths']),(paths,paths),'Exact immutable source paths')
    root=Path(getattr(a,'source_root',None) or Path(__file__).resolve().parents[2]).resolve()
    required={'sircus/mjlab/clear_shot_range_plan.py','sircus/mjlab/clear_shot_range_study.py','sircus/mjlab/clear_shot_range_audit.py'}
    checks.require(required<=set(spec['source_sha256']),'Producer and independent auditor frozen')
    for name,value in spec['source_sha256'].items():checks.equal(sha(contained(root,name)),value,'Frozen source '+name)
    for name,value in spec['frozen_files'].items():checks.equal(sha(contained(study,name)),value,'Frozen input '+name)
    checks.require({'preparation.json','slots.json','preserved-slots.json','source-bank.json','calibration.json','training-prior.json','setup-home.json'}<=set(spec['frozen_files']),'All pre-native inputs bound')
    parent,binding,_=parent_inputs(Path(a.parent_study),Path(a.parent_audit))
    checks.equal((spec['identity'],spec['scene'],spec['parent'],plan['parent'],plan['identity'],plan['scene']),
        (parent['identity'],parent['scene'],binding,binding,parent['identity'],parent['scene']),'Original calibration parent preserved')
    verify_sources(a,plan,bank,spec,checks)
    prior,home=read(study/'training-prior.json'),read(study/'setup-home.json')
    replay.check_calibration(study,spec['identity']['calibration'],checks)
    resolutions={}
    for case in plan['cases']+plan['preserved_cases']:
        directory=study/case['directory'];resolutions[case['directory']]={}
        _proof(case['original'],dict(id=case['id'],bucket_xy_m=case['original']['bucket_xy_m']),directory,prior,home,resolutions[case['directory']],checks)
    case=plan['cases'][0]
    checks.equal(case['directory'],'native-09','Isolated native evidence directory')
    checks.equal(read(study/'preparation-09.json'),case,'Complete final preparation receipt')
    checks.equal(case['status'],'prepared','CPU preparation completed')
    verify_grid(case,checks);verify_records(case,checks)
    checks.equal(case['preparation_order'],[p['id'] for p in case['proposals']],'Exact full grid preparation order')
    verify_admissions(case,study/case['directory'],prior,home,resolutions[case['directory']],checks)
    checks.equal(case['selection_order'],[r['id'] for r in case['candidates']],'Every admitted command retained in order')
    count=len(case['candidates'])
    checks.require(count<=49,'At most one checked shot per grid setting')
    checks.equal((spec['initial_requests'],spec['maximum_confirmation_requests'],spec['maximum_total_requests']),
        (3+count,3*count,30+4*count),'Originals, candidates, fresh confirmations, preservation and controls')
    return spec,plan,root,prior,home,resolutions


def preflight(a):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Hide GPUs for audit')
    checks=Checks();verify_preparation(a,checks);study=Path(a.output);plan=read(study/'preparation.json')
    checks.require(not (study/'native-started.json').exists(),'No native release before independent preflight')
    verify_request_keys(plan,read(study/'slots.json'),checks)
    verify_preservation_slots(plan,read(study/'preserved-slots.json'),checks)
    return dict(version=PREPARATION_VERSION,status='passed',checks_count=checks.count,native_execution_performed=False,
        contract_sha256=sha(study/'contract.json'),preparation_sha256=sha(study/'preparation.json'),plan_sha256=sha(study/'preparation.json'),
        slots_sha256=sha(study/'slots.json'),preserved_slots_sha256=sha(study/'preserved-slots.json'),
        source_bank_sha256=sha(study/'source-bank.json'),verifier_source_sha256=sha(__file__))


def independent_summary(plan,rows):
    """Rebuild grid outcomes directly from the native ledger, not report totals."""
    from .hybrid_miss_audit import classify
    summaries=[]
    for case in plan['cases']:
        observed=[r for r in rows if r.get('context_id')==case['id']]
        wins={r['candidate_id'] for r in observed if r.get('variant')=='search' and qualified(r)}
        groups={}
        for variant,requested in (('original',3),('search',len(case['candidates'])),('repeat',3*len(wins))):
            found=[r for r in observed if r.get('variant')==variant];labels=[classify(r) for r in found]
            groups[variant]=dict(requested=requested,completed=len(found),released=sum(r['native_release'] for r in found),
                captures=sum(qualified(r) for r in found),direct=labels.count('direct'),table=labels.count('table'),
                robot_contacts=sum(r['experimental_route']['robot_contact_steps']>0 for r in found),outcomes=labels,rows=found)
        repeated=[]
        for candidate in case['candidates']:
            if candidate['id'] not in wins:continue
            confirm=[r for r in observed if r.get('variant')=='repeat' and r['candidate_id']==candidate['id']]
            repeated.append(dict(candidate_id=candidate['id'],request=candidate,repeat_captures=sum(qualified(r) for r in confirm),repeat_completed=len(confirm)))
        confirmed=[r for r in repeated if r['repeat_captures']==r['repeat_completed']==3]
        records={r['proposal_id']:r for r in case['records']};candidates={r['proposal_id']:r for r in case['candidates']};grid=[]
        for p in case['proposals']:
            r=records[p['id']];candidate=candidates.get(p['id']);reason=r['reason']
            shots=[x for x in observed if candidate and x.get('candidate_id')==candidate['id']]
            initial=next((x for x in shots if x.get('variant')=='search'),None);confirms=[x for x in shots if x.get('variant')=='repeat']
            checked=r['attempted'] and reason not in ('proposal_preparation_time_limit','case_preparation_time_limit')
            grid.append(dict(proposal_id=p['id'],tilt_deg=p['values'][3],drop_gap_m=p['values'][4],tilt_delta_deg=p['tilt_delta_deg'],gap_delta_m=p['gap_delta_m'],
                setup_status='admitted' if candidate else 'rejected' if checked else 'not_checked',setup_reason=reason,candidate_id=candidate['id'] if candidate else None,
                initial_outcome=classify(initial) if initial else None,initial_capture=qualified(initial) if initial else None,
                robot_contact=initial['experimental_route']['robot_contact_steps']>0 if initial else None,
                repeat_completed=len(confirms),repeat_captures=sum(qualified(x) for x in confirms),
                confirmed=bool(candidate and any(x['candidate_id']==candidate['id'] for x in confirmed))))
        summaries.append(dict(id=case['id'],bucket_xy_m=case['original']['bucket_xy_m'],results=groups,grid=grid,
            proposed=len(case['proposals']),checked=sum(r['attempted'] for r in case['records']),admitted=len(case['admitted']),selected=len(case['candidates']),
            repeated_settings=repeated,confirmed_settings=confirmed,best_candidate=confirmed[0]['request'] if confirmed else None))
    return summaries

def verify_launch_receipt(study, checks):
    marker=read(study/'native-started.json')
    receipt_path=Path(marker['preflight_audit'])
    checks.equal(sha(receipt_path),marker['preflight_audit_sha256'],'Launch bound the unchanged independent preflight')
    receipt=read(receipt_path)
    checks.equal((receipt['version'],receipt['status'],receipt['native_execution_performed']),
                 (PREPARATION_VERSION,'passed',False),'CPU-only preflight passed before native launch')
    checks.require(type(receipt['checks_count']) is int and receipt['checks_count']>0,'Preflight performed actual checks')
    for field,name in (('contract_sha256','contract.json'),('preparation_sha256','preparation.json'),
                       ('slots_sha256','slots.json'),('source_bank_sha256','source-bank.json'),('preserved_slots_sha256','preserved-slots.json')):
        checks.equal(receipt[field],sha(study/name),'Preflight binds final frozen '+name)
    checks.equal(receipt['verifier_source_sha256'],sha(Path(__file__)),'Same independent verifier before and after native')
    checks.equal(marker['slots_sha256'],sha(study/'slots.json'),'Launch binds exact initial command schedule')


def audit(a):
    from . import replay650_verify as replay
    from .cell1050_rebound_study import control_requests
    from .direct_path_first_audit import verify_heading
    from .intergrid_route_labels import annotate_routes
    from .parallel_capture_grid_audit import observed_rows, rescore_packet, verify_release
    from .policy64_verify import audit_native_options
    from .table_training_verify import control_gate, rescore_rows

    study, checks = Path(a.output), Checks()
    spec,plan,source,prior,home,resolutions = verify_preparation(a,checks)
    report = read(study/'report.json')
    checks.equal((report['version'],report['status'],report['phase'],report['scope'],report['physics_substeps']),
                 (STUDY_VERSION,'completed','completed','development',64),'Completed native64 development experiment')
    checks.require(0 < report['gpu_lease_s'] <= 3600, 'Bounded acquired4090 time')
    manifest = replay.check_manifest(study,checks)
    checks.equal(report['evidence_manifest_sha256'],sha(study/'evidence-manifest.json'),'Final report binds all saved evidence')
    for name,value in spec['frozen_files'].items():
        checks.equal(manifest[name],value,'Final manifest preserves frozen preparation/'+name)
    calibration = replay.check_calibration(study,spec['identity']['calibration'],checks)
    verify_launch_receipt(study,checks)
    slots = read(study/'slots.json'); repeat_slots = read(study/'repeat-slots.json')
    verify_request_keys(plan, slots, checks)
    preserved_slots=read(study/'preserved-slots.json'); verify_preservation_slots(plan,preserved_slots,checks)
    checks.equal(read(study/'native-started.json')['slots_sha256'], sha(study/'slots.json'), 'Native launch binds frozen initial commands')
    initial_count = 3+sum(len(c['candidates']) for c in plan['cases'])
    checks.equal((spec['initial_requests'], spec['maximum_confirmation_requests'], spec['maximum_total_requests']),
                 (initial_count, (initial_count-3)*3, initial_count+(initial_count-3)*3+27), 'Explicit maximum native budget')
    ledger, timings = observed_rows(study/'rows.jsonl'), observed_rows(study/'timing.jsonl')
    sources = read(a.parent_study/'control-replays.json')
    used = {c['directory']: set() for c in plan['cases']+plan['preserved_cases']}; seen = set(); phases = []; observed = []; controls = {}
    expected_commands = {s['request']['id']: s for s in preserved_slots+slots+repeat_slots}
    checks.equal(len(expected_commands), len(preserved_slots)+len(slots)+len(repeat_slots), 'Globally unique diagnostic commands')
    for index, path in enumerate(sorted((study/'requests').glob('*.json'))):
        relative = f'requests/batch-{index:04d}.json'
        checks.equal(path.relative_to(study).as_posix(), relative, 'Contiguous native packet ledger')
        packet = read(path); phase = packet['phase']; requests = packet['rows']; directory = packet['adapter_directory']
        control = phase in ('controls_pre', 'controls_post')
        checks.equal(packet['control'], control, relative+'/control declaration')
        checks.require(phase in ('controls_pre', 'preservation', 'diagnostic', 'confirmation', 'controls_post'), relative+'/known phase')
        phases.append(('controls_pre', 'preservation', 'diagnostic', 'confirmation', 'controls_post').index(phase))
        checks.require(0 < len(requests) <= 16, relative+'/bounded native batch')
        timing = next(timings, None); checks.require(timing is not None, relative+'/timing exists')
        checks.equal((timing['packet'], timing['phase']), (relative, phase), relative+'/matching timing identity')
        if control:
            checks.equal(directory, 'native-controls', relative+'/control directory')
            checks.require(phase not in controls, relative+'/one control block per side')
            checks.equal(requests, control_requests(sources, phase), relative+'/exact control requests')
        else:
            checks.require(directory in used, relative+'/known diagnostic directory')
        rows = []
        for request in requests:
            row = next(ledger, None); checks.require(row is not None, request['id']+'/native observation exists')
            verify_release(row, request, relative, manifest[relative], phase, checks)
            if row.get('route_success'):
                checks.require(row.get('shot_route') in ('direct', 'table') or control,
                               row['id']+'/standard success has explicit direct/table route')
            checks.require(row['id'] not in seen, row['id']+'/exactly one native release'); seen.add(row['id'])
            checks.equal(row['adapter_directory'], directory, row['id']+'/native archive namespace')
            if not control:
                slot = expected_commands[request['id']]
                checks.equal((slot['request'], slot['directory']), (request, directory), row['id']+'/frozen exact command')
                for field in ('variant', 'repeat', 'context_id', 'candidate_id'):
                    checks.equal(row[field], request[field], row['id']+'/'+field)
                verify_heading(row, request, study/directory, spec['scene'], home, prior, resolutions[directory], checks)
            rows.append(row)
        if control:
            controls[phase] = rows
        else:
            alias = {'native-heading/'+name[len(directory)+1:]: v for name, v in manifest.items() if name.startswith(directory+'/')}
            list(rescore_packet(study/directory, rows, spec['scene'], calibration, alias, used[directory], checks))
            relabeled = annotate_routes(copy.deepcopy(rows), study/directory)
            for row, other in zip(rows, relabeled, strict=True):
                checks.equal((row['shot_route'], row['table_rebounds_before_entry']),
                             (other['shot_route'], other['table_rebounds_before_entry']), row['id']+'/independent route label')
            observed.extend(rows)
    checks.equal(phases, sorted(phases), 'Controls bracket initial search then confirmations')
    checks.require(next(ledger, None) is None and next(timings, None) is None, 'No unaccounted rows or timings')
    checks.equal([r['id'] for r in observed], [s['request']['id'] for s in preserved_slots+slots+repeat_slots], 'Every scheduled diagnostic executed once in order')
    eligible = verify_confirmations(plan, observed, checks)
    verify_request_keys(plan, repeat_slots, checks, confirmations=True, eligible=eligible)
    for directory, archives in used.items():
        audit_native_options(study/directory, spec['scene'], calibration, a.assets, checks)
        checks.equal(archives, {p.relative_to(study/directory).as_posix() for p in (study/directory).rglob('*.npz')},
                     directory+'/complete native archive inventory')
    control_rows = [r for group in controls.values() for r in group]
    audit_native_options(study/'native-controls', spec['scene'], calibration, a.assets, checks)
    rescore_rows(study/'native-controls', control_rows, spec['scene'], calibration, spec['identity']['trace_schema'], a.assets, checks)
    checks.equal({r['native_cache_file'] for r in control_rows},
                 {p.relative_to(study/'native-controls').as_posix() for p in (study/'native-controls').rglob('*.npz')}, 'Complete native control archives')
    checks.equal(set(controls), {'controls_pre', 'controls_post'}, 'Both matched control blocks present')
    for phase, records in controls.items():
        gate = control_gate(records); checks.equal(gate['status'], 'passed', phase+'/native control behavior')
        checks.equal(report['controls'][phase], gate, phase+'/independent control counts')
    preserved=[r for r in observed if r.get('variant')=='preserved']
    checks.equal(report['preserved_shots'],dict(requested=9,completed=len(preserved),captures=sum(qualified(r) for r in preserved),passed=len(preserved)==9 and all(qualified(r) for r in preserved),rows=preserved),'All nine unchanged successful recipes preserved')
    checks.require(len(preserved)==9 and all(qualified(r) for r in preserved),'Three successful settings still capture3/3')
    summary = independent_summary(plan, observed)
    source_keys=('id','values','bucket_xy_m','board_heading_offset_deg','board_twist_deg','selected_q10','selected_q14')
    checks.equal(report['source_summary'],dict({k:plan['cases'][0]['original'][k] for k in source_keys},prior_no_robot_contact=True),'Explicit immutable source summary')
    checks.require(len(observed)+18 <= 226, 'Authorized total native release cap')
    checks.equal(report['cases'], summary, 'Independent proposal, original, search and confirmation counts')
    accounting=release_accounting(len(slots),len(repeat_slots),len(preserved))
    checks.equal(len(observed),len(slots)+len(repeat_slots)+len(preserved),'Complete diagnostic and preservation denominator')
    checks.equal(report['progress'], accounting['progress'], 'Complete diagnostic denominator')
    checks.equal((report['native_releases'], report['initial_requests'], report['confirmation_requests']),
                 (accounting['native_releases'], len(slots), len(repeat_slots)), 'All native releases including controls accounted')
    return dict(version=VERSION, status='passed', checks_count=checks.count, report_sha256=sha(study/'report.json'),
        plan_sha256=sha(study/'preparation.json'), source_bank_sha256=sha(study/'source-bank.json'),
        contract_sha256=sha(study/'contract.json'), evidence_manifest_sha256=sha(study/'evidence-manifest.json'),
        verifier_source_sha256=sha(Path(__file__)), frozen_source_root=str(source),
        initial_requests=len(slots), confirmation_requests=len(repeat_slots), preservation_requests=len(preserved), control_releases=len(control_rows), preserved_shots={k:v for k,v in report['preserved_shots'].items() if k!='rows'},
        results=[dict(id=c['id'], bucket_xy_m=c['bucket_xy_m'], proposed=c['proposed'], checked=c['checked'],
            admitted=c['admitted'], selected=c['selected'], results={v: {k: x for k, x in g.items() if k != 'rows'}
                for v, g in c['results'].items()}, confirmed_settings=c['confirmed_settings']) for c in summary])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('source-study','source-audit','parent-study','parent-audit','assets','output','audit'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--source-root',type=Path)
    parser.add_argument('--stage',choices=('preflight','final'),default='final')
    args=parser.parse_args()
    roots=[args.output,args.source_study,args.parent_study,args.source_audit,args.parent_audit]
    targets=(args.audit,args.audit.with_suffix(args.audit.suffix+'.tmp'))
    if args.audit.exists() or any(p.resolve()==r.resolve() or p.resolve().is_relative_to(r.resolve()) for p in targets for r in roots):
        parser.error('Write a fresh audit outside all immutable source evidence')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':parser.error('Hide GPUs for this CPU-only audit')
    try:result=preflight(args) if args.stage=='preflight' else audit(args)
    except Exception as error:result=dict(version=PREPARATION_VERSION if args.stage=='preflight' else VERSION,status='failed',error=f'{type(error).__name__}: {error}')
    atomic(args.audit,result)
    print(json.dumps(result),flush=True)
    if result['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
