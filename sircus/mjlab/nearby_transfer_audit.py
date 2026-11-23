"""Read-only audit of nearby successful-shot transfer and native64 repeats.

Preflight verifies immutable teachers, geometry transformations and full setup
proofs. The final pass additionally rescores every saved native ball/contact
array. It never runs physics, allocates CUDA or modifies source evidence.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path

import numpy as np

from .benchmark_verify import Checks, contained, digest, fingerprint, read, sha
from .board_pose_recovery_audit import (exact_repeat, independent_summary,
    recompute_screen, verify_admissions, verify_confirmations, verify_request_keys)
from .sweep_worker import atomic

VERSION = 'sircus-nearby-transfer-independent-audit-v1'
PREPARATION_VERSION = 'sircus-nearby-transfer-preparation-audit-v1'
STUDY_VERSION = 'sircus-nearby-transfer-study-v1'
CASES = tuple('parallel-board-miss-' + i for i in ('03', '09', '13', '15', '19'))
SOURCE_VERSIONS = dict(recovery='sircus-board-pose-recovery-v1',
    target='sircus-parallel-board-study-v1', guarded='sircus-guarded-board-fallback-study-v1')


def qualified(row):
    """A teacher needs observed valid retention, not predicted clearance."""
    return (row.get('native_release') is True and row.get('invalid_physics') is False
            and row.get('route_success') is True and row.get('shot_route') in ('direct', 'table'))


def physical_command(request):
    return {key: request[key] for key in ('values', 'bucket_xy_m', 'board_heading_offset_deg',
        'board_twist_deg', 'selected_q10', 'selected_q14', 'context_resolution_file',
        'context_resolution_sha256', 'prepared_connection_file', 'prepared_connection_sha256')}


def load_source(study, audit_path, version, files, checks):
    """Older recovery contracts predate label_version; their identity still binds it."""
    study, audit_path = Path(study), Path(audit_path)
    report, spec, manifest, receipt = (read(study/'report.json'), read(study/'contract.json'),
        read(study/'evidence-manifest.json'), read(audit_path))
    fingerprint(spec, checks, str(study)+'/source contract')
    checks.equal((spec['version'], spec['physics_substeps'], report['status'], report['phase'], receipt['status']),
                 (version, 64, 'completed', 'completed', 'passed'), str(study)+'/audited native64 source')
    checks.require(type(receipt.get('checks_count')) is int and receipt['checks_count'] > 0,
                   str(study)+'/nonempty independent audit')
    checks.equal(receipt['report_sha256'], sha(study/'report.json'), str(study)+'/audit binds report')
    checks.equal((receipt['evidence_manifest_sha256'], report['evidence_manifest_sha256']),
                 (sha(study/'evidence-manifest.json'),)*2, str(study)+'/report and audit bind evidence')
    names = tuple(dict.fromkeys(('report.json', 'contract.json', 'evidence-manifest.json', *files)))
    binding = {name: sha(contained(study, name)) for name in names}
    binding['audit'] = sha(audit_path)
    for name in ('contract.json', *files):
        checks.equal(manifest[name], binding[name], str(study)+'/source bytes/'+name)
    rows = [json.loads(line) for line in (study/'rows.jsonl').read_text().splitlines() if line.strip()]
    checks.equal(len({r['id'] for r in rows}), len(rows), str(study)+'/unique observed row IDs')
    return dict(study=study, audit_path=audit_path, report=report, spec=spec, manifest=manifest,
                receipt=receipt, binding=binding, rows=rows)


def check_source_trace(source, request, row, checks):
    """Authenticate the requested joint branch and its actual saved native trace."""
    root, manifest = source['study'], source['manifest']
    label = row['id']
    for field in ('values', 'bucket_xy_m', 'board_heading_offset_deg', 'board_twist_deg'):
        checks.equal(row[field], request[field], label+'/source command '+field)
    for field in ('q10', 'q14'):
        checks.equal(row[field], request['selected_'+field], label+'/source exact joint '+field)
    for request_key, row_key in (('context_resolution', 'context_resolution'), ('prepared_connection', 'connection')):
        checks.equal(row[row_key+'_sha256'], request[request_key+'_sha256'], label+'/source proof '+request_key)
        checks.equal(row[row_key+'_file'], request[request_key+'_file'], label+'/source proof filename')
        relative = row['adapter_directory']+'/'+request[request_key+'_file']
        checks.equal((manifest[relative], sha(contained(root, relative))),
                     (request[request_key+'_sha256'],)*2, label+'/immutable source setup proof')
    relative = row['adapter_directory']+'/'+row['native_cache_file']
    checks.equal((manifest[relative], sha(contained(root, relative))),
                 (row['native_cache_sha256'],)*2, label+'/immutable observed native trajectory')
    packet = row['request_packet']
    checks.equal((manifest[packet], sha(contained(root, packet))),
                 (row['request_packet_sha256'],)*2, label+'/immutable native command packet')
    commands = read(contained(root, packet))['rows']
    matches = [r for r in commands if r['id'] == row['id']]
    checks.equal(len(matches), 1, label+'/one matching native command')
    checks.equal(physical_command(matches[0]), physical_command(request), label+'/source command actually released')


def require_teacher_rows(rows, *, repeats, checks, label):
    checks.equal(len(rows), repeats, label+'/declared observed repeat count')
    checks.equal(len({r['id'] for r in rows}), repeats, label+'/distinct native releases')
    checks.require(all(qualified(r) for r in rows), label+'/every teacher release captured without robot assistance')
    if repeats == 3:
        checks.equal(sorted(r['repeat'] for r in rows), [1, 2, 3], label+'/three recorded confirmations')


def source_entry(source, key, request, rows, checks):
    count = 1 if key == 'guarded' else 3
    require_teacher_rows(rows, repeats=count, checks=checks, label=key+':'+request['id'])
    for row in rows:
        check_source_trace(source, request, row, checks)
    checks.require(len({r['adapter_directory'] for r in rows}) == 1, 'Source repeats share proof namespace')
    return dict(id=key+':'+request['id'], study_key=key, source_request=request,
        source_directory=rows[0]['adapter_directory'], source_row_ids=[r['id'] for r in rows],
        source_row_sha256=[digest(r) for r in rows],
        source_proofs={request[k+'_file']: request[k+'_sha256'] for k in ('context_resolution', 'prepared_connection')},
        repeat_captures=count, repeat_completed=count,
        strength='single_capture' if count == 1 else 'confirmed_3_of_3',
        bucket_xy_m=request['bucket_xy_m'], source_binding=source['binding'],
        observed_routes=[r['shot_route'] for r in rows])


def verify_sources(a, plan, bank, spec, checks):
    files = dict(target=('choices.json','layouts.json','rows.jsonl','training-prior.json','calibration.json','setup-home.json'),
        recovery=('preparation.json','rows.jsonl'), guarded=('choices.json','rows.jsonl'))
    sources = {key: load_source(getattr(a,key+'_study'), getattr(a,key+'_audit'), version, files[key], checks)
               for key,version in SOURCE_VERSIONS.items()}
    for key, source in sources.items():
        checks.equal((source['spec']['identity'], source['spec']['scene']),
                     (spec['identity'], spec['scene']), key+'/identical calibrated scene and original asset identity')
        for name in ('calibration.json', 'training-prior.json', 'setup-home.json'):
            value = sha(source['study']/name)
            checks.equal((source['manifest'][name], value),
                         (sha(Path(a.output)/name),)*2, key+'/same exact '+name)
    checks.equal(plan['source'], {key:sources[key]['binding'] for key in ('target','recovery','guarded')},
                 'Preparation binds all source audits and frozen inputs')
    target = sources['target']
    choices = read(target['study']/'choices.json')['slots']
    requests = {s['context_id']: s['request'] for s in choices if s['method'] == 'baseline'}
    layouts = read(target['study']/'layouts.json')['layouts']
    expected_bank, expected_cases = [], []
    for layout in layouts:
        request = requests[layout['id']]
        rows = [r for r in target['rows'] if r.get('context_id') == layout['id'] and r.get('method') == 'baseline']
        checks.equal(sorted(r['repeat'] for r in rows), [1,2,3], layout['id']+'/three prior baseline releases')
        for row in rows:
            check_source_trace(target, request, row, checks)
        if layout['role'] == 'previous_miss':
            checks.require(all(r['native_release'] and not r['invalid_physics'] and not r['route_success'] for r in rows),
                           layout['id']+'/all prior target attempts were valid misses')
            expected_cases.append(dict(id=layout['id'], original=request, historical=rows[0],
                historical_repeats=rows, source_directory=rows[0]['adapter_directory']))
        else:
            checks.equal(layout['role'], 'known_regression', 'Only successful regression control enters source bank')
            expected_bank.append(source_entry(target, 'parallel', request, rows, checks))
    checks.equal([c['id'] for c in expected_cases], list(CASES), 'Exactly the five audited previous misses')
    checks.equal([c['id'] for c in plan['cases']], list(CASES), 'No failed target omitted from new study')
    for case, expected in zip(plan['cases'], expected_cases, strict=True):
        checks.equal({key:case[key] for key in expected}, expected, case['id']+'/all old commands and observations preserved')
    recovery = sources['recovery']
    for case in read(recovery['study']/'preparation.json')['cases']:
        for request in case['candidates']:
            rows = [r for r in recovery['rows'] if r.get('candidate_id') == request['id']]
            initial = [r for r in rows if r.get('variant') == 'search']
            repeats = [r for r in rows if r.get('variant') == 'repeat']
            if len(initial) != 1 or not qualified(initial[0]) or len(repeats) != 3 or not all(map(qualified,repeats)):
                continue
            check_source_trace(recovery, request, initial[0], checks)
            entry = source_entry(recovery, 'recovery', request, repeats, checks)
            entry['initial_search_row_id'] = initial[0]['id']
            expected_bank.append(entry)
    guarded = sources['guarded']; by_id = {r['id']:r for r in guarded['rows']}
    for slot in read(guarded['study']/'choices.json')['slots']:
        request = slot['request']
        if slot['method'] == 'baseline' and request is not None and qualified(by_id[request['id']]):
            expected_bank.append(source_entry(guarded, 'guarded', request, [by_id[request['id']]], checks))
    expected_bank.sort(key=lambda r:r['id'])
    checks.equal([sum(r['study_key'] == key for r in expected_bank) for key in ('recovery','parallel','guarded')],
                 [7,1,15], 'Seven confirmed recovery settings, one repeated control, fifteen single captures')
    checks.equal(bank, expected_bank, 'Independent complete source bank with exact branch, trace, audit provenance')
    return sources


def proposal_values(source, target, fraction, delta):
    old = source['source_request']['values']
    return [old[0]+fraction*(target[0]-source['bucket_xy_m'][0]),
            old[1]+fraction*(target[1]-source['bucket_xy_m'][1]),
            old[2]+delta[0], old[3]+delta[1], old[4]+delta[2]]


def verify_proposals(case, bank, checks):
    """Independently rebuild every transfer, adjustment and source joint hint."""
    target = case['original']['bucket_xy_m']
    sources = sorted(bank, key=lambda r:(math.dist(target,r['bucket_xy_m']),r['id']))
    deltas = ((0.,0.,0.,0.),(-.04,0.,0.,0.),(.04,0.,0.,0.),(0.,-6.,0.,0.),(0.,6.,0.,0.),
        (0.,0.,-.08,0.),(0.,0.,.08,0.),(0.,0.,0.,-8.),(0.,0.,0.,8.),
        (0.,-3.,-.04,0.),(0.,-3.,.04,0.),(0.,3.,-.04,0.),(0.,3.,.04,0.),(-.02,0.,.04,0.),(.02,0.,-.04,0.))
    expected = []
    for seed in sources:
        source = seed['source_request']
        for mode, fraction in (('retain',0.), ('blend',.5), ('translate',1.)):
            for index, delta in enumerate(deltas):
                values = proposal_values(seed,target,fraction,delta)
                heading = source.get('board_heading_offset_deg',0.)+delta[3]
                if (any(v < lo or v > hi for v,lo,hi in zip(values,case['bounds']['low'],case['bounds']['high'],strict=True))
                    or abs(heading)>180 or math.dist(values[:2],target)<=.15):
                    continue
                expected.append(dict(id=case['id']+f'-transfer-{len(expected):04d}',context_id=case['id'],
                    values=values,bucket_xy_m=list(target),board_heading_offset_deg=heading,
                    board_twist_deg=source.get('board_twist_deg',0.),phase='search',role='diagnostic',method='nearby_transfer',
                    intended_route=source.get('intended_route','direct'),group=mode,
                    seed_q10=list(source['selected_q10']),seed_q14=list(source['selected_q14']),
                    source_request_id=source['id'],source_bank_id=seed['id'],source_strength=seed['strength'],
                    source_distance_m=math.dist(target,source['bucket_xy_m']),transfer_mode=mode,transfer_fraction=fraction,
                    parameter_delta=dict(zip(('height_m','tilt_deg','gap_m','heading_deg'),delta,strict=True)),delta_index=index))
    base = [p for p in case['proposals'] if 'retry_of' not in p]
    retries = [p for p in case['proposals'] if 'retry_of' in p]
    checks.equal(base, expected, case['id']+'/complete deterministic nearby transfer menu')
    checks.require(len({p['id'] for p in case['proposals']})==len(case['proposals']),case['id']+'/unique proposal identity')
    originals = {p['id']:p for p in base}
    for retry in retries:
        source = originals[retry['retry_of']]
        checks.equal(retry,dict(source,id=source['id']+'-multistart',retry_of=source['id'],ik_search='multistart16'),
                     retry['id']+'/same teacher and geometry; only IK search changes')
    checks.require(len(retries)<=8,case['id']+'/at most eight multistart retries')
    return base,retries


def score_key(request, predictions):
    score = predictions[request['id']].get('score_m')
    valid = type(score) in (int,float) and math.isfinite(score)
    return (not valid,-score if valid else 0.,request['id'])


def independent_order(case, bank):
    target = case['original']['bucket_xy_m']
    ordered = sorted(bank,key=lambda s:(math.dist(target,s['bucket_xy_m']),s['id']))
    confirmed = [s for s in ordered if s['strength']=='confirmed_3_of_3'][:3]
    reserved_ids = {s['id'] for s in confirmed}
    reserved = (confirmed+[s for s in ordered if s['id'] not in reserved_ids])[:8]
    reserved_ids = {s['id'] for s in reserved}
    available = [p for p in case['proposals'] if 'retry_of' not in p and case['wrist_reach'][p['id']]['passed']]
    chosen = [p for p in available if p['source_bank_id'] in reserved_ids and p['delta_index']==0]
    used = {p['id'] for p in chosen}
    groups = [sorted([p for p in available if p['source_bank_id']==source['id'] and p['transfer_mode']==mode and p['id'] not in used],
                     key=lambda p:score_key(p,case['nominal_predictions']))
              for source in ordered for mode in ('retain','blend','translate')]
    for i in range(max(map(len,groups),default=0)):
        chosen.extend(g[i] for g in groups if i<len(g))
    return [p['id'] for p in chosen[:96]]


def source_summary(bank):
    return dict(settings=len(bank),confirmed_settings=sum(s['strength']=='confirmed_3_of_3' for s in bank),
        single_capture_settings=sum(s['strength']=='single_capture' for s in bank),
        sources=[dict(id=s['id'],bucket_xy_m=s['bucket_xy_m'],study_key=s['study_key'],strength=s['strength'],
            repeat_captures=s['repeat_captures'],repeat_completed=s['repeat_completed'],
            report_sha256=s['source_binding']['report.json'],audit_sha256=s['source_binding']['audit'],
            values=s['source_request']['values'],board_heading_offset_deg=s['source_request'].get('board_heading_offset_deg',0.),
            board_twist_deg=s['source_request'].get('board_twist_deg',0.)) for s in bank])


def independent_candidates(rows, predictions, limit=20):
    from .board_pose_recovery_audit import ranked_diverse
    zero = [r for r in rows if r['delta_index']==0]
    best_by_source = {}
    for row in sorted(zero,key=lambda r:score_key(r,predictions)):
        best_by_source.setdefault(row['source_bank_id'],row)
    nearest = sorted(best_by_source.values(),key=lambda r:(r['source_distance_m'],r['source_bank_id']))
    confirmed = [r for r in nearest if r['source_strength']=='confirmed_3_of_3'][:3]
    ids = {r['source_bank_id'] for r in confirmed}
    reserved = (confirmed+[r for r in nearest if r['source_bank_id'] not in ids])[:8]
    for mode in ('retain','blend','translate'):
        family = [r for r in zero if r['transfer_mode']==mode]
        if family:
            reserved.append(min(family,key=lambda r:(r['source_strength']!='confirmed_3_of_3',score_key(r,predictions))))
    chosen,geometries = [],set()
    for row in reserved+ranked_diverse(rows,predictions,len(rows)):
        key=tuple(round(float(v),10) for v in (*row['values'],row['board_heading_offset_deg'],row['board_twist_deg']))
        if key not in geometries:
            chosen.append(row);geometries.add(key)
        if len(chosen)>=limit:
            break
    return chosen


def verify_preparation(a, checks):
    from . import replay650_verify as replay
    from .cell1050_rebound_study import parent_inputs
    from .direct_path_first_audit import _proof
    from .fixed35_broad_native import reach_models
    from .intergrid_flight_ranker import FlightRanker
    from .intergrid_route_labels import LABEL_VERSION
    from .nearby_transfer_plan import SEARCH_RULE

    study = Path(a.output)
    spec, plan, bank = (read(study/name) for name in ('contract.json','preparation.json','source-bank.json'))
    fingerprint(spec, checks, 'Frozen nearby-shot protocol')
    checks.equal((spec['version'],plan['version'],spec['physics_substeps'],spec['repeats'],spec['controls'],spec['gpu_cap_s']),
                 (STUDY_VERSION,STUDY_VERSION,64,3,18,3600), 'Declared five-target native64 bounded development study')
    checks.equal((spec['label_version'],plan['label_version']), (LABEL_VERSION,)*2, 'Unchanged table-assisted route scoring')
    checks.equal((plan['repeats'],plan['seed']), (3,2026112304), 'Deterministic preparation seed and confirmations')
    checks.equal((spec['search_rule'],plan['search_rule']), (SEARCH_RULE,SEARCH_RULE), 'Exact transfer and preparation rule')
    checks.equal(spec['plan_sha256'],sha(study/'preparation.json'), 'Frozen preparation bytes')
    checks.equal(plan['source_bank_sha256'],sha(study/'source-bank.json'), 'Frozen teacher bank bytes')
    paths={key:str(Path(getattr(a,key)).resolve()) for key in ('target_study','target_audit','recovery_study','recovery_audit',
        'guarded_study','guarded_audit','parent_study','parent_audit')}
    checks.equal((spec['source_paths'],plan['source_paths']), (paths,paths), 'Exact protected source locations')
    source = Path(getattr(a,'source_root',None) or Path(__file__).resolve().parents[2]).resolve()
    required = {'sircus/mjlab/nearby_transfer_plan.py','sircus/mjlab/nearby_transfer_study.py','sircus/mjlab/nearby_transfer_audit.py'}
    checks.require(required <= set(spec['source_sha256']), 'Computational producer and independent verifier frozen')
    for name,value in spec['source_sha256'].items():
        checks.equal(sha(contained(source,name)),value,'Exact frozen source/'+name)
    for name,value in spec['frozen_files'].items():
        checks.equal(sha(contained(study,name)),value,'Immutable pre-native file/'+name)
    checks.require({'source-bank.json','preparation.json','slots.json','training-prior.json','calibration.json','setup-home.json'}
                   <= set(spec['frozen_files']), 'All planning inputs frozen before native')
    native,binding,_ = parent_inputs(Path(a.parent_study),Path(a.parent_audit))
    checks.equal((spec['identity'],spec['scene'],spec['parent'],plan['parent'],plan['identity'],plan['scene']),
                 (native['identity'],native['scene'],binding,binding,native['identity'],native['scene']), 'Original calibrated parent remains intact')
    sources = verify_sources(a,plan,bank,spec,checks)
    checks.equal(spec['source_bindings'],plan['source'],'Contract binds independently audited teachers')
    summary = source_summary(bank)
    checks.equal(plan['source_bank_summary'],summary,'Teacher strength and counts reflect observed source releases')
    prior,home = read(study/'training-prior.json'),read(study/'setup-home.json')
    calibration = replay.check_calibration(study,spec['identity']['calibration'],checks)
    ranker = FlightRanker(spec['scene'],calibration,a.assets,sources['target']['spec']['flight_ranking']['rule'])
    models = reach_models(str(Path(a.assets).resolve()))
    resolutions = {}
    for case in plan['cases']:
        label = case['id']; directory = study/case['directory']
        checks.equal(case['directory'],'native-'+label[-2:],label+'/isolated target proof namespace')
        checks.equal(read(study/f'preparation-{label[-2:]}.json'),case,label+'/complete final case receipt')
        checks.equal(case['status'],'prepared',label+'/CPU stage completed')
        resolutions[case['directory']] = {}
        _proof(case['original'],dict(id=label,bucket_xy_m=case['original']['bucket_xy_m']),directory,prior,home,
               resolutions[case['directory']],checks)
        checks.equal(case['original_prediction'],ranker.predict(case['original']),label+'/original flight prediction recomputed')
        base,retries = verify_proposals(case,bank,checks)
        proposals={p['id']:p for p in case['proposals']}
        checks.equal(set(case['nominal_predictions']),set(proposals),label+'/complete nominal prediction inventory')
        checks.equal(set(case['wrist_reach']),set(proposals),label+'/complete necessary reach screen inventory')
        for proposal in case['proposals']:
            reach,nominal = recompute_screen(ranker,spec['scene'],proposal,models)
            checks.equal((case['wrist_reach'][proposal['id']],case['nominal_predictions'][proposal['id']]),
                         (reach,nominal),proposal['id']+'/independent geometry-only screening')
        order=independent_order(case,bank)
        checks.equal(case['preparation_order'],order+[p['id'] for p in retries],label+'/source-balanced fixed preparation order')
        warm=[r for r in case['records'] if 'retry_of' not in proposals[r['proposal_id']]]
        retry_records=[r for r in case['records'] if 'retry_of' in proposals[r['proposal_id']]]
        checks.equal([r['proposal_id'] for r in warm],order[:len(warm)],label+'/unskipped warm setup prefix')
        rejected={r['proposal_id'] for r in warm if not r['admitted'] and r['reason'] in
            ('finite_ik_unresolved','strict_endpoint_clearance_or_pose_rejected','finite_requested_setup_path_unresolved')}
        retry_sources=[identity for identity in order if identity in rejected][:8]
        checks.equal([p['retry_of'] for p in retries],retry_sources[:len(retries)],label+'/only first eight rejected geometries retried')
        checks.equal([r['proposal_id'] for r in retry_records],[p['id'] for p in retries],label+'/complete retry outcomes')
        checks.require(len(warm)<=96 and len(retry_records)<=8,label+'/bounded setup attempts')
        checks.require(not retries or len(warm)==len(order),label+'/retry only after complete initial menu')
        timeout_records=[r for r in case['records'] if r['reason']=='case_preparation_time_limit']
        checks.equal(bool(timeout_records),case['preparation_timed_out'],label+'/explicit CPU timeout accounting')
        checks.require(case['preparation_timed_out'] or len(warm)==len(order),label+'/no unexplained omitted setup attempts')
        verify_admissions(case,directory,prior,home,resolutions[case['directory']],checks)
        predicted={r['id']:ranker.predict(r) for r in case['admitted']}
        checks.equal(case['predictions'],predicted,label+'/actual admitted joint clearance predictions recomputed')
        checks.equal(case['candidates'],independent_candidates(case['admitted'],predicted),
                     label+'/reserved unadjusted teachers plus diverse predictions; maximum twenty')
        checks.equal(case['selection_order'],[r['id'] for r in case['candidates']],label+'/fixed native candidate order')
        counts={}
        for candidate in case['candidates']:
            counts[candidate['source_bank_id']]=counts.get(candidate['source_bank_id'],0)+1
        checks.equal(case['source_counts'],counts,label+'/actual native candidates by teacher source')
        checks.equal(case['nearest_seed_distance_m'],min(math.dist(case['original']['bucket_xy_m'],s['bucket_xy_m']) for s in bank),
                     label+'/nearest audited successful bucket distance')
    count=sum(len(c['candidates']) for c in plan['cases'])
    checks.require(count<=100,'At most twenty candidates at each of five fixed buckets')
    checks.equal((spec['initial_requests'],spec['maximum_confirmation_requests'],spec['maximum_total_requests']),
                 (15+count,3*count,33+4*count),'All old repeats, candidates, potential confirmations and controls included')
    return spec,plan,source,prior,home,resolutions


def preflight(a):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Preparation audit requires hidden GPUs')
    checks = Checks()
    spec, plan, source, prior, home, resolutions = verify_preparation(a, checks)
    study = Path(a.output)
    checks.require(not (study/'native-started.json').exists(), 'No native release before preparation audit')
    verify_request_keys(plan, read(study/'slots.json'), checks)
    return dict(version=PREPARATION_VERSION, status='passed', checks_count=checks.count,
        scope='CPU preparation only', native_execution_performed=False, development_positions=5,
        plan_sha256=sha(study/'preparation.json'), source_bank_sha256=sha(study/'source-bank.json'),
        preparation_sha256=sha(study/'preparation.json'),
        contract_sha256=sha(study/'contract.json'), slots_sha256=sha(study/'slots.json'),
        verifier_source_sha256=sha(Path(__file__)),
        deferred=['Native64 trajectory/contact scoring', 'Pre/post controls', 'Final evidence binding'])



def verify_launch_receipt(study, checks):
    marker=read(study/'native-started.json')
    receipt_path=Path(marker['preflight_audit'])
    checks.equal(sha(receipt_path),marker['preflight_audit_sha256'],'Launch bound the unchanged independent preflight')
    receipt=read(receipt_path)
    checks.equal((receipt['version'],receipt['status'],receipt['native_execution_performed']),
                 (PREPARATION_VERSION,'passed',False),'CPU-only preflight passed before native launch')
    checks.require(type(receipt['checks_count']) is int and receipt['checks_count']>0,'Preflight performed actual checks')
    for field,name in (('contract_sha256','contract.json'),('preparation_sha256','preparation.json'),
                       ('slots_sha256','slots.json'),('source_bank_sha256','source-bank.json')):
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
    checks.equal(read(study/'native-started.json')['slots_sha256'], sha(study/'slots.json'), 'Native launch binds frozen initial commands')
    initial_count = 15+sum(len(c['candidates']) for c in plan['cases'])
    checks.equal((spec['initial_requests'], spec['maximum_confirmation_requests'], spec['maximum_total_requests']),
                 (initial_count, (initial_count-15)*3, initial_count+(initial_count-15)*3+18), 'Explicit maximum native budget')
    ledger, timings = observed_rows(study/'rows.jsonl'), observed_rows(study/'timing.jsonl')
    sources = read(a.parent_study/'control-replays.json')
    used = {c['directory']: set() for c in plan['cases']}; seen = set(); phases = []; observed = []; controls = {}
    expected_commands = {s['request']['id']: s for s in slots+repeat_slots}
    checks.equal(len(expected_commands), len(slots)+len(repeat_slots), 'Globally unique diagnostic commands')
    for index, path in enumerate(sorted((study/'requests').glob('*.json'))):
        relative = f'requests/batch-{index:04d}.json'
        checks.equal(path.relative_to(study).as_posix(), relative, 'Contiguous native packet ledger')
        packet = read(path); phase = packet['phase']; requests = packet['rows']; directory = packet['adapter_directory']
        control = phase in ('controls_pre', 'controls_post')
        checks.equal(packet['control'], control, relative+'/control declaration')
        checks.require(phase in ('controls_pre', 'diagnostic', 'confirmation', 'controls_post'), relative+'/known phase')
        phases.append(('controls_pre', 'diagnostic', 'confirmation', 'controls_post').index(phase))
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
    checks.equal([r['id'] for r in observed], [s['request']['id'] for s in slots+repeat_slots], 'Every scheduled diagnostic executed once in order')
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
    summary = independent_summary(plan, observed)
    for item, prepared in zip(summary, plan['cases'], strict=True):
        item.update(nearest_seed_distance_m=prepared['nearest_seed_distance_m'],source_counts=prepared['source_counts'])
    checks.equal(report['source_bank_summary'], source_summary(read(study/'source-bank.json')), 'Published source strength matches evidence')
    checks.require(len(observed)+18 <= 433, 'Authorized total native release cap')
    checks.equal(report['cases'], summary, 'Independent proposal, original, search and confirmation counts')
    checks.equal(report['progress'], dict(completed=len(observed), planned=len(slots)+len(repeat_slots)), 'Complete diagnostic denominator')
    checks.equal((report['native_releases'], report['initial_requests'], report['confirmation_requests']),
                 (len(observed)+18, len(slots), len(repeat_slots)), 'All native releases including controls accounted')
    return dict(version=VERSION, status='passed', checks_count=checks.count, report_sha256=sha(study/'report.json'),
        plan_sha256=sha(study/'preparation.json'), source_bank_sha256=sha(study/'source-bank.json'),
        contract_sha256=sha(study/'contract.json'), evidence_manifest_sha256=sha(study/'evidence-manifest.json'),
        verifier_source_sha256=sha(Path(__file__)), frozen_source_root=str(source),
        initial_requests=len(slots), confirmation_requests=len(repeat_slots), control_releases=len(control_rows),
        results=[dict(id=c['id'], bucket_xy_m=c['bucket_xy_m'], proposed=c['proposed'], checked=c['checked'],
            admitted=c['admitted'], selected=c['selected'], results={v: {k: x for k, x in g.items() if k != 'rows'}
                for v, g in c['results'].items()}, confirmed_settings=c['confirmed_settings']) for c in summary])



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('target-study','target-audit','recovery-study','recovery-audit','guarded-study','guarded-audit',
                'parent-study','parent-audit','assets','output','audit'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--source-root',type=Path)
    parser.add_argument('--stage',choices=('preflight','final'),default='final')
    args=parser.parse_args()
    roots=[args.output,args.target_study,args.recovery_study,args.guarded_study,args.parent_study]
    receipts=[args.target_audit,args.recovery_audit,args.guarded_audit,args.parent_audit]
    targets=(args.audit,args.audit.with_suffix(args.audit.suffix+'.tmp'))
    if args.audit.exists() or any(p.resolve()==r.resolve() or p.resolve().is_relative_to(r.resolve())
                                for p in targets for r in (*roots,*receipts)):
        parser.error('Write a new receipt outside every immutable source directory and audit')
    try:
        result=preflight(args) if args.stage=='preflight' else audit(args)
    except Exception as error:
        result=dict(version=PREPARATION_VERSION if args.stage=='preflight' else VERSION,status='failed',
                    error=f'{type(error).__name__}: {error}')
    atomic(args.audit,result)
    print(json.dumps(result),flush=True)
    if result['status']!='passed':
        raise SystemExit(1)


if __name__=='__main__':
    main()
