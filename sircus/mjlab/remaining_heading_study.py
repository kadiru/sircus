"""Aim correction at the one remaining development target, using native64.

Reuse the established native runner, immutable request schedule and scoring.
The one-case proposal protocol and its source binding are new and explicit;
the previous board-pose study and all successful settings remain unchanged.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import time

from .benchmark_verify import contained, digest, read, sha
from .sweep_worker import atomic
from . import board_pose_recovery_study as native
from .hybrid_miss_study import engine_for, native_inputs, source_inputs

VERSION = 'sircus-remaining-heading-study-v1'
AUDIT_VERSION = 'sircus-remaining-heading-independent-audit-v1'
CASE = 'stalled-reference-test-19'
SOURCE_CANDIDATE = CASE + '-pose-00755-b0'
SOURCE_REPORT = '37a630386e4e2ee4b46a362174eb70de6666414b7a3010bc15b1b0a883b1b764'
HEADINGS = (0., -1., 1., -2., 2., -3., 3., -4., 4., 5., 6., 7., 8.)
OFFSETS = ((.02, 0.), (-.02, 0.), (0., .02), (0., -.02),
           (.04, 0.), (-.04, 0.), (0., .04), (0., -.04))
RULE = dict(headings_deg=list(HEADINGS), fallback_xy_offsets_m=[list(x) for x in OFFSETS],
    fallback='Only headings whose unshifted warm IK/path attempt failed receive translated-center attempts',
    fixed=['bucket', 'board height', 'board tilt', 'drop gap', 'board twist', 'gripper yaw rule'],
    retry_limit=24, maximum_setup_checks=160, maximum_native_candidates=40,
    cpu_limit_s=600, repeats=3, controls=18, physics_substeps=64,
    selection='Keep all admitted unshifted heading geometries first, then descending predicted clearance; one branch per geometry')


def bound_source(a):
    """Authenticate the exact observed arm-free near miss, without relabeling it."""
    report = read(a.recovery_study/'report.json'); receipt = read(a.recovery_audit)
    if (sha(a.recovery_study/'report.json') != SOURCE_REPORT or receipt.get('status') != 'passed'
            or receipt.get('report_sha256') != SOURCE_REPORT
            or receipt.get('plan_sha256') != sha(a.recovery_study/'preparation.json')
            or receipt.get('evidence_manifest_sha256') != sha(a.recovery_study/'evidence-manifest.json')
            or report.get('status') != 'completed'):
        raise ValueError('Need the exact independently audited board-pose recovery evidence')
    plan = read(a.recovery_study/'preparation.json')
    case = next(c for c in plan['cases'] if c['id'] == CASE)
    original = next(r for r in case['candidates'] if r['id'] == SOURCE_CANDIDATE)
    found = next(c for c in report['cases'] if c['id'] == CASE)
    observed = next(r for r in found['results']['search']['rows'] if r['candidate_id'] == SOURCE_CANDIDATE)
    if (not observed['native_release'] or observed['invalid_physics'] or observed['route_success']
            or observed['experimental_route']['robot_contact_steps'] != 0):
        raise ValueError('Selected development example must be a valid arm-free miss')
    directory = a.recovery_study/case['directory']
    if sha(contained(directory, observed['native_cache_file'])) != observed['native_cache_sha256']:
        raise ValueError('Recorded source trace changed')
    for field in ('values', 'bucket_xy_m'):
        if observed[field] != original[field]: raise ValueError('Source command does not match observed flight')
    if (observed['q10'], observed['q14']) != (original['selected_q10'], original['selected_q14']):
        raise ValueError('Source joint solution does not match observed flight')
    binding = dict(report_sha256=SOURCE_REPORT, audit_sha256=sha(a.recovery_audit),
        plan_sha256=sha(a.recovery_study/'preparation.json'),
        manifest_sha256=sha(a.recovery_study/'evidence-manifest.json'),
        candidate_id=SOURCE_CANDIDATE, request_sha256=digest(original),
        observation_sha256=digest(observed), trace_sha256=observed['native_cache_sha256'])
    return dict(id=CASE, original=original, historical=observed), binding, directory


def proposal(original, heading, offset, index, group):
    values = list(original['values']); values[0] += offset[0]; values[1] += offset[1]
    return dict(id=f'{CASE}-aim-{index:03d}', context_id=CASE,
        bucket_xy_m=list(original['bucket_xy_m']), values=values,
        board_heading_offset_deg=float(heading), board_twist_deg=original['board_twist_deg'],
        phase='search', role='diagnostic', method='heading_search', intended_route='table',
        group=group, seed_q10=original['selected_q10'], seed_q14=original['selected_q14'],
        source_request_id=original['id'], xy_offset_m=list(offset))


def primary_menu(original):
    return [proposal(original, h, (0., 0.), i, 'heading_only') for i, h in enumerate(HEADINGS)]


def translated_menu(original, primary_records):
    parents = primary_menu(original)
    if len(primary_records) != len(parents) or any(r.get('reason') == 'case_preparation_time_limit' for r in primary_records): return []
    failed = {r['proposal_id'] for r in primary_records if not r['admitted']}
    pairs = [(p['board_heading_offset_deg'], offset) for offset in OFFSETS for p in parents if p['id'] in failed]
    return [proposal(original, h, offset, len(parents)+i, 'center_clearance') for i, (h, offset) in enumerate(pairs)]


def select_candidates(admitted, predictions):
    """One best arm branch per geometry; preserve coverage of primary headings."""
    ordered = sorted(admitted, key=lambda r: native.score_key(r, predictions[r['id']]))
    unique = {}; first = []; others = []
    for r in ordered:
        key = native.geometry_key(r)
        if key in unique: continue
        unique[key] = r
        (first if r['group'] == 'heading_only' else others).append(r)
    first.sort(key=lambda r: (HEADINGS.index(r['board_heading_offset_deg']), r['id']))
    return (first + others)[:RULE['maximum_native_candidates']]


def prepare(a):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('Hide CUDA for CPU preparation')
    import torch
    torch.set_num_threads(1)
    from .arm_check_speed import install_vectorized_gate
    from .arm_path_speed import install_path_short_circuit
    from .cell1050_rebound_study import preparation_time_limit, SearchPreparationExpired
    from .interactive_stalled_planner import copy_prepared_proofs
    from .intergrid_flight_ranker import FlightRanker
    from .intergrid_system_study import require_storage
    case, recovery_binding, source_directory = bound_source(a)
    _, source_contract, binding = source_inputs(a)
    spec, parent_binding, _, ref = native_inputs(a)
    if spec['identity'] != source_contract['identity']: raise ValueError('Native scene changed')
    a.output.mkdir(parents=True, exist_ok=False); require_storage(a.output)
    for name in ('training-prior.json', 'calibration.json', 'setup-home.json'):
        if sha(a.recovery_study/name) != sha(a.source_study/name): raise ValueError('Source input changed: '+name)
        shutil.copyfile(a.recovery_study/name, a.output/name)
    shutil.copyfile(a.parent_study/'verification.json', a.output/'verification.json')
    directory = a.output/'native-19'; old = case['original']
    copy_prepared_proofs(source_directory, directory, old)
    engine = engine_for(a, old, directory, ref)
    install_vectorized_gate(engine.adapter.resolver.planner.gate)
    install_path_short_circuit(engine.adapter.resolver.planner)
    ranker = FlightRanker(ref['scene'], read(a.output/'calibration.json')['score_config'], a.assets, source_contract['flight_ranking']['rule'])
    report = dict(version=VERSION, status='running', phase='preparing', physics_substeps=64,
        audit_status='pending', progress=dict(completed=0, planned=0))
    native.publish(a, report)
    proposals = primary_menu(old); records = []; admitted = []; predictions = {}; timed_out = False
    begin = time.monotonic()
    def attempt(rows, one_per_setting=True):
        nonlocal timed_out
        for start in range(0, len(rows), 4 if one_per_setting else 1):
            if timed_out: break
            batch = rows[start:start+(4 if one_per_setting else 1)]
            if len(records)+len(batch) > RULE['maximum_setup_checks']: raise ValueError('Declared setup budget exceeded')
            try:
                with preparation_time_limit(RULE['cpu_limit_s']-(time.monotonic()-begin)):
                    added, logs = engine.prepare(batch, one_per_setting=one_per_setting)
                    scored = {r['id']: ranker.predict(r) for r in added}
                admitted.extend(added); records.extend(logs); predictions.update(scored)
            except SearchPreparationExpired:
                timed_out = True
                records.extend(dict(proposal_id=p['id'], proposal_sha256=digest(p), admitted=0,
                    admitted_request_ids=[], reason='case_preparation_time_limit', timing={}) for p in batch)
            atomic(a.output/'preparation-progress.json', dict(records=records, admitted=len(admitted), elapsed_s=time.monotonic()-begin))
    attempt(proposals)
    primary_records = list(records)
    translated = translated_menu(old, primary_records) if not timed_out else []
    proposals.extend(translated); attempt(translated)
    rejected = {r['proposal_id'] for r in records if not r['admitted'] and r['reason'] in
        ('finite_ik_unresolved', 'strict_endpoint_clearance_or_pose_rejected', 'finite_requested_setup_path_unresolved')}
    # Prefer failed original-center headings before translated alternatives.
    failed = [p for p in proposals if p['id'] in rejected]
    failed.sort(key=lambda p: (p['group'] != 'heading_only', abs(p['board_heading_offset_deg']), p['id']))
    retries = []
    if not timed_out:
        for p in failed[:RULE['retry_limit']]:
            retries.append(dict(copy.deepcopy(p), id=p['id']+'-multistart', retry_of=p['id'], ik_search='multistart16'))
    proposals.extend(retries); attempt(retries, one_per_setting=False)
    candidates = select_candidates(admitted, predictions)
    case.update(directory=directory.name, proposals=proposals, records=records, admitted=admitted,
        candidates=candidates, predictions=predictions, original_prediction=ranker.predict(old),
        preparation_order=[r['proposal_id'] for r in records], selection_order=[r['id'] for r in candidates],
        preparation_s=time.monotonic()-begin, preparation_timed_out=timed_out,
        bounds=dict(low=engine.adapter.resolver.solver.low.tolist(), high=engine.adapter.resolver.solver.high.tolist()), status='prepared')
    plan = dict(version=VERSION, repeats=3, search_rule=RULE, source=binding, recovery=recovery_binding,
        parent=parent_binding, identity=spec['identity'], scene=spec['scene'], cases=[case])
    if engine.adapter.env is not None or torch.cuda.is_initialized(): raise ValueError('CPU preparation allocated CUDA')
    atomic(a.output/'preparation.json', plan)
    report.update(phase='prepared', cases=native.summarize(plan, [])); native.publish(a, report)


def freeze(a):
    from .corrected_grid import source_manifest
    plan = read(a.output/'preparation.json')
    if plan['version'] != VERSION or [c['id'] for c in plan['cases']] != [CASE] or (a.output/'contract.json').exists():
        raise ValueError('Need fresh complete one-case preparation')
    slots = native.schedule(plan); searches = len(plan['cases'][0]['candidates'])
    atomic(a.output/'slots.json', slots)
    files = {str(p.relative_to(a.output)):sha(p) for p in a.output.rglob('*') if p.is_file() and p.name != 'report.json'}
    spec = dict(version=VERSION, identity=plan['identity'], scene=plan['scene'], parent=plan['parent'],
        plan_sha256=sha(a.output/'preparation.json'), frozen_files=files,
        source_sha256=source_manifest(Path(__file__).resolve().parents[2]), physics_substeps=64, repeats=3, controls=18,
        initial_requests=len(slots), maximum_confirmation_requests=3*searches, maximum_total_requests=len(slots)+3*searches+18,
        search_rule=RULE, gpu_cap_s=3600,
        scope='One observed development near miss; native held endpoints, no training or unseen-position inference claim')
    spec['fingerprint'] = digest(spec); atomic(a.output/'contract.json', spec)
    report = read(a.output/'report.json'); report.update(phase='frozen', progress=dict(completed=0, planned=len(slots)))
    native.publish(a, report)


def audit(a):
    """Rescore saved native samples and verify exact schedules/proofs on CPU."""
    from .benchmark_verify import Checks, fingerprint
    from . import replay650_verify as replay
    from .board_pose_recovery_audit import (verify_admissions, verify_request_keys,
        verify_confirmations, independent_summary)
    from .cell1050_rebound_study import parent_inputs, control_requests
    from .direct_path_first_audit import _proof, verify_heading
    from .parallel_capture_grid_audit import observed_rows, rescore_packet, verify_release
    from .policy64_verify import audit_native_options
    from .table_training_verify import control_gate, rescore_rows
    from .intergrid_flight_ranker import FlightRanker
    from .intergrid_route_labels import annotate_routes
    checks = Checks(); study = a.output
    spec, plan, report = (read(study/n) for n in ('contract.json','preparation.json','report.json'))
    fingerprint(spec, checks, 'Frozen heading correction protocol')
    checks.equal((spec['version'], plan['version'], report['version'], report['status'], report['phase']),
        (VERSION, VERSION, VERSION, 'completed', 'completed'), 'Completed one-case heading correction')
    checks.equal((spec['physics_substeps'], spec['repeats'], spec['controls']), (64,3,18), 'Native64, three confirmations, 18 controls')
    checks.equal((spec['search_rule'], plan['search_rule']), (RULE, RULE), 'Prespecified bounded rule')
    checks.require(0 < report['gpu_lease_s'] <= spec['gpu_cap_s'] <= 3600, 'GPU budget')
    manifest = replay.check_manifest(study, checks)
    checks.equal(report['evidence_manifest_sha256'], sha(study/'evidence-manifest.json'), 'Report binds evidence')
    checks.equal(spec['plan_sha256'], sha(study/'preparation.json'), 'Frozen preparation')
    for name, expected in spec['frozen_files'].items(): checks.equal(manifest[name], expected, 'Frozen input/'+name)
    root = a.source_root or Path(__file__).resolve().parents[2]
    for name, expected in spec['source_sha256'].items(): checks.equal(sha(contained(root,name)), expected, 'Frozen source/'+name)
    parent_spec, parent_binding, _ = parent_inputs(a.parent_study, a.parent_audit)
    checks.equal((spec['identity'], spec['scene'], spec['parent']),
        (parent_spec['identity'], parent_spec['scene'], parent_binding), 'Same native scene/parent')
    _, source_contract, source_binding = source_inputs(a)
    checks.equal(plan['source'], source_binding, 'Exact original planner study binding')
    old, recovery_binding, _ = bound_source(a)
    checks.equal(plan['recovery'], recovery_binding, 'Audited near-miss source binding')
    checks.equal([c['id'] for c in plan['cases']], [CASE], 'One previously observed target only')
    case = plan['cases'][0]; directory = study/case['directory']
    checks.equal((case['original'],case['historical'],case['directory']),
        (old['original'],old['historical'],'native-19'), 'Preserve exact source shot and observation')
    records = case['records']; proposals = case['proposals']; primary = primary_menu(case['original'])
    warm = [r for r in records if not r['proposal_id'].endswith('-multistart')]
    initial_records = warm[:len(primary)]
    shifted = translated_menu(case['original'], initial_records)
    base = primary + shifted
    checks.equal([r['proposal_id'] for r in warm], [p['id'] for p in base][:len(warm)], 'Prescribed warm attempt prefix')
    rejected = {r['proposal_id'] for r in warm if not r['admitted'] and r['reason'] in
        ('finite_ik_unresolved','strict_endpoint_clearance_or_pose_rejected','finite_requested_setup_path_unresolved')}
    failures = sorted([p for p in base if p['id'] in rejected],
        key=lambda p:(p['group'] != 'heading_only',abs(p['board_heading_offset_deg']),p['id']))
    retries = [dict(copy.deepcopy(p),id=p['id']+'-multistart',retry_of=p['id'],ik_search='multistart16')
        for p in failures[:RULE['retry_limit']]] if len(warm) == len(base) and not any(r['reason']=='case_preparation_time_limit' for r in warm) else []
    checks.equal(proposals, base+retries, 'Exact fixed-heading and setup-driven fallback proposal set')
    retry_records = [r for r in records if r['proposal_id'].endswith('-multistart')]
    checks.equal([r['proposal_id'] for r in retry_records], [p['id'] for p in retries][:len(retry_records)], 'Bounded multistart attempt prefix')
    checks.require(len(records) <= 160 and len(case['candidates']) <= 40, 'Setup/native candidate caps')
    checks.equal(case['preparation_order'], [r['proposal_id'] for r in records], 'Every setup attempt retained')
    if len(records) < len(proposals): checks.require(case['preparation_timed_out'], 'Unattempted proposals explicitly timed out')
    prior, home = read(study/'training-prior.json'), read(study/'setup-home.json')
    calibration = replay.check_calibration(study,spec['identity']['calibration'],checks)
    resolutions = {}; _proof(case['original'],dict(id=CASE,bucket_xy_m=case['original']['bucket_xy_m']),directory,prior,home,resolutions,checks)
    verify_admissions(case,directory,prior,home,resolutions,checks)
    ranker = FlightRanker(spec['scene'],calibration,a.assets,source_contract['flight_ranking']['rule'])
    predictions = {r['id']:ranker.predict(r) for r in case['admitted']}
    checks.equal(case['predictions'], predictions, 'Recomputed flight projections, not native labels')
    checks.equal(case['original_prediction'],ranker.predict(case['original']),'Original projection')
    # Independently reconstruct stable ranking and one branch per geometry.
    ordered = sorted(case['admitted'],key=lambda r:(predictions[r['id']]['score_m'] is None,
        -(predictions[r['id']]['score_m'] or 0.),r['id']))
    distinct = {}; primary_choices=[]; other_choices=[]
    for row in ordered:
        key = tuple(round(float(x),10) for x in (*row['values'],row['board_heading_offset_deg'],row['board_twist_deg']))
        if key in distinct: continue
        distinct[key] = row
        (primary_choices if row['group']=='heading_only' else other_choices).append(row)
    primary_choices.sort(key=lambda r:(HEADINGS.index(r['board_heading_offset_deg']),r['id']))
    checks.equal(case['candidates'],(primary_choices+other_choices)[:40],'Frozen selection independent of outcomes')
    checks.equal(case['selection_order'],[r['id'] for r in case['candidates']],'Selected order')
    slots, repeat_slots = read(study/'slots.json'), read(study/'repeat-slots.json')
    verify_request_keys(plan,slots,checks)
    checks.equal(read(study/'native-started.json')['slots_sha256'],sha(study/'slots.json'),'Native launch identity')
    n = len(case['candidates']); checks.equal((spec['initial_requests'],spec['maximum_confirmation_requests'],spec['maximum_total_requests']),
        (3+n,3*n,3+4*n+18),'Maximum complete native budget')
    ledger, timings = observed_rows(study/'rows.jsonl'), observed_rows(study/'timing.jsonl')
    sources = read(a.parent_study/'control-replays.json'); commands={s['request']['id']:s for s in slots+repeat_slots}
    checks.equal(len(commands),len(slots)+len(repeat_slots),'Globally unique commands')
    used=set(); seen=set(); phases=[]; observed=[]; controls={}
    for index,path in enumerate(sorted((study/'requests').glob('*.json'))):
        relative=f'requests/batch-{index:04d}.json';checks.equal(path.relative_to(study).as_posix(),relative,'Contiguous packet ledger')
        packet=read(path);phase=packet['phase'];requests=packet['rows'];dirname=packet['adapter_directory']
        control=phase in ('controls_pre','controls_post');checks.equal(packet['control'],control,'Control declaration')
        checks.require(phase in ('controls_pre','diagnostic','confirmation','controls_post'),'Known phase')
        phases.append(('controls_pre','diagnostic','confirmation','controls_post').index(phase))
        checks.require(0<len(requests)<=16,'Native batch bound')
        timing=next(timings,None);checks.require(timing is not None,'Timing exists')
        checks.equal((timing['packet'],timing['phase']),(relative,phase),'Matching timing')
        if control:
            checks.equal(dirname,'native-controls','Control namespace');checks.require(phase not in controls,'Unique control block')
            checks.equal(requests,control_requests(sources,phase),'Exact matched controls')
        else:checks.equal(dirname,'native-19','Diagnostic namespace')
        rows=[]
        for request in requests:
            row=next(ledger,None);checks.require(row is not None,'Native observation exists')
            verify_release(row,request,relative,manifest[relative],phase,checks)
            checks.require(row['id'] not in seen,'One observation per release');seen.add(row['id'])
            checks.equal(row['adapter_directory'],dirname,'Native archive namespace')
            if not control:
                checks.equal((commands[row['id']]['request'],commands[row['id']]['directory']),(request,dirname),'Frozen exact command')
                for field in ('variant','repeat','context_id','candidate_id'):checks.equal(row[field],request[field],field)
                verify_heading(row,request,directory,spec['scene'],home,prior,resolutions,checks)
            rows.append(row)
        if control:controls[phase]=rows
        else:
            alias={'native-heading/'+name[len(dirname)+1:]:value for name,value in manifest.items() if name.startswith(dirname+'/')}
            list(rescore_packet(directory,rows,spec['scene'],calibration,alias,used,checks))
            relabeled=annotate_routes(copy.deepcopy(rows),directory)
            for row,other in zip(rows,relabeled,strict=True):checks.equal((row['shot_route'],row['table_rebounds_before_entry']),
                (other['shot_route'],other['table_rebounds_before_entry']),'Independent route classification')
            observed.extend(rows)
    checks.equal(phases,sorted(phases),'Pre controls, search, repeats, post controls')
    checks.require(next(ledger,None) is None and next(timings,None) is None,'Complete row/timing inventory')
    checks.equal([r['id'] for r in observed],[s['request']['id'] for s in slots+repeat_slots],'Every scheduled drop executed once')
    eligible=verify_confirmations(plan,observed,checks);verify_request_keys(plan,repeat_slots,checks,confirmations=True,eligible=eligible)
    audit_native_options(directory,spec['scene'],calibration,a.assets,checks)
    checks.equal(used,{p.relative_to(directory).as_posix() for p in directory.rglob('*.npz')},'All diagnostic native archives')
    control_rows=[r for rows in controls.values() for r in rows]
    audit_native_options(study/'native-controls',spec['scene'],calibration,a.assets,checks)
    rescore_rows(study/'native-controls',control_rows,spec['scene'],calibration,spec['identity']['trace_schema'],a.assets,checks)
    checks.equal({r['native_cache_file'] for r in control_rows},{p.relative_to(study/'native-controls').as_posix() for p in (study/'native-controls').rglob('*.npz')},'All control native archives')
    checks.equal(set(controls),{'controls_pre','controls_post'},'Both control blocks')
    for phase,rows in controls.items():
        gate=control_gate(rows);checks.equal(gate['status'],'passed',phase+' behavior');checks.equal(report['controls'][phase],gate,phase+' counts')
    summary=independent_summary(plan,observed);checks.equal(report['cases'],summary,'Independent counts and selected setting')
    checks.equal(report['progress'],dict(completed=len(observed),planned=len(slots)+len(repeat_slots)),'Complete diagnostic denominator')
    checks.equal((report['native_releases'],report['initial_requests'],report['confirmation_requests']),
        (len(observed)+18,len(slots),len(repeat_slots)),'Every native release accounted')
    return dict(version=AUDIT_VERSION,status='passed',checks_count=checks.count,
        report_sha256=sha(study/'report.json'),plan_sha256=sha(study/'preparation.json'),
        evidence_manifest_sha256=sha(study/'evidence-manifest.json'),verifier_source_sha256=sha(Path(__file__)),
        results=[dict(id=c['id'],bucket_xy_m=c['bucket_xy_m'],
            results={v:{k:x for k,x in g.items() if k!='rows'} for v,g in c['results'].items()},
            confirmed_settings=c['confirmed_settings']) for c in summary])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source-study','source-audit','parent-study','parent-audit','recovery-study','recovery-audit','reference','assets','output','gpu-lock'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--gpu-wait-s', type=int, default=1800)
    p.add_argument('--stage', choices=('prepare','freeze','run','audit'), required=True)
    p.add_argument('--audit', type=Path)
    p.add_argument('--source-root', type=Path)
    a = p.parse_args()
    if a.stage != 'audit': return (native.run if a.stage == 'run' else globals()[a.stage])(a)
    if a.audit is None or a.audit.exists() or a.audit.resolve().is_relative_to(a.output.resolve()):
        p.error('Audit requires a new receipt outside the evidence directory')
    try: result=audit(a)
    except Exception as error:result=dict(version=AUDIT_VERSION,status='failed',error=f'{type(error).__name__}: {error}')
    atomic(a.audit,result);print(json.dumps(result),flush=True)
    if result['status']!='passed':raise SystemExit(1)


if __name__ == '__main__': main()
