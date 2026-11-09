"""Byte-bound CPU diagnosis of the completed recovery comparison.

The inspected TEST positions are development evidence now. This module reads
saved traces and independently rescores them; it never executes native physics,
fits a policy, changes source evidence, or selects an action by new outcomes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path

import numpy as np

from .benchmark_verify import contained, digest, read, sha
from .replay650_verify import AuditChecks
from .table_training_verify import captured, check_native_score
from . import table_assisted_score_oracle as oracle
from .xy650_verify import first_flight

VERSION = 'sircus-coverage-regression-diagnosis-v1'
AUDIT_VERSION = 'sircus-coverage-regression-diagnosis-audit-v1'
SOURCE = dict(report_sha256='6a255223820656059fe16a5de2944ba2593f7ac3603b287f4955979cd4677504',
    audit_sha256='e2ba653792b37d302425ba4af6cb22460c3d5b426274b3c1b3cbce270a857a9b',
    rows_sha256='3fba67b7cbb1a1a8769ef773bab3245233e6704641e2f98a037076a11112985b',
    manifest_sha256='809d0aa053d14b7c5d2bca7a558b55c0d68bcb0ac1e6a4d5751d18759c784235',
    contract_fingerprint='f54c341eba5258b427e4e1eb05a1b2e93bf76518d9f94a2fe8cf6bad744f6082')
PARAMETERS = ['board_x_m', 'board_y_m', 'board_height_m', 'board_tilt_deg', 'nominal_plane_gap_m']
LOSS_IDS = ['recovery-test-'+s for s in ('r0-c5-n1', 'r0-c6-n1', 'r1-c1-n1', 'r2-c4-n0', 'r3-c1-n0')]
NAMES = ('board', 'bucket', 'table', 'floor', 'robot', 'unattributed')
SCOPE = ('Posthoc development diagnosis of the original recovery TEST positions. '
    'Historical scores remain unchanged. Contacts are observations, not causal proof. '
    'Fixed64 held endpoints only; no new training, native rollout, noise, driven transit or physical validation.')


def verify_inputs(study, audit, checks):
    study, audit = Path(study).resolve(), Path(audit).resolve()
    for key, path in (('report_sha256', study/'report.json'), ('audit_sha256', audit),
                      ('rows_sha256', study/'rows.jsonl'), ('manifest_sha256', study/'evidence-manifest.json')):
        checks.equal(sha(path), SOURCE[key], 'Exact completed source/'+key)
    report, spec, proof, manifest = map(read, (study/'report.json', study/'contract.json', audit, study/'evidence-manifest.json'))
    checks.equal((report['version'], report['status'], report['phase']),
                 ('sircus-coverage-recovery-v1', 'completed', 'completed'), 'Completed recovery source')
    checks.equal((proof['version'], proof['status'], proof['full_checks_verified'], proof['native_checks_verified'], proof['failures']),
                 ('sircus-coverage-recovery-audit-v1', 'passed', True, True, []), 'Full passed native source audit')
    checks.equal((proof['checks_count'], len(proof['checks'])), (283251, 283251), 'All original audit entries retained')
    checks.require(all(c.get('passed') is True for c in proof['checks']), 'Every original check passed')
    checks.equal((proof['report_sha256'], proof['evidence_manifest_sha256']),
                 (SOURCE['report_sha256'], SOURCE['manifest_sha256']), 'Audit binds original report and manifest')
    unsigned = dict(spec); fp = unsigned.pop('fingerprint')
    checks.equal((digest(unsigned), fp, report['contract_fingerprint'], proof['contract_fingerprint']),
                 (SOURCE['contract_fingerprint'],)*4, 'Exact frozen contract identity')
    checks.equal(report['evidence_sha256'], manifest, 'Original report binds complete manifest')
    checks.equal(spec['protocol']['substeps'], 64, 'Original native64 physics')
    consumed = {}

    def bound(name):
        path = contained(study, name); actual = sha(path)
        checks.equal(actual, manifest[name], 'Authenticated consumed file/'+name)
        consumed[name] = actual
        return path

    bound('contract.json'); bound('rows.jsonl')
    calibration = read(bound('calibration.json'))['score_config']
    rows = [json.loads(line) for line in (study/'rows.jsonl').read_text().splitlines() if line]
    checks.equal((len(rows), len({r['id'] for r in rows})), (970, 970), 'All970 original requests retained')
    tests = [r for r in rows if r.get('phase') == 'test']
    checks.equal(Counter(r['method'] for r in tests), Counter({m:70 for m in ('recovery_bc','parent_bc','parent_ppo','previous_ppo')}), 'All280 original TEST requests')
    by = {(r['method'],r['context_id']):r for r in tests}
    checks.equal(len(by), 280, 'Distinct paired source identities')
    own = sorted((r for r in tests if r['method']=='recovery_bc'), key=lambda r:r['context_id'])
    losses = [r['context_id'] for r in own if not captured(r) and captured(by['parent_bc',r['context_id']])]
    gains = [r['context_id'] for r in own if captured(r) and not captured(by['parent_bc',r['context_id']])]
    checks.equal(losses, LOSS_IDS, 'All five exact parent-BC capture losses')
    checks.equal((sum(captured(r) for r in own), len(gains)), (37,8), 'Historical37 captures and8 gains retained')
    checks.equal(sum(captured(r) for r in tests if r['method']=='parent_bc'),34,'Historical parent34 captures')
    for row in own:
        parent = by['parent_bc', row['context_id']]
        checks.equal(parent['bucket_xy_m'],row['bucket_xy_m'],row['id']+'/paired exact bucket')
    return study, spec, proof, calibration, rows, own, by, bound, consumed


def setup_evidence(row, connection, resolution):
    attempts = connection.get('endpoint_attempts') or {}
    pairs = []
    for pair in attempts.get('pair_attempts', []):
        checked = pair.get('checked') or {}; strict = checked.get('strict_geometry') or {}
        pairs.append(dict(branch10=pair.get('branch10'),branch14=pair.get('branch14'),
            reason=checked.get('reason'),board_position_error_m=checked.get('board_position_error_m'),
            dropper_position_error_m=checked.get('dropper_position_error_m'),board_normal_error_deg=checked.get('board_normal_error_deg'),
            collision_pairs=checked.get('pairs',[]),independent_reject_arms=checked.get('independent_reject_arms',[]),
            pose_rejected=pair.get('pose_rejected',False),native_pose_errors=pair.get('native_pose_errors'),
            minimum_clearance_m=strict.get('minimum_clearance_m'),closest_pair=strict.get('closest_pair'),
            margin_contacts=strict.get('margin_contacts',[])))
    home = resolution.get('home_geometry') or {}; proof = connection.get('proof') or {}
    admitted = [dict(branch10=c.get('branch10'),branch14=c.get('branch14'))
        for c in attempts.get('candidates',[]) if c.get('q10')==row.get('q10') and c.get('q14')==row.get('q14')]
    branches = attempts.get('valid_ik_branches')
    group = connection.get('reason')
    if group=='finite_ik_unresolved' and isinstance(branches,list) and len(branches)==2:
        group = 'finite_ik_'+('both_arms' if branches==[0,0] else 'dropper' if branches[0]==0 else 'board' if branches[1]==0 else 'unknown')
    return dict(reason=connection.get('reason'),group=group or 'admitted',passed=connection['passed'],
        connection_sha256=row['connection_sha256'],resolution_sha256=row['context_resolution_sha256'],
        action_substituted=connection['action_substituted'],valid_ik_branches=branches,
        endpoint_examples=pairs,endpoint_pair_attempts=len(pairs),admitted_branches=admitted,
        layout_eligible=resolution.get('layout_eligible'),anchor_admitted=resolution.get('anchor_admitted'),
        proof_minimum_clearance_m=proof.get('minimum_clearance_m'),
        home_minimum_clearance_m=home.get('minimum_clearance_m'),home_closest_pair=home.get('closest_pair'),
        fixed_base_blockers=home.get('fixed_base_blockers',[]),required_clearance_m=.005,
        scope='Recorded finite admission proof; rejection does not prove unreachability, and a conservative zero bound does not prove penetration.')


def classify(item):
    if not item['native_release']:
        return 'setup_'+item['setup_proof']['group']
    if item['capture']:
        return 'capture_with_table_contact' if item['table_contact_steps'] else 'direct_capture'
    if item['invalid_physics']:
        return 'invalid_physics'
    if not item['whole_ball_entry']:
        return 'no_whole_ball_entry'
    if item['max_retention_s'] < 1.-1e-9:
        return 'entry_without_one_second_retention'
    if item['forbidden_contact_steps']:
        return 'retained_with_forbidden_contact'
    return 'other_route_failure'


def contact_events(force, times, stop):
    """Force interval k starts at state k; exclude post-score contacts."""
    events = []
    for column, name in enumerate(NAMES):
        present = np.asarray(force[:stop,column]) > .01
        starts = np.flatnonzero(present & ~np.r_[False,present[:-1]])
        ends = np.flatnonzero(present & ~np.r_[present[1:],False])+1
        for start,end in zip(starts,ends):
            events.append(dict(category=name,label=name+' contact',time_s=float(times[start]),
                end_time_s=float(times[end]),peak_force_N=float(np.max(force[start:end,column]))))
    return sorted(events,key=lambda e:(e['time_s'],e['category']))


def trace_summary(row, arrays, scene, calibration):
    stop = int(row['physics_steps']); time = arrays['time_s']; position = arrays['position_m']; velocity = arrays['velocity_m_s']
    events = contact_events(arrays['forces_N'],time,stop)
    for field,label in (('whole_ball_entry_time_s','whole-ball entry'),('capture_time_s','one-second retention')):
        if row.get(field) is not None:
            events.append(dict(category=field,label=label,time_s=float(row[field])))
    events.sort(key=lambda e:(e['time_s'],e['category']))
    clipped = {key:(value[:stop] if key in ('forces_N','contact_distance_m') else value[:stop+1]
                   if key in ('position_m','velocity_m_s','time_s') else value) for key,value in arrays.items()}
    flight = first_flight(scene,row['bucket_xy_m'],row['board_normal'],calibration,clipped,valid=not row['invalid_physics'])
    if flight['available']:
        start,end = flight['start_index'],flight['end_index']
        flight.update(rebound_position_m=position[start].astype(float).tolist(),rebound_velocity_m_s=velocity[start].astype(float).tolist(),
            end_position_m=position[end].astype(float).tolist(),end_velocity_m_s=velocity[end].astype(float).tolist())
        crossing = flight.get('first_crossing')
        if crossing:
            direction = np.asarray(row['bucket_xy_m'])-row['values'][:2]; direction /= np.linalg.norm(direction)
            offset = np.asarray(crossing['xy_m'])-row['bucket_xy_m']
            flight.update(forward_error_m=float(offset@direction),lateral_error_m=float(offset@np.array([-direction[1],direction[0]])))
    # Preserve exact observed samples; bounded events prevent huge contact chatter
    # from bloating the display, while full native bytes remain authenticated.
    indices = set(np.linspace(0,stop,min(stop+1,241),dtype=int).tolist())
    for event in events[:80]:
        ix = int(np.searchsorted(time,event['time_s']))
        indices.update((max(0,ix-1),min(stop,ix),min(stop,ix+1)))
    indices = sorted(indices)
    return dict(events=events,first_flight=flight,trace=dict(record_id=row['id'],native_cache_sha256=row['native_cache_sha256'],
        source_indices=indices,time_s=time[indices].astype(float).tolist(),position_m=position[indices].astype(float).tolist(),
        scope='Actual saved native states through original scoring terminal; uniform samples plus event-adjacent states. No generated motion.'))


def compact(row, connection, resolution):
    fields = ('context_id','bucket_xy_m','method','values','q10','q14','native_release','invalid','invalid_physics',
        'board_center_m','board_normal','release_ball_center_m','pose_gate','board_bounce','whole_ball_entry','max_retention_s',
        'whole_ball_entry_time_s','capture_time_s','retention_resets','table_contact_steps','forbidden_contact_steps',
        'contact_peaks_N','minimum_mouth_clearance_mm','closest_opening_distance_m','entry_class','rim_band_contact')
    item = {key:copy.deepcopy(row.get(key)) for key in fields}
    item.update(record_id=row['id'],record_sha256=digest(row),capture=captured(row),setup_proof=setup_evidence(row,connection,resolution),
        events=[],first_flight=None,trace=None)
    item['group'] = classify(item)
    return item


def build(study, audit, output):
    output = Path(output).resolve(); study = Path(study).resolve()
    if output.exists() or output.is_relative_to(study):
        raise ValueError('Fresh diagnosis directory outside immutable source required')
    checks = AuditChecks()
    study,spec,proof,calibration,rows,own,by,bound,consumed = verify_inputs(study,audit,checks)
    selected = [r for r in own if not captured(r)]+[by['parent_bc',i] for i in LOSS_IDS]
    items = {}; groups = {}
    for row in selected:
        namespace = row.get('adapter_directory','native')
        checks.require(namespace in ('native','native-controls'),row['id']+'/known native namespace')
        connection = read(bound(namespace+'/'+row['connection_file']))
        resolution = read(bound(namespace+'/'+row['context_resolution_file']))
        checks.equal((consumed[namespace+'/'+row['connection_file']],consumed[namespace+'/'+row['context_resolution_file']]),
            (row['connection_sha256'],row['context_resolution_sha256']),row['id']+'/exact setup hashes')
        checks.equal((connection['bucket_xy_m'],connection['values'],connection['resolution_sha256']),
            (row['bucket_xy_m'],row['values'],row['context_resolution_sha256']),row['id']+'/same requested action and resolution')
        checks.equal(connection['action_substituted'],False,row['id']+'/no substituted action')
        checks.equal(connection['passed'],row['native_release'],row['id']+'/setup and release agree')
        if row['native_release']:
            checks.equal((connection['q10'],connection['q14']),(row['q10'],row['q14']),row['id']+'/exact admitted joint branch')
            groups.setdefault(namespace+'/'+row['native_cache_file'],[]).append(row)
        items[row['id']] = compact(row,connection,resolution)
    for name,members in sorted(groups.items()):
        path = bound(name)
        with np.load(path,allow_pickle=False) as archive:
            ids = archive['record_ids'].tolist()
            arrays = {k:archive[k] for k in archive.files if k!='record_ids'}
        for row in members:
            column = row['native_cache_row']
            checks.equal((str(ids[column]),consumed[name]),(row['id'],row['native_cache_sha256']),row['id']+'/bound dense trajectory identity')
            trace = {key:value[:,column] if value.ndim>=2 and key!='time_s' else value for key,value in arrays.items()}
            score = oracle.score_trace(spec['scene'],row['bucket_xy_m'],row['board_normal'],calibration,trace)
            check_native_score(row,score,checks)
            items[row['id']].update(trace_summary(row,trace,spec['scene'],calibration))
    failures = [items[r['id']] for r in own if not captured(r)]
    rejects = [r for r in failures if not r['native_release']]; misses = [r for r in failures if r['native_release']]
    checks.equal((len(rejects),len(misses)),(16,17),'All16 setup rejections and17 native misses')
    losses = []
    for ident in LOSS_IDS:
        parent,recovery = (items[by[m,ident]['id']] for m in ('parent_bc','recovery_bc'))
        delta = (np.asarray(recovery['values'])-parent['values']).tolist()
        losses.append(dict(context_id=ident,bucket_xy_m=recovery['bucket_xy_m'],parent_bc=parent,recovery_bc=recovery,
            delta_values=delta,delta_release_height_m=delta[2]+delta[4],
            same_anchor=by['parent_bc',ident]['effective_source_hint']==by['recovery_bc',ident]['effective_source_hint']))
    # Recheck every consumed byte after extraction. The complete source proof is
    # inherited; this separate proof only covers derived diagnosis and rescoring.
    for name,expected in list(consumed.items()):
        checks.equal(sha(contained(study,name)),expected,'Consumed source unchanged/'+name)
    checks.equal(sha(audit),SOURCE['audit_sha256'],'Original proof unchanged after extraction')
    source = dict(SOURCE,full_checks_verified=True,native_checks_verified=True,checks_count=proof['checks_count'],
        producer_commit='522409306b836e39d2108be91c5d57fcf1fd484b',diagnosis_source_sha256=sha(__file__),
        consumed_files_sha256=consumed,helper_source_sha256={p.name:sha(p) for p in
            (Path(oracle.__file__).with_name('xy650_verify.py'),Path(oracle.__file__),Path(oracle.__file__).with_name('table_training_verify.py'))})
    result = dict(version=VERSION,status='completed',scope=SCOPE,source=source,parameter_order=PARAMETERS,
        summary=dict(requested=70,parent_bc_captures=34,recovery_bc_captures=37,gains=8,losses=5,setup_rejections=16,flight_misses=17,
            failure_groups=dict(Counter(r['group'] for r in failures)),
            first_post_board_contact_groups=dict(Counter('+'.join((r['first_flight'] or {}).get('cutoff_contact_channels',[])) or 'unknown' for r in misses)),
            forbidden_contact_misses=sum(bool(r['forbidden_contact_steps']) for r in misses)),
        scene=spec['scene'],losses=losses,setup_rejections=rejects,flight_misses=misses,
        recommended_correction=dict(parameter='board_height_m',parameter_index=2,mode='restore_matched_parent_coordinate',
            context_ids=LOSS_IDS,settings=[dict(context_id=p['context_id'],parent_record_sha256=p['parent_bc']['record_sha256'],
                recovery_record_sha256=p['recovery_bc']['record_sha256'],original_values=p['recovery_bc']['values'],
                corrected_values=[p['parent_bc']['values'][i] if i==2 else p['recovery_bc']['values'][i] for i in range(5)]) for p in losses],
            rule='For all five losses, restore only board height to the matched parent-BC value; retain the other four recovery-BC coordinates and bucket.',
            coupled_geometry='With nominal gap unchanged, restoring board_height also shifts derived absolute release Z by the same height delta. This is one shot-parameter change, not an isolated physical board displacement; no gap compensation.',
            rationale='Three losses became setup rejects, including a low board/table branch collision and a missing board IK branch. Height restoration tests admission sensitivity; it need not fix either released miss. Their larger gaps remain a separate hypothesis, not a second intervention.',
            posthoc_development=True,new_training=False,additional_coordinate_search=False),
        limitations=[SCOPE,'A changed parameter and changed outcome are associated; one-coordinate intervention remains a hypothesis.',
            'Finite IK counts and rejected branch pairs do not prove no feasible solution exists.',
            'Table/bucket force channels are permitted and do not alone identify rim contact or causal assistance.',
            'Every inspected TEST position is now development data; no future result here establishes generalization.'])
    result['diagnosis_fingerprint'] = digest(result)
    output.mkdir(parents=True)
    path = output/'diagnosis.json';path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    derived = dict(version=AUDIT_VERSION,status='passed',full_checks_verified=True,report_sha256=sha(path),
        diagnosis_fingerprint=result['diagnosis_fingerprint'],source=source,checks_count=checks.count,checks=checks.entries,failures=[],
        scope='Source byte binding, complete loss/failure accounting and independent v3 trace rescoring; original native validity inherited from full283251-check proof.')
    (output/'audit.json').write_text(json.dumps(derived,indent=2,allow_nan=False)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study','audit','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args();result = build(args.study,args.audit,args.output)
    print(json.dumps(dict(summary=result['summary'],diagnosis_fingerprint=result['diagnosis_fingerprint']),indent=2))


if __name__=='__main__':
    main()
