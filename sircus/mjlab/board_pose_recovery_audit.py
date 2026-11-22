"""Independently audit the bounded board-pose recovery search on saved evidence.

No GPU allocation, native rollout, source-data edits or policy training. Search
captures and their three fresh confirmations are separate development results.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np

from .benchmark_verify import Checks, contained, digest, fingerprint, read, sha
from .hybrid_miss_audit import classify
from .sweep_worker import atomic

VERSION = 'sircus-board-pose-recovery-independent-audit-v1'
STUDY_VERSION = 'sircus-board-pose-recovery-v1'
CASES = tuple('stalled-reference-test-' + x for x in ('03', '09', '19'))
SOURCE_REPORT = 'e8cd7009362af9c80f7fe6a54b70937b6cbef8f4182eaf04d22e6b7e8dbac2e1'
CAPTURES = ('direct', 'table', 'unclassified_capture')


def captured(row):
    """Physics-invalid and robot-assisted outcomes do not qualify silently."""
    return classify(row) in CAPTURES


def exact_repeat(source, request, checks, label):
    """Only bookkeeping may change between search and confirmation commands."""
    mutable = {'id', 'context_id', 'candidate_id', 'variant', 'repeat', 'method',
               'phase', 'source_request_id', 'role'}
    checks.equal({k: v for k, v in request.items() if k not in mutable},
                 {k: v for k, v in source.items() if k not in mutable},
                 label + '/exact original geometry, joint branches and proofs')


def search_success_ids(rows):
    """Eligibility comes only from initial search results, never later repeats."""
    found = [r['candidate_id'] for r in rows if r.get('variant') == 'search' and captured(r)]
    if len(found) != len(set(found)):
        raise ValueError('Duplicate initial search candidate')
    return found


def counts(rows, requested):
    labels = [classify(r) for r in rows]
    return dict(requested=requested, completed=len(rows), released=len(rows), setup_rejected=0,
                captures=sum(x in CAPTURES for x in labels),
                direct=labels.count('direct'), table=labels.count('table'),
                robot_contacts=sum(r['experimental_route']['robot_contact_steps'] > 0 for r in rows),
                outcomes=labels, rows=rows)


def verify_confirmations(plan, rows, checks):
    """Every qualifying search capture gets exactly three unchanged releases."""
    eligible = set(search_success_ids(rows))
    candidates = {r['id']: r for c in plan['cases'] for r in c['candidates']}
    repeats = [r for r in rows if r.get('variant') == 'repeat']
    checks.equal({r['candidate_id'] for r in repeats}, eligible,
                 'Repeat every and only qualifying initial search capture')
    for identity in eligible:
        records = [r for r in repeats if r['candidate_id'] == identity]
        checks.equal(sorted(r['repeat'] for r in records), [1, 2, 3],
                     identity + '/three fresh nominal confirmations')
        checks.require(identity in candidates, identity + '/belongs to frozen candidate menu')
    return eligible


def verify_request_keys(plan, slots, checks, *, confirmations=False, eligible=()):
    """Check every frozen command and denominator without trusting runner counts."""
    sources = {c['id']: {r['id']: r for r in c['candidates']} for c in plan['cases']}
    originals = {c['id']: c['original'] for c in plan['cases']}
    expected = set()
    if confirmations:
        for case_id, candidates in sources.items():
            expected.update((case_id, 'repeat', identity, repeat)
                            for identity in candidates if identity in eligible for repeat in (1, 2, 3))
    else:
        for case_id, candidates in sources.items():
            expected.update((case_id, 'original', originals[case_id]['id'], repeat) for repeat in range(3))
            expected.update((case_id, 'search', identity, 0) for identity in candidates)
    actual = set(); ids = set()
    for slot in slots:
        request = slot['request']
        checks.require(request is not None, 'Prepared schedule contains only admitted exact commands')
        context = request['context_id']; variant = request['variant']; identity = request['candidate_id']
        key = (context, variant, identity, request['repeat'])
        checks.require(key not in actual and request['id'] not in ids, 'Unique shot/repeat identity')
        actual.add(key); ids.add(request['id'])
        source = originals[context] if variant == 'original' else sources[context][identity]
        exact_repeat(source, request, checks, request['id'])
        checks.equal(request['bucket_xy_m'], originals[context]['bucket_xy_m'], request['id'] + '/fixed bucket')
    checks.equal(actual, expected, 'Complete frozen request set with no substitutions or omissions')


def ranked_diverse(rows, predictions, limit):
    def rank(row):
        value = predictions[row['id']].get('score_m')
        valid = type(value) in (int, float) and math.isfinite(value)
        return (not valid, -value if valid else 0., row['id'])
    first, rest, seen, geometries = [], [], set(), set()
    for row in sorted(rows, key=rank):
        values = row['values']
        geometry = tuple(round(float(x), 10) for x in (*values, row['board_heading_offset_deg'], row['board_twist_deg']))
        if geometry in geometries:
            continue
        geometries.add(geometry)
        key = tuple(round(x/s) for x, s in zip(values, (.05, .05, .04, 8., .08), strict=True)) + (
            round(row['board_heading_offset_deg']/16), round(row['board_twist_deg']/180))
        (rest if key in seen else first).append(row)
        seen.add(key)
    return (first + rest)[:limit]


def independent_summary(plan, rows):
    summaries = []
    for case in plan['cases']:
        records = [r for r in rows if r.get('context_id') == case['id']]
        winners = set(search_success_ids(records)); groups = {}
        for variant, requested in (('original', 3), ('search', len(case['candidates'])), ('repeat', 3*len(winners))):
            groups[variant] = counts([r for r in records if r.get('variant') == variant], requested)
        repeated = []
        for candidate in case['candidates']:
            if candidate['id'] not in winners:
                continue
            confirmations = [r for r in records if r.get('variant') == 'repeat' and r['candidate_id'] == candidate['id']]
            repeated.append(dict(candidate_id=candidate['id'], request=candidate,
                                 repeat_captures=sum(captured(r) for r in confirmations), repeat_completed=len(confirmations)))
        def rank(item):
            score = case['predictions'][item['candidate_id']]['score_m']
            valid = type(score) in (int, float) and math.isfinite(score)
            return (-item['repeat_captures'], (not valid, -score if valid else 0., item['candidate_id']))
        ordered = sorted(repeated, key=rank)
        confirmed = [r for r in repeated if r['repeat_captures'] == r['repeat_completed'] == 3]
        best = ordered[0]['request'] if ordered else (case['candidates'][0] if case['candidates'] else None)
        summaries.append(dict(id=case['id'], bucket_xy_m=case['original']['bucket_xy_m'], results=groups,
            proposed=len(case['proposals']), checked=len(case['records']), admitted=len(case['admitted']),
            selected=len(case['candidates']), repeated_settings=repeated, confirmed_settings=confirmed, best_candidate=best))
    return summaries


def verify_proposals(case, checks):
    """Independently enforce the authorized bounded geometry changes."""
    old = case['original']; bucket = np.asarray(old['bucket_xy_m']); initial = np.asarray(old['values'])
    direction = bucket-initial[:2]; direction /= np.linalg.norm(direction)
    tangent = np.array([-direction[1], direction[0]])
    ids = set(); geometry = set(); originals = {}
    for p in case['proposals']:
        label = p['id']; values = np.asarray(p['values'], float)
        checks.require(label not in ids, label+'/unique proposal'); ids.add(label)
        checks.equal((p['context_id'], p['bucket_xy_m'], p['phase']),
                     (case['id'], old['bucket_xy_m'], 'search'), label+'/fixed bucket and development phase')
        checks.equal((p['seed_q10'], p['seed_q14']), (old['selected_q10'], old['selected_q14']), label+'/explicit original IK hint')
        checks.require(values.shape == (5,) and np.isfinite(values).all(), label+'/five finite shot values')
        shift = values[:2]-initial[:2]
        checks.require(abs(shift@direction) <= .150000001 and abs(shift@tangent) <= .150000001,
                       label+'/board center within declared radial and lateral window')
        checks.require(-30 <= p['board_heading_offset_deg'] <= 30 and p['board_twist_deg'] in (0, 180),
                       label+'/declared board orientation window')
        checks.require(np.linalg.norm(bucket-values[:2]) > .15, label+'/minimum board-to-bucket distance')
        checks.require(np.all(values >= np.asarray(case['bounds']['low'])-1e-9)
                       and np.all(values <= np.asarray(case['bounds']['high'])+1e-9), label+'/admission domain')
        if p['group'] == 'position_heading':
            checks.equal(p['values'][2:], old['values'][2:], label+'/position-heading family preserves height tilt gap')
        else:
            checks.equal(p['group'], 'position_height_tilt_gap', label+'/known search family')
            checks.require(-.120000001 <= values[2]-initial[2] <= .080000001
                           and 12 <= values[3] <= 44 and .08 <= values[4] <= .55,
                           label+'/declared height tilt gap tuning window')
        key = tuple(round(float(x), 10) for x in (*p['values'], p['board_heading_offset_deg'], p['board_twist_deg']))
        if 'retry_of' in p:
            checks.require(p['retry_of'] in originals, label+'/multistart retries an existing geometry')
            expected = dict(originals[p['retry_of']], id=p['retry_of']+'-multistart', retry_of=p['retry_of'], ik_search='multistart16')
            checks.equal(p, expected, label+'/multistart changes only IK search metadata')
        else:
            checks.require(key not in geometry, label+'/unique initial geometry'); geometry.add(key)
            originals[label] = p
    checks.require(len(originals) <= 4782 and len(ids)-len(originals) <= 24, case['id']+'/finite geometry and retry budgets')


def recompute_screen(ranker, scene, proposal, models):
    """Rebuild analytic screening values from geometry, without released data."""
    from .cell1050_heading_native import heading_targets
    from .fixed35_broad_native import wrist_bound
    from .intergrid_flight_ranker import board_launch, rebound
    from .consistent_flight_selector import member_flight, _descending_time
    targets = heading_targets(scene, proposal['bucket_xy_m'], proposal['values'],
                              proposal['board_heading_offset_deg'], proposal['board_twist_deg'])
    arms = dict(dropper=wrist_bound(models['dropper'], targets['dropper_flange_world']),
                bouncer=wrist_bound(models['bouncer'], targets['board_flange_local']))
    reach = dict(passed=all(v['passed'] for v in arms.values()), arms=arms,
                 scope='Necessary outer wrist reach only; not IK or collision/path admission')
    release = np.asarray(proposal['values'][:3], float); release[2] += proposal['values'][4]+ranker.radius
    be, bt, te, tt = ranker.coefficients
    launch = board_launch(targets['board_world'], ranker.board_size, release, ranker.radius, ranker.gravity[2], be, bt)
    prediction = dict(direct=None, table=None, predicted_route=None, score_m=None)
    if launch is None:
        return reach, prediction
    position, velocity = launch; bucket = proposal['bucket_xy_m']
    direct = member_flight(position, velocity, bucket, ranker.geom)['conservative_rim_band_clearance_m']
    if direct is not None:
        prediction['direct'] = float(direct)
    time = _descending_time(position[2], velocity[2], ranker.table_center[2]+ranker.table_half[2]+ranker.radius, ranker.gravity[2])
    if time is not None and time <= 2.:
        impact = position+velocity*time+.5*ranker.gravity*time*time
        edge = float(min(ranker.table_half[:2]-np.abs(impact[:2]-ranker.table_center[:2])))
        if edge >= 0:
            after = rebound(velocity+ranker.gravity*time, np.array([0., 0., 1.]), te, tt)
            table = member_flight(impact, after, bucket, ranker.geom)['conservative_rim_band_clearance_m']
            if table is not None:
                prediction['table'] = float(min(table, edge))
    routes = [r for r in ('direct', 'table') if prediction[r] is not None]
    if routes:
        best = max(routes, key=lambda r: prediction[r])
        prediction.update(predicted_route=best, score_m=prediction[best])
    return reach, prediction


def verify_admissions(case, directory, prior, home, resolutions, checks):
    from .direct_path_first_audit import _proof, verify_seed
    proposals = {p['id']: p for p in case['proposals']}
    admitted = {r['id']: r for r in case['admitted']}
    checks.equal(len(admitted), len(case['admitted']), case['id']+'/unique admitted branch identities')
    ordered = []
    for record in case['records']:
        label = record['proposal_id']; proposal = proposals[label]
        checks.equal(record['proposal_sha256'], digest(proposal), label+'/exact checked proposal')
        ids = record['admitted_request_ids']
        checks.equal(record['admitted'], len(ids), label+'/complete admitted count')
        checks.require(len(ids) <= (4 if 'retry_of' in proposal else 1), label+'/bounded branch budget')
        for index, identity in enumerate(ids):
            request = admitted[identity]
            checks.equal(identity, label+f'-b{index}', label+'/consecutive admitted branch identity')
            checks.equal({k: request[k] for k in proposal if k != 'id'}, {k: v for k, v in proposal.items() if k != 'id'},
                         label+'/admission keeps exact requested geometry and seed')
            checks.equal((request['proposal_id'], request['native_prepared'], request['command_values']),
                         (label, True, proposal['values']), label+'/exact prepared command')
            proof = _proof(request, dict(id=case['id'], bucket_xy_m=case['original']['bucket_xy_m']),
                           directory, prior, home, resolutions, checks)
            verify_seed(proof, proposal, checks)
            ordered.append(identity)
    checks.equal(ordered, list(admitted), case['id']+'/every and only admitted branch accounted')


def audit(a):
    from . import replay650_verify as replay
    from .cell1050_rebound_study import parent_inputs, control_requests
    from .direct_path_first_audit import _proof, verify_heading
    from .intergrid_route_refit_inputs import _audited_source
    from .intergrid_route_labels import annotate_routes
    from .parallel_capture_grid_audit import observed_rows, rescore_packet, verify_release
    from .policy64_verify import audit_native_options
    from .table_training_verify import control_gate, rescore_rows
    from .intergrid_flight_ranker import FlightRanker
    from .fixed35_broad_native import reach_models

    study, checks = a.output, Checks()
    spec, report, plan = (read(study/name) for name in ('contract.json', 'report.json', 'preparation.json'))
    fingerprint(spec, checks, 'Frozen board-pose search contract')
    checks.equal((spec['version'], plan['version'], report['version'], report['status'], report['phase']),
                 (STUDY_VERSION, STUDY_VERSION, STUDY_VERSION, 'completed', 'completed'), 'Completed declared development search')
    checks.equal((spec['physics_substeps'], spec['repeats'], spec['controls']), (64, 3, 18), 'Fixed64 and three confirmations with matched controls')
    checks.require(0 < report['gpu_lease_s'] <= spec['gpu_cap_s'] <= 3600, 'Bounded acquired 4090 time')
    manifest = replay.check_manifest(study, checks)
    checks.equal(report['evidence_manifest_sha256'], sha(study/'evidence-manifest.json'), 'Report binds all immutable evidence')
    checks.equal(spec['plan_sha256'], manifest['preparation.json'], 'Frozen CPU preparation hash')
    for name, value in spec['frozen_files'].items():
        checks.equal(manifest[name], value, 'Every pre-release file unchanged/'+name)
    source = a.source_root or Path(__file__).resolve().parents[2]
    for name, value in spec['source_sha256'].items():
        checks.equal(sha(contained(source, name)), value, 'Exact frozen producer source/'+name)
    parent_spec, parent_binding, _ = parent_inputs(a.parent_study, a.parent_audit)
    checks.equal((spec['identity'], spec['scene'], spec['parent']),
                 (parent_spec['identity'], parent_spec['scene'], parent_binding), 'Unchanged calibrated native scene')
    _, _, _, bindings, source_spec = _audited_source(a.source_study, a.source_audit, 'sircus-stalled-reference-study-v1',
        ('choices.json', 'rows.jsonl', 'training-prior.json', 'calibration.json', 'setup-home.json'))
    checks.equal(bindings['report.json'], SOURCE_REPORT, 'Exact source of the three missed shots')
    checks.equal(plan['source'], bindings, 'Prepared source binding')
    checks.equal(plan['identity'], source_spec['identity'], 'Same source scene identity')
    checks.equal(plan['search_rule'], spec['search_rule'], 'Search rule frozen before release')
    checks.equal((plan['repeats'], plan['seed']), (3, 2026112207), 'Declared deterministic seed and repeats')
    checks.equal([c['id'] for c in plan['cases']], list(CASES), 'Exactly three diagnosed bucket positions')
    original_choices = {s['context_id']: s['request'] for s in read(a.source_study/'choices.json')['slots']
                        if s['method'] == 'hybrid' and s['request'] is not None}
    original_rows = {r['id']: r for r in observed_rows(a.source_study/'rows.jsonl')}
    prior, home = read(study/'training-prior.json'), read(study/'setup-home.json')
    calibration = replay.check_calibration(study, spec['identity']['calibration'], checks)
    ranker = FlightRanker(spec['scene'], calibration, a.assets, source_spec['flight_ranking']['rule'])
    models = reach_models(str(a.assets.resolve()))
    resolutions = {}
    for case in plan['cases']:
        label = case['id']; directory = study/case['directory']; old = original_choices[label]
        checks.equal(case['original'], old, label+'/exact original request preserved')
        checks.equal(case['historical'], original_rows[old['id']], label+'/exact original observation preserved')
        checks.require(not captured(case['historical']) and case['historical']['native_release']
                       and not case['historical']['invalid_physics'], label+'/released valid historical miss')
        checks.equal(case['directory'], 'native-'+label[-2:], label+'/separate proof namespace')
        resolutions[case['directory']] = {}
        _proof(old, dict(id=label, bucket_xy_m=old['bucket_xy_m']), directory, prior, home, resolutions[case['directory']], checks)
        checks.equal(case['original_prediction'], ranker.predict(old), label+'/recomputed original flight prediction')
        verify_proposals(case, checks)
        proposals = {p['id']: p for p in case['proposals']}
        nominal = case['nominal_predictions']; reach = case['wrist_reach']
        checks.equal(set(nominal), set(proposals), label+'/all cheap-screen predictions retained')
        checks.equal(set(reach), set(proposals), label+'/all necessary reach screens retained')
        for p in case['proposals']:
            recomputed_reach, recomputed_nominal = recompute_screen(ranker, spec['scene'], p, models)
            checks.equal((reach[p['id']], nominal[p['id']]), (recomputed_reach, recomputed_nominal), p['id']+'/screening uses only geometry')
        base_proposals = [p for p in case['proposals'] if 'retry_of' not in p]
        retries = [p for p in case['proposals'] if 'retry_of' in p]
        groups = [ranked_diverse([p for p in base_proposals if p['group'] == group and reach[p['id']]['passed']], nominal, 96)
                  for group in ('position_heading', 'position_height_tilt_gap')]
        order = [g[i]['id'] for i in range(max(map(len, groups))) for g in groups if i < len(g)]
        checks.equal(case['preparation_order'], order+[p['id'] for p in retries], label+'/interleaved screen then multistart order')
        warm_records = [r for r in case['records'] if 'retry_of' not in proposals[r['proposal_id']]]
        retry_records = [r for r in case['records'] if 'retry_of' in proposals[r['proposal_id']]]
        checks.equal([r['proposal_id'] for r in warm_records], order[:len(warm_records)], label+'/sequential warm setup attempts')
        rejected = {r['proposal_id'] for r in warm_records if not r['admitted'] and r['reason'] in (
            'finite_ik_unresolved', 'strict_endpoint_clearance_or_pose_rejected', 'finite_requested_setup_path_unresolved')}
        def retry_rank(identity):
            score = nominal[identity]['score_m']; finite = type(score) in (int, float) and math.isfinite(score)
            return (not finite, -score if finite else 0., identity)
        retry_sources = sorted(rejected, key=retry_rank)[:24]
        checks.equal([p['retry_of'] for p in retries], retry_sources[:len(retries)], label+'/only highest-ranked failed warm geometries retried')
        checks.equal([r['proposal_id'] for r in retry_records], [p['id'] for p in retries], label+'/every multistart proposal has an outcome')
        checks.require(len(warm_records) <= 192 and len(retry_records) <= 24, label+'/bounded IK/path attempts')
        checks.require(not retries or len(warm_records) == len(order), label+'/multistart follows completed warm menu')
        verify_admissions(case, directory, prior, home, resolutions[case['directory']], checks)
        predictions = {r['id']: ranker.predict(r) for r in case['admitted']}
        checks.equal(case['predictions'], predictions, label+'/precise clearance and rim predictions recomputed')
        checks.equal(case['candidates'], ranked_diverse(case['admitted'], predictions, 40), label+'/frozen ranked diverse native candidate menu')
        checks.equal(case['selection_order'], [r['id'] for r in case['candidates']], label+'/explicit candidate order')
        checks.require(len(case['candidates']) <= 40, label+'/native candidate budget')

    slots = read(study/'slots.json'); repeat_slots = read(study/'repeat-slots.json')
    verify_request_keys(plan, slots, checks)
    checks.equal(read(study/'native-started.json')['slots_sha256'], sha(study/'slots.json'), 'Native launch binds frozen initial commands')
    initial_count = 9+sum(len(c['candidates']) for c in plan['cases'])
    checks.equal((spec['initial_requests'], spec['maximum_confirmation_requests'], spec['maximum_total_requests']),
                 (initial_count, (initial_count-9)*3, initial_count+(initial_count-9)*3+18), 'Explicit maximum native budget')
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
    checks.equal(report['cases'], summary, 'Independent proposal, original, search and confirmation counts')
    checks.equal(report['progress'], dict(completed=len(observed), planned=len(slots)+len(repeat_slots)), 'Complete diagnostic denominator')
    checks.equal((report['native_releases'], report['initial_requests'], report['confirmation_requests']),
                 (len(observed)+18, len(slots), len(repeat_slots)), 'All native releases including controls accounted')
    return dict(version=VERSION, status='passed', checks_count=checks.count, report_sha256=sha(study/'report.json'),
        plan_sha256=sha(study/'preparation.json'), evidence_manifest_sha256=sha(study/'evidence-manifest.json'),
        verifier_source_sha256=sha(Path(__file__)), frozen_source_root=str(source),
        initial_requests=len(slots), confirmation_requests=len(repeat_slots), control_releases=len(control_rows),
        results=[dict(id=c['id'], bucket_xy_m=c['bucket_xy_m'], proposed=c['proposed'], checked=c['checked'],
            admitted=c['admitted'], selected=c['selected'], results={v: {k: x for k, x in g.items() if k != 'rows'}
                for v, g in c['results'].items()}, confirmed_settings=c['confirmed_settings']) for c in summary])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source-study', 'source-audit', 'parent-study', 'parent-audit', 'assets', 'output', 'audit'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--source-root', type=Path)
    args = parser.parse_args()
    if args.audit.exists() or args.audit.resolve().is_relative_to(args.output.resolve()):
        parser.error('Write a new receipt outside the frozen evidence directory')
    try:
        result = audit(args)
    except Exception as error:
        result = dict(version=VERSION, status='failed', error=f'{type(error).__name__}: {error}')
    atomic(args.audit, result); print(json.dumps(result), flush=True)
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
