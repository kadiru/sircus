"""Authenticated CPU diagnosis of finger bounds and the frozen time horizon.

No simulation stepping, fitting or label changes. The original TEST cases are
now development evidence. Accurate CPU distance queries provide an independent
geometric diagnostic; acceptance still uses auditable separating-plane bounds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import arm_clearance_distance as precise
from . import arm_clearance_selector as old
from . import clearance_horizon_diagnosis as horizon
from .prepare_arm_clearance_report import compact, read, sha
from .benchmark_verify import contained
from .xy650_verify import first_flight

VERSION = 'sircus-clearance-prediction-diagnosis-v1'
AUDIT_VERSION = 'sircus-clearance-prediction-diagnosis-audit-v1'
FINGER_CASES = ('04', '07', '39')


class Checks:
    def __init__(self): self.rows = []
    def require(self, flag, label):
        self.rows.append(dict(label=label, passed=bool(flag)))
        if not flag: raise ValueError(label)


def independent_support_check(points, colliders, radius, allowance, proof, checks, label):
    """Rebuild every saved separating-plane certificate without certify_path."""
    refinements = {item['geom']: item for item in proof['refinements']}
    checks.require(len(refinements) == len(proof['refinements']), label+'/unique refined colliders')
    expected = set(); best = float('inf')
    for collider in colliders:
        planes = np.asarray(collider['planes'])
        lower = np.max(np.dot(points, planes[:, :3].T) + planes[:, 3], axis=1)
        old_bound = lower-radius-allowance-1e-10
        if old_bound.min() < old.MARGIN_M:
            expected.add(collider['name'])
            normals = np.asarray(refinements[collider['name']]['support_normals'])
            checks.require(normals.shape == points.shape and np.isfinite(normals).all()
                and np.max(np.abs(np.sum(normals*normals, axis=1)-1)) < 2e-9, label+'/unit witness normals/'+collider['name'])
            normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
            vertices = np.asarray(collider['vertices'])
            support = np.max(np.dot(vertices, normals.T), axis=0)
            distances = np.sum(points*normals, axis=1)-support
            lower = np.maximum(lower, distances)
            checks.require(np.all(support[None, :] >= vertices@normals.T-1e-12), label+'/support contains every vertex/'+collider['name'])
        best = min(best, float((lower-radius-allowance-1e-10).min()))
    checks.require(set(refinements) == expected, label+'/exact refinement domain')
    checks.require(abs(best-proof['minimum_clearance_lower_bound_m']) < 1e-10, label+'/independently reconstructed bound')
    checks.require(proof['passed'] is (best >= old.MARGIN_M), label+'/unchanged two-mm margin')
    return best


def build(study, audit_path, assets):
    study, audit_path, assets = map(Path, (study, audit_path, assets))
    published = compact(study, audit_path)
    checks = Checks()
    checks.require(published.get('status') == 'verified' and published['audit']['full_checks_verified'] is True,
                   'Complete byte-bound original independent proof')
    manifest = read(study/'evidence-manifest.json')
    def bound(relative):
        path = contained(study, relative)
        checks.require(sha(path) == manifest[relative], 'Consumed source bytes/'+relative)
        return path
    spec = read(bound('contract.json')); calibration = read(bound('calibration.json'))['score_config']
    rows = [json.loads(line) for line in bound('records.jsonl').read_text().splitlines()]
    rankings = {item['context_id']: item for item in read(bound('rankings.json'))}
    gravity = spec['native_reference']['allocated_options']['native']['gravity']
    gate = precise.PreciseArmClearanceGate(spec['scene'], assets)
    ball = gate.ball_geom; ids = [item['index'] for item in gate.local]
    def distances(point):
        gate.d.qpos[gate.ball:gate.ball+3] = point
        gate.mj.mj_kinematics(gate.m, gate.d)
        return np.asarray([gate.mj.mj_geomDistance(gate.m, gate.d, ball, index, 10., None) for index in ids])
    cases = []
    for case in FINGER_CASES:
        context = 'arm-clearance-test-'+case
        ledger = rankings[context]
        row = next(item for item in rows if item.get('context_id') == context and item['method'] == 'consistent')
        entry = next(item for item in ledger['selection']['arm_clearance']['checks'] if item['candidate_index'] == row['candidate_index'])
        connection = read(bound('native/'+entry['connection_file']))
        checks.require(sha(study/'native'/entry['connection_file']) == entry['connection_sha256'], context+'/exact endpoint proof')
        diagnostic = ledger['banks']['arm_clearance']['ranking']['diagnostics'][row['candidate_index']]
        original = old.ArmClearanceGate.check(gate, connection['q10'], connection['q14'], row['bucket_xy_m'], row['values'], diagnostic, gravity=gravity)
        checks.require(original == entry['clearance'], context+'/reproduced immutable original clearance')
        corrected = gate.check(connection['q10'], connection['q14'], row['bucket_xy_m'], row['values'], diagnostic, gravity=gravity)
        colliders = gate.colliders(connection['q10'], connection['q14'], row['bucket_xy_m'])
        members = []
        for predicted, before, after in zip(diagnostic['members'], original['members'], corrected['members']):
            label = context+'/member'+str(predicted['member'])
            checks.require(before['horizon_s'] == after['horizon_s'] and before['samples'] == after['samples']
                and before['between_sample_allowance_m'] == after['between_sample_allowance_m'], label+'/unchanged sampling horizon and allowance')
            times = np.linspace(0., before['horizon_s'], before['samples'])
            points = np.asarray(old.trajectory_for_member(diagnostic['launch']['origin_m'], predicted['velocity_world_m_s'], times, gravity=gravity))
            independent_support_check(points, colliders, gate.radius, before['between_sample_allowance_m'], after, checks, label)
            exact = np.asarray([distances(point).min() for point in points])
            checks.require(np.isfinite(exact).all(), label+'/finite CPU distance oracle')
            oracle_bound = float(exact.min()-before['between_sample_allowance_m'])
            checks.require(after['minimum_clearance_lower_bound_m'] <= oracle_bound+2e-6, label+'/support bound no greater than independent CPU signed distance')
            members.append(dict(member=predicted['member'], old_lower_bound_m=before['minimum_clearance_lower_bound_m'],
                corrected_lower_bound_m=after['minimum_clearance_lower_bound_m'], native_exact_sample_min_m=float(exact.min()),
                native_exact_minus_allowance_m=oracle_bound, between_sample_allowance_m=before['between_sample_allowance_m'],
                closest_geom=after['closest']['geom'], horizon_s=before['horizon_s'], samples=before['samples']))
        cache = bound('native/'+row['native_cache_file'])
        checks.require(sha(cache) == row['native_cache_sha256'], context+'/exact native cache')
        with np.load(cache, allow_pickle=False) as archive:
            col = row['native_cache_row']
            checks.require(str(archive['record_ids'][col]) == row['id'], context+'/native column identity')
            trace = {key: archive[key][:, col] if archive[key].ndim >= 2 else archive[key]
                     for key in ('time_s', 'position_m', 'velocity_m_s', 'forces_N')}
        flight = first_flight(spec['scene'], row['bucket_xy_m'], row['board_normal'], calibration, trace, valid=True)
        checks.require(flight['available'] is True, context+'/observed board rebound')
        start, end = flight['start_index'], flight['end_index']
        # Every recorded first-flight sample, not a claim of continuous clearance.
        observed_min = min(float(distances(point).min()) for point in trace['position_m'][start:end+1])
        observed_robot = bool(np.any(trace['forces_N'][start:row['physics_steps'], 4] > .01))
        native_audit = next(item for item in published['test_positions'] if item['record_id'] == row['id'])
        checks.require(observed_robot is native_audit['robot_contact_before_terminal'], context+'/observed native robot contact agrees with audit')
        origin = np.asarray(diagnostic['launch']['origin_m']); observed_origin = trace['position_m'][start]
        joint_delta = max(float(np.max(np.abs(np.asarray(row[key])-np.asarray(row['native_initial_'+key])))) for key in ('q10', 'q14'))
        weakest = min(original['members'], key=lambda item:item['minimum_clearance_lower_bound_m'])
        cases.append(dict(context_id=context, record_id=row['id'], native_capture=bool(row['route_success']),
            native_robot_contact=observed_robot, old_lower_bound_m=original['minimum_clearance_lower_bound_m'],
            corrected_lower_bound_m=corrected['minimum_clearance_lower_bound_m'], corrected_passed=corrected['passed'],
            closest_geom=weakest['closest']['geom'], bucket_xy_m=row['bucket_xy_m'], values=row['values'],
            observed_first_flight_sample_clearance_m=observed_min,
            observed_distance_scope='CPU collision-distance reconstruction at admitted held joints for every saved first-flight ball center; not measured continuous clearance',
            nominal_origin_m=origin.tolist(), observed_origin_m=observed_origin.tolist(),
            observed_minus_nominal_origin_mm=((observed_origin-origin)*1000).tolist(),
            observed_rebound_velocity_m_s=trace['velocity_m_s'][start].tolist(),
            predicted_member_velocities_m_s=[item['velocity_world_m_s'] for item in diagnostic['members']],
            maximum_initial_joint_rounding_difference_rad=joint_delta, pose_gate=row['pose_gate'],
            max_board_drift_mm=row['max_board_drift_mm'], native_cache_sha256=row['native_cache_sha256'], members=members,
            diagnosis='Conservative face-only corner bound rejects a geometrically clear predicted path.' if corrected['passed'] else
                'Tighter distance alone remains below the unchanged safety margin; predicted flight and close observed geometry also matter.'))
    horizon_report = horizon.build(study, audit_path, assets)
    checks.require(horizon_report['source']['audit_sha256'] == sha(audit_path), 'Horizon report authenticates identical original audit')
    horizon_cases = []
    for item in horizon_report['cases']:
        if item['method'] != 'arm_clearance' or item['first_robot_delay_s'] is None: continue
        relation = horizon.horizon_relation(item['first_robot_delay_s'], item['predicted_rim_start_s'])
        checks.require(relation == item['robot_contact_horizon_relation'], item['record_id']+'/independent horizon coverage arithmetic')
        horizon_cases.append(dict(context_id=item['context_id'], native_capture=item['capture'],
            predicted_horizon_min_s=min(item['predicted_rim_start_s']), predicted_horizon_max_s=max(item['predicted_rim_start_s']),
            observed_first_contact_s=item['first_contact_delay_s'], observed_robot_contact_s=item['first_robot_delay_s'],
            first_contact_channels=item['first_contact_channels'], after_all_horizons=relation['event_after_all_horizons'],
            hypothetical_longer_horizon_passed=item['hypothetical_table_horizon']['passed'],
            diagnosis='Recorded robot contact occurs after all five frozen nominal horizons; extending a ballistic projection does not model intervening contact.'))
    report = dict(version=VERSION, status='completed', source=dict(audit_sha256=sha(audit_path),
        report_sha256=published['report_sha256'], manifest_sha256=published['manifest_sha256'],
        plan_sha256=published['plan_sha256'], native_checks=published['audit']['checks_count'],
        diagnostic_source_sha256=sha(__file__), horizon_source_sha256=sha(horizon.__file__),
        correction_source_sha256=sha(precise.__file__)), finger_cases=cases, horizon_cases=horizon_cases,
        summary=dict(finger_cases=3, recovered_prediction_passes=sum(item['corrected_passed'] for item in cases),
            horizon_robot_contact_shots=len(horizon_cases), events_after_all_horizons=sum(item['after_all_horizons'] for item in horizon_cases),
            correction='Only replace the conservative face-only bound with certified witness-direction support planes.',
            no_training=True, no_new_native_shots=True), correction_rule=precise.RULE, correction_rule_sha256=precise.fingerprint(precise.RULE),
        scope='Post-hoc CPU diagnosis of prior fixed64 development evidence. Recorded outcomes unchanged. Geometry queries do not step physics. The correction preserves all collision geometry, the 2 mm margin, five predictions and existing time horizons.')
    return report, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('study', 'audit', 'assets', 'output'): parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.study.resolve()):
        raise ValueError('New derivative directory outside immutable source required')
    report, checks = build(args.study, args.audit, args.assets)
    args.output.mkdir(parents=True)
    path = args.output/'diagnosis.json'; path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    proof = dict(version=AUDIT_VERSION, status='passed', report_sha256=sha(path),
        auditor_source_sha256=sha(__file__),
        source_audit_sha256=report['source']['audit_sha256'], checks_count=len(checks.rows), checks=checks.rows, failures=[],
        scope='Authenticated original bytes; independent support-plane arithmetic and CPU distance-oracle checks. No new physics rollout or material calibration.')
    (args.output/'diagnosis-audit.json').write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps(report['summary']))


if __name__ == '__main__': main()
