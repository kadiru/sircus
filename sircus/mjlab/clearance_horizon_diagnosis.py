"""CPU-only diagnosis of the frozen clearance horizon against recorded flights.

Reads the already independently audited arm-clearance comparison. All ballistic
extensions are explicitly counterfactual model projections, not native shots or
new capture labels. The original selection rule and native evidence are immutable.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from sircus.mjlab.prepare_arm_clearance_report import compact, read, sha
from sircus.mjlab.benchmark_verify import contained
from sircus.mjlab.xy650_verify import first_flight
from sircus.mjlab.arm_clearance_selector import ArmClearanceGate, path_bound
from sircus.mjlab.consistent_flight_selector import trajectory_for_member


VERSION = 'sircus-clearance-horizon-diagnosis-v1'
FORCE_N = .01
SCOPE = ('Post-hoc CPU diagnosis of previously observed fixed64 TEST evidence. '
    'All times below use the observed first-board-rebound state as time zero; '
    'predicted paths still use the requested nominal tangent origin. Their '
    'different origins are reported explicitly. Hypothetical extensions retain '
    'the same learned velocities and collider bound and do not model bucket '
    'deflection, retention, incoming drop or driven-arm transit. No fitting, '
    'GPU stepping, new native shots or modified success labels.')


def descending_time(origin_z, velocity_z, target_z, gravity_z):
    """Independent descending root; absent crossings remain unknown."""
    values = [origin_z, velocity_z, target_z, gravity_z]
    if any(not isinstance(v, (float, int)) or not math.isfinite(v) for v in values) or gravity_z >= 0:
        raise ValueError('Finite scalar downward-gravity inputs required')
    discriminant = velocity_z**2 - 2*gravity_z*(origin_z-target_z)
    if discriminant < 0: return None
    t = (-velocity_z-math.sqrt(discriminant))/gravity_z
    return t if t > 0 and velocity_z+gravity_z*t < 0 else None


def horizon_relation(event_s, horizons):
    """Describe coverage without equating an extrapolated path to observation."""
    if event_s is None: return None
    if not isinstance(event_s, (float, int)) or not math.isfinite(event_s) or event_s < 0:
        raise ValueError('Finite nonnegative event time required')
    if len(horizons) != 5 or any(not isinstance(h, (float, int)) or not math.isfinite(h) or h <= 0 for h in horizons):
        raise ValueError('All five frozen positive horizons required')
    count = sum(event_s <= h for h in horizons)
    return dict(members_covering_event=count, event_after_all_horizons=count == 0,
        event_before_all_horizons=count == 5,
        after_latest_horizon_ms=1000*(event_s-max(horizons)))


def event_clock(trace, metric):
    """Use interval starts, excluding contact response at interval end."""
    if metric.get('available') is not True: return dict(available=False)
    start, end = metric['start_index'], metric['end_index']
    forces, times = trace['forces_N'], trace['time_s']
    if forces.shape != (len(times)-1, 6) or not 0 <= start <= end < len(times):
        raise ValueError('Aligned observed trace and first-flight boundaries required')
    active = np.flatnonzero(forces[:, 4] > FORCE_N)
    active = active[active >= start]
    first = float(times[end]-times[start]) if metric['cutoff_reason'] == 'contact' else None
    return dict(available=True, observed_rebound_time_s=float(times[start]),
        first_contact_delay_s=first, first_contact_channels=list(metric['cutoff_contact_channels']),
        first_robot_delay_s=float(times[active[0]]-times[start]) if len(active) else None,
        trace_start_index=int(start), first_flight_end_index=int(end))


def build(study, audit, assets):
    study, audit = Path(study), Path(audit)
    verified = compact(study, audit)
    if verified.get('status') != 'verified' or verified['audit']['full_checks_verified'] is not True:
        raise ValueError('Fully authenticated completed native comparison required')
    rows = [json.loads(line) for line in (study/'records.jsonl').read_text().splitlines()]
    rankings = {r['context_id']: r for r in read(study/'rankings.json')}
    spec = read(study/'contract.json'); calibration = read(study/'calibration.json')['score_config']
    gravity = spec['native_reference']['allocated_options']['native']['gravity']
    radius = float(spec['scene']['ball']['radius_m'])
    gate = ArmClearanceGate(spec['scene'], assets)
    # Every actually released new-selector shot plus the three false refusals
    # and original case44. The latter remain old selector observations.
    selected = [r for r in rows if r['role']=='test' and r['native_release'] is True and
        (r['method']=='arm_clearance' or r['context_id'] in
         {'arm-clearance-test-'+x for x in ('04','07','39','44')})]
    groups = {}
    for row in selected: groups.setdefault(row['native_cache_file'], []).append(row)
    items = []; checks = 0
    for filename, group in groups.items():
        path = contained(study/'native', filename); actual_sha = sha(path)
        with np.load(path, allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in ('record_ids','position_m','velocity_m_s','forces_N','time_s')}
        for row in group:
            col = row['native_cache_row']
            if actual_sha != row['native_cache_sha256'] or str(arrays['record_ids'][col]) != row['id']:
                raise ValueError('Bound native trace identity changed')
            checks += 2
            trace = {k: v[:,col] if v.ndim>=2 else v for k,v in arrays.items() if k!='record_ids'}
            metric = first_flight(spec['scene'],row['bucket_xy_m'],row['board_normal'],calibration,trace,
                valid=row['invalid_physics'] is False)
            clock = event_clock(trace,metric)
            if not clock['available']: raise ValueError('Selected recorded shot lacks a board rebound')
            ledger = rankings[row['context_id']]; bank = ledger['banks'][row['method']]
            diagnostic = bank['ranking']['diagnostics'][row['candidate_index']]
            members = diagnostic['members']; horizons = [m['rim_band_start_s'] for m in members]
            p0 = np.asarray(diagnostic['launch']['origin_m']); start,end=metric['start_index'],metric['end_index']
            observed0 = np.asarray(trace['position_m'][start]); observed_v = np.asarray(trace['velocity_m_s'][start])
            times = np.asarray([0.,float(trace['time_s'][end]-trace['time_s'][start])])
            predicted_end = [np.asarray(trajectory_for_member(p0,m['velocity_world_m_s'],times,gravity=gravity))[-1] for m in members]
            observed_end = np.asarray(trace['position_m'][end])
            nominal_bound = gate.check(row['q10'],row['q14'],row['bucket_xy_m'],row['values'],diagnostic,gravity=gravity)
            colliders = gate.colliders(row['q10'],row['q14'],row['bucket_xy_m'])
            extended = []
            for member in members:
                horizon = descending_time(float(p0[2]),float(member['velocity_world_m_s'][2]),radius,float(gravity[2]))
                if horizon is None or not 0 < horizon <= 1.5:
                    extended.append(dict(available=False,reason='no_bounded_table_tangent_crossing'));continue
                result = path_bound(p0,member['velocity_world_m_s'],horizon,gravity,colliders,radius)
                extended.append(dict(member=member['member'],available=True,**result))
            bound_known = len(extended)==5 and all(r['available'] for r in extended)
            origin_error = (p0-observed0)*1000
            first_relation = horizon_relation(clock['first_contact_delay_s'],horizons)
            robot_relation = horizon_relation(clock['first_robot_delay_s'],horizons)
            items.append(dict(record_id=row['id'],context_id=row['context_id'],method=row['method'],
                bucket_xy_m=row['bucket_xy_m'],values=row['values'],capture=bool(row['route_success']),
                clean=bool(row['route_success'] and row['route_quality']==2),native_cache_sha256=actual_sha,
                native_cache_row=col,**clock,predicted_rim_start_s=horizons,
                first_contact_horizon_relation=first_relation,robot_contact_horizon_relation=robot_relation,
                predicted_nominal_origin_m=p0.tolist(),observed_rebound_origin_m=observed0.tolist(),
                nominal_minus_observed_origin_mm=origin_error.tolist(),origin_error_norm_mm=float(np.linalg.norm(origin_error)),
                observed_rebound_velocity_m_s=observed_v.tolist(),
                predicted_member_velocities_m_s=[m['velocity_world_m_s'] for m in members],
                nominal_prediction_error_at_first_contact_mm=[float(np.linalg.norm(p-observed_end)*1000) for p in predicted_end],
                original_clearance=nominal_bound,
                hypothetical_table_horizon=dict(scope='Same nominal path/bound extended to center Z=ball radius; no table extent or contact response modeled',
                    passed=all(m['passed'] for m in extended) if bound_known else None,
                    minimum_clearance_lower_bound_m=min(m['minimum_clearance_lower_bound_m'] for m in extended) if bound_known else None,
                    members=extended)))
    evaluated=[r for r in items if r['method']=='arm_clearance']
    robot=[r for r in evaluated if r['first_robot_delay_s'] is not None]
    return dict(version=VERSION,status='diagnosed',scope=SCOPE,source=dict(
        report_sha256=verified['report_sha256'],audit_sha256=sha(audit),manifest_sha256=verified['manifest_sha256'],
        plan_sha256=verified['plan_sha256'],rule_sha256=verified['rule_sha256'],
        diagnosis_source_sha256=sha(__file__),independent_native_checks=verified['audit']['checks_count'],
        direct_trace_identity_checks=checks),summary=dict(new_selector_native_releases=len(evaluated),
        native_robot_contact_shots=len(robot),robot_events_after_all_predicted_horizons=sum(r['robot_contact_horizon_relation']['event_after_all_horizons'] for r in robot),
        hypothetical_extended_passes=sum(r['hypothetical_table_horizon']['passed'] is True for r in evaluated),
        hypothetical_extended_rejections=sum(r['hypothetical_table_horizon']['passed'] is False for r in evaluated),
        hypothetical_rejected_native_captures=sum(r['capture'] and r['hypothetical_table_horizon']['passed'] is False for r in evaluated)),
        cases=items)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('study','audit','assets','output'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.study.resolve()):
        raise ValueError('New derivative directory outside original evidence required')
    result=build(args.study,args.audit,args.assets)
    args.output.mkdir(parents=True);path=args.output/'horizon-diagnosis.json'
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (args.output/'checksums.json').write_text(json.dumps({path.name:sha(path)},indent=2)+'\n')
    print(json.dumps(result['summary']))

if __name__=='__main__':main()
