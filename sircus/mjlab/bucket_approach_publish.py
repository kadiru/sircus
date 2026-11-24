"""Publish a bound bucket-approach development test without changing planners."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from .benchmark_verify import contained, read, sha
from .nearby_transfer_publish import protect_destination

VERSION = 'sircus-bucket-approach-display-v1'
STUDY_VERSION = 'sircus-bucket-approach-study-v1'
AUDIT_VERSION = 'sircus-bucket-approach-independent-audit-v1'
CASES = tuple('expanded-selector-test-' + n for n in ('04', '10', '19'))
METHODS = ('original', 'corrected')
LIMITS = ('The check follows the nominal predicted flight through one board rebound and, when predicted, '
          'one table rebound until the descending whole-ball rim passage. It does not simulate bucket, '
          'robot or second-board contact response or retention. A warning lowers selection rank; it '
          'does not forbid a later contact-assisted capture. Several settings can change together.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def vector(value, length):
    return isinstance(value, list) and len(value) == length and all(finite(n) for n in value)


def forecast_view(approach):
    if approach is None:
        return dict(summary='No checked selection', collision=None, first_collision_s=None,
                    minimum_margin_m=None, opening_margin_m=None, limits=LIMITS)
    require(approach.get('version') == 'sircus-full-bucket-approach-v1', 'Expected versioned whole-approach forecast')
    route = approach.get('predicted_route')
    result = approach.get('routes', {}).get(route, {})
    contact = result.get('first_predicted_contact')
    margin = result.get('minimum_clearance_lower_bound_m')
    opening = result.get('original_rim_margin_m')
    for value in (margin, opening, None if not contact else contact.get('time_s')):
        require(value is None or finite(value), 'Finite forecast measurements required')
    status = result.get('status', 'unsupported_horizon')
    require(status in ('clear', 'sampled_overlap', 'between_samples_unproven', 'unsupported_horizon'), 'Unknown forecast status')
    if contact:
        require(vector(contact.get('center_m'), 3) and contact['time_s'] >= 0, 'Finite predicted contact required')
        direction = 'rising' if contact.get('rising') else 'descending'
        summary = f"{route}: predicted {contact.get('geom', 'bucket')} contact while {direction}"
    else:
        summary = f'{route or "No route"}: ' + {
            'clear':'nominal approach clears the bucket',
            'between_samples_unproven':'clearance not established between sampled points',
            'unsupported_horizon':'forecast horizon unsupported',
            'sampled_overlap':'predicted contact',
        }[status]
    return dict(summary=summary, collision=bool(contact) if result else None,
                first_collision_s=contact['time_s'] if contact else None,
                minimum_margin_m=margin, opening_margin_m=opening, limits=LIMITS)


def settings_view(request, native):
    if request is None:
        require(not native, 'Declined setup cannot have native drops')
        return None
    require(vector(request.get('values'), 5) and request['values'][4] > 0, 'Finite positive-gap command required')
    require(vector(request.get('selected_q10'), 6) and vector(request.get('selected_q14'), 6), 'Checked joint poses required')
    require(native, 'Selected shot must have its matched native repeats')
    first = native[0]
    for row in native:
        for key in ('values', 'bucket_xy_m', 'selected_q10', 'selected_q14', 'board_heading_offset_deg', 'board_twist_deg'):
            require(row.get(key) == request.get(key), 'Native repeat differs from frozen setting: '+key)
        for key in ('release_ball_center_m', 'native_initial_q10', 'native_initial_q14'):
            require(row.get(key) == first.get(key), 'Matched repeats must use the same held geometry')
    require(vector(first.get('release_ball_center_m'), 3)
            and vector(first.get('native_initial_q10'), 6) and vector(first.get('native_initial_q14'), 6), 'Measured release and held arm poses required')
    return dict(values=request['values'], q10_rad=first['native_initial_q10'], q14_rad=first['native_initial_q14'],
                release_ball_center_m=first['release_ball_center_m'],
                heading_offset_deg=request['board_heading_offset_deg'], twist_deg=request['board_twist_deg'])



def selection_view(prepared):
    without, selected = prepared['selected_without_bucket_check'], prepared.get('corrected')
    fields = ('values', 'selected_q10', 'selected_q14', 'board_heading_offset_deg', 'board_twist_deg')
    require(isinstance(without.get('id'), str) and vector(without.get('values'), 5),
            'Same-menu uncorrected ranking must be recorded')
    changed = selected is None or any(without.get(key) != selected.get(key) for key in fields)
    return dict(changed=changed, without_check_id=without['id'], with_check_id=selected['id'] if selected else None,
                without_check_values=without['values'], with_check_values=selected['values'] if selected else None,
                checked_proposals=len(prepared.get('checked', [])), admitted_candidates=len(prepared.get('candidates', [])),
                statement=('The full-path check chose a different setting from the same admitted candidates.' if changed and selected
                           else 'The full-path check withheld a replacement; no improved setting was selected.' if selected is None
                           else 'Both rankings chose the same setting from the admitted candidates; this selection did not depend on the new check.'))


def count_view(result):
    require(result.get('requested') == 3 and type(result.get('completed')) is int, 'Three matched slots required')
    names = ('released', 'captures', 'setup_rejected', 'invalid_physics', 'robot_contacts', 'bucket_contacts')
    require(all(type(result.get(k)) is int and result[k] >= 0 for k in names), 'Nonnegative measured counts required')
    require(result['completed'] == result['released'] and result['completed'] + result['setup_rejected'] == 3,
            'Three completed drops or declined slots required')
    misses = result['released'] - result['captures'] - result['invalid_physics']
    require(misses >= 0 and result['released'] + result['setup_rejected'] == 3, 'Matched attempt accounting failed')
    require(result['robot_contacts'] <= result['released'] and result['bucket_contacts'] <= result['released'], 'Contact counts exceed releases')
    return dict(requested=3, completed=3, released=result['released'], captures=result['captures'],
                setup_rejections=result['setup_rejected'], invalid_physics=result['invalid_physics'],
                flight_misses=misses, robot_contacts=result['robot_contacts'], bucket_contacts=result['bucket_contacts'])


def compose(study, audit_path):
    study, audit_path = Path(study), Path(audit_path)
    report, audit = read(study/'report.json'), read(audit_path)
    require((report.get('version'), report.get('status'), report.get('phase'), report.get('physics_substeps'),
             audit.get('version'), audit.get('status')) ==
            (STUDY_VERSION, 'completed', 'completed', 64, AUDIT_VERSION, 'passed'), 'Completed native64 study and matching independent audit required')
    require(type(audit.get('checks_count')) is int and audit['checks_count'] > 0
            and audit.get('frozen_before_native') is True, 'Independently audited frozen selection required')
    bindings = {'report_sha256':'report.json', 'evidence_manifest_sha256':'evidence-manifest.json',
                'contract_sha256':'contract.json', 'preparation_sha256':'preparation.json',
                'choices_sha256':'slots.json', 'native_started_sha256':'native-started.json'}
    for key, filename in bindings.items():
        require(audit.get(key) == sha(study/filename), 'Independent audit no longer binds '+filename)
    manifest = read(study/'evidence-manifest.json')
    for filename, expected in manifest.items():
        require(sha(contained(study, filename)) == expected, 'Evidence changed: '+filename)
    plan = read(study/'preparation.json')
    require(all(audit.get('comparison', {}).get(k) == report.get(k) for k in ('cases','preservation','diagnostic_controls')),
            'Independent summary differs from published counts')
    require([c.get('id') for c in report.get('cases', [])] == list(CASES)
            and [c.get('id') for c in plan.get('cases', [])] == list(CASES), 'All three diagnosed positions must remain visible')
    native_rows = [json.loads(line) for line in (study/'rows.jsonl').read_text().splitlines() if line.strip()]
    lookup = {r['id']:r for r in native_rows}
    require(len(lookup) == len(native_rows), 'Native identities must be unique')
    def native(result):
        identities = [r['id'] if isinstance(r, dict) else r for r in result['rows']]
        require(len(set(identities)) == len(identities), 'Duplicate repeat rows')
        rows = [lookup[i] for i in identities]
        require(len(rows) == result['released'], 'Released count differs from bound native rows')
        require(all(r.get('native_release') is True and r.get('board_first_scalar_audit_passed') is True
                    and r.get('board_first_score', {}).get('scoring_version') == 'sircus-board-first-native-capture-v4' for r in rows), 'Native v4 scalar verification required')
        require(all(type(r['board_first_score'].get(k)) is bool for r in rows for k in ('route_success','invalid_physics')),
                'Boolean capture and validity required')
        require(not any(r['board_first_score']['route_success'] and r['board_first_score']['invalid_physics'] for r in rows),
                'Invalid physics cannot capture')
        require(sum(r['board_first_score']['route_success'] for r in rows) == result['captures'], 'Capture counts differ from native evidence')
        require(sum(r['board_first_score']['invalid_physics'] for r in rows) == result['invalid_physics'], 'Invalid physics counts differ')
        return rows
    cases = []
    for r, p in zip(report['cases'], plan['cases'], strict=True):
        require(r['bucket_xy_m'] == p['original']['bucket_xy_m'], 'Wrong diagnostic bucket')
        methods = {}
        for method in METHODS:
            result, request = r['results'][method], p.get(method)
            approach = p.get('original_approach') if method == 'original' else p.get('approaches', {}).get(request['id']) if request else None
            require(request is None or approach is not None, 'Chosen shot needs its frozen approach prediction')
            methods[method] = dict(**count_view(result), settings=settings_view(request, native(result)), forecast=forecast_view(approach))
        cases.append(dict(id=r['id'], bucket_xy_m=r['bucket_xy_m'], original_method='current' if r['id'].endswith('10') else 'expanded', methods=methods, selection_comparison=selection_view(p)))
    preserved = report.get('preservation', [])
    prepared_preserved = {p['id']:p for p in plan.get('preserved_cases', [])}
    require(len(preserved) == len(prepared_preserved) == 4, 'Four original successful shots required')
    preservation = []
    for result in preserved:
        p = prepared_preserved[result['id']]
        require(result['bucket_xy_m'] == p['original']['bucket_xy_m'], 'Wrong preservation bucket')
        preservation.append(dict(id=result['id'], bucket_xy_m=result['bucket_xy_m'], **count_view(result),
            settings=settings_view(p['original'], native(result)), forecast=forecast_view(p['original_approach'])))
    diagnostic = []
    prepared_diagnostic = {p['id']:p for p in plan.get('diagnostic_cases', [])}
    require(len(prepared_diagnostic) == len(report.get('diagnostic_controls', [])) == 1,
            'The fourth unchanged robot-contact miss must remain visible')
    for result in report['diagnostic_controls']:
        p = prepared_diagnostic[result['id']]
        require(result['bucket_xy_m'] == p['original']['bucket_xy_m'], 'Wrong diagnostic control bucket')
        diagnostic.append(dict(id=result['id'], bucket_xy_m=result['bucket_xy_m'], **count_view(result),
            settings=settings_view(p['original'], native(result)), forecast=forecast_view(p['original_approach'])))
    controls = {}
    for phase in ('controls_pre', 'controls_post'):
        c = report.get('controls', {}).get(phase, {})
        require(c.get('status') == 'passed' and c.get('board_first_passed') is True, 'Both native scoring control gates must pass')
        controls[phase] = dict(status='passed', requested=9)
    native_releases = 18 + sum(p['released'] for p in preservation+diagnostic) + sum(c['methods'][m]['released'] for c in cases for m in METHODS)
    require(report.get('native_releases') == native_releases == len(native_rows), 'Native release accounting differs')
    old = sum(c['methods']['original']['captures'] for c in cases)
    new = sum(c['methods']['corrected']['captures'] for c in cases)
    preserved_count = sum(p['captures'] for p in preservation)
    conclusion = (f'Known misses: unchanged {old}/9, approach-checked {new}/9. Unchanged successful settings: {preserved_count}/12. '
                  'Candidate search and ranking both changed. This does not isolate the effect of the new check or establish a new table-wide capture rate. The live planner remains unchanged.')
    return dict(version=VERSION, status='completed', audit_status='passed', scope='development', physics_substeps=64,
        frozen_before_native=True, cases=cases, preservation=preservation, diagnostic_controls=diagnostic, controls=controls,
        native_releases=native_releases, checks_count=audit['checks_count'], conclusion=conclusion, limits=LIMITS,
        provenance={**{key:sha(study/filename) for key, filename in bindings.items()}, 'audit_sha256':sha(audit_path)})


def publish(study, audit, display_report):
    study, audit, display_report = map(Path, (study, audit, display_report))
    protect_destination(study, audit, display_report, read(study/'contract.json').get('source_paths', {}))
    section = compose(study, audit)
    display = read(display_report) if display_report.exists() else {}
    display['bucket_approach_study'] = section
    temporary = display_report.with_suffix(display_report.suffix+'.tmp')
    temporary.write_text(json.dumps(display, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, display_report)
    return dict(status='completed', audit_status='passed', cases=len(section['cases']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'audit', 'display-report'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(publish(a.study, a.audit, a.display_report)))


if __name__ == '__main__':
    main()
