"""Publish audited one-target clear-shot range results without changing the planner."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

from .benchmark_verify import read, sha
from .board_fallback_publish import audited_report
from .nearby_transfer_publish import protect_destination

VERSION = 'sircus-clear-shot-range-v1'
AUDIT_VERSION = 'sircus-clear-shot-range-independent-audit-v1'
CASES = ('parallel-board-miss-09',)
BINDINGS = {'plan_sha256': 'preparation.json', 'source_bank_sha256': 'source-bank.json',
            'contract_sha256': 'contract.json'}
REQUEST_FIELDS = ('id', 'context_id', 'bucket_xy_m', 'values', 'selected_q10', 'selected_q14',
                  'board_heading_offset_deg', 'board_twist_deg')


def compose(study, audit_path):
    study, audit_path = Path(study), Path(audit_path)
    raw, provenance = audited_report(study, audit_path)
    audit, contract = read(audit_path), read(study/'contract.json')
    if audit.get('version') != AUDIT_VERSION:
        raise ValueError('Matching independent clear-shot-range audit required')
    if (raw.get('version'), raw.get('phase'), raw.get('scope'), raw.get('physics_substeps')) != (VERSION, 'completed', 'development', 64):
        raise ValueError('Completed native64 one-target development study required')
    if [row['id'] for row in raw.get('cases', [])] != list(CASES):
        raise ValueError('The single diagnosed target must be visible')
    for field, filename in BINDINGS.items():
        if audit.get(field) != sha(study/filename):
            raise ValueError('Audit does not bind ' + filename)
        provenance[field] = sha(study/filename)
    for phase in ('controls_pre', 'controls_post'):
        control = raw.get('controls', {}).get(phase, {})
        if (control.get('status'), control.get('direct_captures'), control.get('table_assisted_captures'), control.get('negative_captures')) != ('passed', 3, 3, 0):
            raise ValueError('Both full control blocks must pass')
    preserved = raw.get('preserved_shots', {})
    if (preserved.get('requested'), preserved.get('completed'), preserved.get('captures'), preserved.get('passed')) != (9, 9, 9, True):
        raise ValueError('All nine preservation drops must capture')
    if (raw.get('native_releases') != raw.get('initial_requests', -100) + raw.get('confirmation_requests', -100) + 27
        or not 30 <= raw['native_releases'] <= 226
        or not 30 <= contract.get('maximum_total_requests', 1000) <= 226):
        raise ValueError('Native release accounting differs from the bounded protocol')
    case = raw['cases'][0]
    grid = case.get('grid', [])
    if len(grid) != 49 or len({row['proposal_id'] for row in grid}) != 49:
        raise ValueError('All 49 planned configurations must remain visible')
    source = raw.get('source_summary', {})
    if source.get('id') != 'parallel-board-miss-09-clearance-0016-b0' or source.get('prior_no_robot_contact') is not True:
        raise ValueError('Expected the selected prior arm-free source shot')
    if source.get('bucket_xy_m') != case.get('bucket_xy_m'):
        raise ValueError('Source and target bucket differ')
    section = copy.deepcopy(raw)
    section.update(audit_status='passed', published_provenance=provenance)
    # Publish only compact counts and requested geometry. Raw records and proof
    # paths remain in the immutable evidence archive, outside the browser.
    section['preserved_shots'].pop('rows', None)
    for case in section['cases']:
        case.pop('best_candidate', None)
        for result in case['results'].values():
            result.pop('rows', None)
        for group in ('repeated_settings', 'confirmed_settings'):
            for item in case.get(group, []):
                item['request'] = {key: item['request'][key] for key in REQUEST_FIELDS}
    section['next'] = ('Review the observed range and confirmed settings before changing the live planner. '
        'Earlier successful shots are preserved. The other unresolved bucket was not tested in this pass.')
    return section


def publish(study, audit, display_report):
    study, audit, display_report = map(Path, (study, audit, display_report))
    protected_paths = dict(read(study/'contract.json')['source_paths'])
    if protected_paths.get('diagnosis'):
        protected_paths['diagnosis_audit'] = protected_paths['diagnosis']
    protect_destination(study, audit, display_report, protected_paths)
    section = compose(study, audit)
    display = read(display_report) if display_report.exists() else {}
    display['clear_shot_range'] = section
    temporary = display_report.with_suffix(display_report.suffix + '.tmp')
    temporary.write_text(json.dumps(display, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, display_report)
    return dict(status='completed', audit_status='passed', cases=len(section['cases']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('study', 'audit', 'display-report'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.study, args.audit, args.display_report)))


if __name__ == '__main__':
    main()
