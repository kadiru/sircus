"""Publish independently audited nearby-shot development results only.

The raw study, successful source recipes and live planner remain unchanged.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

from .benchmark_verify import read, sha
from .board_fallback_publish import audited_report

VERSION = 'sircus-nearby-transfer-study-v1'
CASES = tuple('parallel-board-miss-' + i for i in ('03', '09', '13', '15', '19'))


def protect_destination(study, audit, destination, paths):
    """Protect source roots and receipts, including a write-through .tmp link."""
    roots = [Path(study).resolve()]
    receipts = {Path(audit).resolve()}
    for key, value in paths.items():
        if value is None:
            continue
        if key.endswith('study'):
            roots.append(Path(value).resolve())
        elif key.endswith('audit'):
            receipts.add(Path(value).resolve())
    for target in (destination, destination.with_suffix(destination.suffix + '.tmp')):
        resolved = target.resolve()
        if resolved in receipts or any(resolved.is_relative_to(root) for root in roots):
            raise ValueError('Derived display cannot overwrite a source study or audit receipt')


def compose(study, audit_path):
    study, audit_path = Path(study), Path(audit_path)
    raw, provenance = audited_report(study, audit_path)
    plan, audit, spec = (read(study/'preparation.json'), read(audit_path), read(study/'contract.json'))
    if (raw.get('version'), raw.get('phase'), raw.get('physics_substeps')) != (VERSION, 'completed', 64):
        raise ValueError('Completed native64 nearby-transfer development study required')
    if [r['id'] for r in raw.get('cases', [])] != list(CASES):
        raise ValueError('All five diagnosed targets must remain visible')
    for phase in ('controls_pre', 'controls_post'):
        if raw.get('controls', {}).get(phase, {}).get('status') != 'passed':
            raise ValueError('Both control gates must pass')
    for name, field in (('preparation.json', 'plan_sha256'), ('source-bank.json', 'source_bank_sha256'),
                        ('contract.json', 'contract_sha256')):
        if audit.get(field) != sha(study/name):
            raise ValueError('Audit does not bind ' + name)
        provenance[field] = sha(study/name)
    if (raw.get('native_releases') != raw.get('initial_requests', -100) + raw.get('confirmation_requests', -100) + 18
        or not 33 <= raw['native_releases'] <= 433
        or spec.get('maximum_total_requests', 1000) > 433):
        raise ValueError('Native release accounting differs from the bounded protocol')
    section = copy.deepcopy(raw)
    section.update(audit_status='passed', scope='development', published_provenance=provenance)
    # The audited report already includes a compact, path-free source summary.
    # Never embed the raw bank's proof files or full source requests in the UI.
    section['source_bank_summary'] = copy.deepcopy(raw['source_bank_summary'])
    for case, prepared in zip(section['cases'], plan['cases'], strict=True):
        for group in case['results'].values():
            group.pop('rows', None)
    section['next'] = ('Review the repeated settings before changing the live planner. '
        'These are five known development misses, not a new generalization score; no model was retrained.')
    return section


def publish(study, audit, display_report):
    study, audit, display_report = map(Path, (study, audit, display_report))
    protect_destination(study, audit, display_report, read(study/'contract.json')['source_paths'])
    section = compose(study, audit)
    display = read(display_report) if display_report.exists() else {}
    display['nearby_shot_transfer'] = section
    temporary = display_report.with_suffix(display_report.suffix + '.tmp')
    temporary.write_text(json.dumps(display, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, display_report)
    return dict(status=section['status'], audit_status='passed', cases=len(section['cases']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'audit', 'display-report'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.study, args.audit, args.display_report)))


if __name__ == '__main__':
    main()
