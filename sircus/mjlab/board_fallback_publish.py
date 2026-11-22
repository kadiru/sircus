"""Publish derived displays only, after binding independent audit receipts."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import os


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audited_report(study, audit_path):
    study, audit_path = Path(study), Path(audit_path)
    report, audit = read(study / 'report.json'), read(audit_path)
    if report.get('status') != 'completed' or audit.get('status') != 'passed':
        raise ValueError('Completed study and passed independent audit required')
    if audit.get('report_sha256') != sha(study / 'report.json'):
        raise ValueError('Audit does not bind the current report')
    if audit.get('evidence_manifest_sha256') != sha(study / 'evidence-manifest.json'):
        raise ValueError('Audit does not bind the evidence manifest')
    return report, dict(report_sha256=sha(study / 'report.json'), audit_sha256=sha(audit_path),
                        evidence_manifest_sha256=sha(study / 'evidence-manifest.json'))


def heading_summary(report, provenance):
    if report.get('physics_substeps') != 64 or len(report.get('cases', [])) != 1:
        raise ValueError('Expected one known aiming target at native64')
    case = report['cases'][0]
    def counts(key):
        row = case['results'][key]
        return dict(released=row['released'], captures=row['captures'],
                    arm_assisted=row['outcomes'].count('arm_assisted_capture'))
    return dict(status='completed', audit_status='passed', physics_substeps=64,
                bucket_xy_m=case['bucket_xy_m'], **{k:counts(k) for k in ('original','search','repeat')},
                confirmed_settings=len(case['confirmed_settings']), provenance=provenance)


def publish(study, audit, display_report, heading_study=None, heading_audit=None):
    display_report = Path(display_report)
    if bool(study) != bool(audit) or bool(heading_study) != bool(heading_audit):
        raise ValueError('Pair each study with its audit receipt')
    if not study and not heading_study:
        raise ValueError('At least one audited result required')
    evidence_roots = [Path(p).resolve() for p in (study, heading_study) if p]
    protected_audits = {Path(p).resolve() for p in (audit, heading_audit) if p}
    temporary = display_report.with_suffix(display_report.suffix+'.tmp')
    # Resolve both the final destination and its write-through temporary path:
    # an existing symlink must not make a derived display overwrite raw evidence.
    for destination in (display_report, temporary):
        resolved = destination.resolve()
        if resolved in protected_audits or any(resolved.is_relative_to(root) for root in evidence_roots):
            raise ValueError('Display must be separate from original evidence')
    display = read(display_report) if display_report.exists() else {}
    prior = display.get('board_fallback_study', {})
    if study:
        raw, provenance = audited_report(study,audit)
        if raw.get('version') != 'sircus-board-fallback-study-v1' or raw.get('physics_substeps') != 64:
            raise ValueError('Expected board fallback native64 comparison')
        section = copy.deepcopy(raw)
        section.update(audit_status='passed', published_provenance=provenance)
        if prior.get('heading_diagnostic'):
            section['heading_diagnostic'] = copy.deepcopy(prior['heading_diagnostic'])
    else:
        section = copy.deepcopy(prior) if prior else dict(version='sircus-board-fallback-study-v1',
            status='running',phase='preparing',audit_status='pending',physics_substeps=64,
            budgets=dict(positions=50,planning_s=30),progress=dict(prepared=0,planned=100,completed_releases=0,planned_releases=None))
    if heading_study:
        raw, provenance = audited_report(heading_study,heading_audit)
        section['heading_diagnostic'] = heading_summary(raw,provenance)
    display['board_fallback_study'] = section
    temporary.write_text(json.dumps(display,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,display_report)
    return dict(status=section['status'],audit_status=section['audit_status'],heading=bool(section.get('heading_diagnostic')))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','audit','heading-study','heading-audit'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--display-report',required=True,type=Path)
    a=p.parse_args()
    print(json.dumps(publish(a.study,a.audit,a.display_report,a.heading_study,a.heading_audit)))


if __name__ == '__main__': main()
