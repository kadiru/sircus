"""Publish the guarded comparison without modifying its frozen evidence."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

from .board_fallback_publish import audited_report, read


def publish(study, audit, display_report):
    study, audit, display_report = map(Path, (study, audit, display_report))
    temporary = display_report.with_suffix(display_report.suffix + '.tmp')
    for destination in (display_report, temporary):
        resolved = destination.resolve()
        if resolved.is_relative_to(study.resolve()) or resolved == audit.resolve():
            raise ValueError('Display must be separate from original evidence')
    raw, provenance = audited_report(study, audit)
    if (raw.get('version') != 'sircus-guarded-board-fallback-study-v1'
            or raw.get('physics_substeps') != 64
            or raw.get('budgets', {}).get('positions') != 20):
        raise ValueError('Expected twenty-position guarded native64 comparison')
    if not all(raw.get('controls', {}).get(name, {}).get('status') == 'passed'
               for name in ('controls_pre', 'controls_post')):
        raise ValueError('Both simulator control blocks must pass')
    section = copy.deepcopy(raw)
    section.update(audit_status='passed', published_provenance=provenance)
    display = read(display_report) if display_report.exists() else {}
    display['guarded_board_fallback_study'] = section
    temporary.write_text(json.dumps(display, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, display_report)
    return dict(status=section['status'], audit_status=section['audit_status'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'audit', 'display-report'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.study, args.audit, args.display_report)))


if __name__ == '__main__':
    main()
