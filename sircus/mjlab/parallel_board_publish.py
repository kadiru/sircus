"""Publish a derived development view only after independent evidence audit."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
from .board_fallback_publish import audited_report,read


def publish(study,audit,display_report):
    study,audit,display_report=map(Path,(study,audit,display_report))
    paths=read(study/'contract.json')['source_paths']
    roots=[study.resolve(),Path(paths['study']).resolve(),Path(paths['parent_study']).resolve()]
    audits={audit.resolve(),Path(paths['audit']).resolve(),Path(paths['parent_audit']).resolve()}
    temporary=display_report.with_suffix(display_report.suffix+'.tmp')
    for destination in (display_report,temporary):
        resolved=destination.resolve()
        if resolved in audits or any(resolved.is_relative_to(root) for root in roots):
            raise ValueError('Derived display cannot overwrite a raw study or audit receipt')
    raw,provenance=audited_report(study,audit)
    if (raw.get('version'),raw.get('scope'),raw.get('physics_substeps'),raw.get('budgets')) != (
        'sircus-parallel-board-study-v1','development',64,dict(positions=6,repeats=3,planning_s=30.)):
        raise ValueError('Expected the six-target native64 development comparison')
    if not all(raw.get('controls',{}).get(k,{}).get('status')=='passed' for k in ('controls_pre','controls_post')):
        raise ValueError('Both control gates must pass')
    section=copy.deepcopy(raw);section.update(audit_status='passed',published_provenance=provenance)
    display=read(display_report) if display_report.exists() else {}
    display['parallel_board_study']=section
    temporary.write_text(json.dumps(display,indent=2,allow_nan=False)+'\n');os.replace(temporary,display_report)
    return dict(status=section['status'],audit_status=section['audit_status'],scope='development')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('study','audit','display-report'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();print(json.dumps(publish(args.study,args.audit,args.display_report)))


if __name__=='__main__':main()
