"""Explicit audit-only correction for omitted historical branch display metadata.

The producer and its immutable native evidence are unchanged. The v1 audit
failed because one preserved six-study recipe retained its exact connection
proof but its emitted row omitted historical branch-slot metadata. This adapter
recovers only those fields from the authenticated original source and identical
connection. It interposes on geometry review of an in-memory copy, then runs all
original strict branch, endpoint, FK, sampled-path and scoring checks.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
from unittest.mock import patch
from . import fixed35_bouncer_verify as original
from .benchmark_verify import EvidenceError,contained,digest,read,sha

VERSION='sircus-fixed35-bouncer-audit-v2'
FIELDS=('ik_branch_slot','branch_key','branch_pool_file','branch_pool_sha256','branch_pool_key','branch_rule_version')
ORIGINAL_GEOMETRY_AUDIT=original.audit_geometry


def enrich_preserved_branch_metadata(study,rows,data,checks):
    """Return an audit-only copy; never alter rows or infer a different branch."""
    copied=copy.deepcopy(rows);sources={r['id']:r for r in data['preserved']};receipts=[]
    for row in copied:
        if row['phase']!='preserved' or row['adapter_directory']!='native-yaw':continue
        path=contained(Path(study)/'native-yaw',row['connection_file'])
        checks.equal(sha(path),row['connection_sha256'],row['id']+'/v2 exact preserved proof bytes')
        connection=read(path)
        if connection.get('branch_rule_version') is None:continue
        source=sources.get(row['parent_row_id'])
        checks.require(source is not None,row['id']+'/v2 authenticated preserved parent exists')
        checks.equal(row['parent_row_sha256'],digest(source),row['id']+'/v2 original row identity')
        checks.equal(row['exact_parent_action'],True,row['id']+'/v2 exact preservation request')
        for field in ('values','bucket_xy_m','q10','q14','connection_file','connection_sha256'):
            checks.equal(row[field],source[field],row['id']+'/v2 exact historical field/'+field)
        supplied={}
        for field in FIELDS:
            checks.equal(connection[field],source[field],row['id']+'/v2 historical source agrees with exact proof/'+field)
            if field in row:checks.equal(row[field],source[field],row['id']+'/v2 never replace conflicting metadata/'+field)
            else:supplied[field]=copy.deepcopy(source[field]);row[field]=copy.deepcopy(source[field])
        receipts.append(dict(record_id=row['id'],parent_row_id=source['id'],parent_row_sha256=digest(source),
            connection_file=row['connection_file'],connection_sha256=row['connection_sha256'],supplied_fields=supplied,
            original_record_sha256=digest(next(r for r in rows if r['id']==row['id'])),
            audit_only_record_sha256=digest(row),original_record_mutated=False))
    return copied,receipts


def audit(*args,original_failure,**kwargs):
    failure_path=Path(original_failure)
    failure=read(failure_path)
    if (failure.get('version')!=original.VERSION or failure.get('status')!='failed'
            or not any('ik_branch_slot' in str(f) for f in failure.get('failures',[]))):
        raise EvidenceError('The separately retained v1 missing-branch-metadata failure is required')
    receipts=[]
    def corrected_geometry(study,rows,scene,assets,data,checks):
        copied,derived=enrich_preserved_branch_metadata(study,rows,data,checks)
        checks.equal(len(derived),1,'v2 correction applies only to the one preserved six-study recipe')
        checks.require(derived[0]['parent_row_id'].endswith('fixed35-dev-grid-01-05'),'v2 correction has the exact known historical context')
        receipts.extend(derived)
        return ORIGINAL_GEOMETRY_AUDIT(study,copied,scene,assets,data,checks)
    with patch.object(original,'audit_geometry',corrected_geometry):
        result=original.audit(*args,**kwargs)
    original_auditor_sha=result['auditor_sha256']
    result.update(version=VERSION,auditor_sha256=sha(Path(__file__)),
        audit_correction=dict(version=VERSION,kind='Explicit audit-only metadata adapter; original v1 geometry audit interposed',
            original_audit_failure_sha256=sha(failure_path),original_audit_failure=failure,
            original_auditor_version=original.VERSION,original_auditor_sha256=original_auditor_sha,
            adapter_sha256=sha(Path(__file__)),derived_metadata=receipts,
            producer_source_unchanged=True,native_rows_unchanged=True,report_unchanged=True,
            physics_rerun=False,scoring_changed=False,admission_checks_relaxed=False))
    result['limitations'].append('The original v1 audit failed on omitted historical branch metadata; v2 supplies authenticated fields only in its audit copy and retains the original failure receipt.')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('study','source','assets','parent-study','parent-audit','original-study','original-audit','recovery-study','recovery-audit','reference','original-failure','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.study.resolve()):
        raise ValueError('Fresh independent proof outside immutable experiment required')
    try:
        kwargs=vars(args).copy();kwargs.pop('output');result=audit(**kwargs)
    except Exception as error:
        args.output.write_text(json.dumps(dict(version=VERSION,status='failed',auditor_sha256=sha(Path(__file__)),
            original_audit_failure_sha256=sha(args.original_failure),failures=[type(error).__name__+': '+str(error)]),indent=2,allow_nan=False)+'\n')
        raise
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','checks_count','requested_trials','native_releases','demonstrated_recipe_coverage')}))


if __name__=='__main__':main()
