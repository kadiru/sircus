"""Audit-only historical metadata recovery never edits evidence or trusts new values."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from sircus.mjlab import fixed35_bouncer_audit_v2 as V
from sircus.mjlab.benchmark_verify import Checks,EvidenceError


def fixture(root):
    branch=dict(ik_branch_slot=0,branch_key='k',branch_pool_file='branch-pools/p.json',branch_pool_sha256='p',branch_pool_key='pk',branch_rule_version='version')
    path=root/'native-yaw/fixed-yaw-connections/k.json';path.parent.mkdir(parents=True);path.write_text(json.dumps(branch))
    source=dict(id='parent-fixed35-dev-grid-01-05',values=[1,2,3,4,5],bucket_xy_m=[1,2],q10=[0]*6,q14=[1]*6,
        connection_file='fixed-yaw-connections/k.json',connection_sha256=V.sha(path),**branch)
    row={k:copy.deepcopy(source[k]) for k in ('values','bucket_xy_m','q10','q14','connection_file','connection_sha256')}
    row.update(id='preserved',phase='preserved',adapter_directory='native-yaw',parent_row_id=source['id'],parent_row_sha256=V.digest(source),exact_parent_action=True)
    return [row],dict(preserved=[source]),path


class CorrectionTests(unittest.TestCase):
    def test_enrichment_is_a_copy_from_authenticated_source_and_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows,data,path=fixture(root);before=copy.deepcopy(rows);raw=path.read_bytes()
            copied,receipts=V.enrich_preserved_branch_metadata(root,rows,data,Checks())
            self.assertEqual(rows,before);self.assertEqual(path.read_bytes(),raw)
            self.assertEqual(copied[0]['ik_branch_slot'],0);self.assertEqual(set(receipts[0]['supplied_fields']),set(V.FIELDS))
            self.assertNotEqual(receipts[0]['original_record_sha256'],receipts[0]['audit_only_record_sha256'])

    def test_conflicting_metadata_or_changed_joints_source_or_proof_is_rejected(self):
        for mutation in ('slot','q14','parent','proof','source_branch','not_exact'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);rows,data,path=fixture(root)
                if mutation=='slot':rows[0]['ik_branch_slot']=1
                if mutation=='q14':rows[0]['q14'][0]=2
                if mutation=='parent':rows[0]['parent_row_sha256']='bad'
                if mutation=='proof':path.write_text('{}')
                if mutation=='source_branch':data['preserved'][0]['ik_branch_slot']=1;rows[0]['parent_row_sha256']=V.digest(data['preserved'][0])
                if mutation=='not_exact':rows[0]['exact_parent_action']=False
                with self.assertRaises(EvidenceError):V.enrich_preserved_branch_metadata(root,rows,data,Checks())

    def test_wrapper_requires_retained_original_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'failure.json';path.write_text(json.dumps(dict(version=V.original.VERSION,status='passed',failures=[])))
            with self.assertRaises(EvidenceError):V.audit(original_failure=path)

    def test_wrapper_discloses_interposition_and_restores_original_function(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows,data,path=fixture(root);failure=root/'failed.json'
            failure.write_text(json.dumps(dict(version=V.original.VERSION,status='failed',failures=["KeyError: 'ik_branch_slot'"])))
            old=V.original.audit_geometry;observed=[]
            def fake_audit(*args,**kwargs):
                V.original.audit_geometry(root,rows,{},root,data,Checks())
                return dict(auditor_sha256='frozen-v1-hash',limitations=[],status='passed')
            def strict(study,copyrows,*args):observed.extend(copyrows);return {}
            with patch.object(V.original,'audit',side_effect=fake_audit),patch.object(V,'ORIGINAL_GEOMETRY_AUDIT',side_effect=strict):
                result=V.audit(original_failure=failure)
            self.assertIs(V.original.audit_geometry,old);self.assertNotIn('ik_branch_slot',rows[0]);self.assertEqual(observed[0]['ik_branch_slot'],0)
            self.assertFalse(result['audit_correction']['physics_rerun']);self.assertFalse(result['audit_correction']['admission_checks_relaxed'])
            self.assertEqual(result['audit_correction']['original_auditor_sha256'],'frozen-v1-hash')


if __name__=='__main__':unittest.main()
