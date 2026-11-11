import copy,json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from sircus.mjlab import fixed35_broad_audit_v2 as V
from sircus.mjlab.benchmark_verify import EvidenceError,sha
from sircus.mjlab.replay650_verify import AuditChecks
from sircus.mjlab.tests.test_certificate_roundoff_verify import fixture


class WrapperTests(unittest.TestCase):
    def files(self,root):
        study=root/'original';study.mkdir()
        pipeline=study.with_suffix('.pipeline.json')
        pipeline.write_text(json.dumps(dict(status='audit_failed',study_exit_code=0,audit_exit_code=1,output=str(study))))
        failure=root/'original-audit.json';failure.write_text(json.dumps(dict(version=V.original.VERSION,status='failed',
            failures=['EvidenceError: fixed35-broad-p0090/v3 numeric exception restricted to near-zero normal angle'])))
        return study,pipeline,failure
    def test_full_audit_runs_with_explicit_adapter_and_restores_original(self):
        with tempfile.TemporaryDirectory() as temporary:
            study,pipeline,failure=self.files(Path(temporary));before=(sha(pipeline),sha(failure));old=V.original.compare_certificates
            def full(*args,**kwargs):
                V.original.compare_certificates(*fixture(),AuditChecks())
                return dict(version=V.original.VERSION,status='passed',report_sha256='report',evidence_manifest_sha256='manifest',
                    producer_source_sha256={'frozen':'source'},limitations=[])
            with patch.object(V.original,'audit',side_effect=full) as original:
                result=V.audit(study=study,original_failure=failure)
            self.assertEqual(original.call_count,1);self.assertIs(V.original.compare_certificates,old)
            self.assertEqual(before,(sha(pipeline),sha(failure)))
            self.assertEqual(result['version'],V.VERSION)
            correction=result['audit_correction']
            self.assertEqual(correction['original_audit_sha256'],sha(failure))
            self.assertEqual(correction['original_pipeline_sha256'],sha(pipeline))
            self.assertEqual(correction['complete_certificates_reproduced'],1)
            self.assertEqual(correction['differing_normal_angle_diagnostics'],1)
            self.assertFalse(correction['admission_changed']);self.assertFalse(correction['physics_changed']);self.assertFalse(correction['scoring_changed'])
    def test_adapter_restored_if_a_later_full_audit_check_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            study,_,failure=self.files(Path(temporary));old=V.original.compare_certificates
            with patch.object(V.original,'audit',side_effect=EvidenceError('later true failure')):
                with self.assertRaises(EvidenceError):V.audit(study=study,original_failure=failure)
            self.assertIs(V.original.compare_certificates,old)
    def test_wrong_failure_or_unfinished_original_supervisor_refuses(self):
        with tempfile.TemporaryDirectory() as temporary:
            study,pipeline,failure=self.files(Path(temporary))
            failure.write_text(json.dumps(dict(version=V.original.VERSION,status='failed',failures=['clearance failure'])))
            with self.assertRaises(EvidenceError):V.audit(study=study,original_failure=failure)
        with tempfile.TemporaryDirectory() as temporary:
            study,pipeline,failure=self.files(Path(temporary));pipeline.write_text(json.dumps(dict(status='running')))
            with self.assertRaises(EvidenceError):V.audit(study=study,original_failure=failure)

if __name__=='__main__':unittest.main()
