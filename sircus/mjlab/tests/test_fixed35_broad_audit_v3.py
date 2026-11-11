import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from sircus.mjlab import fixed35_broad_audit_v3 as V
from sircus.mjlab.benchmark_verify import EvidenceError,read,sha
from sircus.mjlab.replay650_verify import AuditChecks
from sircus.mjlab.tests.test_fixed35_broad_audit_v2 import WrapperTests
from sircus.mjlab.tests.test_certificate_roundoff_verify import fixture as certificates
from sircus.mjlab.tests.test_packet_timing_verify import fixture as packets


class WrapperV3Tests(unittest.TestCase):
    def fixture(self,root):
        study,pipeline,original=WrapperTests().files(root)
        (study/'contract.json').write_text(json.dumps(dict(protocol=dict(worker_limit_s=43200,search_limit_s=36000,audit_reserve_s=7200))))
        (study/'report.json').write_text(json.dumps(dict(elapsed_s=29191.)))
        previous=root/'original-audit-v2.json'
        previous.write_text(json.dumps(dict(version=V.prior.VERSION,status='failed',auditor_sha256=sha(Path(V.prior.__file__)),
            original_audit_failure_sha256=sha(original),failures=['EvidenceError: Monotone native batch timings within wall budget'])))
        rows,_=packets(study)
        return study,pipeline,original,previous,rows
    def test_full_nested_audit_adapters_restore_and_bind_both_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            study,pipeline,original,previous,rows=self.fixture(Path(tmp))
            olds=V.original.compare_certificates,V.original.geometry.audit_request_packets
            hashes=[sha(p) for p in (pipeline,original,previous,study/'report.json')]
            def full(*a,**kw):
                c=AuditChecks();V.original.compare_certificates(*certificates(),c)
                result=V.original.geometry.audit_request_packets(study,rows,c)
                return dict(version=V.original.VERSION,status='passed',report_sha256=sha(study/'report.json'),
                    evidence_manifest_sha256='manifest',producer_source_sha256={'frozen':'source'},limitations=[],request_packets=result)
            with patch.object(V.original,'audit',side_effect=full),patch.object(V.original,'audit_budget',return_value=dict(
                    worker_receipt_sha256='budget',effective_worker_limit_s=43200.,global_limit_s=51300.)) as budget:
                result=V.audit(study=study,original_failure=original,previous_failure=previous)
            self.assertEqual(budget.call_count,1);self.assertEqual(result['version'],V.VERSION)
            self.assertEqual(result['request_packets']['last_native_end_s'],29000.)
            self.assertEqual(result['audit_correction']['original_audit_sha256'],sha(original))
            correction=result['timing_audit_correction']
            self.assertEqual(correction['previous_audit_sha256'],sha(previous));self.assertEqual(correction['previous_audit_failure'],read(previous))
            self.assertEqual(correction['original_budget']['effective_worker_limit_s'],43200.)
            self.assertEqual(hashes,[sha(p) for p in (pipeline,original,previous,study/'report.json')])
            self.assertIs(V.original.compare_certificates,olds[0]);self.assertIs(V.original.geometry.audit_request_packets,olds[1])
    def test_adapter_restores_after_real_later_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            study,_,original,previous,_=self.fixture(Path(tmp));old=V.original.geometry.audit_request_packets
            with patch.object(V.original,'audit',side_effect=EvidenceError('actual scoring failure')):
                with self.assertRaises(EvidenceError):V.audit(study=study,original_failure=original,previous_failure=previous)
            self.assertIs(old,V.original.geometry.audit_request_packets)
    def test_unrelated_failure_or_source_or_original_binding_rejected(self):
        for change in ('reason','source','original'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                study,_,original,previous,_=self.fixture(Path(tmp));v=read(previous)
                if change=='reason':v['failures']=['native failure']
                if change=='source':v['auditor_sha256']='wrong'
                if change=='original':v['original_audit_failure_sha256']='wrong'
                previous.write_text(json.dumps(v))
                with self.assertRaises(EvidenceError):V.audit(study=study,original_failure=original,previous_failure=previous)
    def test_wrong_study_or_unapproved_contract_cannot_receive_larger_cap(self):
        for change in ('study','contract'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                study,_,original,previous,rows=self.fixture(Path(tmp))
                if change=='contract':(study/'contract.json').write_text(json.dumps(dict(protocol=dict(worker_limit_s=99999,search_limit_s=36000,audit_reserve_s=7200))))
                def full(*a,**kw):
                    return V.original.geometry.audit_request_packets(study/'other' if change=='study' else study,rows,AuditChecks())
                with patch.object(V.original,'audit',side_effect=full):
                    with self.assertRaises(EvidenceError):V.audit(study=study,original_failure=original,previous_failure=previous)

if __name__=='__main__':unittest.main()
