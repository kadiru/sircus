import copy,json,tempfile,unittest
from pathlib import Path
from sircus.mjlab import packet_timing_verify as V
from sircus.mjlab import coverage_training_verify as historical
from sircus.mjlab.benchmark_verify import EvidenceError,sha
from sircus.mjlab.replay650_verify import AuditChecks


def fixture(root,*,end=29000.,count=1):
    (root/'requests').mkdir();name='requests/batch-00000.json'
    requests=[dict(id=f'r{i}',phase='search',values=[.4,-.1,.2,30.,.2]) for i in range(count)]
    path=root/name;path.write_text(json.dumps(dict(pre_outcome=True,phase='search',rows=requests)))
    rows=[dict(r,request_packet=name,request_packet_sha256=sha(path)) for r in requests]
    timing=dict(phase='search',request_packet=name,request_packet_sha256=sha(path),acquired_start_s=19000.,acquired_end_s=end)
    (root/'timing.jsonl').write_text(json.dumps(timing)+'\n')
    return rows,timing


class PacketTests(unittest.TestCase):
    def run_audit(self,root,rows,*,limit=43200.,elapsed=29191.):
        return V.audit_request_packets(root,rows,AuditChecks(),effective_worker_limit_s=limit,reported_elapsed_s=elapsed)
    def test_only_correct_clock_unblocks_actual_long_completed_packet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows,_=fixture(root)
            with self.assertRaises(EvidenceError):historical.audit_request_packets(root,rows,AuditChecks())
            self.assertEqual(self.run_audit(root,rows),dict(batches=1,last_native_end_s=29000.))
    def test_early_packet_result_matches_original_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows,_=fixture(root,end=19500.)
            self.assertEqual(self.run_audit(root,rows),historical.audit_request_packets(root,rows,AuditChecks()))
    def test_cap_endpoint_report_and_nonfinite_values_rejected(self):
        for end,elapsed,limit in ((43200,43201,43200),(29192,29191,43200),(float('nan'),29191,43200),
                                  (29000,float('inf'),43200),(29000,29191,float('nan')),(18000,29191,43200)):
            with self.subTest(end=end,elapsed=elapsed,limit=limit),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);rows,_=fixture(root,end=end)
                with self.assertRaises(EvidenceError):self.run_audit(root,rows,limit=limit,elapsed=elapsed)
    def test_hash_phase_request_or_extra_packet_still_rejected(self):
        for field in ('hash','phase','request','extra'):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);rows,t=fixture(root)
                if field=='hash':rows[0]['request_packet_sha256']='wrong'
                if field=='phase':t['phase']='repeat';(root/'timing.jsonl').write_text(json.dumps(t)+'\n')
                if field=='request':rows[0]['values'][0]=.41
                if field=='extra':(root/'requests/extra.json').write_text('{}')
                with self.assertRaises(EvidenceError):self.run_audit(root,rows)
    def test_oversized_packet_still_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows,_=fixture(root,count=33)
            with self.assertRaises(EvidenceError):self.run_audit(root,rows)
    def test_each_existing_non_clock_check_is_preserved(self):
        # Exact historical check strings are a regression against accidental
        # omissions while adapting one legacy literal timing cap.
        import inspect,re
        old=inspect.getsource(historical.audit_request_packets);new=inspect.getsource(V.audit_request_packets)
        literals=re.findall(r"'([^'\n]+)'",old)
        for label in literals:
            if label.startswith(('Every ','Ordered ','Bounded ','Batch ','Timing ','No missing')):
                self.assertIn(repr(label),new)

if __name__=='__main__':unittest.main()
