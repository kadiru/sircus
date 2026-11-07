import copy,json,unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from sircus.mjlab import prepare_flight_residual_report as R
from sircus.mjlab.gap_coverage_study import atomic,sha

class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.data=self.root/'data';self.data.mkdir()
        self.base=self.root/'base.json';atomic(self.base,dict(old_result={'capture':False}))
        self.proof=dict(version=R.AUDIT_VERSION,status='passed',checks_count=1,checks=[dict(passed=True)],
            source_report_sha256=R.SOURCE_HASHES['report'],source_audit_sha256=R.SOURCE_HASHES['audit'],diagnosis_source_sha256='a'*64)
        self.result=dict(version=R.VERSION,status='completed',source=dict(verified=True,report_sha256=R.SOURCE_HASHES['report'],
            audit_sha256=R.SOURCE_HASHES['audit'],manifest_sha256=R.SOURCE_HASHES['manifest'],native_requests_added=0),
            processing=dict(source_sha256='a'*64),audit=dict(status='passed',checks_count=1,full_checks_verified=True))
        self.freeze()
    def freeze(self):
        atomic(self.data/'checks.json',self.proof);self.result['audit']['source_sha256']=sha(self.data/'checks.json')
        atomic(self.data/'diagnosis.json',self.result);atomic(self.data/'summary.json',{})
        atomic(self.data/'checksums.json',{n:sha(self.data/n) for n in ('checks.json','diagnosis.json','summary.json')})
    def test_preserves_other_panels(self):
        value=R.prepare(self.base,self.data,self.root/'derived.json')
        self.assertEqual(value['old_result'],dict(capture=False));self.assertEqual(value[R.KEY],self.result)
    def test_literal_failed_or_empty_checks_rejected(self):
        for checks in ([],[dict(passed=1)],[dict(passed=False)]):
            self.proof['checks']=checks;self.proof['checks_count']=len(checks);self.result['audit']['checks_count']=len(checks);self.freeze()
            with self.assertRaises(ValueError):R.compact(self.data)
    def test_changed_bytes_and_wrong_source_rejected(self):
        atomic(self.data/'diagnosis.json',{})
        with self.assertRaises(ValueError):R.compact(self.data)
        self.result['source']['report_sha256']='b'*64;self.freeze()
        with self.assertRaises(ValueError):R.compact(self.data)
    def test_never_overwrite_source_or_current_display(self):
        for output in (self.base,self.data/'display.json'):
            with self.assertRaises(ValueError):R.prepare(self.base,self.data,output)

if __name__=='__main__':unittest.main()
