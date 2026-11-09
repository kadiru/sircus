import json
from pathlib import Path
import tempfile
import unittest
from sircus.mjlab import prepare_table_training as P

class PublicationTests(unittest.TestCase):
    def test_additive_safe_embed(self):
        original='<html><body><video src="unchanged.mp4"></video><!--other--></body></html>'
        result=P.embed_html(original,dict(version=P.VERSION,status='</script><script>bad()</script>'))
        self.assertIn('<video src="unchanged.mp4"></video><!--other-->',result)
        self.assertNotIn('<script>bad()',result)
        self.assertNotIn('type="module"',result)
        again=P.embed_html(result,dict(version=P.VERSION,status='running'))
        self.assertEqual(again.count(P.START),1)
        self.assertEqual(again[:again.index(P.START)],original[:original.index('</body>')])
    def test_preserves_other_report_keys_and_hides_local_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);study=root/'study';study.mkdir();display=root/'display.json'
            display.write_text(json.dumps(dict(old={'rows':[1,2]},media='previous.mp4')))
            (study/'report.json').write_text(json.dumps(dict(version=P.VERSION,status='running',phase='test',raw_local_file='/home/example/raw.npz',test_positions=[dict(context_id='test01',bucket_xy_m=[.5,.1],native_cache_file='/home/example/raw.npz')],counts=dict(attempts=1))))
            panel=P.publish(study,display)
            value=json.loads(display.read_text());self.assertEqual(value['old'],{'rows':[1,2]});self.assertEqual(value['media'],'previous.mp4')
            self.assertNotIn('/home/',json.dumps(panel));self.assertEqual(panel['audit']['status'],'pending')
    def test_audit_requires_current_report_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);study=root/'study';study.mkdir();report=study/'report.json';proof=root/'study-audit.json'
            report.write_text(json.dumps(dict(version=P.VERSION,status='completed',contract_fingerprint='a'*64)))
            proof.write_text(json.dumps(dict(version=P.AUDIT,status='passed',full_checks_verified=True,contract_fingerprint='a'*64,report_sha256=P.sha(report),failures=[],checks=[dict(passed=True)],checks_count=1)))
            self.assertEqual(P.compact(study)['audit']['status'],'passed')
            report.write_text(report.read_text()+'\n')
            self.assertEqual(P.compact(study)['audit']['status'],'unverified')
    def test_install_refuses_stale_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'index.html';path.write_text('<html><body>existing</body></html>');before=path.read_bytes()
            with self.assertRaises(ValueError):P.install(path,{},'a'*64)
            self.assertEqual(path.read_bytes(),before)
    def test_pending_study_is_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            value=P.compact(Path(tmp)/'not-created',pipeline={'status':'running'})
            self.assertEqual(value['train_requests'],0);self.assertNotEqual(value['status'],'completed')

if __name__=='__main__':unittest.main()
