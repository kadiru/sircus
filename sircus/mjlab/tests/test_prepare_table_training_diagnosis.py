import copy
import json
from pathlib import Path
import tempfile
import unittest
from sircus.mjlab import prepare_table_training_diagnosis as P

class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.study = self.root/'diagnosis'; self.study.mkdir()
        self.path = self.study/'report.json'
        self.raw = dict(version=P.VERSION, status='completed', contract_fingerprint='a'*64,
                        source=dict(P.SOURCE), summary={}, cases=[])
        self.write()
    def write(self): self.path.write_text(json.dumps(self.raw))
    def audit(self, path=None, version=P.AUDIT, raw=None):
        path = path or self.path; raw = raw or self.raw
        proof = dict(version=version,status='passed',full_checks_verified=True,
                     contract_fingerprint=raw['contract_fingerprint'], report_sha256=P.sha(path),
                     failures=[],checks=[dict(passed=True)],checks_count=1)
        target=P.default_audit(path);target.write_text(json.dumps(proof));return target,proof
    def test_audit_bound_to_report_and_nonempty_checks(self):
        target,proof=self.audit();self.assertEqual(P.compact(self.study)['audit']['status'],'passed')
        self.path.write_text(self.path.read_text()+'\n')
        self.assertEqual(P.compact(self.study)['audit']['status'],'unverified')
        self.write();proof['checks']=[];proof['checks_count']=0;target.write_text(json.dumps(proof))
        self.assertEqual(P.compact(self.study)['audit']['status'],'unverified')
    def test_source_must_match_all_four_identifiers(self):
        for key in P.SOURCE:
            changed=copy.deepcopy(self.raw);changed['source'][key]='b'*64
            self.path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'completed 50-position'):P.compact(self.study)
    def test_private_and_unallowed_fields_excluded(self):
        self.raw['native_archive']='/home/private/data.npz'
        self.raw['cases']=[dict(context_id='case01',raw_file='/home/private/file',events=[dict(label='board',time_s=.2,raw='/home/private')],trace=dict(record_id='r1',time_s=[0,.1],position_m=[[0,0,0],[1,0,0]],local_path='/home/private/cache'))]
        self.write();out=P.compact(self.study)
        self.assertNotIn('/home/private',json.dumps(out))
    def test_admission_proof_is_preserved_with_private_fields_removed(self):
        self.raw['cases']=[dict(setup_proof=dict(connection_sha256='d'*64,valid_ik_branches=0,required_clearance_m=.01,local_path='/home/private/proof',home_geometry=dict(minimum_clearance_m=0,closest_pair=['arm','bucket'],fixed_base_blocker_count=2,raw_file='/home/private/scene'),interpretation='Conservative lower bound, not penetration.'))]
        self.write();out=P.compact(self.study);proof=out['cases'][0]['setup_proof']
        self.assertEqual(proof['home_geometry']['minimum_clearance_m'],0)
        self.assertEqual(proof['valid_ik_branches'],0)
        self.assertNotIn('/home/private',json.dumps(out))
    def test_correction_independent_proof_and_count_normalization(self):
        correction=self.root/'correction';correction.mkdir();cp=correction/'report.json'
        counts=dict(attempts=3,native_releases=2,captures=1,setup_rejections=1,native_misses=1,forbidden_contact_shots=1,invalid_physics=0)
        c=dict(version=P.CORRECTION,status='completed',contract_fingerprint='c'*64,
               counts_by_variant=dict(baseline=counts,corrected=counts),cases=[dict(context_id='c1',variants=dict(baseline=counts,corrected=counts))])
        cp.write_text(json.dumps(c));self.audit(cp,P.CORRECTION_AUDIT,c)
        out=P.compact(self.study,correction=correction)
        self.assertEqual(out['audit']['status'],'pending');self.assertEqual(out['correction']['audit']['status'],'passed')
        self.assertEqual([r['variant'] for r in out['correction']['cases']],['baseline','corrected'])
        cp.write_text(cp.read_text()+'\n');self.assertEqual(P.compact(self.study,correction=correction)['correction']['audit']['status'],'unverified')
    def test_additive_embed_escapes_text_and_preserves_others(self):
        before='<html><body><video src="keep.mp4"></video><!--existing--></body></html>'
        after=P.embed_html(before,dict(version=P.VERSION,interpretation='</script><script>bad()</script>'))
        self.assertNotIn('<script>bad()',after);self.assertIn('<video src="keep.mp4"></video><!--existing-->',after)
        again=P.embed_html(after,dict(version=P.VERSION,status='completed'))
        self.assertEqual(again.count(P.START),1);self.assertEqual(after[:after.index(P.START)],again[:again.index(P.START)])
    def test_install_checks_target_version(self):
        target=self.root/'index.html';target.write_text('<html><body>old</body></html>')
        with self.assertRaisesRegex(ValueError,'changed'):P.install(target,{},'0'*64)
        self.assertEqual(target.read_text(),'<html><body>old</body></html>')
        P.install(target,{},P.sha(target));self.assertIn(P.START,target.read_text())
    def test_publish_preserves_existing_keys(self):
        target=self.root/'display.json';target.write_text(json.dumps(dict(old=[1,2],video='saved.mp4')))
        P.publish(self.study,target);value=json.loads(target.read_text());self.assertEqual(value['old'],[1,2]);self.assertEqual(value['video'],'saved.mp4');self.assertIn(P.KEY,value)
    def test_nonfinite_geometry_refused(self):
        self.raw['cases']=[dict(values=[0,0,float('nan'),32,.2])];self.write()
        with self.assertRaises(ValueError):P.compact(self.study)

if __name__=='__main__':unittest.main()
