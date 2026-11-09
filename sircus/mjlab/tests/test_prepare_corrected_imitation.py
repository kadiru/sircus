import copy
import json
from pathlib import Path
import tempfile
import unittest

from sircus.mjlab import prepare_corrected_imitation as P

def write(path,value):Path(path).write_text(json.dumps(value,allow_nan=False))
def counts(captures=30):return dict(attempts=70,native_releases=60,captures=captures,setup_rejections=10,native_misses=60-captures,invalid_physics=0,direct_captures=captures,table_assisted_captures=0)


class PublisherTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.study=self.root/'study';self.study.mkdir();self.report=self.study/'report.json';self.audit=self.root/'audit.json';self.pipeline=self.root/'pipeline.json'
        self.splits=dict(test=[dict(id=f'fresh-{i}',bucket_xy_m=[.1+i/100,-.1],grid_cell=[i%7,i//14]) for i in range(70)])
        self.protocol=dict(substeps=64,test_layouts=70,test_requests=140,maximum_requests=158,actor_steps=800,critic_steps=0,base_teacher_count=92,correction_teacher_count=5,teacher_count=97)
        self.spec=dict(version=P.VERSION,protocol=self.protocol,splits=self.splits);self.spec['fingerprint']=P.digest(self.spec)
        write(self.study/'contract.json',self.spec);write(self.study/'splits.json',self.splits)
        self.manifest={p.name:P.sha(p) for p in self.study.iterdir()};write(self.study/'evidence-manifest.json',self.manifest)
        positions=[]
        for method in P.METHODS:
            for i,layout in enumerate(self.splits['test']):positions.append(dict(context_id=layout['id'],bucket_xy_m=layout['bucket_xy_m'],method=method,values=[.5,0,.4,30,.2],capture=i<(32 if method=='corrected_bc' else 30),native_release=i<60,invalid=None,record_id=f'{method}-{i}'))
        self.raw=dict(version=P.VERSION,status='completed',phase='completed',protocol=self.protocol,contract_fingerprint=self.spec['fingerprint'],counts=dict(attempts=158,native_releases=138,captures=74,setup_rejections=20,native_misses=64),control_gates={k:dict(status='passed') for k in ('pre','post')},
            test_comparison=[dict(method=m,counts=counts(32 if m=='corrected_bc' else 30),artifact_sha256=('a' if m=='corrected_bc' else 'b')*64) for m in P.METHODS],test_positions=positions,
            fit=dict(status='actor_updated',parent_teacher_count=92,new_teacher_count=5,actor_steps=800,critic_steps=0),
            artifacts={m:dict(sha256=('a' if m=='corrected_bc' else 'b')*64,weight_sha256='c'*64,samples=97 if m=='corrected_bc' else 92,source_path='/private/not-publish') for m in P.METHODS},
            teacher_corrections=[dict(context_id=f'old-{i}',bucket_xy_m=[.3,-.2],values=[.5,0,.4,30,.2],source_study='prior-correction',source_record_id=f'teacher-{i}',source_record_sha256='d'*64,source_report_sha256='e'*64,confirmation_captures=3,confirmation_requests=3,private_path='/private/excluded') for i in range(5)],evidence_sha256=self.manifest)
        write(self.report,self.raw);self.proof();write(self.pipeline,dict(status='completed',finished_unix_s=100))
        self.html=self.root/'index.html';self.css=self.root/'style.css';self.snapshot=self.root/'snapshot.json'
        self.html.write_text('<html><body><video src="old.mp4"></video><!--OLD_PANEL-->history</body></html>');self.css.write_text('/*historical css*/body{color:white}')

    def proof(self):
        write(self.audit,dict(version=P.AUDIT,status='passed',full_checks_verified=True,native_checks_verified=True,checks_count=1,checks=[dict(passed=True)],failures=[],report_sha256=P.sha(self.report),contract_fingerprint=self.spec['fingerprint'],evidence_manifest_sha256=P.sha(self.study/'evidence-manifest.json')))

    def compact(self):return P.compact(self.study,audit_path=self.audit,pipeline_path=self.pipeline,now=150)

    def test_complete_paired_results_teacher_lineage_and_private_fields(self):
        s=self.compact();self.assertEqual(s['paired'],dict(gains=2,losses=0,shared_captures=30,shared_failures=38,net_captures=2,positions=70));self.assertEqual(len(s['test_positions']),140);self.assertEqual(len(s['teacher_corrections']),5)
        self.assertEqual(s['fit']['new_teacher_count'],5);self.assertNotIn('/private',json.dumps(s));self.assertLess(len(json.dumps(s)),P.MAX_SNAPSHOT_BYTES)
        self.assertEqual(P.baseline_summary(s)['paired'],s['paired']);self.assertNotIn('publication',P.baseline_summary(s))

    def test_actual_worker_protocol_schema(self):
        from sircus.mjlab import corrected_imitation_contract as C
        protocol=C.protocol();self.spec['protocol']=protocol
        unsigned=dict(self.spec);unsigned.pop('fingerprint');self.spec['fingerprint']=P.digest(unsigned)
        write(self.study/'contract.json',self.spec)
        self.raw.update(protocol=protocol,contract_fingerprint=self.spec['fingerprint'])
        self.manifest['contract.json']=P.sha(self.study/'contract.json');write(self.study/'evidence-manifest.json',self.manifest)
        write(self.report,self.raw);self.proof();out=self.compact()
        self.assertEqual(out['protocol']['correction_teacher_count'],5)
        self.assertEqual(out['protocol']['teacher_count'],97)
        self.assertEqual(out['protocol']['test_seed'],C.TEST_SEED)
        self.assertEqual(tuple(r['method'] for r in out['test_comparison']),C.METHODS)

    def test_no_test_outcomes_or_fitting_claim_before_full_audit(self):
        self.audit.unlink();s=self.compact();self.assertEqual(s['audit']['status'],'pending');self.assertEqual(s['test_comparison'],[]);self.assertEqual(s['test_positions'],[]);self.assertEqual(s['fit'],{});self.assertIsNone(s['paired'])
        with self.assertRaisesRegex(ValueError,'audited'):P.baseline_summary(s)
        self.proof();a=P.read(self.audit);a['checks'][0]['passed']=False;write(self.audit,a);self.assertEqual(self.compact()['audit']['status'],'unverified')

    def test_stale_manifest_report_native_flags_never_pass(self):
        for key in ('native_checks_verified','full_checks_verified'):
            self.proof();a=P.read(self.audit);a[key]=False;write(self.audit,a);self.assertEqual(self.compact()['test_positions'],[])
        self.proof();self.report.write_text(self.report.read_text()+' ');self.assertEqual(self.compact()['audit']['status'],'unverified')
        self.proof();write(self.study/'evidence-manifest.json',{});self.assertEqual(self.compact()['audit']['status'],'unverified')

    def test_duplicate_position_coordinate_mismatch_and_inconsistent_counts_rejected(self):
        for mutation in (lambda r:r['test_positions'].pop(),lambda r:r['test_positions'].__setitem__(0,r['test_positions'][1]),lambda r:r['test_positions'][0].update(bucket_xy_m=[9,9]),lambda r:r['test_comparison'][0]['counts'].update(captures=33,native_misses=27)):
            raw=copy.deepcopy(self.raw);mutation(raw);write(self.report,raw);self.proof()
            with self.assertRaises(ValueError):self.compact()

    def test_declined_or_invalid_capture_rejected(self):
        for mutation in (dict(native_release=False),dict(invalid_physics=True),dict(invalid='native_physics_outside_calibrated_domain')):
            raw=copy.deepcopy(self.raw);raw['test_positions'][0].update(mutation);write(self.report,raw);self.proof()
            with self.assertRaises(ValueError):self.compact()

    def test_wrong_contract_protocol_or_split_rejected(self):
        raw=copy.deepcopy(self.raw);raw['protocol']=dict(self.protocol,substeps=128);write(self.report,raw)
        with self.assertRaises(ValueError):self.compact()
        write(self.report,self.raw);splits=copy.deepcopy(self.splits);splits['test'][0]['bucket_xy_m']=[1,1];write(self.study/'splits.json',splits)
        with self.assertRaisesRegex(ValueError,'hash differs'):self.compact()

    def test_missing_running_and_failed_evidence_is_pending_or_stopped(self):
        self.report.unlink();s=self.compact();self.assertEqual(s['status'],'pending');self.assertEqual(s['test_positions'],[])
        write(self.pipeline,dict(status='failed',finished_unix_s=100));self.assertEqual(self.compact()['status'],'stopped')

    def test_markup_escaping_idempotence_and_old_content_preservation(self):
        s=self.compact();s['phase']='</script>*/<script>bad()</script>';html=self.html.read_text();css=self.css.read_text();new=P.embed_html(html,s);newcss=P.embed_css(css,s)
        self.assertNotIn('<script>bad()',new);self.assertNotIn('*/<script>',newcss);self.assertIn('old.mp4',new)
        self.assertEqual(P.embed_html(new,s),new);self.assertEqual(P.embed_css(newcss,s),newcss)
        self.assertEqual(new[:new.index(P.START)]+new[new.index(P.END)+len(P.END):],html)
        self.assertEqual(newcss[:newcss.index(P.CSS_START)]+newcss[newcss.index(P.CSS_END)+len(P.CSS_END):],css)
        with self.assertRaisesRegex(ValueError,'Ambiguous'):P.embed_html(html+P.START,s)

    def test_refresh_rejects_concurrent_changes_and_foreign_snapshot(self):
        options=dict(snapshot_path=self.snapshot,target_html=self.html,target_css=self.css,audit_path=self.audit,pipeline_path=self.pipeline)
        with self.assertRaisesRegex(ValueError,'HTML changed'):P.refresh(self.study,**options,expected_html_sha256='0'*64)
        self.assertFalse(self.snapshot.exists());s=P.refresh(self.study,**options,expected_html_sha256=P.sha(self.html),expected_css_sha256=P.sha(self.css));self.assertEqual(s['audit']['status'],'passed')
        write(self.snapshot,dict(version='foreign'))
        with self.assertRaisesRegex(ValueError,'another study'):P.refresh(self.study,**options)

    def test_publication_outputs_cannot_overlap_inputs_or_study(self):
        for targets in ((self.report,),(self.audit,),(self.snapshot,self.snapshot)):
            with self.assertRaises(ValueError):P.guard_targets(self.study,(self.audit,),targets)


if __name__=='__main__':unittest.main()
