import copy
import json
from pathlib import Path
import tempfile
import unittest

from sircus.mjlab import prepare_coverage_recovery as P


def write(path,value):path.write_text(json.dumps(value))
def counts(attempts=0,captures=0,rejected=0):
    return dict(attempts=attempts,captures=captures,native_releases=attempts-rejected,setup_rejections=rejected,native_misses=attempts-rejected-captures)

def diagnosis():
    cases=[dict(context_id=f'prior-{i}',bucket_xy_m=[.2+i*.01,0],values=[.5,0,.3,28,.2],reason='fixed_base_clearance_rejected' if i<3 else 'finite_ik_unresolved' if i<22 else None,native_release=i>=22,capture=False,events=[]) for i in range(42)]
    result=dict(version=P.DIAGNOSIS,status='completed',source=dict(audit_sha256='e'*64),summary=dict(requests=70,captures=28,setup_rejections=22,flight_misses=20),cases=cases)
    result['diagnosis_fingerprint']=P.digest(result)
    return result


class RecoveryPublicationTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.study=self.root/'study';self.study.mkdir();self.diagnosis=diagnosis()
        self.protocol=dict(search_layouts=39,fixed_base_skips=3,search_requests=624,repeat_requests_max=78,maximum_requests=1000)
        self.splits=dict(development=[dict(id=f'prior-{i}',bucket_xy_m=[.2+i*.01,0]) for i in range(3,42)],test=[dict(id=f'fresh-{i}',bucket_xy_m=[.201+i*.01,.1],grid_cell=[i%7,(i//7)%5]) for i in range(70)])
        self.contract=dict(version=P.VERSION,protocol=copy.deepcopy(self.protocol),splits=self.splits);self.contract['fingerprint']=P.digest(self.contract)
        write(self.study/'contract.json',self.contract);write(self.study/'splits.json',self.splits)
        self.raw=dict(version=P.VERSION,status='running',phase='repeat',protocol=copy.deepcopy(self.protocol),contract_fingerprint=self.contract['fingerprint'],counts=counts(12,3,4),search_requests=10,repeat_requests=2,confirmed_teachers=1,selected=[dict(context_id='prior-3',discovery_id='s1')],search_coverage=[dict(context_id='prior-3',confirmed=True)],stage_counts=dict(search=counts(10,1,4),repeat=counts(2,2)),fit=dict(status='actor_updated',parent_teacher_count=68,new_teacher_count=1,actor_steps=800),test_comparison=[],test_positions=[],control_gates={},private_path='/home/private/keep-out')
        self.report=self.study/'report.json';write(self.report,self.raw)
        rows=[dict(id='s1',phase='search',context_id='prior-3',native_release=True,route_success=True),*[dict(id=f'r{i}',phase='repeat',context_id='prior-3',native_release=True,route_success=True) for i in range(2)]]
        (self.study/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        self.audit=self.root/'audit.json';self.pipeline=self.root/'pipeline.json';write(self.pipeline,dict(status='running',started_unix_s=1000,global_deadline_unix_s=11800))
        self.html=self.root/'index.html';self.original='<html><body><video src="historical.mp4"></video><!--OTHER-->keep</body></html>';self.html.write_text(self.original)
        self.display=self.root/'display.json';write(self.display,dict(other={'keep':True}));self.snapshot=self.root/'snapshot.json';self.receipt=self.root/'receipt.json'

    def compact(self):return P.compact(self.study,self.diagnosis,audit_path=self.audit,pipeline=P.read(self.pipeline))

    def test_exact_schema_teacher_confirmation_and_fresh_splits(self):
        out=self.compact();self.assertEqual(len(out['test_layouts']),70);self.assertTrue(out['test_layouts'][0]['id'].startswith('fresh'))
        self.assertEqual(out['confirmed_teachers'],1);self.assertEqual(out['training_summary'],dict(status='actor_updated',trained=True,old_teachers=68,new_teachers=1,actor_steps=800))
        self.assertEqual(len(out['recovery_cases']),39);first=out['recovery_cases'][0];self.assertEqual(first['context_id'],'prior-3');self.assertTrue(first['teacher_accepted']);self.assertEqual(first['confirmation']['captures'],2)
        self.assertIsNone(out['recovery_cases'][1]['teacher_accepted']);self.assertNotIn('/home/private',json.dumps(out));self.assertLess(len(json.dumps(out)),P.MAX_SNAPSHOT_BYTES)

    def test_two_repeats_alone_do_not_claim_teacher_until_producer_confirms(self):
        self.raw['search_coverage'][0]['confirmed']=False;self.raw['confirmed_teachers']=0;write(self.report,self.raw)
        self.assertIsNone(self.compact()['recovery_cases'][0]['teacher_accepted'])
        self.raw['phase']='fit'
        write(self.report,self.raw)
        self.assertFalse(self.compact()['recovery_cases'][0]['teacher_accepted'])
        self.raw['fit']=dict(status='no_confirmed_new_teacher',parent_teacher_count=68,new_teacher_count=0,actor_steps=0);write(self.report,self.raw)
        self.assertFalse(self.compact()['training_summary']['trained'])

    def test_producer_reason_string_and_invalid_counts_reach_ui_without_rescoring(self):
        self.raw['test_comparison']=[dict(method='parent_bc',counts=dict(counts(32,9,4),invalid_physics=1))]
        self.raw['stage_counts']['test']=dict(counts(32,9,4),invalid_physics=1)
        self.raw['test_positions']=[dict(context_id='fresh-0',bucket_xy_m=[.2,.1],method='parent_bc',capture=False,native_release=True,invalid='native_physics_outside_calibrated_domain',record_id='invalid-example')]
        write(self.report,self.raw);out=self.compact()
        self.assertEqual(out['test_positions'][0]['invalid'],'native_physics_outside_calibrated_domain')
        self.assertNotIn('invalid_physics',out['test_positions'][0])
        self.assertEqual(out['test_comparison'][0]['counts'],self.raw['test_comparison'][0]['counts'])
        self.assertEqual(next(r for r in out['stage_counts'] if r['stage']=='test')['counts'],self.raw['stage_counts']['test'])

    def test_changed_split_contract_and_diagnosis_rejected(self):
        split=copy.deepcopy(self.splits);split['test'][0]['bucket_xy_m'][0]+=.1;write(self.study/'splits.json',split)
        with self.assertRaisesRegex(ValueError,'split file'):self.compact()
        write(self.study/'splits.json',self.splits);self.raw['protocol']['search_layouts']=38;write(self.report,self.raw)
        with self.assertRaisesRegex(ValueError,'protocol'):self.compact()
        self.raw['protocol']=self.protocol;write(self.report,self.raw);self.diagnosis['summary']['captures']=29
        with self.assertRaisesRegex(ValueError,'fingerprint'):self.compact()

    def complete(self):
        self.raw.update(status='completed',phase='completed');write(self.report,self.raw)
        write(self.audit,dict(version=P.AUDIT,status='passed',full_checks_verified=True,native_checks_verified=True,checks=[dict(passed=True)],checks_count=1,failures=[],report_sha256=P.sha(self.report),contract_fingerprint=self.contract['fingerprint']))

    def test_proof_requires_current_report_and_all_native_checks(self):
        self.complete();self.assertEqual(self.compact()['audit']['status'],'passed');self.report.write_bytes(self.report.read_bytes()+b' ')
        self.assertEqual(self.compact()['audit']['status'],'unverified');self.complete();proof=P.read(self.audit);proof['native_checks_verified']=False;write(self.audit,proof)
        self.assertEqual(self.compact()['audit']['status'],'unverified')

    def test_pending_and_failed_pipeline_do_not_invent_new_scores(self):
        self.report.unlink();value=self.compact();self.assertEqual(value['status'],'pending');self.assertEqual(value['test_comparison'],[]);self.assertEqual(value['clock']['deadline_unix_s'],11800)
        write(self.pipeline,dict(status='failed'));self.assertEqual(self.compact()['status'],'stopped')

    def test_additive_idempotent_safe_embedding_preserves_history(self):
        value=self.compact();value['phase']='</script><script>bad()</script>'
        result=P.embed_html(self.original,value);self.assertIn('historical.mp4',result);self.assertIn('<!--OTHER-->keep',result);self.assertNotIn('<script>bad()',result)
        self.assertEqual(P.embed_html(result,value),result);self.assertEqual(result.count(P.START),1)
        with self.assertRaisesRegex(ValueError,'Ambiguous'):P.embed_html(self.original+P.START,value)

    def test_refresh_keeps_other_display_keys_and_rejects_foreign_snapshot(self):
        value,_=P.refresh(self.study,self.diagnosis,snapshot_path=self.snapshot,target_html=self.html,display=self.display,audit_path=self.audit,pipeline_path=self.pipeline,now=1030)
        self.assertEqual(value['publication']['updated_unix_s'],1030);self.assertEqual(P.read(self.display)['other'],{'keep':True});self.assertIn('historical.mp4',self.html.read_text())
        write(self.snapshot,dict(version='foreign'))
        with self.assertRaisesRegex(ValueError,'another study'):P.refresh(self.study,self.diagnosis,snapshot_path=self.snapshot,target_html=self.html)

    def test_targets_cannot_touch_evidence_or_each_other(self):
        for targets in [(self.report,),(self.snapshot,self.snapshot)]:
            with self.assertRaises(ValueError):P.guarded_targets(self.study,targets)
        with self.assertRaisesRegex(ValueError,'immutable'):P.guarded_targets(self.study,(self.audit,),inputs=(self.audit,))

    def test_watch_waits_for_pipeline_finish_then_stops(self):
        self.complete();clock=[0.];slept=[]
        def sleep(seconds):
            slept.append(seconds);clock[0]+=seconds;write(self.pipeline,dict(status='completed',finished_unix_s=1060))
        result=P.watch(self.study,self.diagnosis,snapshot_path=self.snapshot,target_html=self.html,receipt=self.receipt,audit_path=self.audit,pipeline_path=self.pipeline,max_seconds=100,monotonic=lambda:clock[0],sleep=sleep)
        self.assertEqual(result['status'],'finished');self.assertEqual(result['updates'],2);self.assertEqual(slept,[30]);self.assertEqual(P.read(self.receipt)['audit_status'],'passed')

    def test_relative_evidence_paths_and_hashes_are_enforced(self):
        data=self.study/'native.json';write(data,{'native':'evidence'});self.assertEqual(P.checked_file(self.study,'native.json',P.sha(data)),data.resolve())
        for path,digest in [('../audit.json','0'*64),('/tmp/native.json','0'*64),('native.json','0'*64)]:
            with self.assertRaises(ValueError):P.checked_file(self.study,path,digest)


class TraceDiagnosisTests(unittest.TestCase):
    def setUp(self):
        try:import numpy as np
        except ImportError:self.skipTest('Native trace tests require the project NumPy environment')
        self.np=np;self.times=np.arange(12)/15360;self.positions=np.column_stack((np.linspace(.5,.8,12),np.zeros(12),np.linspace(.7,.2,12)))
        self.forces=np.zeros((11,6));self.row=dict(physics_steps=10,native_initial_ball_center_m=self.positions[0].tolist(),native_cache_sha256='a'*64,forbidden_contact_steps=0,table_contact_steps=1)

    def test_first_post_board_contact_uses_actual_later_hit_and_ignores_postterminal(self):
        self.forces[1,1]=.02;self.forces[2,0]=.02;self.forces[4,1]=.02;self.forces[6,2]=.02;self.forces[10,4]=10
        result=P.summarize_flight(self.row,self.times,self.positions,self.forces)
        self.assertEqual(result['first_post_board_contact']['category'],'bucket');self.assertEqual(result['first_post_board_contact']['time_s'],self.times[4])
        self.assertNotIn('robot',[r['category'] for r in result['events']]);self.assertEqual(result['trace']['position_m'][-1],self.positions[10].tolist())
        self.assertTrue(all(p in self.positions.tolist() for p in result['trace']['position_m']))

    def test_inconsistent_forbidden_classification_and_clock_rejected(self):
        self.forces[2,0]=1;self.forces[3,2]=1;self.forces[4,4]=1
        with self.assertRaisesRegex(ValueError,'forbidden'):P.summarize_flight(self.row,self.times,self.positions,self.forces)
        self.row['forbidden_contact_steps']=1
        with self.assertRaisesRegex(ValueError,'clock'):P.summarize_flight(self.row,self.times*2,self.positions,self.forces)


if __name__=='__main__':unittest.main()
