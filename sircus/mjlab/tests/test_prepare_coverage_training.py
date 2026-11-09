import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sircus.mjlab import prepare_coverage_training as P
from sircus.mjlab.coverage_training_contract import protocol, make_splits


def write(path,value):path.write_text(json.dumps(value))


def counts(attempts=0,captures=0,rejected=0):
    return dict(attempts=attempts,captures=captures,native_releases=attempts-rejected,
                setup_rejections=rejected,native_misses=attempts-rejected-captures,
                direct_captures=captures,table_assisted_captures=0,mean_reward=0.,invalid_physics=0)


class PublicationTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name);self.study=self.root/'study';self.study.mkdir()
        original=[dict(id=f'old-{row}-{col}',bucket_xy_m=[.1+col*.19,-.65+row*.225]) for row in range(5) for col in range(7)]
        self.splits,_=make_splits(dict(train=original),[[.7,.7]])
        contract=dict(version=P.VERSION,protocol=protocol(),splits=self.splits)
        contract['fingerprint']=P.digest(contract)
        self.fingerprint=contract['fingerprint'];write(self.study/'contract.json',contract);write(self.study/'splits.json',self.splits)
        self.raw=dict(version=P.VERSION,status='running',phase='train',contract_fingerprint=self.fingerprint,
            protocol=protocol(),counts=counts(10,2,3),stage_counts={'search':counts(10,2,3)},search_requests=10,
            train_requests=0,updates=0,elapsed_s=30.,search_coverage=[dict(context_id=self.splits['train'][0]['id'],
                bucket_xy_m=self.splits['train'][0]['bucket_xy_m'],counts=counts(2,1),has_teacher=True)],
            bootstrap_summary=dict(teacher_count=1,requests=10,captures=2,successful_layouts=1,layouts_requested=5),
            per_seed={str(protocol()['seeds'][0]):dict(samples=32,updates=1,private_path='/home/private/weights.pt')},
            curves=[dict(method='bc',seed=protocol()['bc_seed'],samples=0,validation=counts(35,9),
                artifact=dict(path='/home/private/bc.pt',sha256='b'*64,samples=0,seed=protocol()['bc_seed']))],
            selected=None,test_comparison=[],test_positions=[],control_gates={},private_path='/home/private/trace.npz')
        self.report=self.study/'report.json';write(self.report,self.raw)
        self.audit=self.root/'study-audit.json';self.pipeline=self.root/'pipeline.json'
        self.state=dict(status='running',started_unix_s=1000.,global_deadline_unix_s=26200.,global_limit_s=25200)
        write(self.pipeline,self.state)
        self.html=self.root/'index.html';self.original='<html><body><video src="historical.mp4"></video><!--OTHER-->kept</body></html>'
        self.html.write_text(self.original)
        self.display=self.root/'display.json';write(self.display,dict(other={'existing':'evidence'}))
        self.snapshot=self.root/'snapshot.json';self.receipt=self.root/'receipt.json'

    def complete(self):
        self.raw.update(status='completed',phase='completed')
        write(self.report,self.raw)
        write(self.audit,dict(version=P.AUDIT,status='passed',full_checks_verified=True,native_checks_verified=True,
            report_sha256=P.sha(self.report),contract_fingerprint=self.fingerprint,
            checks_count=1,checks=[dict(passed=True)],failures=[]))

    def snapshot_value(self):return P.compact(self.study,self.audit,pipeline=self.state)

    def test_normalizes_exact_producer_counts_splits_and_seed_progress(self):
        value=self.snapshot_value()
        self.assertEqual([len(value['layouts'][s]) for s in ('train','validation','test')],[105,35,70])
        self.assertEqual(value['layouts']['train'][0]['grid_cell'],[0,0])
        self.assertEqual(value['layouts']['train'][34]['grid_cell'],[6,4])
        self.assertEqual(value['stage_counts'][1]['counts']['attempts'],10)
        self.assertEqual(value['stage_counts'][1]['maximum_requests'],1260)
        self.assertEqual(value['bootstrap_coverage'][0]['has_teacher'],True)
        self.assertEqual(value['clock']['deadline_unix_s'],26200.)
        self.assertNotIn('/home/private',json.dumps(value))
        self.assertLess(len(json.dumps(value)),P.MAX_SNAPSHOT_BYTES)

    def test_changed_split_contract_or_protocol_cannot_be_published(self):
        split=copy.deepcopy(self.splits);split['test'][0]['bucket_xy_m'][0]+=.01
        write(self.study/'splits.json',split)
        with self.assertRaisesRegex(ValueError,'split file'):self.snapshot_value()
        write(self.study/'splits.json',self.splits)
        self.raw['protocol']['test_layouts']=50;write(self.report,self.raw)
        with self.assertRaisesRegex(ValueError,'protocol'):self.snapshot_value()

    def test_bc_step_zero_selection_keeps_validation_and_identity(self):
        self.raw['selected']=self.raw['curves'][0]['artifact'];write(self.report,self.raw)
        value=self.snapshot_value()
        self.assertEqual(value['selected']['method'],'bc')
        self.assertEqual(value['selected']['samples'],0)
        self.assertEqual(value['selected']['validation']['captures'],9)
        self.assertNotIn('path',value['selected'])

    def test_proof_needs_current_complete_native_evidence(self):
        self.complete();self.assertEqual(self.snapshot_value()['audit']['status'],'passed')
        self.report.write_bytes(self.report.read_bytes()+b' ')
        self.assertEqual(self.snapshot_value()['audit']['status'],'unverified')
        self.complete();audit=json.loads(self.audit.read_text());audit['native_checks_verified']=False;write(self.audit,audit)
        self.assertEqual(self.snapshot_value()['audit']['status'],'unverified')

    def test_pending_report_still_shows_real_global_deadline(self):
        self.report.unlink();value=self.snapshot_value()
        self.assertEqual(value['clock']['deadline_unix_s'],26200.)
        self.assertEqual(value['audit']['status'],'pending')

    def test_additive_idempotent_html_and_cas_keep_unrelated_work(self):
        value=self.snapshot_value();value['phase']='</script><script>bad()</script>'
        result=P.embed_html(self.original,value)
        self.assertTrue(result.startswith(self.original.split('</body>')[0]))
        self.assertNotIn('<script>bad()',result)
        self.assertEqual(P.embed_html(result,value),result)
        with self.assertRaisesRegex(ValueError,'changed'):P.install(self.html,value,'0'*64)
        self.assertEqual(self.html.read_text(),self.original)
        P.publish_display(self.display,value)
        self.assertEqual(json.loads(self.display.read_text())['other'],{'existing':'evidence'})

    def test_sidecar_cannot_replace_foreign_file_or_write_into_evidence(self):
        write(self.snapshot,dict(other='foreign'))
        with self.assertRaisesRegex(ValueError,'another task'):P.publish_sidecar(self.snapshot,self.snapshot_value())
        self.assertEqual(json.loads(self.snapshot.read_text()),{'other':'foreign'})
        with self.assertRaisesRegex(ValueError,'immutable'):P.guarded_targets(self.study,[self.report])
        with self.assertRaisesRegex(ValueError,'distinct'):P.guarded_targets(self.study,[self.snapshot,self.snapshot])

    def test_refresh_updates_own_block_and_retains_other_display_keys(self):
        value,state=P.refresh(self.study,snapshot_path=self.snapshot,target_html=self.html,display=self.display,
            audit_path=self.audit,pipeline_path=self.pipeline,now=1030.)
        self.assertEqual(value['publication']['updated_unix_s'],1030.)
        self.assertEqual(state['status'],'running')
        self.assertIn('historical.mp4',self.html.read_text())
        self.assertEqual(self.html.read_text().count(P.START),1)
        self.assertEqual(json.loads(self.display.read_text())['other'],{'existing':'evidence'})

    def test_bounded_watch_waits_through_audit_for_final_pipeline_video(self):
        self.complete();clock=[0.];sleeps=[]
        def sleep(seconds):
            sleeps.append(seconds);clock[0]+=seconds;state=copy.deepcopy(self.state);state.update(status='completed',finished_unix_s=1030.+clock[0]);write(self.pipeline,state)
        result=P.watch(self.study,snapshot_path=self.snapshot,target_html=self.html,receipt=self.receipt,
            display=self.display,audit_path=self.audit,pipeline_path=self.pipeline,max_seconds=60,
            monotonic=lambda:clock[0],sleep=sleep,now=lambda:1030.+clock[0])
        self.assertEqual(result['status'],'finished');self.assertEqual(result['updates'],2);self.assertEqual(sleeps,[30.])

    def test_watch_timeout_is_finite_and_cannot_touch_a_worker(self):
        clock=[0.]
        result=P.watch(self.study,snapshot_path=self.snapshot,target_html=self.html,receipt=self.receipt,
            pipeline_path=self.pipeline,max_seconds=31,monotonic=lambda:clock[0],
            sleep=lambda seconds:clock.__setitem__(0,clock[0]+seconds),now=lambda:1030.+clock[0])
        self.assertEqual(result['status'],'publication_timeout');self.assertEqual(clock[0],31)
        self.assertEqual(json.loads(self.pipeline.read_text()),self.state)
        with self.assertRaisesRegex(ValueError,'eight hours'):
            P.watch(self.study,snapshot_path=self.snapshot,target_html=self.html,receipt=self.receipt,max_seconds=28801)

    def test_movie_bytes_and_source_bindings_required(self):
        self.complete();manifest,movie=P.video_paths(self.study);manifest.parent.mkdir()
        movie.write_bytes(b'\x00\x00\x00\x18ftypmp42fixture')
        # Header-only media fixture; this test does not claim codec/simulator validation.
        frames=[dict(source_tick=i) for i in range(30)]
        media=dict(version=P.VIDEO,status='completed',native_steps_added=0,held_arm_poses=True,sampled_ball_centers_only=True,
            source=dict(report_sha256=P.sha(self.report),audit_sha256=P.sha(self.audit),contract_fingerprint=self.fingerprint),
            video_bytes=movie.stat().st_size,output_sha256=P.sha(movie),frames=frames,frames_sha256=P.digest(frames),fps=30,duration_s=1,
            clips=[dict(record_id='test-row',label='Actual example',capture=False,start_s=0,end_s=1,events={'board':.2})])
        write(manifest,media);value=self.snapshot_value();self.assertEqual(value['video_status'],'ready')
        self.assertNotIn('frames',value['video'])
        with self.assertRaisesRegex(ValueError,'Verified movie'):P.embed_html(self.original,value)
        self.assertIn('data:video/mp4;base64,',P.embed_html(self.original,value,movie.read_bytes()))
        media['source']['report_sha256']='0'*64;write(manifest,media)
        with self.assertRaisesRegex(ValueError,'source differs'):self.snapshot_value()


if __name__=='__main__':unittest.main()
