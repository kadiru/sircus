import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sircus.mjlab import prepare_gripper_yaw as P
from sircus.mjlab.prepare_imitation_regression import CASES as ALL_CASES


def write(path,value):Path(path).write_text(json.dumps(value,allow_nan=False))
def row(identifier,native=False,capture=False,**extra):
    return dict(id=identifier,record_id=identifier,context_id=P.CASES[0],bucket_xy_m=[.7,-.3],values=[.5,-.2,.4,28,.2],native_release=native,capture=capture,route_success=capture,invalid=None if native else 'finite_ik_unresolved',invalid_physics=False,repeat=0,**extra)


class PublisherTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.study=self.root/'study';self.study.mkdir();self.audit=self.root/'audit.json';self.diagnosis=self.root/'diagnosis.json';self.pipeline=self.root/'pipeline.json'
        d=dict(version=P.DIAGNOSIS,status='completed',source=dict(checks_count=55633,full_checks_verified=True,native_checks_verified=True,report_sha256='a'*64,audit_sha256='b'*64,contract_fingerprint='c'*64),losses=[])
        for i,c in enumerate(ALL_CASES):
            old=row('old-'+c,True,True);new=row('new-'+c,i in (2,3,4),False,forbidden_contact_steps=2 if i in (2,4) else 0)
            old['context_id']=new['context_id']=c
            d['losses'].append(dict(context_id=c,bucket_xy_m=[.7,-.3],current_bc=old,corrected_bc=new,delta_values=[.01,0,.03,1,-.02]))
        d['diagnosis_fingerprint']=P.digest(d);write(self.diagnosis,d)
        self.spec=dict(protocol=dict(substeps=64,variants=list(P.VARIANTS)));self.spec['fingerprint']=P.digest(self.spec);write(self.study/'contract.json',self.spec)
        self.manifest={'contract.json':P.sha(self.study/'contract.json')};write(self.study/'evidence-manifest.json',self.manifest)
        self.rule=dict(mode='bucket_facing_gripper_yaw',contexts=list(P.CASES),diagnosis_sha256=P.sha(self.diagnosis))
        self.report=dict(version=P.STUDY,status='completed',phase='completed',expected_requests=36,contract_fingerprint=self.spec['fingerprint'],evidence_sha256=self.manifest,rule=self.rule,counts=dict(attempts=36,captures=24),counts_by_variant={v:dict(attempts=6,captures=6 if v!='corrected_exact' else 0) for v in P.VARIANTS},cases=[dict(context_id=c,original_yaw_rad=0.,target_yaw_rad=-.5,variant_values={v:[.5,-.2,.4,28,.2] for v in P.VARIANTS}) for c in P.CASES],control_gates={k:dict(status='passed') for k in ('pre','post')})
        write(self.study/'report.json',self.report);self.proof();write(self.pipeline,dict(status='completed',finished_unix_s=123))
        self.records=[]
        for c in P.CASES:
            for v in P.VARIANTS:
                r=row(c+v,variant=v,yaw_evidence=dict(requested_yaw_rad=-.5,board_q14_unchanged=True,release_unchanged=True));r['context_id']=c;r['record_sha256']=P.digest(r);self.records.append(r)
        self.patcher=patch.object(P,'comparison_recordings',return_value=self.records);self.patcher.start();self.addCleanup(self.patcher.stop)
        self.html=self.root/'index.html';self.css=self.root/'style.css';self.snapshot=self.root/'snapshot.json'
        self.html.write_text('<html><body><video src="old.mp4"></video><!--OLD_PANEL-->history</body></html>');self.css.write_text('body{color:white}/*old*/')

    def proof(self):
        write(self.audit,dict(version=P.AUDIT,status='passed',full_checks_verified=True,native_checks_verified=True,failures=[],checks=[dict(passed=True)],checks_count=1,report_sha256=P.sha(self.study/'report.json'),evidence_manifest_sha256=P.sha(self.study/'evidence-manifest.json'),contract_fingerprint=self.spec['fingerprint']))

    def compact(self):return P.compact(self.study,self.diagnosis,audit_path=self.audit,pipeline_path=self.pipeline,now=150)

    def test_exact_two_collision_pairs_and_six_first_repeats(self):
        s=self.compact();self.assertEqual(len(s['diagnosis']['losses']),2);self.assertEqual(len(s['correction']['recordings']),6);self.assertEqual(P.baseline_summary(s)['expected_requests'],36)
        self.assertEqual(s['correction']['cases'][0]['target_yaw_rad'],-.5)
        self.assertTrue(P.baseline_summary(s)['first_repeat_realizations'][0]['yaw_evidence']['board_q14_unchanged'])

    def test_actual_protocol_contract_is_accepted(self):
        from sircus.mjlab.gripper_yaw_contract import protocol
        self.spec['protocol']=protocol();self.spec.pop('fingerprint');self.spec['fingerprint']=P.digest(self.spec);write(self.study/'contract.json',self.spec)
        self.manifest['contract.json']=P.sha(self.study/'contract.json');write(self.study/'evidence-manifest.json',self.manifest)
        self.report['contract_fingerprint']=self.spec['fingerprint'];write(self.study/'report.json',self.report);self.proof()
        self.assertEqual(self.compact()['correction']['audit']['status'],'passed')
        self.assertEqual(tuple(self.spec['protocol']['contexts']),P.CASES)
        self.assertEqual(tuple(self.spec['protocol']['variants']),P.VARIANTS)

    def test_actual_rotation_rule_and_native_invariance_survive_compaction(self):
        rule=P.read(Path(P.__file__).with_name('gripper-yaw-correction.json'))
        self.report['rule'].update({key:rule[key] for key in ('rotation','preserve_board_joints')})
        write(self.study/'report.json',self.report);self.proof()
        shown=self.compact()['correction']['rule']
        self.assertEqual(shown['rotation'],rule['rotation']);self.assertTrue(shown['preserve_board_joints'])
        compact=P.compact_row(row('native',True,yaw_evidence=dict(board_q14_unchanged=True,board_native_initial_q14_unchanged=True,nominal_source_q14=[1,2,3])))
        self.assertTrue(compact['yaw_evidence']['board_native_initial_q14_unchanged'])
        self.assertNotIn('nominal_source_q14',compact['yaw_evidence'])

    def test_first_repeat_loader_uses_distinct_yaw_native_namespace(self):
        import numpy as np
        self.patcher.stop()
        try:
            rows=[];manifest={}
            for adapter,variants in (('native',P.VARIANTS[:2]),('native-yaw',P.VARIANTS[2:])):
                folder=self.study/adapter;folder.mkdir();members=[]
                for context in P.CASES:
                    for variant in variants:
                        r=row(context+'-'+variant,True,False,phase='diagnostic',variant=variant,adapter_directory=adapter,native_cache_file='same-name.npz',native_cache_row=len(members));r['context_id']=context;members.append(r)
                path=folder/'same-name.npz';np.savez(path,time_s=[0.,.1],position_m=np.zeros((2,len(members),3)),forces_N=np.zeros((2,len(members),4)),record_ids=np.asarray([r['id'] for r in members]))
                manifest[adapter+'/same-name.npz']=P.sha(path)
                for r in members:r['native_cache_sha256']=P.sha(path)
                rows.extend(members)
            self.study.joinpath('rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));manifest['rows.jsonl']=P.sha(self.study/'rows.jsonl')
            with patch.object(P,'summarize_flight',side_effect=lambda row,*args:dict(trace=dict(time_s=[0.,.1],position_m=[[0.,0.,0.],[0.,0.,.1]],native_cache_sha256=row['native_cache_sha256']))):
                actual=P.comparison_recordings(self.study,dict(evidence_sha256=manifest))
            self.assertEqual(len(actual),6)
            self.assertEqual({r['trace']['native_cache_sha256'] for r in actual if r['variant']=='bucket_facing'},{manifest['native-yaw/same-name.npz']})
        finally:self.patcher.start()

    def test_real_archive_summary_binds_actual_video_producer(self):
        """Exercise the producer boundary: no mocked or hand-filled display trace."""
        import numpy as np
        from sircus.mjlab.gripper_yaw_video import frame_plan
        self.patcher.stop()
        try:
            rows=[];bundles=[];times=np.arange(1537)/15360
            for adapter,variants in (('native',P.VARIANTS[:2]),('native-yaw',P.VARIANTS[2:])):
                folder=self.study/adapter;folder.mkdir();members=[]
                for context in P.CASES:
                    for variant in variants:
                        r=row(context+'-'+variant,True,False,phase='diagnostic',variant=variant,
                            adapter_directory=adapter,native_cache_file='native-0000.npz',native_cache_row=len(members),
                            physics_steps=1536,native_initial_ball_center_m=[0.,0.,0.],
                            native_initial_q10=[0.]*6,native_initial_q14=[0.]*6,whole_ball_entry=False)
                        r['context_id']=context;members.append(r)
                path=folder/'native-0000.npz'
                np.savez(path,time_s=times,position_m=np.zeros((len(times),len(members),3)),
                    forces_N=np.zeros((len(times),len(members),6)),record_ids=np.asarray([r['id'] for r in members]))
                self.manifest[adapter+'/'+path.name]=P.sha(path)
                for r in members:
                    r['native_cache_sha256']=P.sha(path)
                    if r['context_id']==P.CASES[0]:
                        bundles.append(dict(row=r,label=r['variant'],record_sha256=P.digest(r),
                            cache_sha256=P.sha(path),time_s=times,events={},scored_until_s=float(times[-1])))
                rows.extend(members)
            self.study.joinpath('rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            self.manifest['rows.jsonl']=P.sha(self.study/'rows.jsonl')
            write(self.study/'evidence-manifest.json',self.manifest)
            write(self.study/'report.json',self.report);self.proof()
            snapshot=self.compact()
            for actual in snapshot['correction']['recordings']:
                original=next(r for r in rows if r['id']==actual['id'])
                self.assertEqual(actual['trace']['native_cache_sha256'],original['native_cache_sha256'])
                self.assertEqual(actual['trace']['native_cache_row'],original['native_cache_row'])
                self.assertEqual(actual['trace']['record_id'],original['id'])
            frames,clips=frame_plan(bundles,fps=10)
            movie=self.root/'producer.mp4';movie.write_bytes(b'0000ftyp0000recorded-fixture')
            manifest=self.root/'producer.json';r=snapshot['correction']
            raw=dict(version=P.VIDEO,status='completed',source=dict(report_sha256=r['report_sha256'],
                audit_sha256=r['audit']['proof_sha256'],contract_fingerprint=r['contract_fingerprint']),
                native_steps_added=0,held_arm_poses=True,sampled_ball_centers_only=True,interpolation=False,
                backend=dict(software_only=True,cuda_visible_devices=''),video_bytes=movie.stat().st_size,
                output_sha256=P.sha(movie),fps=10,duration_s=len(frames)/10,frames=frames,
                frames_sha256=P.digest(frames),clips=clips)
            write(manifest,raw)
            self.assertEqual(P.movie_summary(manifest,movie,snapshot)['context_id'],P.CASES[0])
            for field,value in (('native_cache_sha256','0'*64),('native_cache_row',999)):
                changed=copy.deepcopy(raw);changed['clips'][0][field]=value;write(manifest,changed)
                with self.assertRaisesRegex(ValueError,'exact native'):P.movie_summary(manifest,movie,snapshot)
        finally:self.patcher.start()

    def test_new_outcomes_hidden_until_exact_native_audit(self):
        for field in ('native_checks_verified','full_checks_verified'):
            self.proof();a=P.read(self.audit);a[field]=False;write(self.audit,a);s=self.compact()
            self.assertEqual(s['correction']['cases'],[]);self.assertEqual(s['correction']['counts'],{'attempts':36});self.assertEqual(s['correction']['counts_by_variant'],{});self.assertNotIn('recordings',s['correction'])
            with self.assertRaises(ValueError):P.baseline_summary(s)
        self.audit.unlink();self.assertEqual(self.compact()['correction']['audit']['status'],'pending')

    def test_stale_report_and_manifest_fail_closed(self):
        self.study.joinpath('report.json').write_text(self.study.joinpath('report.json').read_text()+' ');self.assertEqual(self.compact()['correction']['audit']['status'],'unverified')
        self.proof();write(self.study/'evidence-manifest.json',{});self.assertEqual(self.compact()['correction']['audit']['status'],'unverified')

    def test_wrong_rule_context_or_variant_rejected(self):
        for changes in (dict(mode='midpoint'),dict(contexts=list(P.CASES[:-1]))):
            r=copy.deepcopy(self.report);r['rule'].update(changes);write(self.study/'report.json',r);self.proof()
            with self.assertRaises(ValueError):self.compact()
        write(self.study/'report.json',self.report);self.spec['protocol']['variants'].append('extra');self.spec.pop('fingerprint');self.spec['fingerprint']=P.digest(self.spec);write(self.study/'contract.json',self.spec)
        with self.assertRaises(ValueError):self.compact()

    def test_historical_collision_selection_is_bound(self):
        d=P.read(self.diagnosis);d['losses'][2]['corrected_bc']['forbidden_contact_steps']=0;d.pop('diagnosis_fingerprint');d['diagnosis_fingerprint']=P.digest(d);write(self.diagnosis,d)
        with self.assertRaisesRegex(ValueError,'robot-contact'):P.checked_diagnosis(self.diagnosis)

    def test_escape_idempotence_and_preservation(self):
        s=self.compact();s['correction']['phase']='</script>*/<script>bad()</script>'
        old=self.html.read_text();h=P.embed_html(old,s);css=P.embed_css(self.css.read_text(),s)
        self.assertNotIn('<script>bad()',h);self.assertNotIn('*/<script>',css);self.assertEqual(P.embed_html(h,s),h);self.assertEqual(P.embed_css(css,s),css)
        self.assertEqual(h[:h.index(P.START)]+h[h.index(P.END)+len(P.END):],old)
        movie=P.VIDEO_START+'<video src="new.mp4"></video>'+P.VIDEO_END
        h=h.replace('<div data-gripper-yaw="movies"></div>',movie);self.assertIn(movie,P.embed_html(h,s))
        with self.assertRaises(ValueError):P.embed_html(h+P.VIDEO_START,s)

    def test_targets_and_concurrent_work_preserved(self):
        kwargs=dict(snapshot_path=self.snapshot,target_html=self.html,target_css=self.css,audit_path=self.audit,pipeline_path=self.pipeline)
        with self.assertRaises(ValueError):P.refresh(self.study,self.diagnosis,**kwargs,expected_html_sha256='0'*64)
        self.assertFalse(self.snapshot.exists());P.refresh(self.study,self.diagnosis,**kwargs,expected_html_sha256=P.sha(self.html),expected_css_sha256=P.sha(self.css))
        self.assertEqual(P.read(self.snapshot)['correction']['audit']['status'],'passed')
        for targets in [(self.diagnosis,),(self.study/'output.json',),(self.snapshot,self.snapshot)]:
            with self.assertRaises(ValueError):P.guard_targets(self.study,(self.diagnosis,),targets)

    def movie_fixture(self):
        snapshot=self.compact();clips=[];frames=[]
        for i,r in enumerate(self.records[:3]):
            clips.append(dict(record_id=r['id'],context_id=r['context_id'],variant=r['variant'],label='Recorded <case>',start_s=i,end_s=i+1,values=r['values'],bucket_xy_m=r['bucket_xy_m'],capture=False,native_release=False,no_flight=True,record_sha256=r['record_sha256'],events={}))
            frames.append(dict(frame=i,video_time_s=i,clip_index=i,record_id=r['id'],source_tick=None,source_time_s=None,segment='no_flight'))
        movie=self.root/'movie.mp4';movie.write_bytes(b'0000ftyp0000recorded-fixture');manifest=self.root/'movie.json';r=snapshot['correction']
        raw=dict(version=P.VIDEO,status='completed',source=dict(report_sha256=r['report_sha256'],audit_sha256=r['audit']['proof_sha256'],contract_fingerprint=r['contract_fingerprint']),native_steps_added=0,held_arm_poses=True,sampled_ball_centers_only=True,interpolation=False,backend=dict(software_only=True,cuda_visible_devices=''),video_bytes=movie.stat().st_size,output_sha256=P.sha(movie),fps=1,duration_s=3,frames=frames,frames_sha256=P.digest(frames),clips=clips)
        write(manifest,raw);return snapshot,manifest,movie,raw

    def test_media_bindings_and_no_invented_refusal_samples(self):
        s,meta,movie,raw=self.movie_fixture();self.assertEqual(P.movie_summary(meta,movie,s)['context_id'],P.CASES[0])
        for mutate in (lambda r:r['source'].update(report_sha256='0'*64),lambda r:r.update(native_steps_added=1),lambda r:r['clips'][0].update(record_sha256='0'*64),lambda r:r['clips'][0].update(variant='bucket_facing'),lambda r:r['backend'].update(cuda_visible_devices='0')):
            changed=copy.deepcopy(raw);mutate(changed);write(meta,changed)
            with self.assertRaises(ValueError):P.movie_summary(meta,movie,s)
        changed=copy.deepcopy(raw);changed['frames'][0]['source_tick']=0;changed['frames_sha256']=P.digest(changed['frames']);write(meta,changed)
        with self.assertRaisesRegex(ValueError,'invented'):P.movie_summary(meta,movie,s)
        with self.assertRaises(ValueError):P.movie_summary(meta,movie,s,historical=True)

    def test_unstarted_queue_is_not_a_failed_shot(self):
        (self.study/'report.json').unlink();write(self.pipeline,dict(status='waiting_for_4090'))
        s=self.compact();self.assertEqual(s['correction']['status'],'waiting_for_4090');self.assertEqual(s['correction']['counts'],{'attempts':0})
        write(self.pipeline,dict(status='failed',worker_started=False,finished_unix_s=123));s=self.compact();self.assertEqual(s['correction']['status'],'not_run');self.assertEqual(s['correction']['counts']['attempts'],0);self.assertTrue(s['publication']['terminal']);self.assertNotIn('recordings',s['correction'])

    def test_inline_movies_preserve_other_panels(self):
        s,meta,movie,_=self.movie_fixture();write(self.snapshot,s);self.html.write_text(P.embed_html(self.html.read_text(),s));before=self.html.read_text()
        out=P.publish_movies(self.snapshot,self.html,[(meta,movie)],expected_html_sha256=P.sha(self.html));h=self.html.read_text();self.assertEqual(len(out['videos']),1);self.assertIn('old.mp4',h);self.assertIn('&lt;case&gt;',h)
        self.assertEqual(h[:h.index(P.VIDEO_START)]+h[h.index(P.VIDEO_END)+len(P.VIDEO_END):],before.replace('<div data-gripper-yaw="movies"></div>',''))
        with self.assertRaises(ValueError):P.publish_movies(self.snapshot,self.html,[(meta,movie),(meta,movie)],expected_html_sha256=P.sha(self.html))

if __name__=='__main__':unittest.main()
