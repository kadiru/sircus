import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sircus.mjlab import prepare_imitation_regression as P


def write(path,value):Path(path).write_text(json.dumps(value,allow_nan=False))
def row(identifier,native=False,capture=False,**extra):
    return dict(id=identifier,record_id=identifier,context_id=P.CASES[0],bucket_xy_m=[.7,-.3],values=[.5,-.2,.4,28,.2],native_release=native,capture=capture,route_success=capture,invalid=None if native else 'finite_ik_unresolved',invalid_physics=False,repeat=0,**extra)


class PublisherTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.study=self.root/'study';self.study.mkdir();self.audit=self.root/'audit.json';self.diagnosis=self.root/'diagnosis.json';self.pipeline=self.root/'pipeline.json'
        d=dict(version=P.DIAGNOSIS,status='completed',source=dict(checks_count=55633,full_checks_verified=True,native_checks_verified=True,report_sha256='a'*64,audit_sha256='b'*64,contract_fingerprint='c'*64),losses=[])
        for i,c in enumerate(P.CASES):
            old=row('old-'+c,True,True);new=row('new-'+c,i>=3,False)
            old['context_id']=new['context_id']=c
            d['losses'].append(dict(context_id=c,bucket_xy_m=[.7,-.3],current_bc=old,corrected_bc=new,delta_values=[.01,0,.03,1,-.02]))
        d['diagnosis_fingerprint']=P.digest(d);write(self.diagnosis,d)
        protocol=dict(substeps=64,variants=list(P.VARIANTS));self.spec=dict(protocol=protocol);self.spec['fingerprint']=P.digest(self.spec);write(self.study/'contract.json',self.spec)
        self.manifest={'contract.json':P.sha(self.study/'contract.json')};write(self.study/'evidence-manifest.json',self.manifest)
        self.rule=dict(mode='matched_current_action_midpoint',interpolation_alpha=.5,parameter_indices=[0,1,2,3,4],contexts=list(P.CASES),diagnosis_sha256=P.sha(self.diagnosis))
        self.report=dict(version=P.STUDY,status='completed',phase='completed',expected_requests=72,contract_fingerprint=self.spec['fingerprint'],evidence_sha256=self.manifest,rule=self.rule,counts=dict(attempts=72,captures=48),counts_by_variant={v:dict(attempts=18,captures=18 if v!='corrected_exact' else 0) for v in P.VARIANTS},cases=[dict(context_id=c,variant_values={v:[.5,-.2,.4,28,.2] for v in P.VARIANTS}) for c in P.CASES],control_gates={k:dict(status='passed') for k in ('pre','post')})
        write(self.study/'report.json',self.report);self.proof();write(self.pipeline,dict(status='completed',finished_unix_s=123))
        self.records=[]
        for c in P.CASES:
            for v in P.VARIANTS:
                r=row(c+v,variant=v);r['context_id']=c;r['record_sha256']=P.digest(r);self.records.append(r)
        self.patcher=patch.object(P,'comparison_recordings',return_value=self.records);self.patcher.start();self.addCleanup(self.patcher.stop)
        self.html=self.root/'index.html';self.css=self.root/'style.css';self.snapshot=self.root/'snapshot.json'
        self.html.write_text('<html><body><video src="old.mp4"></video><!--OLD_PANEL-->history</body></html>');self.css.write_text('body{color:white}/*old*/')

    def proof(self):
        write(self.audit,dict(version=P.AUDIT,status='passed',full_checks_verified=True,native_checks_verified=True,failures=[],checks=[dict(passed=True)],checks_count=1,report_sha256=P.sha(self.study/'report.json'),evidence_manifest_sha256=P.sha(self.study/'evidence-manifest.json'),contract_fingerprint=self.spec['fingerprint']))

    def compact(self):return P.compact(self.study,self.diagnosis,audit_path=self.audit,pipeline_path=self.pipeline,now=150)

    def test_six_historical_pairs_and_three_variants_with_all_requests(self):
        s=self.compact();self.assertEqual(len(s['diagnosis']['losses']),6);self.assertEqual(len(s['correction']['recordings']),18);self.assertEqual(s['correction']['audit']['status'],'passed');self.assertEqual(P.baseline_summary(s)['expected_requests'],72)

    def test_actual_producer_protocol_schema_is_consumed(self):
        from sircus.mjlab.imitation_regression_contract import protocol
        self.spec['protocol']=protocol();unsigned=dict(self.spec);unsigned.pop('fingerprint');self.spec['fingerprint']=P.digest(unsigned)
        write(self.study/'contract.json',self.spec);self.manifest['contract.json']=P.sha(self.study/'contract.json');write(self.study/'evidence-manifest.json',self.manifest)
        self.report['contract_fingerprint']=self.spec['fingerprint'];write(self.study/'report.json',self.report);self.proof()
        out=self.compact();self.assertEqual(out['correction']['audit']['status'],'passed');self.assertEqual(tuple(self.spec['protocol']['variants']),P.VARIANTS);self.assertEqual(tuple(self.spec['protocol']['contexts']),P.CASES)

    def test_new_outcomes_and_recordings_hidden_until_exact_native_audit(self):
        for field in ('native_checks_verified','full_checks_verified'):
            self.proof();a=P.read(self.audit);a[field]=False;write(self.audit,a);s=self.compact()
            self.assertEqual(s['correction']['cases'],[]);self.assertEqual(s['correction']['counts'],{'attempts':72});self.assertEqual(s['correction']['counts_by_variant'],{});self.assertNotIn('recordings',s['correction'])
            with self.assertRaises(ValueError):P.baseline_summary(s)
        self.audit.unlink();self.assertEqual(self.compact()['correction']['audit']['status'],'pending')

    def test_stale_report_manifest_and_diagnosis_fail_closed(self):
        self.study.joinpath('report.json').write_text(self.study.joinpath('report.json').read_text()+' ');self.assertEqual(self.compact()['correction']['audit']['status'],'unverified')
        self.proof();write(self.study/'evidence-manifest.json',{});self.assertEqual(self.compact()['correction']['audit']['status'],'unverified')
        d=P.read(self.diagnosis);d['losses'][0]['context_id']='different';write(self.diagnosis,d)
        with self.assertRaises(ValueError):self.compact()

    def test_no_extra_variant_or_changed_correction_rule(self):
        for changes in (dict(mode='other'),dict(interpolation_alpha=.6),dict(parameter_indices=[2]),dict(contexts=list(P.CASES[:-1]))):
            r=copy.deepcopy(self.report);r['rule'].update(changes);write(self.study/'report.json',r);self.proof()
            with self.assertRaises(ValueError):self.compact()

    def test_fingerprint_bound_six_loss_grouping(self):
        d=P.read(self.diagnosis);d['losses'][0]['current_bc']['capture']=False;d.pop('diagnosis_fingerprint');d['diagnosis_fingerprint']=P.digest(d);write(self.diagnosis,d)
        with self.assertRaisesRegex(ValueError,'loss pair'):P.checked_diagnosis(self.diagnosis)

    def test_additive_escaping_idempotence_and_movies_survive_refresh(self):
        s=self.compact();s['correction']['phase']='</script>*/<script>bad()</script>'
        old=self.html.read_text();h=P.embed_html(old,s);css=P.embed_css(self.css.read_text(),s)
        self.assertNotIn('<script>bad()',h);self.assertNotIn('*/<script>',css)
        self.assertEqual(P.embed_html(h,s),h);self.assertEqual(P.embed_css(css,s),css)
        self.assertEqual(h[:h.index(P.START)]+h[h.index(P.END)+len(P.END):],old)
        movie=P.VIDEO_START+'<video src="new.mp4"></video>'+P.VIDEO_END
        h=h.replace('<div data-imitation-regression="movies"></div>',movie);h2=P.embed_html(h,s)
        self.assertIn(movie,h2);self.assertIn('old.mp4',h2)
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
        movie=self.root/'movie.mp4';movie.write_bytes(b'0000ftyp0000recorded-fixture')
        manifest=self.root/'movie.json';r=snapshot['correction']
        raw=dict(version=P.VIDEO,status='completed',source=dict(report_sha256=r['report_sha256'],audit_sha256=r['audit']['proof_sha256'],contract_fingerprint=r['contract_fingerprint']),native_steps_added=0,held_arm_poses=True,sampled_ball_centers_only=True,interpolation=False,backend=dict(software_only=True,cuda_visible_devices=''),video_bytes=movie.stat().st_size,output_sha256=P.sha(movie),fps=1,duration_s=3,frames=frames,frames_sha256=P.digest(frames),clips=clips)
        write(manifest,raw);return snapshot,manifest,movie,raw

    def test_media_bindings_and_no_invented_refusal_frames(self):
        s,meta,movie,raw=self.movie_fixture();self.assertEqual(P.movie_summary(meta,movie,s)['context_id'],P.CASES[0])
        for mutate in (lambda r:r['source'].update(report_sha256='0'*64),lambda r:r.update(native_steps_added=1),lambda r:r['clips'][0].update(record_sha256='0'*64),lambda r:r['clips'][0].update(variant='blend50'),lambda r:r['backend'].update(cuda_visible_devices='0')):
            changed=copy.deepcopy(raw);mutate(changed);write(meta,changed)
            with self.assertRaises(ValueError):P.movie_summary(meta,movie,s)
        changed=copy.deepcopy(raw);changed['frames'][0]['source_tick']=0;changed['frames_sha256']=P.digest(changed['frames']);write(meta,changed)
        with self.assertRaisesRegex(ValueError,'invented'):P.movie_summary(meta,movie,s)
        write(meta,raw);movie.write_bytes(movie.read_bytes()+b'extra')
        with self.assertRaisesRegex(ValueError,'checksum'):P.movie_summary(meta,movie,s)

    def test_queue_phase_reflects_actual_pipeline_without_inventing_results(self):
        (self.study/'report.json').unlink();write(self.pipeline,dict(status='waiting_for_4090'))
        s=self.compact();self.assertEqual(s['correction']['status'],'waiting_for_4090');self.assertEqual(s['correction']['counts'],{'attempts':0});self.assertIn('no new shots',s['correction']['phase']);self.assertEqual(len(s['diagnosis']['losses']),6)

    def test_historical_movies_bind_parent_while_correction_is_pending(self):
        s,meta,movie,raw=self.movie_fixture();s['correction']=dict(status='waiting_for_4090',audit={'status':'pending'})
        pair=s['diagnosis']['losses'][0];frames=[];clips=[]
        for i,(method,variant) in enumerate((('current_bc','current_exact'),('corrected_bc','corrected_exact'))):
            q=pair[method];q['record_sha256']=P.digest(q)
            clip=dict(record_id=q['record_id'],context_id=q['context_id'],variant=variant,label='Historical original',start_s=i,end_s=i+1,values=q['values'],bucket_xy_m=q['bucket_xy_m'],capture=q['capture'],native_release=q['native_release'],no_flight=not q['native_release'],record_sha256=q['record_sha256'],events={})
            if q['native_release']:q['trace']={'native_cache_sha256':'c'*64};clip['native_cache_sha256']='c'*64
            clips.append(clip);frames.append(dict(frame=i,video_time_s=i,clip_index=i,record_id=q['record_id'],source_tick=0 if q['native_release'] else None,source_time_s=0. if q['native_release'] else None,segment='normal' if q['native_release'] else 'no_flight'))
        raw.update(version=P.HISTORY_VIDEO,source=P.take(s['diagnosis']['source'],('report_sha256','audit_sha256','contract_fingerprint')),frames=frames,frames_sha256=P.digest(frames),clips=clips,duration_s=2)
        write(meta,raw);self.assertEqual(len(P.movie_summary(meta,movie,s,historical=True)['clips']),2)
        with self.assertRaisesRegex(ValueError,'Audited correction'):P.movie_summary(meta,movie,s)
        write(self.snapshot,s);self.html.write_text(P.embed_html(self.html.read_text(),s))
        P.publish_movies(self.snapshot,self.html,[(meta,movie)],expected_html_sha256=P.sha(self.html),historical=True)
        h=self.html.read_text();self.assertIn('Original historical TEST pair; not a new correction result.',h);self.assertIn(P.HISTORY_START,h);self.assertNotIn(P.VIDEO_START,h)
        refreshed=P.embed_html(h,s);self.assertIn(P.HISTORY_START,refreshed);self.assertIn('data:video/mp4;base64,',refreshed);self.assertIn('old.mp4',refreshed)
        raw['source']['report_sha256']='f'*64;write(meta,raw)
        with self.assertRaisesRegex(ValueError,'another report'):P.movie_summary(meta,movie,s,historical=True)

    def test_inline_movies_are_escaped_and_preserve_old_video(self):
        s,meta,movie,_=self.movie_fixture();write(self.snapshot,s);self.html.write_text(P.embed_html(self.html.read_text(),s))
        before=self.html.read_text();out=P.publish_movies(self.snapshot,self.html,[(meta,movie)],expected_html_sha256=P.sha(self.html));h=self.html.read_text()
        self.assertEqual(len(out['videos']),1);self.assertIn('old.mp4',h);self.assertIn('data:video/mp4;base64,',h);self.assertIn('&lt;case&gt;',h);self.assertNotIn('Recorded <case>',h)
        self.assertEqual(h[:h.index(P.VIDEO_START)]+h[h.index(P.VIDEO_END)+len(P.VIDEO_END):],before.replace('<div data-imitation-regression="movies"></div>',''))
        with self.assertRaises(ValueError):P.publish_movies(self.snapshot,self.html,[(meta,movie),(meta,movie)],expected_html_sha256=P.sha(self.html))
        self.assertEqual(self.html.read_text(),h)


if __name__=='__main__':unittest.main()
