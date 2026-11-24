"""New replay publication binds paired records and preserves every old panel."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sircus.mjlab import expanded_selector_replay_publish as pub
from sircus.mjlab.benchmark_verify import sha


class ReplayPublishTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.study=self.root/'study';self.study.mkdir()
        self.replay=self.root/'replays';self.replay.mkdir()
        self.media=self.replay/'media';self.media.mkdir()
        self.diag=self.replay/'diagnosis.json';self.audit=self.root/'audit.json'
        self.display=self.root/'display.json';self.index=self.root/'index.html'
        self.audit.write_text('{}')
        self.rows=[];self.slots=[];self.cases=[];self.layouts=[]
        for i,case_id in enumerate(pub.CASES):
            xy=[i/10,0.];lost=i in (0,2);methods={};self.layouts.append(dict(id=case_id,bucket_xy_m=xy))
            for method in pub.METHODS:
                identifier=case_id+'-'+method;v=[.4,0.,.3,28.,.35];q=[0.]*6
                request=dict(id=identifier,phase="test",values=v,bucket_xy_m=xy,selected_q10=q,selected_q14=q)
                trace=self.study/(identifier+'.npz');trace.write_bytes(identifier.encode())
                capture=lost if method=='current' else not lost
                score=dict(route_success=capture,invalid_physics=False,route_class='direct' if capture else 'miss',max_retention_s=1. if capture else 0.)
                row=dict(request,phase="test_"+method,calibrated_geometry_sha256="a"*64,effective_options_sha256="b"*64,board_center_m=v[:3],board_heading_offset_deg=0.,board_twist_deg=0.,release_ball_center_m=[.4,0.,.68],native_initial_q10=q,native_initial_q14=q,
                    adapter_directory='.',native_cache_file=trace.name,native_cache_sha256=sha(trace),native_cache_row=0,physics_steps=100,board_first_score=score)
                self.rows.append(row);self.slots.append(dict(id=identifier,bucket_xy_m=xy,request=request))
                s=dict(command_board_center_m=v[:3],board_center_m=v[:3],board_height_m=v[2],tilt_deg=v[3],drop_gap_m=v[4],heading_offset_deg=0.,twist_deg=0.,release_ball_center_m=row['release_ball_center_m'],q10_rad=q,q14_rad=q)
                methods[method]=dict(id=identifier,selection={},settings=s,outcome=dict(capture=capture,**score),trace=dict(relative_file='./'+trace.name,sha256=sha(trace),column=0,terminal_tick=100,duration_s=1.,time_step_s=.01))
            self.cases.append(dict(id=case_id,bucket_xy_m=xy,change='expanded_lost' if lost else 'expanded_gained',methods=methods,findings=['Observed outcome'],hypotheses=['Untested idea']))
        for name,value in [('report.json',{}),('contract.json',{}),('layouts.json',dict(layouts=self.layouts)),('choices.json',dict(slots=self.slots))]:self.write(self.study/name,value)
        (self.study/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        manifest={p.name:sha(p) for p in self.study.iterdir()}
        manifest.update({'./'+p.name:sha(p) for p in self.study.glob('*.npz')})
        self.write(self.study/'evidence-manifest.json',manifest)
        self.diagnosis=dict(version=pub.DIAGNOSIS_VERSION,status='completed',cpu_only=True,physics_substeps=64,checks_count=100,cases=self.cases,
            diagnostic_source_sha256=sha(Path(pub.__file__).with_name('expanded_selector_diagnosis.py')),verified_artifacts=manifest,
            source={k:sha(p) for k,p in [('report_sha256',self.study/'report.json'),('audit_sha256',self.audit),('manifest_sha256',self.study/'evidence-manifest.json'),('choices_sha256',self.study/'choices.json'),('contract_sha256',self.study/'contract.json')]})
        self.write(self.diag,self.diagnosis)
        self.mock=patch.object(pub,'parent_display',return_value={'methods':[{'id':m,'captures':44} for m in pub.METHODS]});self.parent=self.mock.start()
        self.index.write_text('<script id="oldVideos">{"keep":1}</script><script type="application/json" id="expandedSelectorReplayVideoAssets">{}</script><p>keep</p>')
        self.write(self.display,{'planner':'hybrid','expanded_selector_study':{'keep':True},'historical_video':'keep'})

    def tearDown(self):self.mock.stop();self.tmp.cleanup()
    def write(self,path,value):path.write_text(json.dumps(value))
    def compose(self):return pub.compose(self.study,self.audit,self.diag)
    def add_movie(self,case):
        stem='expanded-selector-case-'+case['id'][-2:];movie=self.media/(stem+'.mp4')
        movie.write_bytes(b'\x00\x00\x00\x18ftypisom'+b'\x00'*24)
        receipt=dict(version="sircus-expanded-selector-recorded-pairs-v1",renderer_source_sha256=sha(Path(pub.__file__).with_name("expanded_selector_video.py")),playback_speed=.25,fps=30,width=1280,height=720,interpolation=False,sampled_ball_centers_only=True,ball_spin_recorded=False,historical_labels_unchanged=True,duration_s=165/30,frames=[dict(frame=i,video_time_s=i/30,segment=segment,playback_speed=.25 if segment=="slow" else 0,common_time_s=t,methods={m:dict(source_tick=round(t/.01),source_time_s=round(t/.01)*.01,terminal_hold=t>=1.) for m in pub.METHODS}) for i,(t,segment) in enumerate([(0.,"title")]*15+[(i/30*.25,"slow") for i in range(120)]+[(1.,"outcome")]*30)],status='completed',case_id=case['id'],native_steps_added=0,held_arm_poses=True,video_bytes=movie.stat().st_size,output_sha256=sha(movie),
            methods={m:dict(record_id=case['methods'][m]['id'],native_archive_sha256=case['methods'][m]['trace']['sha256'],native_column=0,record_sha256=case["methods"][m]["record_sha256"],input_sha256=case["methods"][m]["video_source_hashes"],calibrated_geometry_sha256="a"*64,capture=case["methods"][m]["outcome"]["capture"],scoring_version="sircus-board-first-native-capture-v4",q10=[0.]*6,q14=[0.]*6,bucket_xy_m=case["bucket_xy_m"],values=[.4,0.,.3,28.,.35],source_interval_s=[0,1.]) for m in pub.METHODS})
        path=movie.with_suffix('.json');self.write(path,receipt);return movie,path,receipt

    def test_original_50_position_audit_is_required_and_all_eight_settings_are_bound(self):
        section=self.compose();self.assertEqual(len(section['cases']),4)
        self.parent.assert_called_once_with(self.study,self.audit)
        self.assertEqual(section['parent_summary']['positions'],50)
        for mutate in [lambda d:d['cases'][0]['methods']['current']['settings']['q10_rad'].__setitem__(0,.1),lambda d:d['cases'][0]['methods']['expanded']['outcome'].__setitem__('capture',True),lambda d:d['cases'][0]['methods']['current']['trace'].__setitem__('column',1),lambda d:d['source'].__setitem__('audit_sha256','0'*64),lambda d:d.__setitem__('diagnostic_source_sha256','0'*64)]:
            d=copy.deepcopy(self.diagnosis);mutate(d);self.write(self.diag,d)
            with self.assertRaises(ValueError):self.compose()
        self.write(self.diag,self.diagnosis)
        with patch.object(pub,'parent_display',side_effect=ValueError('parent audit changed')):
            with self.assertRaises(ValueError):self.compose()

    def test_changed_native_evidence_rejected(self):
        path=next(self.study.glob('*.npz'));path.write_bytes(b'changed')
        with self.assertRaises(ValueError):self.compose()

    def test_video_is_bound_to_both_record_ids_columns_and_bytes(self):
        section=self.compose();movie,path,receipt=self.add_movie(section['cases'][0])
        for case in section["cases"][1:]:self.add_movie(case)
        assets=pub.media_assets(section,self.media);self.assertEqual(len(assets),4)
        for mutate in [lambda r:r.__setitem__('native_steps_added',1),lambda r:r.__setitem__('video_bytes',1),lambda r:r['methods']['expanded'].__setitem__('native_column',2),lambda r:r['methods']['current'].__setitem__('record_id','swapped'),lambda r:r.__setitem__('output_sha256','0'*64),lambda r:r["methods"]["current"]["q10"].__setitem__(0,1.),lambda r:r["methods"]["expanded"].__setitem__("capture",True),lambda r:r.__setitem__("playback_speed",1.),lambda r:r["frames"][0]["methods"]["current"].__setitem__("source_tick",10)]:
            r=copy.deepcopy(receipt);mutate(r);self.write(path,r)
            with self.assertRaises(ValueError):pub.media_assets(self.compose(),self.media)
        path.unlink()
        with self.assertRaises(ValueError):pub.media_assets(self.compose(),self.media)

    def test_publish_preserves_other_panels_and_keeps_video_bytes_out_of_polled_json(self):
        for case in self.compose()["cases"]:self.add_movie(case)
        before=json.loads(self.display.read_text())
        result=pub.publish(self.study,self.audit,self.diag,self.display,self.index,self.media)
        self.assertEqual(result,dict(status='completed',cases=4,videos=4))
        after=json.loads(self.display.read_text());self.assertEqual({k:after[k] for k in before},before)
        self.assertNotIn('base64',self.display.read_text())
        self.assertIn('data:video/mp4;base64,',self.index.read_text())
        self.assertIn('<script id="oldVideos">{"keep":1}</script>',self.index.read_text())
        self.assertTrue(self.index.read_text().endswith('<p>keep</p>'))

    def test_fail_before_publication_if_media_corrupt_or_placeholder_ambiguous(self):
        movie,_,_=self.add_movie(self.compose()["cases"][0]);movie.write_bytes(b'wrong')
        before=(self.display.read_bytes(),self.index.read_bytes())
        with self.assertRaises(ValueError):pub.publish(self.study,self.audit,self.diag,self.display,self.index,self.media)
        self.assertEqual((self.display.read_bytes(),self.index.read_bytes()),before)
        for html in ('<p>no owned placeholder</p>',self.index.read_text()*2):
            with self.assertRaises(ValueError):pub.embed_html(html,{})

    def test_outputs_cannot_overwrite_source_or_media(self):
        for output in (self.diag,self.study/'report.json',self.audit,self.media/'display.json'):
            with self.assertRaises(ValueError):pub.publish(self.study,self.audit,self.diag,output,self.index)
        with self.assertRaises(ValueError):pub.publish(self.study,self.audit,self.diag,self.display,self.display)

    def test_json_embedding_escapes_html_script_terminators(self):
        html=pub.embed_html(self.index.read_text(),{'label':'</script><script>bad</script>'})
        self.assertNotIn('<script>bad',html)
        self.assertIn('\\u003c/script>',html)


if __name__=='__main__':unittest.main()
