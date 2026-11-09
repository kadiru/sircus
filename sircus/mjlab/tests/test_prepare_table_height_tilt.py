import json
import copy
import contextlib
import importlib.util
import io
import shutil
import subprocess
from pathlib import Path
import tempfile
import unittest
from sircus.mjlab import prepare_table_height_tilt as P

class PublisherTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        self.path=self.root/'study'/'report.json';self.path.parent.mkdir()
        self.raw=dict(version=P.VERSION,status='completed',contract_fingerprint='a'*64,cases=[],admission_matrix_sha256='d'*64)
        self.write()
    def write(self):self.path.write_text(json.dumps(self.raw))
    def audit(self):
        path=P.default_audit(self.path);value=dict(version=P.AUDIT,status='passed',full_checks_verified=True,contract_fingerprint=self.raw['contract_fingerprint'],report_sha256=P.sha(self.path),checks_count=1,checks=[dict(passed=True)],failures=[]);path.write_text(json.dumps(value));return path,value
    def test_report_hash_and_nonempty_audit_required(self):
        path,value=self.audit();self.assertEqual(P.compact(self.path)['audit']['status'],'passed')
        self.path.write_text(self.path.read_text()+'\n');self.assertEqual(P.compact(self.path)['audit']['status'],'unverified')
        self.write();value['checks']=[];value['checks_count']=0;path.write_text(json.dumps(value));self.assertEqual(P.compact(self.path)['audit']['status'],'unverified')
    def test_private_paths_excluded_and_null_discovery_preserved(self):
        self.raw['cases']=[dict(context_id='case',local_path='/home/private',grid=[dict(candidate_id='c1',grid_index=0,values=[.1,.2,.32,16,.3],local_path='/home/private',admission=dict(passed=False,reason='finite_ik_unresolved',local_file='/home/private'),discovery=None)],selection=None,repeats={})]
        self.write();out=P.compact(self.path);self.assertNotIn('/home/private',json.dumps(out));self.assertIsNone(out['cases'][0]['grid'][0]['discovery']);self.assertIsNone(out['cases'][0]['selection']);self.assertEqual(out['admission_matrix_sha256'],'d'*64)
    def test_actual_report_field_names_and_both_grid_indices_preserved(self):
        self.raw.update(cpu_elapsed_s=20.,native_elapsed_s=40.,native_requested=31,requested_configurations=150,
            budget=dict(maximum_requested=162,candidate_proposals=120,original_requests=12,control_requests=18,maximum_selected_repeat_requests=12),
            admission_summary=dict(proposals=120,completed=120,admitted=1,declined=119,unattempted=0),
            cases=[dict(context_id='table-training-test-18',grid=[dict(candidate_id='g60',grid_index=60,case_grid_index=0,values=[0,0,.32,16,.3],admission=dict(passed=True),discovery=None)],selection=dict(candidate_id='g60',grid_index=60,case_grid_index=0,row_sha256='b'*64,connection_sha256='c'*64,values=[0,0,.32,16,.3],route_quality=1,learning_reward=3.),repeats={})])
        self.write();out=P.compact(self.path)
        self.assertEqual(out['budget']['maximum_requested'],162)
        self.assertEqual(out['admission_summary']['declined'],119)
        self.assertEqual(out['cases'][0]['grid'][0]['grid_index'],60)
        self.assertEqual(out['cases'][0]['grid'][0]['case_grid_index'],0)
        self.assertEqual(out['cases'][0]['selection']['row_sha256'],'b'*64)
        self.assertEqual(out['native_requested'],31)
    def test_existing_diagnosis_html_preserved_and_embed_escaped(self):
        before='<html><body><!--TABLE_TRAINING_DIAGNOSIS_START--><div>existing</div><!--TABLE_TRAINING_DIAGNOSIS_END--></body></html>'
        after=P.embed_html(before,dict(description='</script><script>bad()</script>'))
        self.assertIn('<!--TABLE_TRAINING_DIAGNOSIS_START--><div>existing</div><!--TABLE_TRAINING_DIAGNOSIS_END-->',after);self.assertNotIn('<script>bad()',after)
        again=P.embed_html(after,{})
        self.assertEqual(again.count(P.START),1);self.assertEqual(again[:again.index(P.START)],after[:after.index(P.START)])
    def test_compare_and_swap_preserves_concurrent_change(self):
        html=self.root/'index.html';html.write_text('<html><body>existing</body></html>')
        with self.assertRaisesRegex(ValueError,'changed'):P.install(html,{},'0'*64)
        self.assertEqual(html.read_text(),'<html><body>existing</body></html>')
    def test_old_report_keys_preserved(self):
        display=self.root/'display.json';display.write_text(json.dumps(dict(table_training_diagnosis={'old':'proof'},other=[1,2])))
        P.publish(self.path,display);out=json.loads(display.read_text());self.assertEqual(out['table_training_diagnosis'],{'old':'proof'});self.assertEqual(out['other'],[1,2]);self.assertIn(P.KEY,out)
    def test_publication_cannot_overwrite_study_artifacts(self):
        with self.assertRaisesRegex(ValueError,'outside immutable'):P.publish(self.path,self.path)
        self.assertEqual(json.loads(self.path.read_text()),self.raw)
    @unittest.skipUnless(importlib.util.find_spec('numpy') and shutil.which('node'), 'Uses existing numerical Python and Node; no native simulation')
    def test_actual_producer_report_compacts_into_verified_ui(self):
        # Fixture-only accounting integration. Never published or run on a GPU.
        from sircus.mjlab import height_tilt_contract as C, height_tilt_study as S, table_assisted_score as score
        from sircus.mjlab.tests.test_table_correction_native import fixture
        sources=C.sources_from_parent(fixture()[0]);plan=C.grid_plan(sources);matrix=C.empty_matrix(plan)
        for index,cell in enumerate(matrix):
            cell.update(status='completed',passed=index==0,reason=None if index==0 else 'finite_ik_unresolved',
                valid_ik_branches=[1,1] if index==0 else [1,0],connection_sha256='e'*64,resolution_sha256='f'*64)
        def observed(request,capture=False,table=False):
            row=copy.deepcopy(request);row.update(native_release=True,route_success=capture,invalid=None,invalid_physics=False,
                route_quality=2 if capture else 0,direct_capture=capture and not table,table_assisted_capture=capture and table,
                diagnostic_robot_contact_steps=0,forbidden_contact_steps=0,table_contact_steps=int(table),board_bounce=True,
                closest_opening_distance_m=.01 if capture else .2,max_retention_s=1.1 if capture else 0.,
                single_board_bounce_success=capture,scoring_version=score.VERSION,route_version=score.ROUTE_VERSION,
                reward_version=score.REWARD_VERSION,connection_file='fixture-only.json',connection_sha256='e'*64,q10=[0.]*6,q14=[0.]*6)
            row['learning_reward']=score.reward(row);return row
        rows=[observed(r,r['role']!='negative_control',r['role']=='table_control') for r in C.fixed_requests(sources,'controls_pre')]
        rows += [observed(r) for r in C.fixed_requests(sources,'baseline')]
        rows += [observed(plan[0],True)]
        selection=C.select_captures(rows);rows += [observed(r,True) for r in C.selected_requests(plan,selection)]
        rows += [observed(r,r['role']!='negative_control',r['role']=='table_control') for r in C.fixed_requests(sources,'controls_post')]
        gate=dict(status='passed',counts=dict(attempts=9,native_releases=9,captures=6,setup_rejections=0,native_misses=3),direct_captures=3,table_assisted_captures=3,negative_captures=0)
        spec=dict(fingerprint='a'*64,admission_matrix_sha256='d'*64,parent_bindings=dict(report_sha256='4605bcb801351ae91ad16b71a033ceb3866027db621f9d84227f23b1ef257e95',audit_sha256='6e50d87ddb7b2d4b37debbc2a5ac96507ff40bc236223277121acd385b8ec91b'))
        (self.path.parent/'parent-rows.json').write_text(json.dumps(sources))
        with contextlib.redirect_stdout(io.StringIO()):
            self.raw=S.report_state(self.path.parent,spec,matrix,rows,selection,status='completed',phase='completed',cpu_elapsed_s=1.,native_elapsed_s=2.,gates=dict(pre=gate,post=gate))
        self.audit();snapshot=P.compact(self.path)
        script="import fs from 'node:fs'; import {studyReady} from './sircus/mjlab/web/table-height-tilt.js'; const d=JSON.parse(fs.readFileSync(0,'utf8')); if(!studyReady(d))throw Error('Actual producer report is not renderable'); if(d.cases[3].grid[0].grid_index!==90||d.cases[3].grid[0].case_grid_index!==0)throw Error('Index mapping changed'); if(d.repeats.selected.attempts!==3)throw Error('Selection denominator changed');"
        result=subprocess.run(['node','--input-type=module','-e',script],input=json.dumps(snapshot),text=True,capture_output=True,cwd=Path(__file__).resolve().parents[3])
        self.assertEqual(result.returncode,0,result.stderr)
    def test_unknown_version_and_nonfinite_values_refused(self):
        self.raw['version']='other';self.write()
        with self.assertRaises(ValueError):P.compact(self.path)
        self.raw['version']=P.VERSION;self.raw['cases']=[dict(baseline_values=[0,0,float('nan'),32,.3])];self.write()
        with self.assertRaises(ValueError):P.compact(self.path)
if __name__=='__main__':unittest.main()
