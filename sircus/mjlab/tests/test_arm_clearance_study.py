"""Synthetic accounting and safety checks; no hardware/native GPU calls."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from sircus.mjlab import arm_clearance_study as study
from sircus.mjlab import arm_clearance_verify as audit
from sircus.mjlab import physics_ensemble_study as old

class FrozenStudyTests(unittest.TestCase):
    def test_split_repeatability_and_independent_regeneration(self):
        previous=dict(excluded=[[.1,-.65],[1.24,.25]],splits=dict(test=[dict(bucket_xy_m=[.55,-.2])]))
        splits,excluded=study.fresh_splits(previous)
        expected=audit.expected_splits(previous)
        self.assertEqual((splits,excluded),(expected['splits'],expected['excluded']))
        self.assertNotEqual(study.SEED,20261801)
        xy=np.array([c['bucket_xy_m'] for c in splits['test']])
        self.assertEqual(len(xy),50)
        self.assertGreaterEqual(np.linalg.norm(xy[:,None]-np.array(excluded),axis=-1).min(),.01)
    def execute_mock(self,foreign=False,tamper=False,abstain=False):
        temp=TemporaryDirectory();self.addCleanup(temp.cleanup);out=Path(temp.name);native=out/'native';native.mkdir()
        (out/'prediction-plan.json').write_text('{}');frozen=study.sha(out/'prediction-plan.json')
        (out/'model.pt').write_bytes(b'model');artifact=dict(path='model.pt',sha256=study.sha(out/'model.pt'))
        connection=dict(passed=True,endpoint_eligible=True,q10=[0.]*6,q14=[0.]*6)
        (native/'connection.json').write_text(json.dumps(connection))
        check=dict(candidate_index=0,passed=True,connection_file='connection.json',connection_sha256=study.sha(native/'connection.json'),clearance=dict(passed=False))
        requests=[];rankings=[]
        for phase in ('controls_pre','test','controls_post'):
            if phase=='test':
                for m in study.METHODS:
                    for i in range(50):requests.append(dict(id=f'{m}{i}',role='test',phase=phase,method=m,context_id=f'c{i}',bucket_xy_m=[.5,-.2],values=[.6,-.3,.3,30,.2],policy_abstain=abstain and m=='arm_clearance'))
                rankings=[dict(context_id=f'c{i}',selection=dict(arm_clearance=dict(selected_index=0,checks=[check]))) for i in range(50)]
            else:
                for role in ('positive_control','negative_control'):
                    for i in range(3):requests.append(dict(id=f'{phase}{role}{i}',role=role,phase=phase,method='control'))
        calls=[]
        class Engine:
            def evaluate(self,rs,phase):
                selfassert=any(r.get('policy_abstain') for r in rs)
                if selfassert:raise AssertionError('Abstention must never enter native engine')
                calls.append((phase,[r['id'] for r in rs]));rows=[]
                for r in rs:
                    success=r['role']!='negative_control'
                    rows.append(dict(r,native_release=True,route_success=success,route_quality=2 if success else 0,
                        invalid=None,invalid_physics=False,scalar_score_audit_passed=True,whole_ball_entry=success,max_retention_s=1.2 if success else 0))
                return rows,dict(wall_s=0)
        state=dict(output=out,engine=Engine(),report=dict(artifacts={m:artifact for m in study.METHODS},control_gates={},comparisons=[],test_positions=[]),requests=requests,prediction_sha256=frozen,rankings=rankings)
        if tamper:(out/'prediction-plan.json').write_text('{"changed":true}')
        api=SimpleNamespace(gpu=SimpleNamespace(foreign_processes=lambda:[123] if foreign else []),reward=lambda r:-1 if r.get('invalid') else int(r['route_success']))
        with patch.object(study.shutil,'disk_usage',return_value=SimpleNamespace(free=20*1024**3)):
            if foreign or tamper:
                with self.assertRaises((RuntimeError,ValueError)):study.execute(state,api,old,SimpleNamespace(wall_time_limit=3600))
                self.assertEqual(calls,[])
            else:
                study.execute(state,api,old,SimpleNamespace(wall_time_limit=3600))
                self.assertEqual(sum(len(ids) for _,ids in calls),62 if abstain else 112)
                self.assertEqual(state['report']['status'],'completed')
                self.assertEqual(state['report']['counts']['requested'],112)
                self.assertEqual(len(state['report']['test_positions']),100)
                self.assertEqual([r['counts']['requested'] for r in state['report']['comparisons']],[50]*2)
                rows=[json.loads(line) for line in (out/'records.jsonl').read_text().splitlines()]
                self.assertEqual([r['id'] for r in rows],[r['id'] for r in requests])
                if abstain:
                    rejected=[json.loads(line) for line in (out/'policy-rejections.jsonl').read_text().splitlines()]
                    self.assertEqual(len(rejected),50)
                    self.assertTrue(all(not r['native_release'] and r['max_retention_s'] is None and r['external_contact_steps'] is None and r['invalid_physics'] is None for r in rejected))
                    self.assertEqual(rejected,[r for r in rows if r.get('policy_abstain')])
                    from sircus.mjlab import calibrated_benchmark_verify as calibrated
                    for row in rejected:row['learning_reward']=calibrated.row_reward(row)
                    calibrated.check_rows(rejected,audit.AuditChecks())
                    self.assertEqual(calibrated.rescore_rows(native,rejected,{}, {},audit.AuditChecks(),'unexecuted',{'native_dt_s':1/15360,'horizon_s':4}),{})
    def test_bounded_mock_execution(self):self.execute_mock()
    def test_foreign_job_no_execution(self):self.execute_mock(foreign=True)
    def test_tampered_plan_no_execution(self):self.execute_mock(tamper=True)
    def test_abstentions_preserve112_slots_and_no_native_calls(self):self.execute_mock(abstain=True)
    def test_cannot_invent_abstention_for_clear_or_unbound_endpoint(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);(root/'c.json').write_text('{}')
            request=dict(policy_abstain=True,method='arm_clearance')
            entry=dict(candidate_index=0,passed=True,clearance=dict(passed=True),connection_file='c.json',connection_sha256=study.sha(root/'c.json'))
            with self.assertRaises(ValueError):study.abstained(request,dict(selected_index=0,checks=[entry]),root)
            entry['clearance']['passed']=False;entry['connection_sha256']='0'*64
            with self.assertRaises(ValueError):study.abstained(request,dict(selected_index=0,checks=[entry]),root)

if __name__=='__main__':unittest.main()
