import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from sircus.mjlab import consistent_flight_study as study
from sircus.mjlab import physics_ensemble_study as old
from sircus.mjlab.prepare_consistent_flight_report import compact,prepare

class FrozenStudyTests(unittest.TestCase):
    def test_split_repeatability_excludes_used_positions(self):
        previous=dict(excluded=[[.1,-.65],[1.24,.25]],splits=dict(test=[dict(bucket_xy_m=[.55,-.2])]))
        a,excluded=study.fresh_splits(previous);b,_=study.fresh_splits(previous)
        self.assertEqual(a,b);self.assertEqual(len(a['test']),30)
        pts=np.array([c['bucket_xy_m'] for c in a['test']])
        self.assertGreaterEqual(np.linalg.norm(pts[:,None]-np.array(excluded),axis=-1).min(),.01)
        self.assertGreaterEqual((np.linalg.norm(pts[:,None]-pts[None,:],axis=-1)+np.eye(30)).min(),.01)

    def execute_mock(self,foreign=False,tamper=False):
        tmp=TemporaryDirectory();self.addCleanup(tmp.cleanup);out=Path(tmp.name)
        (out/'prediction-plan.json').write_text('{}');frozen=study.sha(out/'prediction-plan.json')
        (out/'model.pt').write_bytes(b'model');artifact=dict(path='model.pt',sha256=study.sha(out/'model.pt'))
        requests=[]
        for phase in ('controls_pre','test','controls_post'):
            if phase=='test':
                for m in study.METHODS:
                    for i in range(30):requests.append(dict(id=f'{m}{i}',role='test',phase=phase,method=m,context_id=f'c{i}',bucket_xy_m=[.5,-.2],values=[.6,-.3,.3,30,.2]))
            else:
                for role in ('positive_control','negative_control'):
                    for i in range(3):requests.append(dict(id=f'{phase}{role}{i}',role=role,phase=phase,method='control'))
        calls=[]
        class Engine:
            def evaluate(self,rs,phase):
                calls.append((phase,[r['id'] for r in rs]));rows=[]
                for r in rs:
                    success=r['role']!='negative_control'
                    rows.append(dict(r,native_release=True,route_success=success,route_quality=2 if success else 0,
                        invalid=None,invalid_physics=False,scalar_score_audit_passed=True,whole_ball_entry=success,max_retention_s=1.2 if success else 0))
                return rows,dict(wall_s=0)
        state=dict(output=out,engine=Engine(),report=dict(artifacts={m:artifact for m in study.METHODS},control_gates={},comparisons=[],test_positions=[]),requests=requests,prediction_sha256=frozen)
        if tamper:(out/'prediction-plan.json').write_text('{"changed":true}')
        api=SimpleNamespace(gpu=SimpleNamespace(foreign_processes=lambda:[123] if foreign else []),reward=lambda r:int(r['route_success']))
        with patch.object(study.shutil,'disk_usage',return_value=SimpleNamespace(free=20*1024**3)):
            if foreign or tamper:
                with self.assertRaises((RuntimeError,ValueError)):study.execute(state,api,old,SimpleNamespace(wall_time_limit=3600))
                self.assertEqual(calls,[])
            else:
                study.execute(state,api,old,SimpleNamespace(wall_time_limit=3600))
                self.assertEqual(sum(len(ids) for _,ids in calls),132)
                self.assertEqual(state['report']['status'],'completed')
                self.assertEqual(len(state['report']['test_positions']),120)
                self.assertEqual([r['counts']['requested'] for r in state['report']['comparisons']],[30]*4)
    def test_bounded_mock_execution(self):self.execute_mock()
    def test_foreign_job_no_execution(self):self.execute_mock(foreign=True)
    def test_tampered_frozen_plan_no_execution(self):self.execute_mock(tamper=True)

class PublisherTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.study=self.root/'study';self.study.mkdir()
        self.base=self.root/'base.json';self.base.write_text('{"old_result":{"capture":false}}')
        spec=dict(protocol=study.protocol());spec['fingerprint']=study.digest(spec)
        study.atomic(self.study/'contract.json',spec)
        study.atomic(self.study/'development.json',dict(new_native_requests=0,new_training_requests=0,cases=[],changed_top_candidates=0,contexts=0))
        raw=dict(version=study.VERSION,status='queued',phase='waiting_for_4090',scope=study.SCOPE,protocol=study.protocol(),contract_fingerprint=spec['fingerprint'],
            development_sha256=study.sha(self.study/'development.json'),counts=old.counts([]),comparisons=[],source_report_sha256='a'*64,selector_sha256='b'*64,
            artifacts=dict(ensemble=dict(sha256='c'*64)),queue=dict(reason=[999],wait_s=3))
        study.atomic(self.study/'report.json',raw)
    def test_queue_preserves_old_data_and_missing_results(self):
        result=prepare(self.base,self.study,self.root/'display.json')
        self.assertEqual(result['old_result'],dict(capture=False));self.assertEqual(result['consistent_flight_study']['comparisons'],[])
        self.assertIsNone(result['consistent_flight_study']['audit']['passed'])
        self.assertNotIn('999',json.dumps(result))
    def test_no_in_place_evidence_or_display_write(self):
        for dest in (self.base,self.study/'new.json'):
            with self.assertRaises(ValueError):prepare(self.base,self.study,dest)
    def test_forged_audit_or_changed_development_fails(self):
        audit=self.root/'audit.json';study.atomic(audit,dict(status='passed',version='sircus-consistent-flight-audit-v1',checks=[]))
        with self.assertRaises(ValueError):compact(self.study,audit)
        (self.study/'development.json').write_text('{}')
        with self.assertRaises(ValueError):compact(self.study)

if __name__=='__main__':unittest.main()
