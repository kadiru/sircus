"""264-slot orchestration without Torch, physics, GPUs, SSH or service changes."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sircus.mjlab import physics_ensemble_study as worker


class StudyTests(unittest.TestCase):
    def fixture(self,root,*,fail_controls=False):
        parentdir=root/'parent';parentdir.mkdir();(parentdir/'native').mkdir();(parentdir/'checkpoints').mkdir()
        for name in ('calibration.json','verification.json','setup-home.json','prior.json'):
            worker.atomic(parentdir/name,dict(score_config={}) if name=='calibration.json' else {'fixture':True})
        native_options=dict(timestep=1/15360,solver=2,cone=0,integrator=3,iterations=100,tolerance=1e-8,gravity=[0.,0.,-9.81])
        worker.atomic(parentdir/'native/effective-options.json',dict(version='sircus-calibrated-native-warp-options-v1',
            substeps=64,native=native_options,warp={'fixture':True}))
        artifacts={}
        for method in ('bc','ppo'):
            path=parentdir/'checkpoints'/(method+'.pt');path.write_bytes((method+'-frozen').encode())
            artifacts['bc' if method=='bc' else 'selected']=dict(path='checkpoints/'+path.name,sha256=worker.sha(path),samples=0 if method=='bc' else 1024)
        source=dict(id='source',values=[.6,-.2,.3,30.,.2],bucket_xy_m=[.4,-.3],q10=[0.]*6,q14=[0.]*6,
                    source_record_sha256='a'*64)
        train=[dict(id=f'parent-train-{i:04d}',role='train',context_id='parent-grid',values=source['values']) for i in range(1024)]
        controls=[dict(id='parent-'+role,bucket_xy_m=[.4,-.3],values=source['values'])
                  for role in ('positive','negative')]
        parent=dict(scene={'fixture':True},fingerprint='f'*64,identity=dict(calibration=dict(score_config={})),
                    packages={},source_sha256={})
        bundle=dict(spec=parent,report=dict(artifacts=artifacts),train_rows=train,prior=dict(records=[source]),
                    calibration=dict(score_config={}),home={'fixture':True},controls=controls,bindings={'authenticated':True})
        args=SimpleNamespace(parent_study=parentdir,parent_audit=root/'audit.json',native_source=root/'native-source',
                             assets=root/'assets',output=root/'study',wall_time_limit=7200,queue_timeout=86400)
        owner=self;self.native_calls=[];self.fit_calls=[];self.dataset_calls=[];self.geometry_calls=[];self.metric_calls=[]
        class Contract:
            VERSION='sircus-physics-ensemble-v1';SCOPE='CPU mock only'
            @staticmethod
            def make_splits(parent):
                return dict(splits={role:[dict(id=f'physics-ensemble-{role}-{i:02d}',split=role,
                    bucket_xy_m=[.1+i*.01,-.6 if role=='validation' else -.4]) for i in range(n)]
                    for role,n in (('validation',24),('test',60))},excluded=[[.4,-.3]])
            @staticmethod
            def protocol():return dict(maximum_fresh_requests=264,fixed_model_no_validation_selection=True)
        class Model:
            @staticmethod
            def inputs(bucket,values):return [0.]*7
            @staticmethod
            def build_dataset(rows,directory,scene,calibration,require_train=True):
                owner.dataset_calls.append((copy.deepcopy(rows),require_train))
                if require_train and any(r['role']!='train' for r in rows):raise ValueError('TRAIN only')
                return [dict(id=r['id'],role=r['role'],x=[0.]*7,y=[0.]*9,mask=[1.]*9) for r in rows]
            @staticmethod
            def fit(dataset,directory,fingerprint):
                owner.fit_calls.append(copy.deepcopy(dataset));path=Path(directory)/'ensemble.pt';path.write_bytes(b'new-ensemble')
                return dict(path='ensemble.pt',sha256=worker.sha(path),fit_device='cpu',ensemble_members=5)
            @staticmethod
            def load(path):
                class Predictor:
                    def predict_inputs(self,x):return [dict(capture_probability=.4,uncertainty=.1) for _ in x]
                return Predictor()
            @staticmethod
            def candidates(context,anchor,bc_values,ppo_values,train_rows):
                return [dict(id=f'candidate-{i:02d}',values=[.6+.001*i,-.2,.3,30.,.2],source_kind='mock') for i in range(16)]
            @staticmethod
            def rank(predictor,context,anchor,items):
                return dict(order=[2,3,4,5,0,1,*range(6,16)],predictions=[{'ranking_score':i} for i in range(16)],candidate_count=16)
            @staticmethod
            def prediction_metrics(predictor,dataset,split):
                owner.metric_calls.append(split);return [dict(split=split,target='capture',support=len(dataset),mae=.1)]
        class Resolver:
            def __init__(self,directory):self.directory=directory;self.home={'fixture':True}
            def connect(self,resolution,values):
                owner.geometry_calls.append((resolution['context_id'],copy.deepcopy(values)))
                index=round((values[0]-.6)/.001)
                passed=index==3 and not resolution['context_id'].endswith('-00')
                connection=dict(passed=passed,reason=None if passed else 'finite_ik_unresolved',endpoint_eligible=passed)
                name='connections/'+worker.digest([resolution['context_id'],values])+'.json'
                path=self.directory/name;path.parent.mkdir(exist_ok=True)
                worker.atomic(path,connection)
                return dict(connection,file=name,sha256=worker.sha(path))
        class Engine:
            def __init__(self,ref,assets,directory,**kwargs):
                self.ref=ref;self.directory=Path(directory);self.directory.mkdir();self.sources=[source]
                self.adapter=SimpleNamespace(resolver=Resolver(self.directory));self.resolved={}
                owner.engine=self
            def resolve(self,context):
                key=worker.digest(context['bucket_xy_m'])
                if key not in self.resolved:
                    name='resolutions/'+key+'.json';path=self.directory/name;path.parent.mkdir(exist_ok=True)
                    worker.atomic(path,dict(context_id=context['id'],fixture=True))
                    self.resolved[key]=(name,worker.sha(path))
                name,fingerprint=self.resolved[key]
                meta=dict(source_hint=source,effective_source_hint=source,context_resolution_file=name,
                    context_resolution_sha256=fingerprint,context_home_sha256='b'*64,
                    context_layout_eligible=True,context_anchor_admitted=True,context_hard_reject=False)
                return copy.deepcopy(source),copy.deepcopy(meta),dict(wall_s=.001)
            def evaluate(self,requests,phase):
                frozen=worker.read(root/'study/prediction-plan.json')
                owner.assertEqual(len(frozen['requests']),264)
                owner.native_calls.append((copy.deepcopy(requests),phase));rows=[]
                for r in requests:
                    native=not (r['method']=='ensemble' and r['context_id'].endswith('-00'))
                    capture=native and (r['role']=='positive_control' or (r['role'] in ('validation','test') and r['method']=='ensemble'))
                    if fail_controls and r['role']=='positive_control':capture=False
                    rows.append(dict(r,native_release=native,route_success=capture,route_quality=2 if capture else 0,
                        invalid=None if native else 'finite_ik_unresolved',invalid_physics=False,
                        scalar_score_audit_passed=True,whole_ball_entry=capture,board_bounce=True,max_retention_s=1. if capture else 0.,
                        trajectory_m=[[.6,-.2,.6],[.6,-.2,.3],[.4,-.3,.1]],trajectory_dt_s=1/60,
                        board_center_m=[.6,-.2,.3],board_normal=[0.,0.,1.],native_cache_sha256='e'*64,native_cache_row=0))
                worker.append(self.directory/'shots.jsonl',rows)
                return rows,dict(requested=len(rows),wall_s=.001)
        api=SimpleNamespace(C=Contract,M=Model,Engine=Engine,observation=lambda *a:np.zeros(9),
            verify_resolution=lambda *a:True,verify_connection=lambda *a:True,
            torch=SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:False)),
            reward=lambda r:4. if worker.captured(r) else -1. if r.get('invalid') else 0.,
            gpu=SimpleNamespace(foreign_processes=lambda:[]))
        return args,bundle,api

    def prepare(self,args,bundle,api):
        def actor_values(api,actor,obs,anchor):
            values=[.601 if actor=='bc.pt' else .602,-.2,.3,30.,.2]
            return values,[0.]*5,[0.]*5
        with patch.object(worker,'load_actor',side_effect=lambda api,path:Path(path).name),\
             patch.object(worker,'actor_values',side_effect=actor_values),patch('builtins.print'):
            return worker.prepare(args,api=api,bundle=bundle)

    def test_cpu_fit_and_all252_rankings_freeze_before_any_native_all_rejects_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,bundle,api=self.fixture(Path(tmp));state,returned=self.prepare(args,bundle,api)
            self.assertIs(returned,api);self.assertEqual(self.native_calls,[])
            self.assertEqual(len(self.fit_calls),1);self.assertEqual(len(self.fit_calls[0]),1024)
            self.assertTrue(all(r['role']=='train' for r in self.fit_calls[0]));self.assertEqual(self.metric_calls,[])
            self.assertEqual(state['report']['status'],'queued');self.assertEqual(state['report']['train_requests'],0)
            self.assertEqual(len(state['requests']),264);self.assertEqual(len(self.geometry_calls),172)
            rankings=worker.read(args.output/'predictions/rankings.json')
            self.assertEqual(len(rankings),84);self.assertTrue(all(r['native_queries']==0 for r in rankings))
            for r in rankings:
                if r['context_id'].endswith('-00'):
                    self.assertEqual(len(r['checks']),4);self.assertEqual(r['selected_index'],2)
                    self.assertTrue(r['all_geometry_checks_rejected'])
                else:self.assertEqual(len(r['checks']),2);self.assertEqual(r['selected_index'],3)
            self.assertEqual(self.engine.ref['public']['effective_native_options']['tolerance'],1e-8)
            self.assertTrue(worker.read(args.output/'prediction-plan.json')['frozen_before_any_native_outcome'])
            with patch('builtins.print'):report=worker.execute_prepared(state,api,args)
            rows=[json.loads(x) for x in (args.output/'records.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),264);self.assertEqual(report['status'],'completed')
            self.assertEqual(report['counts']['requested'],264);self.assertEqual(report['counts']['rejected'],2)
            self.assertEqual(self.metric_calls,['validation','test']);self.assertEqual(len(self.fit_calls),1)
            self.assertEqual(len(report['comparisons']),6)
            self.assertEqual(len(report['test_comparison']),3);self.assertEqual(len(report['test_positions']),180)
            self.assertTrue(all('actual' in r and 'model_prediction' in r for r in report['test_positions']))
            self.assertEqual(report['dataset']['train_requests'],1024);self.assertEqual(report['fit_status'],'model_fit_completed')
            self.assertEqual(report['training']['status'],'completed');self.assertEqual(report['training']['fit_device'],'cpu')
            self.assertTrue(report['trace_examples']);self.assertEqual(report['trace_examples'][0]['position_m'],[ [.6,-.2,.6],[.6,-.2,.3],[.4,-.3,.1]])
            self.assertTrue(all('predicted_position_m' not in r for r in report['trace_examples']))
            self.assertTrue(report['gate']['predictions_frozen_before_any_evaluation'])
            self.assertEqual([r['id'] for r in rows],[r['id'] for r in state['requests']])
            manifest=worker.read(args.output/'evidence-manifest.json')
            self.assertEqual(manifest,report['evidence_sha256'])
            for name,fingerprint in manifest.items():self.assertEqual(worker.sha(args.output/name),fingerprint)

    def test_bad_initial_controls_prevent_validation_test_and_posthoc_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,bundle,api=self.fixture(Path(tmp),fail_controls=True);state,_=self.prepare(args,bundle,api)
            with patch('builtins.print'),self.assertRaisesRegex(RuntimeError,'Initial native controls'):
                worker.execute_prepared(state,api,args)
            self.assertEqual(len(self.native_calls),1);self.assertEqual(self.native_calls[0][1],'controls_pre')
            self.assertEqual(self.metric_calls,[]);self.assertEqual(len(self.dataset_calls),1)
            self.assertEqual(worker.read(args.output/'report.json')['status'],'failed')
            self.assertFalse((args.output/'evidence-manifest.json').exists())

    def test_frozen_plan_or_checkpoint_tampering_stops_before_first_native_call(self):
        for kind in ('plan','model'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                args,bundle,api=self.fixture(Path(tmp));state,_=self.prepare(args,bundle,api)
                path=args.output/('prediction-plan.json' if kind=='plan' else 'model/ensemble.pt')
                path.write_bytes(path.read_bytes()+b' ')
                with patch('builtins.print'),self.assertRaises(ValueError):worker.execute_prepared(state,api,args)
                self.assertEqual(self.native_calls,[])

    def test_heldout_source_rows_cannot_be_fit_or_create_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,bundle,api=self.fixture(Path(tmp));bundle['train_rows'][0]['role']='test'
            with self.assertRaisesRegex(ValueError,'TRAIN'):self.prepare(args,bundle,api)
            self.assertFalse(args.output.exists());self.assertEqual(self.fit_calls,[])


if __name__=='__main__':unittest.main()
