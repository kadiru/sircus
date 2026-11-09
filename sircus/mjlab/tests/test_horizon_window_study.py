import copy
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from sircus.mjlab import horizon_window_study as W

class Planning(unittest.TestCase):
    def test_abstention_is_not_a_fake_zero_measurement(self):
        row=W.abstained(dict(id='x',q10=[0.]*6,q14=[0.]*6,native_cache_file='stale.npz',native_initial_q10=[0.]*6))
        self.assertFalse(row['native_release']);self.assertTrue(row['geometry_eligible'])
        for k in ('invalid_physics','route_quality','board_bounce','whole_ball_entry','max_retention_s','closest_opening_distance_m'):
            self.assertIsNone(row[k])
        self.assertNotIn('native_cache_file',row);self.assertNotIn('native_initial_q10',row)
        self.assertEqual(W.counts([row]),dict(requested=1,native_releases=0,captures=0,clean=0,rejected=1,native_misses=0,invalid=0))
    def test_same_candidate_only_and_no_alternative_geometry_or_forecasts(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'native').mkdir()
            path=p/'native/c14.json';W.atomic(path,dict(passed=True,q10=[.1]*6,q14=[.2]*6))
            prior=dict(candidate_index=14,passed=True,connection_file=path.name,connection_sha256=W.sha(path))
            old=dict(checks=[prior],selected_index=14,policy_abstain=False,all_geometry_checks_rejected=False)
            bank=dict(candidates=[dict(values=[float(i)]*5) for i in range(16)],ranking=dict(order=[15,14],diagnostics=list(range(16))))
            ledger=dict(context_id=W.CONTEXT_ID,bucket_xy_m=[.5,-.2],selection={'arm_clearance':old},banks={'arm_clearance':bank})
            calls=[]
            class Gate:
                def check(self,q10,q14,bucket,values,d,**kw):
                    calls.append((q10,q14,bucket,values,d,kw));return dict(passed=False)
            frozen=copy.deepcopy(ledger)
            got=W.select(ledger,'long_window',Gate(),p,[0,0,-9.81])
            self.assertEqual(got['selected_index'],14);self.assertEqual(len(got['checks']),1);self.assertTrue(got['policy_abstain'])
            self.assertEqual(calls,[([.1]*6,[.2]*6,[.5,-.2],[14.]*5,14,{'gravity':[0,0,-9.81]})])
            self.assertEqual(ledger,frozen)
            old['selected_index']=15
            with self.assertRaisesRegex(ValueError,'candidate14'):W.select(ledger,'long_window',Gate(),p,[0,0,-9.81])
            old['selected_index']=14;path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'proof changed'):W.select(ledger,'long_window',Gate(),p,[0,0,-9.81])

    def test_exact18_slot_plan_preserves_pairs_endpoints_and_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'native').mkdir();path=p/'native/c14.json'
            W.atomic(path,dict(passed=True,q10=[.1]*6,q14=[.2]*6))
            source=[];rows=[]
            for name,role in (('positive','positive_control'),('negative','negative_control'),(W.CONTEXT_ID,'comparison')):
                source.append(dict(id=name,context_id=name,role=role,phase='controls_pre' if role!='comparison' else 'test',repeat=0,
                    method='arm_clearance',bucket_xy_m=[.8,.1],values=[.6,-.1,.32,18.,.27],command_values=[.6,-.1,.32,18.,.27]))
                rows.append(dict(source[-1],q10=[.1]*6,q14=[.2]*6,connection_file=path.name,connection_sha256=W.sha(path)))
            candidate=dict(id='candidate14',values=[.6,-.1,.32,18.,.27]);bank=dict(candidates=[candidate]*16,predictions=[dict(p=.9)]*16)
            ledger=dict(banks={'arm_clearance':bank});selections={}
            for method in W.METHODS:
                clearance=dict(rule_sha256='c'*64,minimum_clearance_lower_bound_m=.01 if method=='short_window' else -.03)
                proof=dict(candidate_index=14,connection_file=path.name,connection_sha256=W.sha(path),clearance=clearance)
                selections[method]=dict(selected_index=14,policy_abstain=method=='long_window',checks=[proof])
            plan=W.freeze_requests(source,rows,{W.CONTEXT_ID:ledger},{W.CONTEXT_ID:selections},p)
            self.assertEqual(len(plan),18);self.assertEqual(len({r['id'] for r in plan}),18)
            self.assertTrue(all(r['role'] in ('positive_control','negative_control') for r in plan[:6]+plan[-6:]))
            for i in range(3):
                short,long=plan[6+i],plan[9+i]
                self.assertFalse(short['policy_abstain']);self.assertTrue(long['policy_abstain'])
                for key in ('values','command_values','bucket_xy_m','q10','q14','connection_file','connection_sha256','candidate_index','model_prediction'):
                    self.assertEqual(short[key],long[key],key)
                self.assertEqual(short['repeat'],long['repeat']);self.assertEqual(short['geometry_checks'],1)
            self.assertTrue(all(r['substeps']==64 and not r['selection_or_learning'] for r in plan))

    def test_window_summary_retains_unknown_missing_certificate_members(self):
        certificate=dict(passed=False,members=[],minimum_clearance_lower_bound_m=None,required_margin_m=.002)
        result=W.window_summary('long_window',certificate)
        self.assertIsNone(result['horizon_min_s']);self.assertIsNone(result['horizon_max_s'])
        self.assertIsNone(result['minimum_clearance_lower_bound_m']);self.assertFalse(result['passed'])

class NativeRuntime(unittest.TestCase):
    def fixture(self,root,fail=False,wrong=False):
        p=root/'parent';(p/'native/resolutions').mkdir(parents=True);(p/'native/connections').mkdir()
        resolution=p/'native/resolutions/a.json';W.atomic(resolution,{'fixture':True})
        connection=p/'native/connections/a.json';W.atomic(connection,dict(passed=True,q10=[.1]*6,q14=[.2]*6))
        options=dict(version='sircus-calibrated-native-warp-options-v1',substeps=64,native={'fixture':1},warp={'fixture':1})
        W.atomic(p/'native/effective-options.json',options);W.atomic(p/'native/native-compiled-geometry.json',{'fixture':True})
        spec=dict(scene=dict(ball=dict(radius_m=.0375)),identity=dict(setup=dict(home={}),calibration=dict(score_config={})),native_source_sha256={},packages={})
        W.atomic(p/'contract.json',spec)
        out=root/'study';out.mkdir();(out/'native').mkdir();W.atomic(out/'prior.json',{'records':[]})
        base=dict(values=[.6,-.2,.3,30.,.2],bucket_xy_m=[.4,-.3],q10=[.1]*6,q14=[.2]*6,
            context_resolution_file='resolutions/a.json',context_resolution_sha256=W.sha(resolution),
            connection_file='connections/a.json',connection_sha256=W.sha(connection),policy_abstain=False,kind='exact_replay')
        plan=[]
        def add(ctx,method,phase,role,repeat,abstain=False):
            plan.append(dict(copy.deepcopy(base),id=f'{ctx}-{method}-{phase}-{repeat}',context_id=ctx,method=method,variant=method,phase=phase,role=role,repeat=repeat,policy_abstain=abstain))
        def controls(phase):
            for role in ('positive_control','negative_control'):
                for i in range(3):add(role,'control_exact',phase,role,i)
        controls('controls_pre')
        for code in W.CONTEXTS:
            for method in W.METHODS:
                for i in range(3):add('arm-clearance-test-'+code,method,'diagnostic','diagnostic',i,method=='long_window')
        controls('controls_post');W.atomic(out/'prediction-plan.json',{'requests':plan})
        initial=dict(status='prepared',control_gates={},contexts=[]);owner=self;self.runs=[]
        class Resolver:
            def __init__(self,ref,assets,directory,sources,home):self.sources=sources;self.home=home
            def connect(self,*args):raise AssertionError('No new geometry queries during native evaluation')
        class Env:
            def __init__(self,*args,**kw):self.compiled_geometry={'fixture':True};self.effective_options=copy.deepcopy(options);self.fixed=None
            def run(self,bucket,values,locks,record):
                owner.runs.append(len(bucket));n=len(bucket);p=np.zeros((3,n,3));p[0,:,:2]=values[:,:2];p[0,:,2]=values[:,2]+values[:,4]+.0375
                self.last_arrays=dict(position_m=p,velocity_m_s=np.zeros_like(p),forces_N=np.zeros((2,n,6)),time_s=np.array([0.,.1,.2]))
                rows=[]
                for i in range(n):
                    capture=len(owner.runs) in (1,3) and i<3 and not fail;q10,q14=self.fixed
                    rows.append(dict(native_release=True,values=values[i].tolist(),bucket_xy_m=bucket[i].tolist(),q10=(q10[i]+(.001 if wrong else 0)).tolist(),q14=q14[i].tolist(),
                        native_initial_q10=q10[i].astype(np.float32).astype(float).tolist(),native_initial_q14=q14[i].astype(np.float32).astype(float).tolist(),board_normal=[0.,0.,1.],
                        route_success=capture,route_quality=2 if capture else 0,whole_ball_entry=capture,invalid=None,invalid_physics=False,scalar_score_audit_passed=True,contact_protocol={'substeps':64},board_bounce=True,max_retention_s=float(capture)))
                return rows,{'fixture':True}
        @contextmanager
        def fixed(env,bucket,values,q10,q14,valid):env.fixed=(q10,q14);yield
        runtime=SimpleNamespace(GeometryResolver=Resolver,verify_resolution=lambda *a:True,verify_connection=lambda *a:True,source_geometry=copy.deepcopy,Environment=Env,
            fixed_endpoints=fixed,bounds={},constraints=lambda n:None,first_flight=lambda *a,**kw:dict(available=kw.get('valid',True)),gpu=SimpleNamespace(foreign_processes=lambda:[]))
        return SimpleNamespace(parent_study=p,output=out,native_source=root/'source',assets=root/'assets',wall_time_limit=3600),{},initial,runtime
    def test_all18_requests_withheld_never_reach_environment_or_native_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,contract,initial,runtime=self.fixture(Path(tmp))
            with patch('builtins.print'):result=W.run(args,contract=contract,report=initial,runtime=runtime)
            self.assertEqual(result['status'],'completed');self.assertEqual(self.runs,[6,3,6])
            records=[json.loads(x) for x in (args.output/'records.jsonl').read_text().splitlines()]
            native=[json.loads(x) for x in (args.output/'native/shots.jsonl').read_text().splitlines()]
            refused=[json.loads(x) for x in (args.output/'policy-rejections.jsonl').read_text().splitlines()]
            self.assertEqual((len(records),len(native),len(refused)),(18,15,3))
            self.assertTrue(all(not r.get('policy_abstain') for r in native));self.assertTrue(all(r['max_retention_s'] is None for r in refused))
            self.assertEqual(result['counts']['rejected'],3)
            for name,value in result['evidence_sha256'].items():self.assertEqual(W.sha(args.output/name),value)
    def test_control_or_joint_branch_failure_stops_without_completion_manifest(self):
        for mode in ('controls','branch'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                args,c,r,runtime=self.fixture(Path(tmp),fail=mode=='controls',wrong=mode=='branch')
                with patch('builtins.print'),self.assertRaises(RuntimeError):W.run(args,contract=c,report=r,runtime=runtime)
                self.assertEqual(self.runs,[6]);self.assertFalse((args.output/'evidence-manifest.json').exists())
    def test_foreign4090_work_blocks_before_native_allocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,c,r,runtime=self.fixture(Path(tmp));runtime.gpu.foreign_processes=lambda:[{'pid':123}]
            with patch('builtins.print'),self.assertRaisesRegex(RuntimeError,'Foreign4090'):W.run(args,contract=c,report=r,runtime=runtime)
            self.assertEqual(self.runs,[])

if __name__=='__main__':unittest.main()
