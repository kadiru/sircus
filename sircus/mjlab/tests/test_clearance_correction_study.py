import copy
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from sircus.mjlab import clearance_correction_study as W

class Planning(unittest.TestCase):
    def test_abstention_is_not_a_fake_zero_measurement(self):
        row=W.abstained(dict(id='x',q10=[0.]*6,q14=[0.]*6,native_cache_file='stale.npz',native_initial_q10=[0.]*6))
        self.assertFalse(row['native_release']);self.assertTrue(row['geometry_eligible'])
        for k in ('invalid_physics','route_quality','board_bounce','whole_ball_entry','max_retention_s','closest_opening_distance_m'):
            self.assertIsNone(row[k])
        self.assertNotIn('native_cache_file',row);self.assertNotIn('native_initial_q10',row)
        self.assertEqual(W.counts([row]),dict(requested=1,native_releases=0,captures=0,clean=0,rejected=1,native_misses=0,invalid=0))
    def test_tighter_gate_uses_same_order_and_no_new_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'native').mkdir();checks=[]
            for i in range(4):
                path=p/'native'/f'c{i}.json';W.atomic(path,dict(passed=True,q10=[float(i)]*6,q14=[0.]*6))
                checks.append(dict(candidate_index=i,passed=True,connection_file=path.name,connection_sha256=W.sha(path),clearance={'passed':False}))
            old=dict(checks=checks,selected_index=0,policy_abstain=True,all_geometry_checks_rejected=False)
            ledger=dict(bucket_xy_m=[.5,-.2],selection={'arm_clearance':old},banks={'arm_clearance':dict(candidates=[dict(values=[0.]*5) for i in range(16)],ranking=dict(order=list(range(16)),diagnostics=list(range(16))))})
            class Gate:
                def check(self,q10,q14,bucket,values,d,**kw):return dict(passed=d==1)
            got=W.select(ledger,'corrected',Gate(),p,[0,0,-9.81]);self.assertEqual(got['selected_index'],1);self.assertEqual(len(got['checks']),2);self.assertFalse(got['policy_abstain'])
            old['checks']=checks[:1]
            with self.assertRaisesRegex(ValueError,'new geometry'):W.select(ledger,'corrected',Gate(),p,[0,0,-9.81])

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
                for i in range(3):add('arm-clearance-test-'+code,method,'diagnostic','diagnostic',i,method=='original')
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
                    capture=len(owner.runs) in (1,8) and i<3 and not fail;q10,q14=self.fixed
                    rows.append(dict(native_release=True,values=values[i].tolist(),bucket_xy_m=bucket[i].tolist(),q10=(q10[i]+(.001 if wrong else 0)).tolist(),q14=q14[i].tolist(),
                        native_initial_q10=q10[i].astype(np.float32).astype(float).tolist(),native_initial_q14=q14[i].astype(np.float32).astype(float).tolist(),board_normal=[0.,0.,1.],
                        route_success=capture,route_quality=2 if capture else 0,whole_ball_entry=capture,invalid=None,invalid_physics=False,scalar_score_audit_passed=True,contact_protocol={'substeps':64},board_bounce=True,max_retention_s=float(capture)))
                return rows,{'fixture':True}
        @contextmanager
        def fixed(env,bucket,values,q10,q14,valid):env.fixed=(q10,q14);yield
        runtime=SimpleNamespace(GeometryResolver=Resolver,verify_resolution=lambda *a:True,verify_connection=lambda *a:True,source_geometry=copy.deepcopy,Environment=Env,
            fixed_endpoints=fixed,bounds={},constraints=lambda n:None,first_flight=lambda *a,**kw:dict(available=kw.get('valid',True)),gpu=SimpleNamespace(foreign_processes=lambda:[]))
        return SimpleNamespace(parent_study=p,output=out,native_source=root/'source',assets=root/'assets',wall_time_limit=3600),{},initial,runtime
    def test_all48_requests_withheld_never_reach_environment_or_native_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,contract,initial,runtime=self.fixture(Path(tmp))
            with patch('builtins.print'):result=W.run(args,contract=contract,report=initial,runtime=runtime)
            self.assertEqual(result['status'],'completed');self.assertEqual(self.runs,[6,3,3,3,3,3,3,6])
            records=[json.loads(x) for x in (args.output/'records.jsonl').read_text().splitlines()]
            native=[json.loads(x) for x in (args.output/'native/shots.jsonl').read_text().splitlines()]
            refused=[json.loads(x) for x in (args.output/'policy-rejections.jsonl').read_text().splitlines()]
            self.assertEqual((len(records),len(native),len(refused)),(48,30,18))
            self.assertTrue(all(not r.get('policy_abstain') for r in native));self.assertTrue(all(r['max_retention_s'] is None for r in refused))
            self.assertEqual(result['counts']['rejected'],18)
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
