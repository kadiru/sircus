"""CPU-only paired-plan, raw byte bindings and full96-controller mocks."""
import copy
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sircus.mjlab import gap_flight_correction as worker


def source_row(identity,*,context='control',method='control',role='positive_control',native=True,capture=True):
    hint=dict(id='hint',values=[.6,-.2,.3,30.,.2],bucket_xy_m=[.4,-.3],q10=[.1]*6,q14=[.2]*6,
              source_record_sha256='a'*64)
    row=dict(id=identity,phase='controls_pre' if method=='control' else 'test',role=role,repeat=0,
        method=method,context_id=context,bucket_xy_m=[.4,-.3],values=[.6,-.2,.3,30.,.2],observation=[0.]*9,
        native_release=native,route_success=capture,route_quality=2 if capture else 0,
        invalid=None if native else 'finite_ik_unresolved',invalid_physics=False,
        source_teacher_id='hint',source_hint=hint,effective_source_hint=copy.deepcopy(hint),
        context_resolution_file='resolutions/common.json',context_resolution_sha256='b'*64,
        context_home_sha256='c'*64,context_layout_eligible=True,context_anchor_admitted=True,context_hard_reject=False,
        connection_file='connections/common.json',connection_sha256='d'*64,
        q10=[.1]*6,q14=[.2]*6,native_cache_file='original.npz',native_cache_sha256='e'*64,
        native_cache_row=0,whole_ball_entry=capture)
    return row


def parent_rows():
    rows=[source_row('positive',context='positive_control'),source_row('negative',context='negative_control',role='negative_control',capture=False)]
    for code in worker.CONTEXTS:
        rows.append(source_row('consistent-'+code,context='gap-coverage-test-'+code,method='consistent',role='test',capture=False))
    return rows

class PlanTests(unittest.TestCase):
    def test_only_gap_changes_and_exact_original_joint_branches_are_frozen(self):
        rows=parent_rows();before=copy.deepcopy(rows);plan=worker.make_plan(rows)
        self.assertEqual(rows,before);self.assertEqual(len(plan),30);self.assertEqual(len({r['id'] for r in plan}),30)
        byid={r['id']:r for r in rows}
        self.assertEqual(sum(r['phase'].startswith('controls_') for r in plan),12)
        for r in plan:
            source=byid[r['parent_record_id']]
            self.assertEqual(r['parent_record_sha256'],worker.digest(source))
            self.assertEqual(r['bucket_xy_m'],source['bucket_xy_m'])
            self.assertEqual(r['values'][:4],source['values'][:4])
            if r['variant']=='gap_plus60':
                self.assertEqual(r['values'][4],source['values'][4]+.060)
                self.assertEqual(r['swapped_coordinates'],[4]);self.assertNotIn('q10',r)
            else:
                self.assertEqual(r['values'],source['values']);self.assertEqual(r['q10'],source['q10'])
                self.assertEqual(r['q14'],source['q14']);self.assertEqual(r['kind'],'exact_replay')
        for code in worker.CONTEXTS:
            cohort=[r for r in plan if r['context_id']=='gap-coverage-test-'+code]
            self.assertEqual([r['variant'] for r in cohort],['original']*3+['gap_plus60']*3)

    def test_missing_duplicate_or_wrong_source_fails_closed(self):
        for edit in (lambda r:r.pop(),lambda r:r.append(copy.deepcopy(r[-1])),
                     lambda r:r[-1].update(route_success=True),lambda r:r[-1].update(native_release=False)):
            rows=parent_rows();edit(rows)
            with self.assertRaises(ValueError):worker.make_plan(rows)

class RuntimeTests(unittest.TestCase):
    def fixture(self,root,*,fail_controls=False,wrong_q=False):
        parentdir=root/'parent';(parentdir/'native/resolutions').mkdir(parents=True);(parentdir/'native/connections').mkdir()
        resolution=parentdir/'native/resolutions/common.json';worker.atomic(resolution,dict(fixture=True))
        connection=parentdir/'native/connections/common.json';worker.atomic(connection,dict(fixture=True))
        rows=parent_rows()
        for r in rows:r.update(context_resolution_sha256=worker.sha(resolution),connection_sha256=worker.sha(connection))
        for name in ('calibration.json','verification.json','setup-home.json'):worker.atomic(parentdir/name,{})
        original_options=dict(version='sircus-calibrated-native-warp-options-v1',substeps=64,native={'mock':1},warp={'mock':1})
        worker.atomic(parentdir/'native/effective-options.json',original_options)
        worker.atomic(parentdir/'native/native-compiled-geometry.json',{'fixture':True})
        worker.atomic(parentdir/'prior.json',dict(records=[rows[0]['source_hint']]))
        spec=dict(scene=dict(ball=dict(radius_m=.0375)),identity=dict(setup=dict(home={'sha256':'c'*64}),
            calibration=dict(score_config={})),source_sha256={},native_source_sha256={},packages={})
        worker.atomic(parentdir/'contract.json',spec)
        args=SimpleNamespace(parent_study=parentdir,parent_audit=root/'audit.json',native_source=root/'native-source',
                             assets=root/'assets',output=root/'study',wall_time_limit=3600)
        args.output.mkdir();(args.output/'native').mkdir()
        worker.atomic(args.output/'prior.json',dict(records=[rows[0]['source_hint']]))
        contract=dict(plan=worker.make_plan(rows));worker.atomic(args.output/'contract.json',contract)
        report=dict(status='prepared',counts=worker.counts([]),control_gates={},contexts=[])
        self.connections=[];self.runs=[]
        owner=self
        class Resolver:
            def __init__(self,ref,assets,directory,sources,home):self.directory=Path(directory);self.sources=sources;self.home=home
            def connect(self,resolution,values):
                owner.connections.append(copy.deepcopy(values));index=len(owner.connections)
                c=dict(passed=index!=1,reason=None if index!=1 else 'strict_clearance_reject',
                       endpoint_eligible=index!=1,q10=[.31]*6,q14=[.41]*6)
                filename=f'connections/new-{index}.json';worker.atomic(self.directory/filename,c)
                return dict(c,file=filename,sha256=worker.sha(self.directory/filename))
        class Environment:
            def __init__(self,*a,**kw):
                owner.plan_frozen_before_native=len(worker.read(root/'study/contract.json')['plan'])==30
                self.effective_options=copy.deepcopy(original_options)
                self.compiled_geometry={'fixture':True};self.fixed=None
            def run(self,bucket,values,locks,record):
                owner.runs.append((bucket.copy(),values.copy(),self.fixed[0].copy(),self.fixed[1].copy()))
                n=len(bucket);p=np.zeros((3,n,3));p[0,:,:2]=values[:,:2];p[0,:,2]=values[:,2]+values[:,4]+.0375
                self.last_arrays=dict(position_m=p,velocity_m_s=np.zeros_like(p),forces_N=np.zeros((2,n,6)),time_s=np.array([0.,.1,.2]))
                result=[]
                for i in range(n):
                    # Positive/negative original controls use the same nominal
                    # recipe in this fixture; alternate the first six slots.
                    capture=i<3 and (len(owner.runs) in (1,5)) and not fail_controls
                    q10,q14=self.fixed
                    result.append(dict(native_release=True,values=values[i].tolist(),bucket_xy_m=bucket[i].tolist(),
                        q10=(q10[i]+(.001 if wrong_q else 0)).tolist(),q14=q14[i].tolist(),
                        native_initial_q10=q10[i].astype(np.float32).astype(float).tolist(),native_initial_q14=q14[i].astype(np.float32).astype(float).tolist(),
                        board_normal=[0.,0.,1.],route_success=capture,route_quality=2 if capture else 0,
                        whole_ball_entry=capture,invalid=None,invalid_physics=False,scalar_score_audit_passed=True,
                        contact_protocol=dict(substeps=64),board_bounce=True,max_retention_s=1. if capture else 0.))
                return result,dict(mock=True)
        @contextmanager
        def fixed(env,bucket,values,q10,q14,valid):env.fixed=(q10,q14);yield
        runtime=SimpleNamespace(GeometryResolver=Resolver,verify_resolution=lambda *a:True,verify_connection=lambda *a:True,
            source_geometry=copy.deepcopy,Environment=Environment,fixed_endpoints=fixed,bounds={},
            constraints=lambda n:None,first_flight=lambda *a,**kw:dict(available=kw.get('valid',True),fixture=True),
            gpu=SimpleNamespace(foreign_processes=lambda:[]))
        return args,contract,report,runtime

    def test_full30_flow_keeps_failed_admissions_and_exact_original_branches(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,contract,initial,runtime=self.fixture(Path(tmp))
            with patch('builtins.print'):report=worker.run(args,contract=contract,report=initial,runtime=runtime)
            records=[json.loads(x) for x in (args.output/'records.jsonl').read_text().splitlines()]
            self.assertEqual(report['status'],'completed');self.assertEqual(len(records),30)
            self.assertTrue(self.plan_frozen_before_native)
            self.assertEqual(len(self.connections),9);self.assertEqual(len(self.runs),5)
            self.assertEqual(report['counts']['rejected'],1)
            exact=[r for r in records if r['kind']=='exact_replay']
            self.assertTrue(all(r['q10']==[.1]*6 and r['q14']==[.2]*6 for r in exact))
            new=[r for r in records if r['kind']=='coordinate_swap' and r['native_release']]
            self.assertTrue(all(r['q10']==[.31]*6 and r['q14']==[.41]*6 for r in new))
            manifest=worker.read(args.output/'evidence-manifest.json')
            self.assertEqual(manifest,report['evidence_sha256'])
            for name,fingerprint in manifest.items():self.assertEqual(worker.sha(args.output/name),fingerprint)

    def test_control_failure_and_joint_branch_substitution_stop_execution(self):
        for mode in ('control','branch'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                args,contract,initial,runtime=self.fixture(Path(tmp),fail_controls=mode=='control',wrong_q=mode=='branch')
                with patch('builtins.print'),self.assertRaises(RuntimeError):worker.run(args,contract=contract,report=initial,runtime=runtime)
                self.assertEqual(len(self.connections),0);self.assertEqual(len(self.runs),1)
                self.assertEqual(worker.read(args.output/'report.json')['status'],'failed')
                self.assertFalse((args.output/'evidence-manifest.json').exists())

    def test_foreign_gpu_workload_stops_before_native_allocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,contract,initial,runtime=self.fixture(Path(tmp))
            runtime.gpu.foreign_processes=lambda:[dict(pid=1234)]
            with patch('builtins.print'),self.assertRaisesRegex(RuntimeError,'Foreign4090'):
                worker.run(args,contract=contract,report=initial,runtime=runtime)
            self.assertEqual(self.runs,[])

if __name__=='__main__':unittest.main()
