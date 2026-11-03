"""V2 contract controls and complete CPU fixture: six workers → audit → export.

The fixture uses real CPU PPO/Adam and real producer/oracle scoring, but fabricated
trajectories and geometry. It is explicitly synthetic, never native evidence.
"""
import contextlib
import copy
import importlib.util
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import sys
import unittest
from unittest.mock import patch

import numpy as np

from sircus.mjlab import calibrated_benchmark_contract as contract
from sircus.mjlab.calibrated_benchmark_export import export
from sircus.mjlab.calibrated_benchmark_verify import build_run_summary
from sircus.mjlab.corrected_contract import sha
from sircus.mjlab.learning_contract import digest
from sircus.mjlab.tests.test_calibrated_admission import FakeSolver, FakePlanner, home
from sircus.mjlab.tests.test_calibrated_score import config

ROOT=Path(__file__).resolve().parents[3]
HAS_TORCH=importlib.util.find_spec('torch') is not None


def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,allow_nan=False)+'\n')


class ContractTests(unittest.TestCase):
    def test_fresh_layouts_exclude_all_historical_coordinates(self):
        pools=contract.make_splits(ROOT)
        self.assertEqual([len(pools[k]) for k in ('train','validation','test')],[35,12,24])
        old=np.asarray(contract.historical_coordinates(ROOT))
        new=np.asarray([c['bucket_xy_m'] for role in ('validation','test') for c in pools[role]])
        self.assertGreaterEqual(np.linalg.norm(new[:,None]-old[None],axis=-1).min(),.005)
        self.assertGreaterEqual((np.linalg.norm(new[:,None]-new[None],axis=-1)+np.eye(36)).min(),.005)

    def test_native_budget_cannot_silently_be_fixture_budget(self):
        with self.assertRaisesRegex(ValueError,'canonical'):contract.protocol(budget=32,milestones=[0,32])
        fixture=contract.protocol(budget=32,milestones=[0,32],synthetic=True)
        self.assertEqual(fixture['study_kind'],'synthetic_fixture')
        self.assertIn('geometry-only',fixture['setup_resolution'].lower())

    def test_duplicate_native_release_summary_argument_regression(self):
        rows=[dict(native_release=True,route_success=False,invalid=None,route_quality=0,
                   board_bounce=False,closest_opening_distance_m=1.,external_contact_steps=0,
                   max_retention_s=0.,single_bounce_success=False)]
        value=build_run_summary('sweep',7,32,rows,dict(native_releases=1,provenance={}))
        self.assertEqual(value['native_releases'],1)

    def test_auditor_requires_dense_contact_evidence(self):
        from sircus.mjlab.calibrated_benchmark_verify import rescore_rows, Checks, EvidenceError
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);times=np.arange(1601)*.001;path=root/"native.npz"
            np.savez_compressed(path,position_m=np.zeros((1601,1,3)),velocity_m_s=np.zeros((1601,1,3)),
                forces_N=np.zeros((1600,1,6)),time_s=times,native_time_s=times[:,None],record_ids=np.array(["a"]))
            row=dict(id="a",native_release=True,native_cache_file="native.npz",native_cache_row=0,native_cache_sha256=sha(path))
            with self.assertRaisesRegex(EvidenceError,"contact evidence"):
                rescore_rows(root,[row],{}, {},Checks(),"missing-contact-fixture",contract.trace_schema(synthetic=True))

    def test_zero_normal_requires_narrow_independent_geometry_check(self):
        from sircus.mjlab.calibrated_benchmark_verify import rescore_rows, Checks, EvidenceError, contact_geometry_summary
        from sircus.mjlab.calibrated_score import score_trace
        from sircus.mjlab import verify_capture_calibration
        scene=read(ROOT/"sircus/isaac/scene.json");cfg=config(scene)
        values=np.array([.6,-.2,.3,30.,.2]);bucket=[.4,-.4]
        delta=np.asarray(bucket)-values[:2];tilt=np.deg2rad(values[3])
        normal=np.r_[np.sin(tilt)*delta/np.linalg.norm(delta),np.cos(tilt)]
        release=values[:3].copy();release[2]+=values[4]+scene["ball"]["radius_m"]
        times=np.arange(1601)*.001
        arrays=dict(position_m=np.tile(release,(1601,1,1)),velocity_m_s=np.zeros((1601,1,3)),
            forces_N=np.zeros((1600,1,6)),time_s=times,native_time_s=times[:,None],record_ids=np.array(["a"]),
            contact_distance_m=np.full((1600,1,2),np.inf),contact_geom_ids=np.full((1600,1,2,2),-1,dtype=np.int32),
            contact_position_m=np.zeros((1600,1,2,3)),contact_normal=np.zeros((1600,1,2,3)))
        arrays["contact_distance_m"][0,0,1]=0.;arrays["contact_geom_ids"][0,0,1]=[99,100]
        trace={k:(v[:,0] if v.ndim>=2 else v) for k,v in arrays.items() if k!="record_ids"}
        row=score_trace(scene,bucket,normal,cfg,trace)
        row.update(id="a",values=values.tolist(),bucket_xy_m=bucket,native_release=True,native_cache_file="native.npz",native_cache_row=0,
            board_center_m=values[:3].tolist(),board_normal=normal.tolist(),release_ball_center_m=release.tolist(),
            native_initial_ball_center_m=release.tolist(),native_initial_bucket_center_m=[*bucket,0.],
            pose_gate=dict(board_position_error_mm=0.,board_normal_error_deg=0.,dropper_error_mm=0.),bucket_collider_error_m=0.,calibrated_geometry_sha256="x")
        model=SimpleNamespace(jnt_qposadr=np.array([0]),body_mocapid=np.array([0]),
            geom=lambda name:SimpleNamespace(id=0 if name=="ball/geom" else 1),
            joint=lambda name:SimpleNamespace(id=0),body=lambda name:SimpleNamespace(id=0))
        data=SimpleNamespace(qpos=np.zeros(7),qvel=np.zeros(6),mocap_pos=np.zeros((1,3)))
        distance=[0.];calls=[]
        def query(*args):calls.append(args[2:4]);return distance[0]
        native=SimpleNamespace(MjData=lambda m:data,mj_forward=lambda m,d:None,mj_geomDistance=query)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);np.savez_compressed(root/"native.npz",**arrays);row["native_cache_sha256"]=sha(root/"native.npz")
            write(root/"native-compiled-geometry.json",dict(compiled_planes=cfg["compiled_planes"],geometry_sha256="x",
                contact_geom_ids=dict(ball=99,base=100,walls=list(range(64)))))
            spec=contract.trace_schema(synthetic=True);spec["version"]="sircus-dense-native-interval-trace-v2"
            with patch.dict(sys.modules,{"mujoco":native}),patch.object(verify_capture_calibration,"compile_cpu_model",return_value=(model,{})):
                result=rescore_rows(root,[row],scene,cfg,Checks(),"zero-frame",spec,assets=root)
                self.assertEqual(contact_geometry_summary(result)["zero_frame_contacts"],1)
                self.assertEqual(calls,[(0,1)])
                distance[0]=.001
                with self.assertRaisesRegex(EvidenceError,"independently verified"):
                    rescore_rows(root,[row],scene,cfg,Checks(),"bad-zero-frame",spec,assets=root)

    def test_calibration_kind_and_exact_verification_binding(self):
        scene=read(ROOT/'sircus/isaac/scene.json');cfg=config(scene)
        with tempfile.TemporaryDirectory() as d:
            a,v=Path(d)/'calibration.json',Path(d)/'verification.json'
            write(a,dict(version='synthetic-control',status='passed',checks=1,failures=[],synthetic_fixture=True,
                         score_config=cfg,score_config_sha256=digest(cfg)))
            write(v,dict(version='synthetic-verification',status='passed',checks=1,failures=[],synthetic_fixture=True,
                         calibration_sha256=sha(a),score_config_sha256=digest(cfg)))
            self.assertTrue(contract.checked_calibration(a,v,synthetic=True)['synthetic_fixture'])
            with self.assertRaisesRegex(ValueError,'kind'):contract.checked_calibration(a,v)
            a.write_bytes(a.read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError,'exact'):contract.checked_calibration(a,v,synthetic=True)


@unittest.skipUnless(HAS_TORCH,'CPU Torch required for complete six-worker integration gate')
class CompleteFixtureTests(unittest.TestCase):
    def test_six_workers_full_audit_and_compact_export(self):
        import torch
        from sircus.mjlab import calibrated_benchmark_run as worker
        from sircus.mjlab import calibrated_benchmark_smoke as smoke
        from sircus.mjlab import calibrated_benchmark_study as controller
        from sircus.mjlab import calibrated_benchmark_verify as auditor
        from sircus.mjlab import sweep_worker
        from sircus.mjlab.calibrated_adapter import CalibratedAdapter
        from sircus.mjlab.calibrated_admission import GeometryResolver
        from sircus.mjlab.calibrated_score import score_trace
        from sircus.mjlab.benchmark_resources import ResourceMeter
        from sircus.mjlab.expanded_actions import WIDE_BOUNDS

        scene=read(ROOT/'sircus/isaac/scene.json')
        ref=dict(scene=scene,contract=dict(fingerprint='a'*64),source_hashes={},asset_hashes={},packages={})
        cfg=config(scene)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);a=root/'calibration.json';v=root/'verification.json';h=root/'home.json'
            write(a,dict(version='synthetic-control',status='passed',checks=1,failures=[],synthetic_fixture=True,
                         score_config=cfg,score_config_sha256=digest(cfg)))
            write(v,dict(version='synthetic-verification',status='passed',checks=1,failures=[],synthetic_fixture=True,
                         calibration_sha256=sha(a),score_config_sha256=digest(cfg)))
            write(h,home(scene))

            class Solver(FakeSolver):
                def solve(self,bucket,values,hint):
                    x=np.asarray(values)
                    if ((x<np.asarray(WIDE_BOUNDS['low']))|(x>np.asarray(WIDE_BOUNDS['high']))).any():
                        return dict(passed=False,reason='synthetic_global_bound_reject',candidates=[])
                    if np.linalg.norm(x[:2]-bucket)<=.15:
                        return dict(passed=False,reason='board_bucket_distance_constraint_rejected',candidates=[])
                    return super().solve(bucket,values,hint)

            def native(candidates):
                times=np.arange(1601,dtype=float)*.001
                positions=[];velocities=[];forces=[];rows=[]
                for requested in candidates:
                    values=np.asarray(requested['values']);xy=np.asarray(requested['bucket_xy_m'])
                    delta=xy-values[:2];tilt=np.deg2rad(values[3]);normal=np.r_[np.sin(tilt)*delta/np.linalg.norm(delta),np.cos(tilt)]
                    release=values[:3].copy();release[2]+=values[4]+scene['ball']['radius_m']
                    rim=scene['bucket']['assumed_bottom_thickness_m']+scene['bucket']['assumed_interior_depth_m']
                    floor=scene['bucket']['assumed_bottom_thickness_m']+scene['ball']['radius_m']
                    # The artificial capture/miss mechanism is only a test oracle,
                    # deliberately action-dependent so real PPO receives variance.
                    captured=values[3]>31.
                    target=xy.copy();target[0]+=0 if captured else .3
                    knots=[release,values[:3]+[0,0,scene['ball']['radius_m']],np.r_[target,rim+.07],np.r_[target,floor]]
                    xyz=np.column_stack([np.interp(times,[0,.08,.30,.48],[p[k] for p in knots]) for k in range(3)])
                    velocity=np.gradient(xyz,times,axis=0)
                    # Synthetic velocities remain in calibrated logic-test domain.
                    speed=np.linalg.norm(velocity,axis=1);velocity*=np.minimum(1.,6./np.maximum(speed,1e-12))[:,None]
                    velocity[79]=-normal;velocity[82]=normal
                    force=np.zeros((1600,6));force[80,0]=1.
                    arrays=dict(position_m=xyz,velocity_m_s=velocity,forces_N=force,time_s=times)
                    measured=score_trace(scene,xy,normal,cfg,arrays)
                    measured.update(values=values.tolist(),bucket_xy_m=xy.tolist(),native_release=True,
                        board_center_m=values[:3].tolist(),board_normal=normal.tolist(),
                        release_ball_center_m=release.tolist(),native_initial_ball_center_m=xyz[0].tolist(),
                        native_initial_bucket_center_m=[*xy,0.],q10=requested['q10'],q14=requested['q14'],
                        native_initial_q10=requested['q10'],native_initial_q14=requested['q14'],
                        bucket_collider_error_m=0.,pose_gate=dict(board_position_error_mm=0.,board_normal_error_deg=0.,dropper_error_mm=0.),
                        synthetic_fixture=True)
                    positions.append(xyz);velocities.append(velocity);forces.append(force);rows.append(measured)
                arrays=dict(position_m=np.stack(positions,axis=1),velocity_m_s=np.stack(velocities,axis=1),
                            forces_N=np.stack(forces,axis=1),time_s=times,native_time_s=np.tile(times[:,None],(1,len(rows))),
                            contact_distance_m=np.full((1600,len(rows),2),np.inf),
                            contact_geom_ids=np.full((1600,len(rows),2,2),-1,dtype=np.int32),
                            contact_position_m=np.zeros((1600,len(rows),2,3)),contact_normal=np.zeros((1600,len(rows),2,3)))
                return rows,arrays,dict(native_rollout_s=0.)

            def adapter(ref,assets,directory,*,prior_records,calibration,home,batch):
                resolver=GeometryResolver(ref,assets,directory,prior_records,home=home,solver=Solver(),planner=FakePlanner(home))
                return CalibratedAdapter(ref,assets,directory,prior_records=prior_records,calibration=calibration,
                                         home=home,batch=batch,resolver=resolver,native_executor=native)

            class Meter(worker.ResourceMeter):
                def _watch(self):
                    self._record('nvidia_process_sample',dict(used_vram_mib=None))
                    self.stop_event.wait()

            class Process:
                pid=12345
                def __init__(self,command,**kwargs):
                    def arg(k):return command[command.index('--'+k)+1]
                    worker.run(SimpleNamespace(study=Path(arg('study')),reference=Path(arg('reference')),assets=Path(arg('assets')),
                        gpu_lock=Path(arg('gpu-lock')),method=arg('method'),seed=int(arg('seed')),queue_timeout=10.,wall_time_limit=120.))
                    self.returncode=0
                def poll(self):return self.returncode

            def audit_process(command,**kwargs):
                def arg(k):return Path(command[command.index('--'+k)+1])
                result=auditor.audit(arg('study'),arg('reference'),arg('assets'),repository=arg('repository'))
                write(arg('output'),result)
                return SimpleNamespace(returncode=0)

            common_args=dict(reference=root/'reference',assets=root/'assets',gpu_lock=root/'lock',
                             prior_bundle=ROOT/contract.PRIOR_PATH,calibration_report=a,verification_report=v,setup_home=h)
            with contextlib.ExitStack() as stack:
                for module in (worker,smoke,controller,sweep_worker):stack.enter_context(patch.object(module,'load_reference',return_value=ref))
                for module in (worker,smoke):
                    stack.enter_context(patch.object(module,'BenchmarkAdapter',side_effect=adapter))
                    stack.enter_context(patch.object(module,'gpu_slot',return_value=contextlib.nullcontext()))
                stack.enter_context(patch.object(worker,'ResourceMeter',Meter))
                stack.enter_context(patch.object(worker,'foreign_processes',return_value=[]))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                smoke.run(SimpleNamespace(output=root/'controls',**common_args),synthetic=True)
                controls_audit=auditor.audit_controls(root/'controls',root/'reference',root/'assets',repository=ROOT)
                self.assertGreater(controls_audit['checks'],0);write(root/'controls-audit.json',controls_audit)
                stack.enter_context(patch.object(controller,'subprocess',SimpleNamespace(Popen=Process,run=audit_process,STDOUT=-2,TimeoutExpired=TimeoutError)))
                args=SimpleNamespace(output=root/'study',preflight_report=root/'controls/report.json',controls_audit=root/'controls-audit.json',
                    seeds=contract.SEEDS,budget=32,milestones=[0,32],resume=False,prepare_only=False,queue_timeout=10.,
                    worker_wall_time_limit=120.,**common_args)
                controller.run(args,synthetic=True)
            audit_path=root/'study-audit.json';audited=read(audit_path)
            self.assertEqual(audited['status'],'passed');self.assertEqual(len(audited['runs']),6)
            self.assertEqual(audited['requested_trials'],6*(32+72+72))
            self.assertTrue(audited['synthetic_fixture']);self.assertGreater(audited['checks'],1000)
            fp=read(root/'study/contract.json')['fingerprint']
            # Native publication must reject fixture evidence, even if audited.
            with self.assertRaisesRegex(ValueError,'Synthetic'):
                export(root/'study',audit_path,ROOT,root/'refused',expected_contract_fingerprint=fp,expected_audit_sha256=sha(audit_path))
            published=export(root/'study',audit_path,ROOT,root/'bundle',expected_contract_fingerprint=fp,
                             expected_audit_sha256=sha(audit_path),allow_synthetic=True)
            self.assertEqual(published['files'],34)
            self.assertEqual(len(list((root/'bundle/selected').rglob('*.pt'))),3)
            self.assertFalse(list((root/'bundle').rglob('*.npz')))
            self.assertIn('Synthetic software-test evidence only',(root/'bundle/README.md').read_text())
            with self.assertRaisesRegex(ValueError,'Pinned audit'):
                export(root/'study',audit_path,ROOT,root/'tampered',expected_contract_fingerprint=fp,
                       expected_audit_sha256='0'*64,allow_synthetic=True)


if __name__=='__main__':unittest.main()
