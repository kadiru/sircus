"""Mock-only guard adapter tests; no simulator, real model or arm is created."""

import copy
import gzip
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import candidate_continuation_native as subject
from candidate_continuation_journal import Journal
from candidate_continuation_ledger import Ledger, Outcome, SourceBindings, canonical_hash
from candidate_continuation_policy import freeze_support
from continuous_feasibility import decode_action, LOW, HIGH, canonical_hash as geometry_hash
from prepare_assets import chain
from continuous_policy import torch

SCENE = dict(simulation=dict(physics_dt_s=1/240), ball=dict(radius_m=.0375),
    bucket=dict(inner_top_diameter_m=.177, inner_bottom_diameter_m=.15,
        assumed_interior_depth_m=.175, assumed_bottom_thickness_m=.008, segments=64))
METRIC = dict(position_error_mm=0., normal_error_deg=0., joint_speed_rad_s=0.,
    position_drift_mm=0., normal_drift_deg=0., dropper_position_error_mm=0., dropper_drift_mm=0.)


class FakeBase:
    def prepare(self, actions): raise AssertionError("No real prepare in fixture")


class FakeEnv(FakeBase):
    def prepare(self, actions): return super().prepare(actions)
    def home(self):
        self.calls.append("home")
        return dict(synthetic=True, path_eligible=[True]*35)
    def shot_batch(self, actions, *, record_video, enabled):
        self.calls.append("shot")
        self.seen = copy.deepcopy(dict(actions=actions, seeds=self.proposal_ik_seeds,
            cells=self.base_cells, enabled=list(map(bool,enabled))))
        if self.raise_during_shot:
            self.physics_samples = [dict(time_step=1, position_m=np.full((35,3),np.nan))]
            raise RuntimeError("synthetic interrupted native call")
        self.qualifications=[dict(accepted=bool(use)) for use in enabled]
        self.path_reasons=[None]*35; self.path_checks=[dict(accepted=bool(use)) for use in enabled]
        self.preflight_records=[dict(accepted=bool(use), actual_start_rechecked=True) for use in enabled]
        self.scoring_normals=np.tile([0.,0.,1.],(35,1))
        self.transit_history=[dict(step=0)]; self.correction_history=[]; self.release_step=0
        traces=[]
        for i,cell in enumerate(self.base_cells):
            xy=np.asarray(cell['bucket_xy_m'])
            if self.capture and i==0:
                positions=[[xy[0]+.4,xy[1],.283],[xy[0]+.4,xy[1],.283],[xy[0],xy[1],.183]]+[[xy[0],xy[1],.08]]*245
                velocities=[[0,0,-1],[0,0,1]]+[[0,0,0]]*(len(positions)-2)
                forces=[[1,0,0,0,0,0]]+[[0]*6]*(len(positions)-1)
            else:
                positions=[[xy[0]+.4,xy[1],.5],[xy[0]+.4,xy[1],-.3]]
                velocities=[[0,0,-1]]*2; forces=[[0]*6]*2
            traces.append((positions,velocities,forces))
        count=max(len(t[0]) for t in traces)
        self.physics_samples=[]
        for step in range(count):
            self.physics_samples.append(dict(time_step=step+1,
                position_m=np.asarray([t[0][min(step,len(t[0])-1)] for t in traces]),
                velocity_m_s=np.asarray([t[1][min(step,len(t[1])-1)] for t in traces]),
                forces_N=np.asarray([t[2][min(step,len(t[2])-1)] for t in traces])))
        rows=[]
        for i,cell in enumerate(self.base_cells):
            features=subject.build_features(self.cfg,cell['bucket_xy_m'], *map(np.asarray,traces[i]),self.dt,[0,0,1])
            row=dict(features['observed'],route_quality=2 if features['observed']['route_success'] else 0,
                invalid=None if enabled[i] and i not in self.reject_slots else 'synthetic_guard_decline',
                settled=copy.deepcopy(METRIC),release_ball_center_m=traces[i][0][0])
            if row['invalid']:
                row.update(route_success=False,route_quality=0)
            rows.append(row)
        self.steps=count
        if self.bad_release_metric: rows[0]['settled']['joint_speed_rad_s']=.006
        return rows,dict(settling=[copy.deepcopy(METRIC)]*35,trajectory_m=[t[0] for t in traces],wall_s=0.)
    def tick(self): raise AssertionError("No actual stepping in mock")
    def render(self): pass
    def approach_endpoints(self,*args): raise AssertionError("No motion in mock")
    def _wait_stable(self,*args): pass
    def _measure(self,*args): pass
    def _joint_status(self,*args): pass
    def _ik(self,*args): pass
    def _state(self): pass
    def board_poses(self): pass
    def assert_buckets_fixed(self): pass
    def set_balls(self,*args): raise AssertionError("No ball placement in mock")
    def forces(self): pass


def fixture_methods():
    names=('prepare','home','shot_batch','tick','render','approach_endpoints','_wait_stable',
        '_measure','_joint_status','_ik','_state','board_poses','assert_buckets_fixed','set_balls','forces')
    return dict(__prepare_chain__=(FakeEnv.prepare,FakeBase.prepare),
        __shot_batch_chain__=(FakeEnv.shot_batch,), **{n:getattr(FakeEnv,n) for n in names})


class NativeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.run=self.root/'run'; self.run.mkdir()
        self.parent=self.root/'parent.pt'; self.parent.write_bytes(b'synthetic parent')
        self.camera=self.root/'camera.json'; self.camera.write_text('{}')
        self.poses=self.root/'poses.json'; self.poses.write_text(json.dumps(dict(dropper_joint_rad=[0.]*6,board_joint_rad=[0.]*6)))
        assets=self.root/'assets'; assets.mkdir()
        xml='<robot>'+''.join(f'<joint name="joint{i}" type="revolute"><origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-6.28" upper="6.28" effort="1"/></joint>' for i in range(1,7))+'</robot>'
        self.assets={a:assets/f for a,f in [('Arm10','dropper.urdf'),('Arm14','board_arm.urdf')]}
        for p in self.assets.values(): p.write_text(xml)
        self.cells=[dict(id=f'train-{i:03d}',split='train',bucket_xy_m=[.3+i*.01,-.1]) for i in range(35)]
        self.supports=[]; self.records={}
        for c in self.cells:
            values=((LOW+HIGH)/2).tolist()
            parent=decode_action(c,values,'parent-'+c['id'])
            alternate=list(values); alternate[0]=.1
            rows=[dict(candidate_id=c['id']+'-parent',action=parent),
                  dict(candidate_id=c['id']+'-other',action=decode_action(c,alternate,'other-'+c['id']))]
            support=freeze_support(c['id'],rows,parent,[0.]*5);self.supports.append(support)
            for row in rows:
                decision=dict(accepted=True,release_authorized=False,qualification=dict(ik={a:dict(q=[0.]*6) for a in self.assets}))
                decision['fingerprint']=geometry_hash(decision)
                self.records[row['candidate_id']]=dict(row,cell_id=c['id'],accepted=True,decision=decision)
        self.env=FakeEnv()
        self.env.n=35;self.env.cfg=copy.deepcopy(SCENE);self.env.spec=dict(fingerprint='a'*64,cells=copy.deepcopy(self.cells))
        self.env.base_cells=copy.deepcopy(self.cells); self.env.dest=self.run;self.env.dt=1/240
        self.env.continuous_links={a:chain(ET.fromstring(p.read_bytes())) for a,p in self.assets.items()}
        self.env.transit_limits=np.ones(6);self.env.origins=np.zeros((35,3));self.env.steps=0
        self.env.arms={a:types.SimpleNamespace(data=types.SimpleNamespace(default_joint_pos=np.zeros((35,8)))) for a in self.assets}
        self.env.poses=json.loads(self.poses.read_text())
        self.env.calls=[];self.env.reject_slots={1};self.env.capture=True
        self.env.raise_during_shot=self.env.bad_release_metric=False;self.env.physics_samples=[]
        for name in ('probes.json','rim-rays.json'): (self.run/name).write_text('{}')
        self.binding=subject.make_binding(scene=self.env.cfg,geometry_fingerprint='a'*64,
            camera_sha256=subject.digest(self.camera),assets_sha256={a:subject.digest(p) for a,p in self.assets.items()},
            poses_sha256=subject.digest(self.poses),
            cells=self.cells,canonical_home_rad={a:[0.]*6 for a in self.assets},supports=self.supports,prepared_records=self.records,
            startup_control_sha256={name:subject.digest(self.run/name) for name in ('probes.json','rim-rays.json')})
        self.ledger=Ledger.new(tuple(c['id'] for c in self.cells),bindings=SourceBindings(train_support_sha256=self.binding['support_sha256']))
        real_digest=subject.digest
        def fixture_digest(path):
            if Path(path)==self.parent and self.parent.read_bytes()==b'synthetic parent': return subject.PARENT_CHECKPOINT_SHA256
            return real_digest(path)
        self.patchers=[patch.object(subject,'_trusted_methods',side_effect=fixture_methods),
            patch.object(subject,'_frozen_identity',return_value={k:copy.deepcopy(self.binding[k]) for k in
                ('scene_sha256','geometry_fingerprint','camera_sha256','assets_sha256','poses_sha256')}),
            patch.object(subject,'digest',side_effect=fixture_digest),
            patch('candidate_native_audit.verify_controls',return_value=dict(synthetic_fixture=True))]
        for patcher in self.patchers: patcher.start();self.addCleanup(patcher.stop)

    def reserve(self,ledger=None):
        ledger=self.ledger if ledger is None else ledger
        batch=ledger.next_batch();ledger=ledger.reserve_batch(batch)
        by_support={s.context_id:s for s in self.supports}
        choices=[]
        for i,r in enumerate(batch.requests):
            s=by_support[r.context_id]; index=i%2
            choices.append(dict(request_id=r.request_id,context_id=r.context_id,candidate_id=s.candidate_ids[index],
                index=index,support_fingerprint=s.support_fingerprint,action=s.action(index),latent=list(s.latents[index]),
                old_log_probability=-.7,old_value=0.))
        journal=Journal(self.run/'journal')
        journal.append('start',dict(synthetic=True))
        journal.append('reserve',dict(ledger=ledger.to_dict(),choices=choices,policy_weight_hash='b'*64))
        runner=types.SimpleNamespace(directory=self.run,journal=journal,ledger=ledger,pending_choices=copy.deepcopy(choices),
            engine=types.SimpleNamespace(stage='ppo',state_dict=lambda:dict(weight_hash='b'*64)),
            last_weight='b'*64,failed=False,finished=False,synthetic=True,
            bindings=dict(source_sha256=self.binding['source_sha256'],support_sha256=self.binding['support_sha256']))
        return runner,choices

    def producer(self):
        return subject.GuardedNativeProducer(self.env,self.binding,self.cells,self.supports,self.records,
            parent_checkpoint_path=self.parent,camera_path=self.camera,assets_paths=self.assets,
            poses_path=self.poses,
            frozen_experiment_root=self.root/'frozen')

    def test_reserved_exact_actions_guarded_call_once_and_original_rewards(self):
        runner,choices=self.reserve();producer=self.producer()
        results=producer.execute_reserved(runner,choices,expected_journal_head=runner.journal.head)
        self.assertEqual(self.env.calls,['home','shot']);self.assertEqual(len(results),35)
        self.assertEqual(results[0]['return'],1.6);self.assertTrue(results[0]['captured']);self.assertTrue(results[0]['clean'])
        self.assertEqual(results[1]['return'],-1.);self.assertFalse(results[1]['released'])
        self.assertEqual(results[2]['return'],-.8)
        self.assertEqual(self.env.seen['actions'],[c['action'] for c in choices])
        for i,c in enumerate(choices):
            s=next(s for s in self.supports if s.context_id==c['context_id'])
            expected=None if c['index']==s.parent_index else self.records[c['candidate_id']]['decision']['qualification']['ik']
            self.assertEqual(self.env.seen['seeds'][i],expected)
        packet=json.loads(gzip.decompress((self.run/'native/train-000000.json.gz').read_bytes()))
        self.assertTrue(packet['synthetic']);self.assertEqual(packet['rows'][0]['settled'],METRIC)
        self.assertEqual(packet['rows'][0]['release_ball_center_m'],[.4+self.env.base_cells[0]['bucket_xy_m'][0],-.1,.283])
        self.assertEqual(results[0]['source_sha256'],subject.digest(self.run/'native/train-000000.json.gz'))
        with self.assertRaisesRegex(ValueError,'duplicate'):producer.execute_reserved(runner,choices,expected_journal_head=runner.journal.head)
        self.assertEqual(self.env.calls,['home','shot'])

    def test_new_adapter_cannot_reexecute_durable_dispatch_claim(self):
        runner,choices=self.reserve();self.producer().execute_reserved(runner,choices,expected_journal_head=runner.journal.head)
        with self.assertRaises((ValueError,FileExistsError)):self.producer().execute_reserved(runner,choices,expected_journal_head=runner.journal.head)
        self.assertEqual(self.env.calls,['home','shot'])

    def test_altered_choice_stale_token_and_unreserved_state_stop_before_home(self):
        runner,choices=self.reserve();producer=self.producer()
        changed=copy.deepcopy(choices);changed[0]['action']['height_m']+=.001
        for selected,token in ((changed,runner.journal.head),(choices,'f'*64)):
            with self.assertRaises(ValueError):producer.execute_reserved(runner,selected,expected_journal_head=token)
        runner.ledger=self.ledger
        with self.assertRaises(ValueError):producer.execute_reserved(runner,choices,expected_journal_head=runner.journal.head)
        self.assertEqual(self.env.calls,[])

    def test_changed_parent_camera_asset_caps_and_method_route_stop_before_home(self):
        original_cells=copy.deepcopy(self.cells); original_binding=copy.deepcopy(self.binding)
        expected=dict(parent='checkpoint',camera='camera or asset',asset='camera or asset',caps='effort caps',
            method='resolution',home='default home',cells='bank bytes',poses='pose bytes',
            measure='_measure',joint_status='_joint_status',home_definition='pose definition')
        for mode in ('parent','camera','asset','caps','method','home','cells','poses','measure','joint_status','home_definition'):
            with self.subTest(mode=mode):
                if mode=='parent':self.parent.write_bytes(b'changed')
                if mode=='camera':self.camera.write_text('changed')
                if mode=='asset':self.assets['Arm10'].write_text('changed')
                if mode=='caps':self.env.transit_limits[0]=2.
                if mode=='method':self.env.shot_batch=lambda *a,**kw:None
                if mode=='home':self.env.arms['Arm10'].data.default_joint_pos[0,0]=.1
                if mode=='cells':self.cells[0]['bucket_xy_m'][0]+=.1
                if mode=='poses':self.poses.write_text('{}')
                if mode=='measure':self.env._measure=lambda *a,**kw:None
                if mode=='joint_status':self.env._joint_status=lambda *a,**kw:None
                if mode=='home_definition':
                    self.binding['canonical_home_rad']['Arm10'][0]=.1
                    self.env.poses['dropper_joint_rad'][0]=.1
                    self.env.arms['Arm10'].data.default_joint_pos[:,0]=.1
                    self.binding['fingerprint']=canonical_hash({k:v for k,v in self.binding.items() if k!='fingerprint'})
                with self.assertRaisesRegex((ValueError,ET.ParseError),expected[mode]):self.producer()
                self.parent.write_bytes(b'synthetic parent');self.camera.write_text('{}')
                if mode=='asset':self.assets['Arm10'].write_bytes(self.assets['Arm14'].read_bytes())
                self.env.transit_limits=np.ones(6)
                if mode=='method':del self.env.shot_batch
                if mode=='home':self.env.arms['Arm10'].data.default_joint_pos[:]=0
                self.cells=copy.deepcopy(original_cells)
                self.binding=copy.deepcopy(original_binding)
                if mode=='poses':self.poses.write_text(json.dumps(self.env.poses))
                if mode=='measure':del self.env._measure
                if mode=='joint_status':del self.env._joint_status
                if mode=='home_definition':self.env.poses['dropper_joint_rad'][0]=0.
        self.assertEqual(self.env.calls,[])

    def test_prepared_warm_start_tamper_is_rejected_before_home(self):
        runner,choices=self.reserve();producer=self.producer()
        producer.prepared_records[choices[0]['candidate_id']]['decision']['qualification']['ik']['Arm10']['q'][0]=.1
        with self.assertRaisesRegex(ValueError,'bank bytes'):producer.execute_reserved(runner,choices,expected_journal_head=runner.journal.head)
        self.assertEqual(self.env.calls,[])

    def test_partial_final_native_batch_is_padded_but_only_one_request_is_returned(self):
        ledger=self.ledger
        for index in range(40):
            batch=ledger.next_batch()
            outcomes=tuple(Outcome(r.request_id,not (index==0 and j==0),'synthetic' if index==0 and j==0 else None)
                for j,r in enumerate(batch.requests))
            ledger=ledger.reserve_batch(batch).record_batch(outcomes)
        runner,choices=self.reserve(ledger)
        self.env.reject_slots=set();results=self.producer().execute_reserved(runner,choices,expected_journal_head=runner.journal.head)
        self.assertEqual(len(choices),1);self.assertEqual(len(results),1)
        self.assertEqual(self.env.seen['enabled'],[True]+[False]*34)
        self.assertTrue(all(c['split']=='inactive' for c in self.env.seen['cells'][1:]))
        self.assertEqual(runner.ledger.summary()['train']['requested'],1401)

    def test_bad_release_metric_stops_and_preserves_dispatch_without_retry(self):
        runner,choices=self.reserve();self.env.bad_release_metric=True
        with self.assertRaisesRegex(ValueError,'settling gates'):
            self.producer().execute_reserved(runner,choices,expected_journal_head=runner.journal.head)
        self.assertTrue((self.run/'native/train-000000.failure.json').exists())
        with self.assertRaises((ValueError,FileExistsError)):self.producer().execute_reserved(runner,choices,expected_journal_head=runner.journal.head)
        self.assertEqual(self.env.calls,['home','shot'])

    def test_interrupted_native_call_preserves_invalid_raw_samples_and_no_retry(self):
        runner,choices=self.reserve();self.env.raise_during_shot=True
        with self.assertRaisesRegex(RuntimeError,'interrupted'):self.producer().execute_reserved(runner,choices,expected_journal_head=runner.journal.head)
        partial=json.loads((self.run/'native/train-000000.incomplete-physics.json').read_text())
        self.assertEqual(partial['samples'][0]['position_m'][0][0],dict(nonfinite_numeric_repr='nan'))
        self.assertEqual(runner.ledger.summary()['train']['unresolved'],35)
        with self.assertRaises(FileExistsError):self.producer().execute_reserved(runner,choices,expected_journal_head=runner.journal.head)
        self.assertEqual(self.env.calls,['home','shot'])

    def test_production_trusted_inventory_has_no_fixture_bypass(self):
        source=(Path(subject.__file__)).read_text()
        self.assertNotIn('allow_fixture',source);self.assertNotIn('AppLauncher',source)
        self.assertNotIn('optimizer',source.replace('optimizer callback',''))
        # A subclass substitution is rejected even with the synthetic inventory.
        class Changed(FakeEnv):
            def prepare(self,actions): return None
        other=Changed();other.__dict__.update(self.env.__dict__)
        with self.assertRaisesRegex(ValueError,'delegation|resolution'):subject.validate_environment(other,self.binding)


@unittest.skipIf(torch is None, "configured CPU PyTorch environment unavailable")
class RealMethodResolutionTests(unittest.TestCase):
    def test_real_native_mro_passes_unmocked_guard_inventory_without_initialization(self):
        from continuous_pilot import ContinuousShots
        from feasibility_runtime import FeasibilityShots
        from parallel_runtime import ParallelShots
        from settled_transit import SettledTransitShots
        from shot_benchmark_native import BenchmarkShots

        # FilmShots adds filming behavior above FeasibilityShots. Its native
        # shot/guard resolution is reproduced here without importing Isaac or
        # invoking any application, scene or articulation constructor.
        class InspectionOnly(BenchmarkShots, FeasibilityShots):
            def __init__(self): raise AssertionError("No runtime initialization")
        env=object.__new__(InspectionOnly)
        env.n=35;env.cfg=copy.deepcopy(SCENE);env.spec=dict(fingerprint='a'*64)
        env.dt=1/240;env.continuous_links={'Arm10':[],'Arm14':[]};env.transit_limits=[]
        binding=dict(scene_sha256=geometry_hash(env.cfg),geometry_fingerprint='a'*64)
        trusted=subject._trusted_methods()
        self.assertIs(trusted['prepare'],BenchmarkShots.prepare)
        self.assertIs(trusted['approach_endpoints'],SettledTransitShots.approach_endpoints)
        self.assertIs(trusted['_measure'],SettledTransitShots._measure)
        self.assertIs(trusted['_joint_status'],SettledTransitShots._joint_status)
        self.assertIs(trusted['home'],ContinuousShots.home)
        self.assertIs(trusted['shot_batch'],FeasibilityShots.shot_batch)
        self.assertEqual(trusted['__shot_batch_chain__'],(FeasibilityShots.shot_batch,ParallelShots.shot_batch))
        subject.validate_environment(env,binding)
        # Substituting the older approach implementation must fail before any
        # native code can be called, even though all scene snapshot fields fit.
        class WrongApproach(InspectionOnly):
            approach_endpoints=ContinuousShots.approach_endpoints
        wrong=object.__new__(WrongApproach);wrong.__dict__.update(env.__dict__)
        with self.assertRaisesRegex(ValueError,'approach_endpoints'):
            subject.validate_environment(wrong,binding)
        for attribute in ('_measure','_joint_status'):
            overridden=type('WrongMetrics',(InspectionOnly,),{attribute:lambda *a,**kw:None})
            wrong=object.__new__(overridden);wrong.__dict__.update(env.__dict__)
            with self.assertRaisesRegex(ValueError,attribute):subject.validate_environment(wrong,binding)


if __name__=='__main__':unittest.main()
