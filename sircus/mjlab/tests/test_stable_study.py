"""Protocol tests for branch-seeded search; mocks never establish physics results."""

import contextlib
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from sircus.mjlab import stable_study as study
from sircus.mjlab.learning_contract import digest
from sircus.mjlab.learning_search import VERSION as BANK_VERSION


def row(target="old-test-10", identity="source", success=False, external=0):
    return dict(id=identity,target_id=target,phase="search",bound_phase="wide",
        source_phase="targeted_development",invalid=None,board_bounce=True,
        route_success=success,route_quality=2 if success else 0,success=success,
        whole_ball_entry=success,max_retention_s=1. if success else 0.,
        external_contact_steps=external,closest_opening_distance_m=.02,
        values=[.6,-.1,.32,28.,.3],bucket_xy_m=[.9,-.5],q10=[0.]*6,q14=[0.]*6,
        contact_peaks_N={name:0. for name in ("board","bucket","table","floor","robot","unattributed")},
        board_contact_episodes=1,single_bounce_success=success,minimum_mouth_clearance_mm=20.,
        native_release=True,physics_steps=480,prior_origin={"study_directory":"mock","record_sha256":"mock"})


class ProtocolTests(unittest.TestCase):
    def scene(self):
        return dict(ball=dict(radius_m=.0375),bucket=dict(assumed_interior_depth_m=.175,
            assumed_bottom_thickness_m=.008))

    def test_actual_ten_derive_from_prior_cache(self):
        indices=(8,9,10,12,19,21,23,26,35,39,41,48)
        targets=[dict(id=f"old-test-{i:02d}",original_test_index=i,bucket_xy_m=[float(i),0.]) for i in indices]
        mask=np.zeros((12,17),bool);mask[[4,11],0]=True
        actual=study.targets_from_cache(targets,[t["bucket_xy_m"] for t in targets],mask)
        self.assertEqual(tuple(t["original_test_index"] for t in actual),study.EXPECTED)
        mask[0,1]=True
        with self.assertRaises(ValueError):study.targets_from_cache(targets,[t["bucket_xy_m"] for t in targets],mask)

    def test_direct_partial_retention_beats_table_assisted_capture(self):
        assisted=row(success=False,external=1);assisted.update(whole_ball_entry=True,max_retention_s=1.,closest_opening_distance_m=.001)
        direct=row();direct.update(whole_ball_entry=True,max_retention_s=.63,closest_opening_distance_m=.04)
        self.assertGreater(study.acquisition(direct),study.acquisition(assisted))
        assisted["airborne_closest_opening_distance_m"]=.3
        assisted["bucket_contact_before_external"]=False
        closer=copy.deepcopy(assisted);closer["airborne_closest_opening_distance_m"]=.01
        self.assertGreater(study.acquisition(closer),study.acquisition(assisted))
        self.assertGreater(study.acquisition(row(success=True)),study.acquisition(direct))

    def test_hints_include_original_request_not_jittered_request(self):
        source=row(success=True);target=dict(bucket_xy_m=source["bucket_xy_m"])
        first=study.proposals(target,40,np.random.default_rng(8),self.scene(),[source],0)
        second=study.proposals(target,40,np.random.default_rng(8),self.scene(),[source],0)
        for a,b in zip(first[:5],second[:5]):np.testing.assert_array_equal(a,b)
        values,q10,q14,known_values,known_bucket,metadata=first
        np.testing.assert_array_equal(values[0],source["values"])
        local=next(i for i,m in enumerate(metadata) if m["proposal_family"]=="known-branch-continuation")
        np.testing.assert_array_equal(known_values[local],source["values"])
        self.assertFalse(np.array_equal(values[local],known_values[local]))
        np.testing.assert_array_equal(known_bucket[local],source["bucket_xy_m"])
        self.assertTrue(np.isnan(q10[4]).all())

    def test_pre_external_distance_ignores_later_assisted_entry(self):
        f=np.zeros((6,6));f[0,0]=10;f[2,2]=10;f[4,1]=10
        p=np.asarray([[.6,0,.3],[.7,0,.2],[.8,0,.1],[.9,0,.2],[1.,0,.1455],[1.,0,.1]])
        r=row();r.update(bucket_xy_m=[1.,0.],physics_steps=6)
        d=study.flight_summary(f,p,r,self.scene())
        self.assertFalse(d["bucket_contact_before_external"])
        self.assertEqual(d["first_external_contact_s"],3/240)
        self.assertGreater(d["airborne_closest_opening_distance_m"],.25)
        r["physics_steps"]=2
        terminal=study.flight_summary(f,p,r,self.scene())
        self.assertIsNone(terminal["first_external_contact_s"])

    def test_solver_comparison_retains_failed_masks(self):
        first=dict(mask=np.asarray([True,False]),q10=np.zeros((2,6)),q14=np.zeros((2,6)),reasons=np.asarray([0,3]))
        changed=copy.deepcopy(first);changed["mask"][1]=True
        self.assertFalse(study.compare_arrays(first,changed)["passed"])
        changed=copy.deepcopy(first);changed["q14"][0,0]=.1
        self.assertFalse(study.compare_arrays(first,changed)["passed"])

    def test_full_mock_screen10_then_final20_uses_real_qualification(self):
        targets=[dict(id=f"old-test-{i:02d}",original_test_index=i,bucket_xy_m=[.9,-.5],role="targeted_development") for i in study.EXPECTED]
        controls=[row(target=f"old-test-{i:02d}",identity=f"prior/control-{i}",success=True) for i in study.CONTROL_INDICES]
        seeds={t["id"]:[row(t["id"],"prior/"+t["id"],True)] for t in targets}
        sources=controls+[r for group in seeds.values() for r in group]
        bank=dict(version=BANK_VERSION,templates=[dict(source_trial_id=f"original-{i}") for i in range(17)],action_bounds=study.WIDE_BOUNDS)
        bank["sha256"]=digest(bank)
        def audit(scene,assets,xy,values,**kwargs):
            shape=values.shape[:2];n=np.prod(shape)
            detail=dict(q10=np.zeros((*shape,6)),q14=np.zeros((*shape,6)),reasons=np.zeros(shape,np.uint8),branch10=np.zeros(shape,int),branch14=np.zeros(shape,int),known_pair_preserved=np.zeros(shape,bool),known_request_match=np.zeros(shape,bool),preserved_exact_nominal=np.zeros(shape,bool),refined_continuation=np.ones(shape,bool))
            eligible=np.ones(shape,bool)
            if shape[0]>20:eligible[::10]=False
            return eligible,detail
        def audit_all(scene,assets,queries,output):
            n=len(queries);cache={}
            for method in ("old","stable_matched","stable_free","stable_seeded"):
                cache[method+"_q10"]=np.zeros((n,6));cache[method+"_q14"]=np.zeros((n,6));cache[method+"_mask"]=np.ones(n,bool)
            np.savez_compressed(output/"solver-audit.npz",**cache)
            return dict(order_batch_invariance=dict(passed=True),known_branch_preservation=dict(passed=True)),cache
        class Env:
            def __init__(self,scene,assets,n,bounds):self.n=n;self.sensors=[object()]
        class Recorder:
            def __init__(self,env):self.env=env
            def arrays(self):return np.zeros((2,self.env.n,6)),np.zeros((2,self.env.n,3))
        def native(env,xy,values,q10,q14,eligible,record=False):
            result=[]
            for i in range(len(xy)):
                r=row(success=bool(eligible[i]));r.update(bucket_xy_m=xy[i].tolist(),values=values[i].tolist(),q10=q10[i].tolist(),q14=q14[i].tolist(),invalid=None if eligible[i] else "no_eligible_bank_endpoint",native_release=bool(eligible[i]),board_bounce=bool(eligible[i]))
                result.append(r)
            return result
        @contextlib.contextmanager
        def slot(*args,**kwargs):yield None
        stubs={"torch":SimpleNamespace(set_num_threads=lambda n:None),
            "sircus.mjlab.stable_geometry":SimpleNamespace(VERSION="test",audit_values=audit),
            "sircus.mjlab.stable_ik":SimpleNamespace(VERSION="test"),
            "sircus.mjlab.expanded_env":SimpleNamespace(ExpandedEnvironment=Env)}
        with tempfile.TemporaryDirectory() as tmp:
            a=SimpleNamespace(parent_study=Path(tmp),prior_study=Path(tmp),prior_expansion=Path(tmp),assets=Path(tmp),output=Path(tmp)/"new",gpu_lock=Path(tmp)/"lock",source_study=[],native_per_target=1,proposal_rounds=1,proposals=1,seed_sources=1,top=1,control_repeats=1,batch=128,gpu_wait_timeout=1,audit_only=False)
            loaded=(self.scene(),targets,targets,bank,sources,controls,seeds,{})
            with mock.patch.object(study,"load_inputs",return_value=loaded),mock.patch.object(study,"solver_audit",side_effect=audit_all),mock.patch.dict(sys.modules,stubs),mock.patch("sircus.mjlab.gpu_slot.gpu_slot",slot),mock.patch("sircus.mjlab.bank_study.run_bank_rows",side_effect=native),mock.patch("sircus.mjlab.repeat_diagnostic.ContactRecorder",Recorder),mock.patch("sircus.mjlab.repeat_diagnostic.ObservedSensor",side_effect=lambda sensor,recorder:sensor),mock.patch.object(study.importlib.metadata,"version",return_value="mock"):
                study.run(a)
            report=json.loads((a.output/"report.json").read_text());new=json.loads((a.output/"confirmed-bank.json").read_text())
            self.assertEqual(report["native_search_attempts"],10)
            self.assertEqual(report["confirmation_attempts"],100)
            self.assertEqual(report["final_confirmation_attempts"],200)
            self.assertEqual(report["control_attempts"],30)
            self.assertEqual(report["additions"],10)
            self.assertEqual(new["templates"][:17],bank["templates"])
            for template in new["templates"][17:]:
                self.assertEqual(template["first_confirmation"]["attempts"],10)
                self.assertEqual(template["first_confirmation"]["captures"],9)
                self.assertGreaterEqual(template["final_confirmation"]["captures"],16)
                self.assertEqual(template["final_confirmation"]["attempts"],20)
                self.assertEqual(template["confirmation"]["attempts"],20)
            self.assertTrue(report["no_training"])


if __name__=="__main__":unittest.main()
