import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

from sircus.mjlab import corrected_grid as grid
from sircus.mjlab.learning_contract import digest


class GridTests(unittest.TestCase):
    def test_home_fixed_base_category_requires_actual_negative_contact(self):
        pair=["Arm10/link_base/collision/0","bucket/wall00"]
        self.assertEqual(grid.home_layout_summary(dict(passed=False,reason="clearance",margin_contacts=[dict(pair=pair,distance_m=-.01)]))["category"],"fixed_base_bucket_penetration")
        self.assertEqual(grid.home_layout_summary(dict(passed=False,reason="clearance",margin_contacts=[dict(pair=pair,distance_m=.001)]))["category"],"staging_clearance_blocked")
        self.assertEqual(grid.home_layout_summary(dict(passed=False,reason="unproven_clearance",unresolved_clearance_pairs=[dict(pair=pair,lower_bound_m=0)]))["category"],"staging_clearance_unproven")
        self.assertEqual(grid.home_layout_summary(dict(passed=True,reason=None))["category"],"home_clear")

    def test_proposals_are_fixed_unique_geometry_without_old_qualification(self):
        hint=dict(source_id="old",source_namespace="history",source_record_sha256="oldsha",
            values=[.6,-.2,.3,30,.3],q10=[0]*6,q14=[0]*6,bucket_xy_m=[.4,-.5])
        bank=dict(version="sircus-relative-search-bank-v1",templates=[dict(source_trial_id=f"old-{i}",
            relative_board_xy_m=[.2,-.3],height_m=.3+i*.002,tilt_deg=30,gap_m=.3,qualification="old_success") for i in range(17)])
        bank["sha256"]=digest(bank)
        inputs=dict(scene=dict(bucket=dict(segments=64)),bank=bank,template_hints=[hint]*17,seed_hints=[hint])
        plan=grid.proposal_plan(inputs,extra=0)
        self.assertEqual(len(plan),35*272);self.assertEqual(len({p["id"] for p in plan}),len(plan))
        self.assertTrue(all("qualification" not in p["proposal"] for p in plan))
        self.assertTrue(all("route_success" not in p for p in plan))
        self.assertEqual(plan,grid.proposal_plan(inputs,extra=0))

    def test_mock_flow_keeps_motion_reject_in_fresh20_and_qualifies_only_new_outcomes(self):
        cell=dict(id="dev-grid-00-00",row=0,column=0,role="development_then_train",bucket_xy_m=[.2,-.6])
        hint=dict(source_id="old",source_namespace="history",source_record_sha256="oldsha",
            values=[.5,-.2,.4,30,.3],q10=[0]*6,q14=[0]*6,bucket_xy_m=cell["bucket_xy_m"])
        requests=[dict(id=f"request-{i}",cell_id=cell["id"],cell_index=0,candidate_id=f"candidate-{i}",candidate_index=i,
            bucket_xy_m=cell["bucket_xy_m"],values=[.5+i*.01,-.2,.4,30,.3],proposal=dict(proposal_family="mock"),geometry_hint=hint) for i in range(2)]
        inputs=dict(scene={},parent_fingerprint="parent",bank=dict(sha256="bank"),bank_file_sha256="bankfile",
            source_studies=[],historical_studies=[],historical_coordinates=[cell["bucket_xy_m"]],asset_sha256={},
            reference_sha256="reference",reference_fingerprint="referencefp")
        class Geometry:
            def __init__(self,*args,**kw):pass
            def audit_values(self,xy,values,**kw):
                n=len(xy);mask=np.ones((n,1),bool)
                if n==2:mask[1,0]=False
                return mask,dict(q10=np.zeros((n,1,6)),q14=np.zeros((n,1,6)),reasons=np.where(mask,0,7).astype(np.uint8))
        class Admission:
            def __init__(self,*args,**kw):self.home=dict(sha256="home",passed=True,q10=[0]*6,q14=[0]*6)
            def check_home(self,xy):return dict(passed=True,reason=None,minimum_clearance_m=.01,common_home_sha256="home")
            def admit(self,row):return dict(passed=True,scope="mock_common_home",reason=None)
            def connect_variation(self,source,row):
                passed=not row["id"].endswith("confirmation-00")
                return dict(passed=passed,scope="mock_local_edge",reason=None if passed else "mock_path_reject")
        class Environment:
            def __init__(self,*args,**kw):
                self.last_arrays=None
                assert kw["substeps"]==64 and kw["gravity_mode"]=="passive_only" and kw["board_mode"]=="articulated"
                opt=SimpleNamespace(timestep=1/15360,solver=2,cone=0,integrator=3,iterations=100,tolerance=1e-8,gravity=np.asarray([0,0,-9.81]))
                self.sim=SimpleNamespace(mj_model=SimpleNamespace(opt=opt))
        def score(valid=True):
            return dict(success=valid,route_success=valid,route_quality=1 if valid else 0,complete_success=valid,
                board_bounce=valid,whole_ball_entry=valid,board_contact_episodes=1 if valid else 0,
                external_contact_steps=0,single_bounce_success=valid,physics_steps=3,
                minimum_mouth_clearance_mm=1,closest_opening_distance_m=0)
        def native(env,xy,values,q10,q14,eligible,record=False):
            k=int(eligible.sum());env.last_arrays=dict(position_m=np.zeros((3,k,3)),velocity_m_s=np.zeros((3,k,3)),forces_N=np.zeros((3,k,6)))
            return [dict(**score(bool(ok)),native_release=bool(ok),invalid=None if ok else "no_eligible_bank_endpoint",board_normal=[0,0,1]) for ok in eligible]
        class Scalar:
            def __init__(self,*args):pass
            def step(self,*args):pass
            def result(self):return score()
        @contextlib.contextmanager
        def slot(*args,**kw):
            kw["notify"](dict(status="waiting_for_4090",reason=["mock_shared_lock"]))
            yield None
        mj=SimpleNamespace(mjtSolver=SimpleNamespace(mjSOL_NEWTON=2),mjtCone=SimpleNamespace(mjCONE_PYRAMIDAL=0),mjtIntegrator=SimpleNamespace(mjINT_IMPLICITFAST=3))
        stubs={"sircus.mjlab.contact_env":SimpleNamespace(ContactEnvironment=Environment),
            "sircus.mjlab.bank_study":SimpleNamespace(run_bank_rows=native),"mujoco":mj}
        with tempfile.TemporaryDirectory() as tmp:
            args=SimpleNamespace(parent_study=Path(tmp),bank=Path(tmp),contact_reference=Path(tmp),assets=Path(tmp),
                output=Path(tmp)/"output",gpu_lock=Path(tmp)/"lock",source_study=[],historical_study=[],
                batch=128,extra_proposals=0,native_per_cell=1,top=1,queue_timeout=1,cpu_only=False)
            with (mock.patch.object(grid,"prepare_inputs",return_value=inputs),mock.patch.object(grid,"proposal_plan",return_value=requests),
                mock.patch.object(grid,"grid_layouts",return_value=[cell]),mock.patch.object(grid,"source_manifest",return_value={}),
                mock.patch("sircus.mjlab.corrected_geometry.CorrectedGeometry",Geometry),mock.patch("sircus.mjlab.corrected_admission.TransitAdmission",Admission),
                mock.patch("sircus.mjlab.gpu_slot.gpu_slot",slot),mock.patch("sircus.mjlab.contract.ScoreStream",Scalar),
                mock.patch.object(grid,"validate_packages",side_effect=lambda x:x),
                mock.patch.object(grid.importlib.metadata,"version",return_value="mock"),mock.patch.dict("sys.modules",stubs),contextlib.redirect_stdout(io.StringIO())):
                grid.run(args)
            report=json.loads((args.output/"report.json").read_text())
            rows=[json.loads(s) for s in (args.output/"shots.jsonl").read_text().splitlines()]
            fresh=[r for r in rows if r["phase"]=="confirmation"]
            self.assertEqual(report["status"],"completed");self.assertTrue(report["no_training"])
            self.assertEqual(len(fresh),20);self.assertEqual(sum(bool(r["invalid"]) for r in fresh),1)
            self.assertEqual(sum(r["route_success"] for r in fresh),19)
            self.assertEqual(len(report["qualified_sources"]),1)
            self.assertEqual(report["qualified_sources"][0]["source"]["id"],"request-0")
            self.assertTrue(all(r["confirmation_source_id"]=="request-0" for r in fresh))
            self.assertEqual(sum(r["status"]=="geometry_rejected" for r in rows),1)
            self.assertEqual(sum(r["status"]=="motion_rejected" for r in rows),1)
            self.assertEqual(report["native_attempts"],24)
            self.assertTrue((args.output/"common-home-layouts.json").is_file())
            self.assertEqual(report["counts"]["home_clear_positions"],1)


if __name__=="__main__":unittest.main()
