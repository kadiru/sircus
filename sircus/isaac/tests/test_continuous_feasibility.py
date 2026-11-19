"""Geometry-only sampling and independent qualification contract checks."""

import copy
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
import continuous_feasibility as cf
from expanded_farm import make_spec as original_spec
from expanded_pose_menu import endpoint_targets
from prepare_assets import matrix
import json


class FeasibilityTests(unittest.TestCase):
    def setUp(self):
        self.scene = json.loads((SOURCE / "scene.json").read_text())
        self.source = original_spec(self.scene, 10)
        self.spec = cf.make_spec(self.source, 128)
        self.cell = self.source["cells"][1]

    def action(self, values=None):
        return cf.decode_action(self.cell, values if values is not None else [0, .4, .35, .24, math.radians(29)])

    def test_signed_new_contract_preserves_original_distinct_layout_and_uncertainty(self):
        self.assertEqual(self.spec["cells"], self.source["cells"])
        self.assertEqual(self.spec["scene"], self.scene)
        self.assertNotEqual(self.spec["fingerprint"], self.source["fingerprint"])
        self.assertEqual(self.spec["total_requests"], 4480)
        self.assertEqual(self.spec["scene"]["bucket"]["height_interpretation"], "unresolved: overall height or interior depth")
        unsigned = {k:v for k,v in self.spec.items() if k != "fingerprint"}
        self.assertEqual(cf.canonical_hash(unsigned), self.spec["fingerprint"])
        assignment = list(self.spec["split_assignment"].values())
        self.assertEqual([assignment.count(k) for k in ("train","validation","test")], [25,5,5])
        self.assertEqual(len({tuple(c["bucket_xy_m"]) for c in self.spec["cells"]}),35)

    def test_ik_protocol_changes_fingerprint_without_changing_sampled_requests(self):
        strong=cf.make_spec(self.source,128,"full_robust_v2")
        self.assertNotEqual(strong["fingerprint"],self.spec["fingerprint"])
        self.assertEqual(strong["ik_protocol"],"full_robust_v2")
        self.assertEqual(strong["ik_budget"]["version"],"sircus-endpoint-ik-v2")
        self.assertEqual(len(strong["ik_budget"]["normal_only_budgets"]),12)
        np.testing.assert_array_equal(cf.sample_values(strong,0),cf.sample_values(self.spec,0))
        with self.assertRaises(ValueError):cf.make_spec(self.source,128,"unbounded")

    def test_sobol_sampling_reproducible_broad_and_no_outcomes(self):
        first = cf.sample_values(self.spec, 0)
        np.testing.assert_array_equal(first, cf.sample_values(self.spec, 0))
        self.assertFalse(np.array_equal(first, cf.sample_values(self.spec, 1)))
        self.assertEqual(first.shape,(128,5))
        self.assertTrue((first >= cf.LOW).all() and (first <= cf.HIGH).all())
        # Every stratified marginal visits all quarters of its global range.
        scaled = (first-cf.LOW)/(cf.HIGH-cf.LOW)
        for dim in range(5):
            self.assertEqual(set((scaled[:,dim]*4).astype(int)), {0,1,2,3})
        changed = copy.deepcopy(self.source)
        changed["capture_outcomes"] = [True]*35
        np.testing.assert_array_equal(cf.sample_values(cf.make_spec(changed),0),first)
        for bad in (0,3,129,True,8192):
            with self.assertRaises(ValueError): cf.make_spec(self.source,bad)

    def test_relative_bearing_and_radian_tilt_decode_exact_geometry(self):
        first = self.action()
        turned = self.action([math.pi/2,.4,.35,.24,math.radians(29)])
        bucket = np.array(self.cell["bucket_xy_m"])
        a = np.array([first["board_x_m"],first["board_y_m"]])-bucket
        b = np.array([turned["board_x_m"],turned["board_y_m"]])-bucket
        np.testing.assert_allclose(b,[-a[1],a[0]],atol=1e-12)
        self.assertAlmostEqual(first["tilt_deg"],29)
        self.assertAlmostEqual(first["tilt_rad"],math.radians(29))
        self.assertAlmostEqual(first["azimuth_offset_deg"],0)
        t = endpoint_targets(self.scene,self.cell,first)
        self.assertAlmostEqual(t["release_center_m"][2],.35+.24+.0375)
        self.assertAlmostEqual(t["Arm10"][2,3],.35+.24+.0375+.095)

    def test_nonfinite_bounds_and_inconsistent_geometry_rejected(self):
        for v in ([0,.4,.35,.24,29],[0,.4,.35,float("nan"),.5],[0]*4):
            with self.assertRaises(ValueError): self.action(v)
        a = self.action(); a["board_x_m"] += .001
        with self.assertRaisesRegex(ValueError,"disagrees"):
            cf.geometry_check(self.scene,self.cell,a)

    def test_board_can_be_below_rim_and_release_uses_ball_center(self):
        a = self.action([0,.6,.14,.4,math.radians(12)])
        result = cf.geometry_check(self.scene,self.cell,a)
        self.assertLess(a["height_m"], result["metrics"]["bucket_rim_z_m"])
        self.assertTrue(result["accepted"],result)
        self.assertGreater(result["metrics"]["release_ball_center_m"][2],.183+.0375)

    def test_board_bounds_reject_table_and_bucket_proxy_overlap(self):
        a = self.action([0,.28,.14,.4,math.radians(52)])
        result = cf.geometry_check(self.scene,self.cell,a)
        self.assertIn("board_bucket_proxy_overlap",result["reasons"])
        lowered_scene = copy.deepcopy(self.scene)
        lowered_scene["table"]["center_m"][2] = .2
        result = cf.geometry_check(lowered_scene,self.cell,a)
        self.assertIn("board_table_clearance",result["reasons"])

    def test_segment_distance_handles_crossing_parallel_and_degenerate(self):
        self.assertAlmostEqual(cf.segment_distance([0,0,0],[1,0,0],[.5,-1,0],[.5,1,0]),0)
        self.assertAlmostEqual(cf.segment_distance([0,0,0],[1,0,0],[0,2,0],[1,2,0]),2)
        self.assertAlmostEqual(cf.segment_distance([0,0,0],[0,0,0],[0,2,0],[0,3,0]),2)
        self.assertAlmostEqual(cf.segment_distance([0,0,0],[1,0,0],[2,1,0],[2,1,0]),math.sqrt(2))

    def test_unresolved_ik_keeps_request_and_does_not_change_pose(self):
        a = self.action()
        links={name:[(matrix([0,0,1]),np.array([0,0,1]),-math.pi,math.pi)]*6 for name in ("Arm10","Arm14")}
        failure=dict(solved=False,q=[0]*6,residual=.1)
        with patch.object(cf,"outer_bounds",return_value=dict(eligible=True)), patch.object(cf,"bounded_solve",return_value=failure):
            row = cf.qualify_action(self.scene,self.cell,a,links,{name:[0]*6 for name in links},solver=lambda *args:failure)
        self.assertFalse(row["accepted"])
        self.assertEqual(row["reason"],"endpoint_ik_unresolved_Arm10")
        self.assertEqual(row["action"],a)
        self.assertEqual(len(row["ik"]),1)

    def test_successful_solver_claim_requires_independent_fk(self):
        a = self.action()
        links={name:[(matrix([0,0,1]),np.array([0,0,1]),-math.pi,math.pi)]*6 for name in ("Arm10","Arm14")}
        liar=dict(solved=True,q=[0]*6,residual=0)
        with patch.object(cf,"outer_bounds",return_value=dict(eligible=True)):
            with self.assertRaisesRegex(ValueError,"original residual"):
                cf.qualify_action(self.scene,self.cell,a,links,{name:[0]*6 for name in links},solver=lambda *args:liar)

    def test_geometry_coverage_is_diverse_deterministic_and_never_uses_rewards(self):
        records = []
        for i, cell in enumerate(self.source["cells"]):
            for j, radius in enumerate((.32, .4, .6, .8)):
                records.append(dict(cell_id=cell["id"], accepted=i != 0,
                    action=cf.decode_action(cell, [j*.3, radius, .2+.07*j, .12+.06*j, math.radians(20+4*j)], f"option-{j}"),
                    success=bool(j%2),ik={"Arm10":dict(q=[j]*6),"Arm14":dict(q=[-j]*6)}))
        feasible = dict(cells=[dict(cell_id=c["id"]) for c in self.source["cells"]],records=records)
        plan = cf.geometry_coverage(feasible,self.source["cells"],4)
        self.assertEqual(len(plan),4)
        self.assertTrue(all(not batch["enabled"][0] for batch in plan))
        self.assertTrue(all(sum(batch["enabled"])==34 for batch in plan))
        self.assertTrue(all(batch["fits"][0] is None for batch in plan))
        for batch in plan:
            chosen=int(batch["proposal_ids"][1].split("-")[-1])
            self.assertEqual(batch["fits"][1]["Arm10"]["q"],[chosen]*6)
        self.assertEqual(len({batch["proposal_ids"][1] for batch in plan}),4)
        self.assertTrue(all(batch["mode"] == "geometry_coverage" for batch in plan))
        for r in feasible["records"]: r["success"] = not r["success"]
        self.assertEqual(plan,cf.geometry_coverage(feasible,self.source["cells"],4))
        self.assertTrue(all("no policy updates" in batch["scope"] for batch in plan))
        with self.assertRaises(ValueError): cf.geometry_coverage(feasible,self.source["cells"],0)

    def test_proxy_path_preserves_all_samples_and_exposes_scope(self):
        links={name:[(matrix([0,0,.1]),np.array([0,0,1]),-math.pi,math.pi)]*6 for name in ("Arm10","Arm14")}
        starts={name:[0]*6 for name in links}; goals={name:[.3]*6 for name in links}
        with patch.object(cf,"approximate_clearance",return_value=dict(reasons=[],min_inter_arm_clearance_m=.2,min_arm_table_clearance_m=.1)), patch.object(cf,"_board_proxy_at_joints",return_value=dict(reasons=[],board_bottom_z_m=.3)):
            r=cf.approximate_path_check(self.scene,links,starts,goals,16,self.cell)
        self.assertTrue(r["accepted"])
        self.assertEqual(len(r["checks"]),16)
        self.assertEqual(r["checks"][0]["fraction"],0)
        self.assertEqual(r["checks"][-1]["fraction"],1)
        self.assertIn("not certified",r["scope"])
        for n in (0,1,129,True):
            with self.assertRaises(ValueError): cf.approximate_path_check(self.scene,links,starts,goals,n)


if __name__ == "__main__": unittest.main()
