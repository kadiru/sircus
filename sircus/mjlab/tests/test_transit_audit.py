import unittest
import os
import json
from pathlib import Path
import numpy as np
from sircus.mjlab.transit_audit import GeometricGate, joint_samples, pose_samples, audit_segment
from types import SimpleNamespace
from itertools import product
try:
    import mujoco
except ImportError:
    mujoco=None


class TransitTests(unittest.TestCase):
    def test_joint_sample_bound_includes_endpoints(self):
        a, b = np.zeros(12), np.linspace(-1., 1., 12)
        q = joint_samples(a, b)
        np.testing.assert_array_equal(q[0], a)
        np.testing.assert_array_equal(q[-1], b)
        self.assertLessEqual(np.abs(np.diff(q, axis=0)).max(), np.deg2rad(.5)+1e-12)

    def test_joint_malformed_or_unsafe_step_rejected(self):
        for first, step in [(np.zeros(6), .01), (np.full(12, np.nan), .01), (np.zeros(12), 0), (np.zeros(12), .1)]:
            with self.assertRaises(ValueError): joint_samples(first, np.zeros(12), step)

    def test_pose_translation_bound(self):
        a, b = np.eye(4), np.eye(4); b[0, 3] = .201
        path = pose_samples(a, b)
        self.assertEqual(len(path), 9)
        np.testing.assert_allclose(path[-1], b)
        self.assertLessEqual(np.diff(np.r_[0, path[:, 0, 3]]).max(), .025)

    def test_mid_path_collision_rejected_despite_clear_endpoints(self):
        class Gate:
            def check(self, q, corridor=None):
                hit = .4 < q[0] < .6
                return dict(passed=not hit, reason="clearance" if hit else None,
                    minimum_clearance_m=-.01 if hit else .1, closest_pair=["Arm10", "Arm14"])
        result = audit_segment(Gate(), np.zeros(12), np.ones(12))
        self.assertFalse(result["passed"])
        self.assertGreater(result["sample"], 0)
        self.assertLess(result["sample"], result["total_samples"]-1)

    def test_corridor_rejection_is_not_collision_success(self):
        class Gate:
            def check(self, q, corridor=None):
                return dict(passed=not bool(corridor), reason="corridor", minimum_clearance_m=.2)
        self.assertFalse(audit_segment(Gate(), np.zeros(12), np.ones(12)*.01, corridor={"board_top_max_m":.3})["passed"])

    def test_convex_separation_is_conservative_and_overlap_unknown(self):
        gate=GeometricGate.__new__(GeometricGate)
        cube=np.asarray(list(product((-1.,1.),repeat=3)))*.1
        gate.world_points=[cube,cube+[.208,0,0]]
        gate.local_axes=[np.eye(3),np.eye(3)]
        gate.d=SimpleNamespace(geom_xmat=np.asarray([np.eye(3).ravel()]*2))
        self.assertAlmostEqual(gate.separation_lower_bound(0,1),.008)
        gate.world_points[1]=cube+[.19,0,0]
        self.assertEqual(gate.separation_lower_bound(0,1),0.)


@unittest.skipUnless(mujoco is not None and os.environ.get("SIRCUS_IK_ASSETS"),"Optional pinned MuJoCo/native assets required")
class NativeTransitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[3]
        cls.scene=json.loads(Path(os.environ.get("SIRCUS_TRANSIT_SCENE",root/"sircus/isaac/scene.json")).read_text())
        catalogue=json.loads(Path(os.environ.get("SIRCUS_TRANSIT_CATALOGUE",root/"baselines/mjlab-ik-stability-20261102/candidate-catalogue.json")).read_text())
        cls.rows={r["target"]:r for r in catalogue["candidates"] if r.get("values")}

    def test_carried_ball_world_and_nonadjacent_self_pairs_exist(self):
        row=self.rows["old-test-21"]
        gate=GeometricGate(self.scene,os.environ["SIRCUS_IK_ASSETS"],row["bucket_xy_m"])
        pairs={frozenset([gate.names[i],gate.names[j]]) for i,j in gate.pairs}
        for name in ("table","floor","bucket/base"):
            self.assertIn(frozenset(["ball/geom",name]),pairs)
        self.assertIn(frozenset(["Arm14/link4/collision/0","Arm14/link6/collision/0"]),pairs)
        self.assertTrue(gate.check(np.r_[row["q10"],row["q14"]])["passed"])

    def test_old_self_disabled_endpoint_is_now_rejected(self):
        row=self.rows["old-test-39"]
        gate=GeometricGate(self.scene,os.environ["SIRCUS_IK_ASSETS"],row["bucket_xy_m"])
        result=gate.check(np.r_[row["q10"],row["q14"]])
        self.assertFalse(result["passed"])
        self.assertTrue(any(c["distance_m"]<0 and all(n.startswith("Arm14/") for n in c["pair"]) for c in result["margin_contacts"]))


if __name__ == "__main__": unittest.main()
