"""CPU mocks exercise admission logic; none of these rows are native evidence."""

import copy
from types import SimpleNamespace
import unittest

import numpy as np

from sircus.mjlab.clean650_geometry import CleanGeometry, MAX_PAIRS
from sircus.mjlab.learning_contract import digest


class Array:
    def __init__(self, value): self.value = np.asarray(value)
    def cpu(self): return self
    def numpy(self): return self.value


class Solver:
    def __init__(self, valid=True): self.calls, self.valid = [], valid
    def solve_candidates(self, targets, **kwargs):
        self.calls.append(dict(targets=targets.copy(), **kwargs))
        q = np.zeros((len(targets), 3, 6))
        q[:, :, 0] = np.arange(3)
        q[:, :, 1] = targets[:, 2, 3, None]
        return dict(q=Array(q), valid=Array(np.full(q.shape[:2], self.valid)))


class Gate:
    def __init__(self, mode="accept", clearance=.006): self.mode, self.clearance, self.calls = mode, clearance, []
    def check(self, q10, q14, bucket, values, targets):
        self.calls.append(dict(q10=q10.copy(), q14=q14.copy(), values=values.copy(), bucket=bucket.copy(), targets=copy.deepcopy(targets)))
        failed = (self.mode == "reject" or self.mode == "second_reject" and len(self.calls) > 1
                  or self.mode == "independent14" and q14[0] == 0
                  or self.mode == "cross" and q10[0] == q14[0] == 0)
        return dict(reason=7 if failed else 0, pairs=[], minimum_clearance_lower_bound_m=self.clearance,
                    independent_reject_arms=[14] if failed and self.mode == "independent14" else [])


def scene():
    return dict(arms=dict(base_spacing_m=1.33731), ball=dict(radius_m=.0375),
                board=dict(tool_offset_m=.1, mount_yaw_rad=np.pi/2),
                table=dict(center_m=[.5, -.3, -.02], size_m=[2., 2., .04]),
                bucket=dict(outer_top_diameter_m=.2, outer_bottom_diameter_m=.158))


def seed():
    return dict(id="source", role="candidate", values=[.47, -.27, .38, 28., .3868],
                bucket_xy_m=[.1, -.6], q10=[9.]*6, q14=[8.]*6,
                source_record_id="historical-source", source_record_sha256="a"*64)


class CleanGeometryTests(unittest.TestCase):
    def instance(self, mode="accept", valid=True, clearance=.006):
        ik = SimpleNamespace(dropper=Solver(valid), board=Solver(valid))
        gate = Gate(mode, clearance)
        geometry = CleanGeometry(scene(), "unused-mock-assets", seed(), ik=ik, gate=gate)
        return geometry, ik, gate

    def changed(self, ident="request"):
        request = seed()
        request.update(id=ident, role="coarse", phase="discovery")
        request["values"][2] += .02
        return request

    def test_only_exact_seed_request_preserves_its_original_pair(self):
        geometry, ik, gate = self.instance()
        original = seed()
        out = geometry.check_many([original])[0]
        self.assertTrue(out["eligible"])
        self.assertEqual(out["q10"], original["q10"])
        self.assertEqual(out["q14"], original["q14"])
        self.assertFalse(ik.dropper.calls)
        self.assertEqual(geometry.proofs[0]["source_mode"], "exact_seed")
        request = self.changed()
        out = geometry.check_many([request])[0]
        self.assertNotEqual(out["q10"], request["q10"])
        self.assertEqual(geometry.proofs[1]["source_mode"], "solved_cpu")
        self.assertFalse(ik.dropper.calls[0]["preserve_known"].any())
        self.assertEqual(ik.dropper.calls[0]["starts"], 16)
        self.assertEqual(ik.board.calls[0]["iterations"], 180)
        self.assertEqual(request["q10"], original["q10"])

    def test_explicit_controls_and_verification_are_checked_exactly(self):
        geometry, ik, gate = self.instance()
        control = self.changed("control")
        control.update(role="positive_control", q10=[1.]*6, q14=[2.]*6)
        clone = self.changed("verify")
        clone.update(role="candidate", phase="verification", freeze_joints=True, q10=[3.]*6, q14=[4.]*6)
        out = geometry.check_many([control, clone])
        self.assertEqual([r["q10"] for r in out], [[1.]*6, [3.]*6])
        self.assertEqual([r["q14"] for r in out], [[2.]*6, [4.]*6])
        self.assertEqual([p["source_mode"] for p in geometry.proofs], ["exact_control", "frozen_verification"])
        self.assertFalse(ik.dropper.calls)
        self.assertEqual(len(gate.calls), 2)

    def test_role_target_caching_deduplicates_math_but_never_collision_checks(self):
        geometry, ik, gate = self.instance()
        a = self.changed("first")
        b = copy.deepcopy(a); b["id"] = "gap"; b["values"][4] += .04
        c = copy.deepcopy(a); c["id"] = "duplicate"
        out = geometry.check_many([a, b, c])
        self.assertTrue(all(r["eligible"] for r in out))
        self.assertEqual([len(c["targets"]) for c in ik.dropper.calls], [2])
        self.assertEqual([len(c["targets"]) for c in ik.board.calls], [1])
        self.assertEqual(len(gate.calls), 3)
        geometry.check_many([dict(a, id="later")])
        self.assertEqual(len(ik.dropper.calls), 1)
        self.assertEqual(len(ik.board.calls), 1)
        self.assertEqual(len(gate.calls), 4)
        self.assertEqual([r["geometry_row"] for r in out], [0, 1, 2])
        for result, proof in zip(out, geometry.proofs):
            self.assertEqual(result["geometry_sha256"], digest(proof))

    def test_rejected_request_does_not_inherit_a_cached_collision_success(self):
        geometry, ik, gate = self.instance("second_reject")
        a = self.changed("accepted")
        b = dict(a, id="rejected")
        out = geometry.check_many([a, b])
        self.assertEqual([r["id"] for r in out], ["accepted", "rejected"])
        self.assertTrue(out[0]["eligible"])
        self.assertFalse(out[1]["eligible"])
        self.assertEqual(geometry.proofs[1]["reason"], "strict_endpoint_rejected")
        self.assertEqual(len(geometry.proofs[1]["attempts"]), 9)
        self.assertLessEqual(len(geometry.proofs[1]["attempts"]), MAX_PAIRS)
        self.assertEqual(len(ik.dropper.calls), 1)

    def test_independent_failure_prunes_that_arm_only(self):
        geometry, _, gate = self.instance("independent14")
        result = geometry.check_many([self.changed()])[0]
        self.assertTrue(result["eligible"])
        self.assertEqual(result["q10"][0], 0)
        self.assertEqual(result["q14"][0], 1)
        self.assertEqual(len(gate.calls), 2)
        self.assertEqual(geometry.proofs[0]["independent_pairs_pruned"], 2)
        geometry, _, gate = self.instance("cross")
        result = geometry.check_many([self.changed()])[0]
        self.assertTrue(result["eligible"])
        self.assertEqual(result["q10"][0], 1)
        self.assertEqual(result["q14"][0], 0)
        self.assertEqual(geometry.proofs[0]["independent_pairs_pruned"], 0)

    def test_bounds_footprint_and_separation_failures_remain_without_ik(self):
        geometry, ik, gate = self.instance()
        bounds = self.changed("bounds"); bounds["values"][2] = .119
        footprint = self.changed("footprint"); footprint["bucket_xy_m"] = [1.395001, -.3]
        distance = self.changed("distance"); distance["bucket_xy_m"] = [.595, -.27]
        out = geometry.check_many([bounds, footprint, distance])
        self.assertEqual([r["id"] for r in out], ["bounds", "footprint", "distance"])
        self.assertFalse(any(r["eligible"] for r in out))
        self.assertEqual([p["reason"] for p in geometry.proofs], ["action_bounds_or_invalid", "bucket_table_footprint", "board_bucket_distance"])
        self.assertFalse(ik.dropper.calls)
        self.assertFalse(gate.calls)

    def test_no_outer_sphere_prior_can_veto_a_full_endpoint_check(self):
        geometry, ik, gate = self.instance()
        request = self.changed()
        request["values"][2:] = [.58, 24., .6]
        out = geometry.check_many([request])[0]
        self.assertTrue(out["eligible"])
        self.assertTrue(ik.dropper.calls)
        self.assertEqual(len(gate.calls), 1)

    def test_solver_exhaustion_and_missing_frozen_joints_are_explicit(self):
        geometry, _, gate = self.instance(valid=False)
        result = geometry.check_many([self.changed()])[0]
        self.assertFalse(result["eligible"])
        self.assertEqual(geometry.proofs[0]["reason"], "ik_unresolved")
        self.assertFalse(gate.calls)
        geometry, ik, gate = self.instance()
        control = self.changed(); control.update(role="negative_control"); control.pop("q14")
        result = geometry.check_many([control])[0]
        self.assertFalse(result["eligible"])
        self.assertEqual(geometry.proofs[0]["reason"], "invalid_frozen_joints")
        self.assertFalse(gate.calls)

    def test_success_without_required_clearance_proof_fails_closed(self):
        geometry, _, _ = self.instance(clearance=.0049)
        with self.assertRaisesRegex(RuntimeError, "5 mm clearance"):
            geometry.check_many([self.changed()])


if __name__ == "__main__":
    unittest.main()
