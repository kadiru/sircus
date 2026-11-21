"""CPU controls for opt-in path rejection without reduced accepted-path checks."""
import copy
from types import MethodType
import unittest

import numpy as np

from sircus.mjlab.calibrated_admission import SerializedPlanner
from sircus.mjlab.arm_path_speed import install_path_short_circuit


class RecordingGate:
    def __init__(self, fail_call=None):
        self.calls = []
        self.fail_call = fail_call

    def check(self, joints, *, corridor=None):
        self.calls.append(np.asarray(joints).tolist())
        passed = len(self.calls) != self.fail_call
        return dict(
            passed=passed,
            reason=None if passed else "clearance",
            minimum_clearance_m=.006 if passed else .004,
            closest_pair=["arm", "bucket"],
            board_top_m=.3,
            dropper_flange_z_m=.7,
        )


def planner(fail_call=None):
    # The audit only requires a gate. Avoid constructing models or GPU state.
    instance = object.__new__(SerializedPlanner)
    instance.gate = RecordingGate(fail_call)
    return instance


def points():
    return [np.full(12, value) for value in (0., .012, .024)]


class PathShortCircuitTests(unittest.TestCase):
    def test_accepted_proof_and_every_sample_match_original_exactly(self):
        baseline, faster = planner(), planner()
        original_class_method = SerializedPlanner._audit_path
        install_path_short_circuit(faster)
        expected = baseline._audit_path(points())
        actual = faster._audit_path(points())
        self.assertTrue(actual["passed"])
        self.assertEqual(actual, expected)
        self.assertEqual(faster.gate.calls, baseline.gate.calls)
        self.assertEqual(len(actual["edges"]), 2)
        self.assertEqual(len(actual["reverse_edges"]), 2)
        self.assertIs(SerializedPlanner._audit_path, original_class_method)
        self.assertIs(baseline._audit_path.__func__, original_class_method)

    def test_first_failed_forward_segment_stops_remaining_checks(self):
        baseline, faster = planner(fail_call=2), planner(fail_call=2)
        expected = baseline._audit_path(points())
        install_path_short_circuit(faster)
        actual = faster._audit_path(points())
        self.assertFalse(expected["passed"])
        self.assertFalse(actual["passed"])
        self.assertEqual(actual["edges"], expected["edges"][:1])
        self.assertEqual(actual["reverse_edges"], [])
        self.assertEqual(len(faster.gate.calls), 2)
        self.assertLess(len(faster.gate.calls), len(baseline.gate.calls))
        self.assertEqual(actual["unchecked_segment_count"], 3)
        self.assertFalse(actual["audit_complete"])
        self.assertEqual(actual["minimum_clearance_m"], .004)

    def test_failure_on_second_forward_segment_keeps_first_certificate(self):
        faster = install_path_short_circuit(planner(fail_call=5))
        result = faster._audit_path(points())
        self.assertFalse(result["passed"])
        self.assertTrue(result["edges"][0]["passed"])
        self.assertFalse(result["edges"][1]["passed"])
        self.assertEqual(result["reverse_edges"], [])
        self.assertEqual(len(faster.gate.calls), 5)
        self.assertEqual(result["unchecked_segment_count"], 2)

    def test_reverse_failure_stops_only_after_all_forward_checks(self):
        baseline, faster = planner(fail_call=8), planner(fail_call=8)
        expected = baseline._audit_path(points())
        install_path_short_circuit(faster)
        result = faster._audit_path(points())
        self.assertFalse(result["passed"])
        self.assertEqual(result["edges"], expected["edges"])
        self.assertEqual(result["reverse_edges"], expected["reverse_edges"][:1])
        self.assertEqual(len(faster.gate.calls), 8)
        self.assertEqual(result["unchecked_segment_count"], 1)

    def test_empty_and_single_point_match_historical_behavior(self):
        for path in ([], [np.zeros(12)]):
            with self.subTest(points=len(path)):
                baseline, faster = planner(), install_path_short_circuit(planner())
                self.assertEqual(faster._audit_path(path), baseline._audit_path(path))
                self.assertEqual(faster.gate.calls, [])

    def test_installation_is_instance_local_idempotent_and_preserves_inputs(self):
        instance = planner()
        path = points()
        before = copy.deepcopy(path)
        self.assertIs(install_path_short_circuit(instance), instance)
        bound = instance._audit_path
        self.assertIs(install_path_short_circuit(instance)._audit_path, bound)
        instance._audit_path(path)
        for actual, expected in zip(path, before):
            np.testing.assert_array_equal(actual, expected)

    def test_unknown_type_or_existing_override_is_not_replaced(self):
        with self.assertRaises(TypeError):
            install_path_short_circuit(object())
        instance = planner()
        instance._audit_path = MethodType(lambda self, points: {}, instance)
        original = instance._audit_path
        with self.assertRaises(ValueError):
            install_path_short_circuit(instance)
        self.assertIs(instance._audit_path, original)


if __name__ == "__main__":
    unittest.main()
