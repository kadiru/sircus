import unittest

from sircus.mjlab.table_assisted_groups import grouping


class CohortTests(unittest.TestCase):
    def classify(self, ident, **fields):
        return grouping("/study/native/shots.jsonl", dict(id=ident, **fields), "study")

    def test_actual_policy64_native_ids(self):
        for ident, method, phase in (
            ("validation-ppo-0000-policy64-validation-00", "ppo", "validation"),
            ("policy64-train-00000", "ppo", "train"),
            ("test-bc-0000-policy64-test-00", "bc", "test"),
            ("test-prior-0000-policy64-test-00", "prior", "test"),
            ("test-nearest-0000-policy64-test-00", "nearest", "test"),
            ("test-ppo-1024-policy64-test-00", "ppo", "test"),
        ):
            with self.subTest(ident=ident):
                out = self.classify(ident)
                self.assertEqual((out["method"], out["phase"]), (method, phase))

    def test_actual_benchmark_ids(self):
        for ident, method, phase in (
            ("ppo-20261411-train-00221", "ppo", "train"),
            ("ppo-20261411-test-1024-calibrated-test-00-r0", "ppo", "test"),
            ("sweep-20261411-validation-0000-calibrated-validation-00-r0", "sweep", "validation"),
        ):
            out = self.classify(ident)
            self.assertEqual((out["method"], out["phase"]), (method, phase))

    def test_later_comparison_and_control_ids(self):
        for ident, method, phase in (
            ("physics-ensemble-validation-bc-physics-ensemble-validation-00", "bc", "validation"),
            ("consistent-flight-ensemble-consistent-flight-test-00", "ensemble", "test"),
            ("gap-coverage-gap_coverage-gap-coverage-test-00", "gap_coverage", "test"),
            ("arm-clearance-arm_clearance-arm-clearance-test-01", "arm_clearance", "test"),
            ("controls_pre-positive_control-r0", "control", "controls_pre"),
            ("physics-ensemble-controls_post-negative_control-r0", "control", "controls_post"),
        ):
            out = self.classify(ident)
            self.assertEqual((out["method"], out["phase"]), (method, phase))

    def test_explicit_fields_are_preserved(self):
        out = self.classify("unknown", method="fixed_search", phase="controls", role="positive_control", substeps=128)
        self.assertEqual(out, dict(source_study="study", method="fixed_search", phase="controls", substeps=128))
        self.assertEqual(self.classify("unknown", contact_protocol={"substeps":64})["substeps"], 64)

    def test_unknown_and_ambiguous_stay_unknown(self):
        for ident in ("my-ppo-test", "test-ppo-0000-policy64-validation-00", "physics-ensemble-test-bc-physics-ensemble-validation-00",
                      "ppo-20261411-test-1024-calibrated-validation-00-r0"):
            out = self.classify(ident)
            self.assertEqual((out["method"], out["phase"], out["substeps"]), ("unspecified", "unspecified", "unknown"))
        out = grouping("/ppo-seed20261411/native/shots.jsonl", {"id":"opaque"}, "study")
        self.assertEqual((out["method"],out["phase"]), ("ppo", "unspecified"))
        self.assertEqual(self.classify("policy64-train-001", substeps=True)["substeps"], "unknown")


if __name__ == "__main__":
    unittest.main()
