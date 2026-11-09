"""Mock native adapter checks: the rollout runs once, and inputs stay untouched."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from sircus.mjlab import calibrated_score, table_assisted_env as adapter
from sircus.mjlab.tests.test_calibrated_score import config, trace


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.scene = json.loads((Path(__file__).resolve().parents[3] / "sircus/isaac/scene.json").read_text())
        self.calibration = config(self.scene)
        observed = trace(self.scene)
        observed["forces_N"][100, 2] = 1.
        self.arrays = {k: v[:, None] if v.ndim >= 2 else v for k, v in observed.items()}
        score = calibrated_score.score_trace(self.scene, self.scene["bucket"]["center_xy_m"], [0., 0., 1.], self.calibration, observed)
        self.row = dict(score, id="mock-native-1", native_release=True,
                        native_initial_ball_center_m=observed["position_m"][0].tolist(),
                        bucket_xy_m=self.scene["bucket"]["center_xy_m"], board_normal=[0., 0., 1.],
                        q10=[1.] * 6, q14=[2.] * 6, values=[.6, -.2, .3, 28., .2])

    def test_rescoring_preserves_native_identity_and_old_reward(self):
        before = copy.deepcopy(self.row)
        row = adapter.rescore_native_rows(self.scene, self.calibration, [self.row], self.arrays)[0]
        self.assertFalse(before["route_success"])
        self.assertTrue(row["route_success"])
        self.assertTrue(row["table_assisted_capture"])
        self.assertEqual(row["historical_v2_score"]["reward"], before["reward"])
        self.assertEqual(self.row, before)
        for key in ("id", "q10", "q14", "values", "native_initial_ball_center_m"):
            self.assertEqual(row[key], before[key])

    def test_mixin_calls_original_rollout_once_and_does_not_change_native_timing(self):
        fixture = self
        class FakeNative:
            def run(self, *args, **kwargs):
                self.calls = getattr(self, "calls", 0) + 1
                self.last_arrays = fixture.arrays
                return [fixture.row], dict(native_rollout_s=7., seconds=8., score_and_oracle_s=.5)
        class Environment(adapter.TableAssistedScoringMixin, FakeNative):
            scene, calibration = fixture.scene, fixture.calibration
        env = Environment()
        rows, timing = env.run(None)
        self.assertEqual(env.calls, 1)
        self.assertTrue(rows[0]["route_success"])
        self.assertEqual(timing["native_rollout_s"], 7.)
        self.assertGreaterEqual(timing["seconds"], 8.)

    def test_missing_identity_or_independent_disagreement_fails_closed(self):
        bad = dict(self.row, native_release=False)
        with self.assertRaises(ValueError):
            adapter.rescore_native_rows(self.scene, self.calibration, [bad], self.arrays)
        with patch("sircus.mjlab.table_assisted_score_oracle.score_trace", return_value={"route_success": False}):
            with self.assertRaisesRegex(ValueError, "Independent v3"):
                adapter.rescore_native_rows(self.scene, self.calibration, [self.row], self.arrays)


if __name__ == "__main__":
    unittest.main()
