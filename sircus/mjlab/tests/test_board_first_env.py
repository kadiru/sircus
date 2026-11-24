"""Mock native adapter checks: rollout once, provenance kept, audit required."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from sircus.mjlab import calibrated_score, table_assisted_score, board_first_env as adapter
from sircus.mjlab.tests.test_calibrated_score import config, trace


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.scene = json.loads((Path(__file__).resolve().parents[3] / "sircus/isaac/scene.json").read_text())
        self.calibration = config(self.scene)
        observed = trace(self.scene)
        observed["forces_N"][100, 4] = 1.
        self.arrays = {k: v[:, None] if v.ndim >= 2 else v for k, v in observed.items()}
        score = calibrated_score.score_trace(self.scene, self.scene["bucket"]["center_xy_m"], [0., 0., 1.], self.calibration, observed)
        self.row = dict(score, id="mock-native-1", native_release=True,
                        native_initial_ball_center_m=observed["position_m"][0].tolist(),
                        bucket_xy_m=self.scene["bucket"]["center_xy_m"], board_normal=[0., 0., 1.],
                        q10=[1.] * 6, q14=[2.] * 6, values=[.6, -.2, .3, 28., .2])

    def test_preserves_identity_old_scores_and_inputs(self):
        before = copy.deepcopy(self.row)
        out = adapter.rescore_native_rows(self.scene, self.calibration, [self.row], self.arrays)[0]
        self.assertFalse(before["route_success"])
        self.assertTrue(out["route_success"])
        self.assertTrue(out["robot_assisted_capture"])
        self.assertEqual(out["historical_pre_board_first_score"]["reward"], before["reward"])
        self.assertEqual(self.row, before)
        for key in ("id", "q10", "q14", "values", "native_initial_ball_center_m"):
            self.assertEqual(out[key], before[key])

    def test_v3_original_is_supported_without_overwriting_prior_v2_snapshot(self):
        observed = {k: v[:, 0] if v.ndim >= 2 else v for k, v in self.arrays.items()}
        old = dict(self.row)
        old.update(table_assisted_score.score_trace(self.scene, old["bucket_xy_m"], old["board_normal"], self.calibration, observed))
        old["historical_v2_score"] = {"reward": .15}
        out = adapter.rescore_native_rows(self.scene, self.calibration, [old], self.arrays)[0]
        self.assertEqual(out["historical_v2_score"], {"reward": .15})
        self.assertEqual(out["historical_pre_board_first_score"]["scoring_version"], table_assisted_score.VERSION)

    def test_mixin_rollout_once_preserves_native_timing(self):
        fixture = self
        class FakeNative:
            def run(self, *args, **kwargs):
                self.calls = getattr(self, "calls", 0) + 1
                self.last_arrays = fixture.arrays
                return [fixture.row], dict(native_rollout_s=7., seconds=8., score_and_oracle_s=.5)
        class Environment(adapter.BoardFirstScoringMixin, FakeNative):
            scene, calibration = fixture.scene, fixture.calibration
        env = Environment()
        rows, timing = env.run(None)
        self.assertEqual(env.calls, 1)
        self.assertTrue(rows[0]["route_success"])
        self.assertEqual(timing["native_rollout_s"], 7.)
        self.assertGreaterEqual(timing["seconds"], 8.)

    def test_identity_old_fact_and_oracle_disagreement_fail_closed(self):
        for field, value in (("native_release", False), ("success", False), ("native_initial_ball_center_m", [0., 0., 0.])):
            with self.subTest(field=field), self.assertRaises(ValueError):
                adapter.rescore_native_rows(self.scene, self.calibration, [dict(self.row, **{field: value})], self.arrays)
        with self.assertRaisesRegex(ValueError, "identities"):
            adapter.rescore_native_rows(self.scene, self.calibration, [self.row], dict(self.arrays, record_ids=np.array(["wrong"])))
        with patch("sircus.mjlab.board_first_score_oracle.score_trace", return_value={"route_success": False}):
            with self.assertRaisesRegex(ValueError, "Independent v4"):
                adapter.rescore_native_rows(self.scene, self.calibration, [self.row], self.arrays)


if __name__ == "__main__":
    unittest.main()
