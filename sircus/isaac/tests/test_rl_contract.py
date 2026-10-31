import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl_contract import HIGH, LOW, ShotScore, action_values, fingerprint


class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads(
            (Path(__file__).resolve().parents[1] / "scene.json").read_text()
        )
        self.b = self.cfg["bucket"]
        self.r = self.cfg["ball"]["radius_m"]
        self.inside = [*self.b["center_xy_m"], 0.06]

    def bounce(self, s):
        s.step([0.67, -0.1, 0.4], 30, 0.01, -1)
        s.step([0.67, -0.1, 0.41], 0, 0.01, 1)

    def test_contact_without_rebound_or_bucket_only_is_not_success(self):
        s = ShotScore(self.b, self.r)
        s.step([0.67, -0.1, 0.4], 30, 0.01, -1)
        for _ in range(200):
            s.step(self.inside, 0, 0.01, 0)
        self.assertFalse(s.success)

    def test_continuous_second_and_partial_exit_reset(self):
        s = ShotScore(self.b, self.r)
        self.bounce(s)
        for _ in range(70):
            s.step(self.inside, 0, 0.01)
        self.assertFalse(s.success)
        s.step([0.67, -0.42, 0.16], 0, 0.01)
        for _ in range(99):
            s.step(self.inside, 0, 0.01)
        self.assertFalse(s.success)
        s.step(self.inside, 0, 0.01)
        self.assertTrue(s.success)

    def test_sidewall_whole_sphere(self):
        s = ShotScore(self.b, self.r)
        self.bounce(s)
        for _ in range(200):
            s.step([0.73, -0.42, 0.06], 0, 0.01)
        self.assertFalse(s.success)

    def test_actions_and_fingerprint(self):
        np.testing.assert_allclose(action_values(np.zeros(3)), (LOW + HIGH) / 2)
        self.assertTrue(np.all(action_values(np.ones(3) * 100) <= HIGH))
        first = fingerprint(self.cfg)
        self.cfg["ball"]["restitution"] += 0.01
        self.assertNotEqual(first, fingerprint(self.cfg))

    def test_later_floor_rebound_cannot_substitute_for_board_bounce(self):
        s = ShotScore(self.b, self.r)
        s.step([0.67, -0.1, 0.4], 30, 0.01, -1)
        for _ in range(10):
            s.step([0.67, -0.1, 0.3], 0, 0.01, -1)
        s.step([0.67, -0.1, 0.2], 0, 0.01, 1)
        self.assertFalse(s.bounced)

    def test_wall_assisted_rest_uses_actual_polygon_planes(self):
        s = ShotScore(self.b, self.r)
        self.bounce(s)
        # Measured synthetic resting contact: fits a real wall plane, although
        # a smaller inscribed circular proxy falsely rejects the sphere.
        for _ in range(100):
            s.step([0.6633171439, -0.3803637326, 0.0455000028], 0, 0.01)
        self.assertTrue(s.success)

    def test_float32_contact_jitter_is_not_a_partial_exit(self):
        s = ShotScore(self.b, self.r)
        self.bounce(s)
        for i in range(100):
            s.step(
                [0.6633171439, -0.3803637326 + (2e-7 if i % 2 else 0), 0.0455000028],
                0,
                0.01,
            )
        self.assertTrue(s.success)
        s = ShotScore(self.b, self.r)
        self.bounce(s)
        for _ in range(100):
            s.step([0.6633171439, -0.3803637326 + 1e-4, 0.0455000028], 0, 0.01)
        self.assertFalse(s.success)


if __name__ == "__main__":
    unittest.main()
