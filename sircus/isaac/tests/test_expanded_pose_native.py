"""Do not trust cached IK flags in place of exact endpoint residual checks."""

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from expanded_pose_native import verified_fit


class NativeFitTests(unittest.TestCase):
    def setUp(self):
        self.links = [
            (np.eye(4), np.array([0.0, 0.0, 1.0]), -np.pi, np.pi) for _ in range(6)
        ]
        self.fit = dict(q=[0.0] * 6, solved=True, residual=0.0)

    def test_valid_exact_fit_can_seed_cache(self):
        self.assertEqual(verified_fit(np.eye(4), self.fit, self.links, False), self.fit)

    def test_claimed_solved_flag_does_not_override_residual(self):
        target = np.eye(4)
        target[0, 3] = 0.001
        with self.assertRaisesRegex(ValueError, "residual"):
            verified_fit(target, self.fit, self.links, True)

    def test_board_normal_freedom_does_not_relax_dropper_rotation(self):
        target = np.eye(4)
        target[:3, :3] = Rotation.from_euler("z", 0.5).as_matrix()
        verified_fit(target, self.fit, self.links, True)
        with self.assertRaisesRegex(ValueError, "residual"):
            verified_fit(target, self.fit, self.links, False)

    def test_joint_limit_and_nonfinite_rejections(self):
        for value in (np.nan, 4.0):
            with self.assertRaisesRegex(ValueError, "limits"):
                verified_fit(
                    np.eye(4), dict(self.fit, q=[value] + [0.0] * 5), self.links, True
                )


if __name__ == "__main__":
    unittest.main()
