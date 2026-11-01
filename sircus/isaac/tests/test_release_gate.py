"""Execute the real shot method with fake tensors/arms to prove fail-closed release."""

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class Tensor(np.ndarray):
    def clone(self):
        return self.copy().view(Tensor)


def tensor(value, **_):
    return np.array(value).view(Tensor)


class ReleaseGateTests(unittest.TestCase):
    def test_unsettled_board_never_releases_ball(self):
        fake = SimpleNamespace(zeros_like=np.zeros_like, tensor=tensor)
        spec = importlib.util.spec_from_file_location(
            "fixed_rl_gate_test", ROOT / "fixed_rl.py"
        )
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"torch": fake, "PIL": SimpleNamespace(Image=None)}):
            spec.loader.exec_module(module)

        class BlockedShot(module.FixedShot):
            def ik(self, target, name):
                return np.zeros(6)

            def tick(self, render=False):
                self.ticks += 1

            def before_release(self, target):
                self.checked_at = self.ticks
                return {
                    "invalid": "board_not_settled",
                    "settled_board": [{"position_error_mm": 9}],
                }

        class Arm:
            data = SimpleNamespace(default_joint_pos=tensor(np.zeros((1, 6))))
            joint_names = tuple(f"joint{i}" for i in range(1, 7))

            def write_joint_state_to_sim(self, *args):
                pass

            def set_joint_position_target(self, *args):
                pass

            def reset(self):
                pass

        writes = []
        env = BlockedShot.__new__(BlockedShot)
        env.cfg = json.loads((ROOT / "scene.json").read_text())
        env.episode = 0
        env.ticks = 0
        env.arms = {"Arm10": Arm(), "Arm14": Arm()}
        env.sim = SimpleNamespace(device="cpu")
        env.contact = SimpleNamespace(reset=lambda: None)
        env.ball = SimpleNamespace(
            data=SimpleNamespace(default_root_state=tensor(np.zeros((1, 13)))),
            write_root_state_to_sim=lambda state: writes.append(state.copy()),
            reset=lambda: None,
        )
        result = env.shot(np.array([24.0, 0.0, 0.15]))
        self.assertFalse(result["success"])
        self.assertEqual(result["invalid"], "board_not_settled")
        self.assertEqual(env.checked_at, 120)
        self.assertEqual(len(writes), 1)  # Parking only; no release-state write.
        np.testing.assert_allclose(writes[0][0, :3], [0.1, -0.8, 0.04])
        self.assertEqual(module.FixedShot.before_release(env, np.eye(4)), {})
