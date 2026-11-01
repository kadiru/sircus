import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from farm_contract import HIGH, LOW, actions, contract, observation, reward, splits


class FarmContractTests(unittest.TestCase):
    def test_fixed_distinct_targets_and_disjoint_reproducible_splits(self):
        groups = splits()
        self.assertEqual(groups, splits())
        self.assertEqual([len(g) for g in groups.values()], [32, 32, 32])
        self.assertEqual(
            len({tuple(c["bucket_xy_m"]) for g in groups.values() for c in g}), 96
        )
        for g in groups.values():
            for c in g:
                self.assertEqual(observation(c).shape, (5,))
                self.assertTrue(np.isfinite(observation(c)).all())

    def test_actions_remain_bounded_with_height_gap_tilt_order(self):
        result = actions([[-100] * 3, [100] * 3, [0] * 3])
        for r, expected in zip(result, [LOW, HIGH, (LOW + HIGH) / 2]):
            np.testing.assert_allclose(
                [r["height_m"], r["gap_m"], r["tilt_deg"]], expected
            )

    def test_illegal_route_and_invalid_never_receive_capture_reward(self):
        row = dict(
            invalid=None,
            external_contact_steps=0,
            board_bounce=True,
            closest_opening_distance_m=0,
            max_retention_s=1,
            route_success=True,
            route_quality=2,
            single_bounce_success=True,
        )
        self.assertGreater(reward(row), 1)
        for change, value in [
            ("invalid", "endpoint_ik_Arm10"),
            ("external_contact_steps", 1),
            ("board_bounce", False),
        ]:
            bad = copy.deepcopy(row)
            bad[change] = value
            self.assertLess(reward(bad), 0)

    def test_contract_bounds_and_scene_fingerprint(self):
        cfg = json.loads(
            (Path(__file__).resolve().parents[1] / "scene.json").read_text()
        )
        c = contract(cfg, 16, 4)
        self.assertEqual(c["training_throws"], 2048)
        self.assertFalse(c["bootstrap"])
        cfg["ball"]["radius_m"] += 0.001
        self.assertNotEqual(c["fingerprint"], contract(cfg, 16, 4)["fingerprint"])
        with self.assertRaises(ValueError):
            contract(cfg, 10000, 4)


@unittest.skipUnless(
    importlib.util.find_spec("torch"), "PPO tensor test runs in Isaac container"
)
class FarmPPOTests(unittest.TestCase):
    def test_real_ppo_update_changes_actor_and_frozen_inference_does_not(self):
        import torch
        from farm_policy import Policy, update, weight_hash

        torch.set_num_threads(1)
        torch.manual_seed(42)
        policy = Policy()
        obs = torch.tensor(np.array([observation(c) for c in splits()["train"]]))
        with torch.no_grad():
            d = policy.distribution(obs)
            z = d.sample()
            old = d.log_prob(z).sum(-1)
        original = {k: v.clone() for k, v in policy.state_dict().items()}
        spec = dict(
            epochs=4, clip=0.2, value_coef=0.5, entropy_coef=0.001, grad_norm=1.0
        )
        ret = torch.linspace(-1, 1, 32)
        log = update(
            policy,
            torch.optim.Adam(policy.parameters(), lr=0.003),
            obs,
            z,
            old,
            ret,
            spec,
        )
        self.assertNotEqual(log["before"], log["after"])
        self.assertGreater(
            sum(
                float((v - original[k]).abs().sum())
                for k, v in policy.state_dict().items()
                if k.startswith("actor.")
            ),
            0,
        )
        self.assertTrue(all(np.isfinite(r["loss"]) for r in log["epochs"]))
        before = weight_hash(policy)
        with torch.no_grad():
            one = policy.actor(obs)
            two = policy.actor(obs)
        self.assertTrue(torch.equal(one, two))
        self.assertEqual(before, weight_hash(policy))
        for r in actions(one.numpy()):
            self.assertTrue(LOW[0] <= r["height_m"] <= HIGH[0])


if __name__ == "__main__":
    unittest.main()
