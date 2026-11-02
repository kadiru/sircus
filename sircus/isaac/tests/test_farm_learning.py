import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import farm_learning as learning

try:
    import torch
except ImportError:
    torch = None


def row(**kwargs):
    return dict(
        invalid=None,
        board_bounce=True,
        external_contact_steps=0,
        closest_opening_distance_m=0.2,
        max_retention_s=0,
        route_success=False,
        route_quality=0,
        single_bounce_success=False,
        **kwargs,
    )


class TestLearningContract(unittest.TestCase):
    def setUp(self):
        from expanded_farm import HIGH, LOW, make_spec

        self.low, self.high = LOW, HIGH
        scene = json.loads(
            Path(__file__).resolve().parents[1].joinpath("scene.json").read_text()
        )
        self.spec = make_spec(scene, 10)

    def test_new_budget_identity_does_not_relabel_old_task(self):
        spec = copy.deepcopy(self.spec)
        short = learning.study_contract(spec, "a" * 64, 8, 2)
        longer = learning.study_contract(spec, "a" * 64, 128, 16)
        self.assertNotEqual(short["fingerprint"], longer["fingerprint"])
        self.assertEqual(spec, self.spec)
        self.assertEqual(longer["training_throws"], 4480)
        for invalid in (7, 9, 514):
            with self.assertRaises(ValueError):
                learning.study_contract(spec, "a" * 64, invalid, 16)

    def test_duplicate_bucket_layout_is_rejected(self):
        spec = copy.deepcopy(self.spec)
        spec["cells"][1]["bucket_xy_m"] = spec["cells"][0]["bucket_xy_m"]
        with self.assertRaises(ValueError):
            learning.study_contract(spec, "a" * 64)

    def test_bearing_wrap_is_continuous(self):
        action = dict(zip(learning.PARAMETERS, [179.999, 0.4, 0.3, 0.2, 28]))
        a = learning.shot_features(action, self.low, self.high)
        action["azimuth_offset_deg"] = -179.999
        b = learning.shot_features(action, self.low, self.high)
        self.assertLess(np.linalg.norm(a - b), 0.0001)

    def test_candidate_features_exclude_identity_and_outcomes(self):
        from expanded_farm import shot

        cells = self.spec["cells"][:2]
        action = [0, 0.4, 0.3, 0.2, 28]
        menus = [[shot(cells[0], action)], [shot(cells[1], action)] * 2]
        obs, features, mask = learning.candidate_tensor_data(
            cells, menus, self.low, self.high
        )
        self.assertEqual(features.shape, (2, 2, 11))
        self.assertEqual(mask.tolist(), [[True, False], [True, True]])
        np.testing.assert_array_equal(features[0, 0, :5], obs[0])
        changed = copy.deepcopy(menus)
        for options in changed:
            for entry in options:
                entry.update(id="different-id", route_success=True, reward=1000)
        np.testing.assert_array_equal(
            features,
            learning.candidate_tensor_data(cells, changed, self.low, self.high)[1],
        )

    def test_padded_or_noninteger_indices_are_rejected(self):
        menus = [[{"id": "one"}], [{"id": "two"}, {"id": "three"}]]
        self.assertEqual(
            [a["id"] for a in learning.choose_actions(menus, [0, 1])],
            ["one", "three"],
        )
        for invalid in ([1, 0], [-1, 0], [0.0, 1.0]):
            with self.assertRaises(ValueError):
                learning.choose_actions(menus, invalid)


class TestLearningReward(unittest.TestCase):
    def test_external_misses_still_have_distance_gradient_without_capture_bonus(self):
        near = row()
        near.update(external_contact_steps=1, closest_opening_distance_m=0.01)
        far = dict(near, closest_opening_distance_m=1.0)
        self.assertGreater(
            learning.learning_reward(near), learning.learning_reward(far)
        )
        self.assertLessEqual(learning.learning_reward(near), 0)
        near["max_retention_s"] = 1
        self.assertLessEqual(learning.learning_reward(near), 0)

    def test_unqualified_entry_does_not_get_success_bonus(self):
        miss = row()
        miss.update(closest_opening_distance_m=0, max_retention_s=1)
        capture = dict(
            miss, route_success=True, route_quality=2, single_bounce_success=True
        )
        self.assertAlmostEqual(
            learning.learning_reward(capture) - learning.learning_reward(miss), 1.15
        )
        with self.assertRaises(ValueError):
            learning.learning_reward(dict(capture, external_contact_steps=1))

    def test_no_board_contact_cannot_gain_proximity_reward(self):
        miss = row()
        miss.update(board_bounce=False, closest_opening_distance_m=0, max_retention_s=1)
        self.assertEqual(learning.learning_reward(miss), -0.25)
        self.assertEqual(
            learning.learning_reward(dict(miss, invalid="not_settled")), -1
        )


class TestHeadlessIsolation(unittest.TestCase):
    def test_original_methods_are_restored_after_native_failure(self):
        from farm_learning_runtime import endpoint_mode

        class Original:
            training_active = True
            preparation_enabled = [True]
            record_video = True

            def prepare(self, actions):
                return "original-prepare"

            def approach_endpoints(self, targets):
                return "original-drive"

            def tick(self):
                return "original-tick"

            def render(self):
                return "original-render"

        class Headless:
            def prepare(self, actions):
                return "endpoint-prepare"

            def approach_endpoints(self, targets):
                return "endpoint-reset"

            def tick(self):
                return "native-physics"

        obj = Original()
        with mock.patch.dict(
            sys.modules,
            {
                "feasibility_runtime": SimpleNamespace(FeasibilityShots=Headless),
                "parallel_runtime": SimpleNamespace(ParallelShots=Headless),
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "native failure"):
                with endpoint_mode(obj):
                    self.assertEqual(obj.prepare([]), "endpoint-prepare")
                    self.assertEqual(obj.approach_endpoints({}), "endpoint-reset")
                    self.assertEqual(obj.tick(), "native-physics")
                    self.assertIsNone(obj.render())
                    self.assertFalse(obj.training_active)
                    self.assertFalse(obj.record_video)
                    raise RuntimeError("native failure")
        self.assertEqual(obj.prepare([]), "original-prepare")
        self.assertEqual(obj.approach_endpoints({}), "original-drive")
        self.assertEqual(obj.tick(), "original-tick")
        self.assertEqual(obj.render(), "original-render")
        self.assertTrue(obj.training_active)
        self.assertTrue(obj.record_video)


@unittest.skipIf(
    torch is None, "Torch unavailable on this Mac; run in Ubuntu CPU environment"
)
class TestCategoricalPolicy(unittest.TestCase):
    def test_uniform_masked_exploration_and_actual_ppo_update(self):
        from farm_policy import weight_hash

        torch.manual_seed(4)
        policy = learning.make_policy()
        features = torch.randn(8, 4, 11)
        mask = torch.tensor([[True, True, True, False]] * 8)
        observations = features[:, 0, :5]
        dist = policy.distribution(features, mask)
        np.testing.assert_allclose(
            dist.probs.detach().numpy(), [[1 / 3, 1 / 3, 1 / 3, 0]] * 8, atol=1e-7
        )
        indices = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
        old_logp = dist.log_prob(indices).detach()
        returns = torch.tensor([1.0, 0.0, -0.5, 1, 0, -0.5, 1, 0])
        before = weight_hash(policy)
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
        report = learning.ppo_update(
            policy,
            optimizer,
            observations,
            features,
            mask,
            indices,
            old_logp,
            returns,
            dict(epochs=4, clip=0.2, value_coef=0.5, entropy_coef=0.01, grad_norm=1.0),
        )
        self.assertEqual(report["before"], before)
        self.assertNotEqual(report["after"], before)
        self.assertTrue(all(math.isfinite(e["loss"]) for e in report["epochs"]))
        with self.assertRaises(ValueError):
            policy.distribution(features, torch.zeros_like(mask))


if __name__ == "__main__":
    unittest.main()
