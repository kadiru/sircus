import copy
import importlib.util
from pathlib import Path
import unittest

import numpy as np

from sircus.mjlab.benchmark_contract import (
    VERSION,
    load_prior,
    make_splits,
    schedule,
    anchor_for,
    observation,
    action_from_latent,
    perturbation,
    protocol,
    validate_report,
)
from sircus.mjlab.learning_contract import digest


def signed(value):
    value = copy.deepcopy(value)
    value.pop("fingerprint", None)
    value["fingerprint"] = digest(value)
    return value


def completed_report():
    p = protocol(seeds=[7], budget=32, milestones=[0, 32])
    counts = dict(
        attempts=36,
        captures=20,
        native_releases=30,
        invalid=6,
        clean=10,
        mean_reward=1.0,
    )
    final = dict(
        attempts=72,
        captures=40,
        native_releases=60,
        invalid=12,
        clean=20,
        mean_reward=1.0,
    )
    value = dict(
        version=VERSION,
        status="completed",
        protocol=p,
        contract_fingerprint="a" * 64,
        curves=[
            dict(
                method=m,
                seed=7,
                samples=b,
                validation=copy.deepcopy(counts),
                resources={},
                artifact_sha256="b" * 64,
            )
            for m in ("sweep", "ppo")
            for b in (0, 32)
        ],
        final_test=[
            dict(
                method=m,
                seed=7,
                selected_samples=32,
                counts=copy.deepcopy(final),
                frozen_artifact_sha256="b" * 64,
            )
            for m in ("sweep", "ppo")
        ],
    )
    return signed(value)


class ContractTests(unittest.TestCase):
    def test_shared_prior_is_all_training_and_includes_old_wider_y(self):
        root = Path(__file__).resolve().parents[3]
        p = load_prior(root / "baselines/mjlab-grid35-expanded-20261103")
        self.assertEqual(len(p["records"]), 17)
        self.assertLess(min(r["values"][1] for r in p["records"]), -0.30)
        self.assertFalse(p["full_historical_cost_known"])
        self.assertEqual(p["latest_study_native_releases"], 4234)

    def test_fresh_splits_are_deterministic_distinct_and_spaced(self):
        pools = make_splits()
        self.assertEqual(pools, make_splits())
        self.assertEqual(
            [len(pools[k]) for k in ("train", "validation", "test")], [35, 12, 24]
        )
        xy = np.asarray([r["bucket_xy_m"] for group in pools.values() for r in group])
        distances = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
        distances += np.eye(len(xy)) * 100
        self.assertGreaterEqual(distances.min(), 0.005)

    def test_method_schedules_have_equal_budgets_and_balanced_contexts(self):
        a = schedule(7)
        self.assertEqual(a, schedule(7))
        self.assertEqual(len(a), 1024)
        counts = np.bincount(a, minlength=35)
        self.assertLessEqual(counts.max() - counts.min(), 1)

    def test_anchor_ties_are_fixed_by_id_without_outcomes(self):
        rows = [dict(id="b", bucket_xy_m=[0, 1]), dict(id="a", bucket_xy_m=[0, -1])]
        self.assertEqual(anchor_for([0, 0], rows)["id"], "a")

    def test_policy_transform_and_features_preserve_all_five_fields(self):
        a = dict(bucket_xy_m=[0.48, -0.65], values=[0.61, -0.444, 0.25, 37.0, 0.19])
        np.testing.assert_array_equal(
            action_from_latent(a["values"], np.zeros(5)), a["values"]
        )
        self.assertEqual(observation([0.5, -0.6], a).shape, (9,))
        result = action_from_latent(a["values"], [0, 0, 0, 0, 1])
        self.assertGreater(result[4], a["values"][4])
        np.testing.assert_array_equal(result[:4], a["values"][:4])

    def test_paired_perturbations_do_not_depend_on_method_or_training_seed(self):
        np.testing.assert_array_equal(perturbation("ctx", 0), np.zeros(5))
        np.testing.assert_array_equal(perturbation("ctx", 1), perturbation("ctx", 1))
        self.assertFalse(np.array_equal(perturbation("ctx", 1), perturbation("ctx", 2)))

    def test_completed_report_requires_every_method_seed_and_trial(self):
        validate_report(completed_report())
        for mutate in (
            lambda r: r["final_test"].__setitem__(1, copy.deepcopy(r["final_test"][0])),
            lambda r: r["final_test"][0]["counts"].__setitem__("attempts", 71),
            lambda r: r["final_test"][0].__setitem__("selected_samples", 16),
            lambda r: r["curves"][0]["validation"].__setitem__("attempts", 35),
        ):
            value = completed_report()
            mutate(value)
            with self.assertRaises(ValueError):
                validate_report(signed(value))


@unittest.skipUnless(
    importlib.util.find_spec("torch"), "Pinned Torch test runs on Ubuntu CPU"
)
class PolicyTests(unittest.TestCase):
    def test_real_cpu_ppo_changes_actor_using_saved_behavior_likelihood(self):
        import torch
        from sircus.mjlab.benchmark_policy import ResidualPolicy, update, weight_hash

        torch.manual_seed(11)
        policy = ResidualPolicy()
        x = torch.linspace(-1, 1, 32 * 9).reshape(32, 9)
        with torch.no_grad():
            dist = policy.distribution(x)
            z = dist.sample()
            logp = dist.log_prob(z).sum(-1)
            self.assertLess(
                float((policy.distribution(x).log_prob(z).sum(-1) - logp).abs().max()),
                1e-6,
            )
        before = weight_hash(policy.actor)
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.0003)
        update(policy, optimizer, x, z, logp, torch.linspace(-1, 3, 32))
        self.assertNotEqual(before, weight_hash(policy.actor))

    def test_sweep_can_only_observe_train_and_prediction_is_read_only(self):
        from sircus.mjlab.benchmark_policy import GuidedSweep

        contexts = [dict(id="a", bucket_xy_m=[0, 0]), dict(id="b", bucket_xy_m=[1, 1])]
        sweep = GuidedSweep(7, contexts)
        with self.assertRaises(ValueError):
            sweep.observe(dict(role="test"))
        anchor = dict(values=[0.6, -0.2, 0.3, 30, 0.2])
        before = digest(sweep.state())
        values, metadata = sweep.predict(dict(bucket_xy_m=[0.1, 0.1]), anchor)
        np.testing.assert_array_equal(values, anchor["values"])
        self.assertEqual(metadata["source_training_ids"], [])
        self.assertFalse(metadata["search_at_inference"])
        self.assertEqual(digest(sweep.state()), before)


if __name__ == "__main__":
    unittest.main()
