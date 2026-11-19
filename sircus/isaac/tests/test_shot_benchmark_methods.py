"""Synthetic CPU method evidence only; no native capture claim."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shot_benchmark_contract as contract
import shot_benchmark_methods as methods
import continuous_policy


def train_cells():
    return contract.make_splits([[2., 2.]])["train"]


def observed(metadata, reward=-1.):
    return dict(proposal_id=metadata["proposal_id"], cell_id=metadata["cell_id"],
        bucket_xy_m=metadata["bucket_xy_m"], actions_normalized=metadata["actions_normalized"],
        reward=reward, split="train", invalid="synthetic_rejection" if reward < 0 else None)


class SweepTests(unittest.TestCase):
    def setUp(self):
        self.cells = train_cells()
        self.sweep = methods.GuidedSweep(self.cells, contract.SEEDS[0])

    def test_declared_prior_and_continuous_decoder_are_shared(self):
        np.testing.assert_array_equal(methods.INITIAL_NORMALIZED, continuous_policy.INITIAL_NORMALIZED)
        np.testing.assert_array_equal(methods.INITIAL_LOGSTD, continuous_policy.INITIAL_LOGSTD)
        queries = [dict(bucket_xy_m=[.2, -.5]), dict(bucket_xy_m=[.67, -.1]), dict(bucket_xy_m=[2., -.1])]
        predicted = self.sweep.predict(queries)
        prior = methods.geometry_prior(queries)
        np.testing.assert_array_equal(predicted["actions_normalized"], prior["actions_normalized"])
        physical = methods.decode_normalized(prior["actions_normalized"])
        np.testing.assert_allclose(physical[:, 1], [np.linalg.norm([.47, .4]), .29, 1.19])
        np.testing.assert_allclose(physical[:, [0, 2, 3, 4]], np.tile([0, .35, .24, np.radians(29)], (3, 1)))
        changed = copy.deepcopy(queries)
        for q in changed:
            q.update(id="new", reward=999, old_success=True)
        np.testing.assert_array_equal(self.sweep.predict(changed)["actions_normalized"], predicted["actions_normalized"])
        first = methods.decode_normalized([[0.] * 5])
        next_value = methods.decode_normalized([[1e-8] * 5])
        self.assertTrue((next_value > first).all())

    def test_all_rejects_are_paid_and_best_is_not_filtered(self):
        request = self.sweep.propose(self.cells)
        self.assertEqual(self.sweep.issued, 35)
        with self.assertRaises(ValueError):
            self.sweep.propose(self.cells[:1])
        self.sweep.observe([observed(p) for p in request["proposal_metadata"]])
        self.assertEqual(self.sweep.observed, 35)
        self.assertEqual(len(self.sweep.best), 35)
        self.assertFalse(self.sweep.pending)
        self.assertTrue(all(value["reward"] == -1 for value in self.sweep.best.values()))
        with self.assertRaises(ValueError):
            self.sweep.observe([observed(request["proposal_metadata"][0])])

    def test_prescheduled_phases_full_domain_and_training_dependencies(self):
        phases, proposals = [], []
        for index in range(49):
            request = self.sweep.propose(self.cells[:1])
            proposal = request["proposal_metadata"][0]
            phases.append(proposal["phase"])
            proposals.append(proposal)
            self.assertTrue(np.isfinite(request["actions_normalized"]).all())
            self.assertTrue((np.abs(request["actions_normalized"]) <= 1).all())
            self.sweep.observe([observed(proposal, 2. if index == 22 else -1.)])
        self.assertEqual(phases[0], "shared_geometry_prior")
        self.assertEqual(set(phases[1:11]), {"signed_prior_axis"})
        self.assertEqual(set(phases[11:43]), {"local_sobol_prior"})
        self.assertEqual(phases[43], "global_domain_sobol")
        self.assertEqual(phases[47], "global_domain_sobol")
        self.assertEqual(proposals[44]["source_training_proposal_id"], proposals[22]["proposal_id"])
        self.assertEqual(proposals[44]["refinement_scale"], 1.)
        self.assertEqual(len({p["proposal_id"] for p in proposals}), 49)
        self.assertGreater(np.max(np.abs(np.asarray(proposals[43]["actions_normalized"]) - methods.geometry_prior(self.cells[:1])["actions_normalized"][0])), .25)

    def test_observation_tampering_or_heldout_data_is_atomic(self):
        request = self.sweep.propose(self.cells[:2])
        rows = [observed(p) for p in request["proposal_metadata"]]
        snapshot = self.sweep.state()
        for field, value in [("split", "test"), ("cell_id", "test-000"), ("reward", float("nan")),
                             ("actions_normalized", [0.] * 5), ("bucket_xy_m", [99., 99.])]:
            bad = copy.deepcopy(rows)
            bad[1][field] = value
            with self.assertRaises(ValueError):
                self.sweep.observe(bad)
            self.assertEqual(self.sweep.state(), snapshot)
        with self.assertRaises(ValueError):
            self.sweep.observe(rows + rows[:1])
        self.sweep.observe(rows)
        with self.assertRaises(ValueError):
            self.sweep.propose([dict(id="test-000", split="test", bucket_xy_m=[.4, -.3])])

    def test_pending_and_frozen_serialization_reproduce_next_proposals(self):
        request = self.sweep.propose(self.cells[:3])
        pending = methods.GuidedSweep.from_state(json.loads(json.dumps(self.sweep.state())), train_cells=self.cells)
        rows = [observed(p, i - 1.) for i, p in enumerate(request["proposal_metadata"])]
        self.sweep.observe(rows)
        pending.observe(rows)
        self.assertEqual(pending.state(), self.sweep.state())
        copy_state = methods.GuidedSweep.from_state(self.sweep.state())
        first, second = self.sweep.propose(self.cells[:3]), copy_state.propose(self.cells[:3])
        np.testing.assert_array_equal(first["actions_normalized"], second["actions_normalized"])
        self.assertEqual(first["proposal_metadata"], second["proposal_metadata"])

    def test_frozen_unseen_prediction_transfers_residual_without_mutation(self):
        for index in range(2):
            request = self.sweep.propose(self.cells[:1])
            self.sweep.observe([observed(request["proposal_metadata"][0], float(index))])
        query = dict(id="never-seen", split="test", bucket_xy_m=(np.asarray(self.cells[0]["bucket_xy_m"]) + [.00001, .00001]).tolist())
        before = self.sweep.state()
        result = self.sweep.predict([query])
        expected_latent = methods.geometry_prior([query])["latent"][0] + np.asarray(self.sweep.best[self.cells[0]["id"]]["latent_residual"])
        np.testing.assert_array_equal(result["actions_normalized"][0], np.tanh(expected_latent))
        self.assertEqual(result["prediction_metadata"][0]["nearest_training_cell_id"], self.cells[0]["id"])
        self.assertEqual(result["prediction_metadata"][0]["ball_flight_queries"], 0)
        self.assertFalse(result["prediction_metadata"][0]["search_at_inference"])
        self.assertEqual(self.sweep.state(), before)

    def test_state_context_guards_and_paid_count_guards(self):
        request = self.sweep.propose(self.cells[:1])
        self.sweep.observe([observed(request["proposal_metadata"][0])])
        for mutate in [lambda d: d["observed_counts"].update({self.cells[0]["id"]: 0}),
                       lambda d: d["best"][self.cells[0]["id"]]["latent_residual"].__setitem__(0, 5.),
                       lambda d: d["best"][self.cells[0]["id"]].update(actions_normalized=[[0.] * 5]),
                       lambda d: d["best"][self.cells[0]["id"]].update(test_outcome=1.),
                       lambda d: d.update(domain_fingerprint="a" * 64),
                       lambda d: d["best"].update({"validation-000": {}})]:
            state = self.sweep.state()
            state.pop("fingerprint")
            mutate(state)
            with self.assertRaises(ValueError):
                methods.GuidedSweep.from_state(contract.signed(state))
        altered = copy.deepcopy(self.cells)
        altered[0]["bucket_xy_m"][0] += .01
        with self.assertRaises(ValueError):
            methods.GuidedSweep.from_state(self.sweep.state(), train_cells=altered)
        with self.assertRaises(ValueError):
            methods.GuidedSweep.from_state(self.sweep.state(), protocol_fingerprint="b" * 64)

    def test_frozen_file_is_exclusive_and_requires_all_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sweep.json"
            request = self.sweep.propose(self.cells[:1])
            with self.assertRaises(ValueError):
                self.sweep.save(path)
            self.sweep.observe([observed(request["proposal_metadata"][0])])
            self.sweep.save(path)
            loaded = methods.GuidedSweep.load(path, train_cells=self.cells)
            self.assertEqual(loaded.state(), self.sweep.state())
            with self.assertRaises(FileExistsError):
                self.sweep.save(path)

    def test_complete_paid_budget_resume_and_sampler_scales(self):
        """Execute method bookkeeping only, with explicitly synthetic returns."""
        phases, scales = set(), set()
        for batch in contract.batch_plan(contract.make_protocol(), contract.SEEDS[0]):
            cells = [self.cells[i] for i in batch["site_indices"]]
            picked = self.sweep.propose(cells)
            rows = []
            for proposal in picked["proposal_metadata"]:
                phases.add(proposal["phase"])
                if proposal["refinement_scale"] is not None:
                    scales.add(proposal["refinement_scale"])
                # A deterministic unit-only objective; these are not simulator labels.
                rows.append(observed(proposal, -float(np.square(proposal["actions_normalized"]).sum())))
            self.sweep.observe(rows)
            self.assertEqual(self.sweep.observed, batch["attempt_stop"])
            if batch["milestone_after"]:
                state = json.loads(json.dumps(self.sweep.state()))
                self.sweep = methods.GuidedSweep.from_state(state, train_cells=self.cells)
                self.assertEqual(self.sweep.state(), state)
        self.assertEqual(self.sweep.observed, 10000)
        self.assertEqual(self.sweep.issued, 10000)
        self.assertEqual(sorted(self.sweep.counts.values()), [285] * 10 + [286] * 25)
        self.assertEqual(phases, {"shared_geometry_prior", "signed_prior_axis", "local_sobol_prior",
                                  "global_domain_sobol", "train_best_refinement"})
        self.assertEqual(scales, {1., .5, .25})
        with self.assertRaises(ValueError):
            self.sweep.propose(self.cells[:1])


@unittest.skipIf(continuous_policy.torch is None, "PyTorch CPU environment is unavailable")
class SharedPPOTests(unittest.TestCase):
    def test_fresh_cpu_actor_matches_shared_prior_and_preserves_rng(self):
        torch = continuous_policy.torch
        torch.set_num_threads(1)
        before = torch.random.get_rng_state().clone()
        policy = methods.make_cpu_ppo(contract.SEEDS[0])
        torch.testing.assert_close(before, torch.random.get_rng_state(), rtol=0, atol=0)
        cells = [dict(bucket_xy_m=[.2, -.5]), dict(bucket_xy_m=[.67, -.1]), dict(bucket_xy_m=[2., -.1])]
        predicted = methods.predict_cpu_ppo(policy, cells)
        np.testing.assert_allclose(predicted["actions_normalized"], methods.geometry_prior(cells)["actions_normalized"], atol=3e-7, rtol=0)
        self.assertEqual(predicted["ball_flight_queries"], 0)
        self.assertTrue(all(p.device.type == "cpu" for p in policy.parameters()))
        second = methods.make_cpu_ppo(contract.SEEDS[0])
        self.assertEqual(continuous_policy.weight_hash(policy), continuous_policy.weight_hash(second))


if __name__ == "__main__":
    unittest.main()
