"""CPU-only protocol and independent ensemble audit controls."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sircus.mjlab import physics_ensemble_verify as audit
from sircus.mjlab.benchmark_verify import Checks, EvidenceError

TORCH = importlib.util.find_spec("torch") is not None


def row(native=True, valid=True):
    return dict(id="training", role="train", values=[.5, -.2, .3, 28., .4], bucket_xy_m=[.5, -.6],
                native_release=native, invalid=None if valid else "native_physics_outside_calibrated_domain",
                invalid_physics=not valid, board_bounce=True, whole_ball_entry=True, route_success=True, max_retention_s=1.2)


class EnsembleAuditTests(unittest.TestCase):
    def test_protocol_has_no_new_training_and_all_evaluation_predictions_frozen(self):
        from sircus.mjlab.physics_ensemble_contract import protocol
        audit.audit_protocol(protocol(), Checks())
        for key, value in (("maximum_fresh_requests", 776), ("new_train_requests", 512), ("maximum_geometry_checks", 16), ("fit_device", "cuda")):
            changed = protocol();changed[key] = value
            with self.assertRaises(EvidenceError): audit.audit_protocol(changed, Checks())

    def test_layout_split_reconstruction_excludes_all_declared_old_positions(self):
        from sircus.mjlab.physics_ensemble_contract import make_splits
        root = Path(__file__).resolve().parents[3]
        parent = json.loads((root/"baselines/mjlab-calibrated-benchmark-20261104/contract.json").read_text())
        # The current parent adds another fresh12/50 cohort to these old splits.
        parent["exclusions"] = [c["bucket_xy_m"] for role in ("train", "validation", "test") for c in parent["splits"][role]]
        actual = audit.expected_splits(parent)
        self.assertEqual(actual, make_splits(parent))
        self.assertEqual([len(actual["splits"][key]) for key in ("validation", "test")], [24, 60])
        xy = np.asarray([context["bucket_xy_m"] for role in ("validation", "test") for context in actual["splits"][role]])
        self.assertTrue((np.linalg.norm(xy[:, None]-np.asarray(actual["excluded"])[None], axis=-1) >= .01).all())
        distances = np.linalg.norm(xy[:, None]-xy[None], axis=-1)+np.eye(84)
        self.assertTrue((distances >= .01).all())

    def test_rejected_or_physics_invalid_requests_do_not_train_fake_physical_zeros(self):
        first = dict(available=True, closest_opening_distance_m=.1)
        for source in (row(native=False), row(valid=False)):
            _, target, mask = audit.target_row(source, first, [1., 2., 3.])
            np.testing.assert_equal(mask, [1., 0., 0., 0., 0., 0., 0., 0., 0.])
            np.testing.assert_equal(target[1:], np.zeros(8))
            self.assertEqual(target[0], float(source["native_release"]))
        self.assertFalse(audit.physical_target_valid(row(valid=False)))

    def test_missing_rebound_masks_only_auxiliary_targets_and_preserves_capture_labels(self):
        _, target, mask = audit.target_row(row(), dict(available=False), None)
        np.testing.assert_equal(mask, [1., 1., 1., 1., 1., 0., 0., 0., 0.])
        np.testing.assert_equal(target[:5], np.ones(5))
        _, target, mask = audit.target_row(row(), dict(available=True, closest_opening_distance_m=.1), [1., 2., 3.])
        np.testing.assert_equal(mask, np.ones(9))
        self.assertAlmostEqual(float(target[5]), np.log(2), places=6)
        # Bearing is -Y: forward=-Vy, lateral=+Vx, vertical=Vz.
        np.testing.assert_allclose(target[6:], [-2., 1., 3.], rtol=0, atol=1e-6)
        with self.assertRaises(EvidenceError):
            audit.target_row(row(), dict(available=True, closest_opening_distance_m=.1), None)

    def test_literal264_requested_slots_prevent_hidden_training_or_retries(self):
        rows = []
        for role, count in (("validation", 24), ("test", 60)):
            for method in audit.METHODS:
                rows.extend(dict(id=f"{role}-{method}-{i}", role=role, method=method, context_id=f"{role}-{i}") for i in range(count))
        for role in ("positive_control", "negative_control"):
            rows.extend(dict(id=f"{role}-{i}", role=role, method="control", context_id="control") for i in range(6))
        audit.literal_budget_checks(rows, Checks())
        changed = copy.deepcopy(rows);changed[1]["context_id"] = changed[0]["context_id"]
        with self.assertRaises(EvidenceError): audit.literal_budget_checks(changed, Checks())
        with self.assertRaises(EvidenceError): audit.literal_budget_checks(rows+[dict(rows[0], id="unlogged-retry")], Checks())

    @unittest.skipUnless(TORCH, "Pinned CPU Torch needed for model/candidate parity")
    def test_candidate_bank_and_masked_targets_match_literal_producer_without_fitting_heldout(self):
        from sircus.mjlab import physics_ensemble_model as model
        source = row();source.update(context_id="old", effective_source_hint=dict(values=source["values"]), route_quality=2, learning_reward=4.)
        anchor = dict(values=source["values"]);context = dict(id="fresh-query", bucket_xy_m=[.65, -.4])
        bc = np.asarray(anchor["values"])+[.001, -.002, .01, 1., .02]
        ppo = np.asarray(anchor["values"])+[-.001, .002, -.01, -1., -.02]
        self.assertEqual(audit.candidates(context, anchor, bc, ppo, [source]), model.candidates(context, anchor, bc, ppo, [source]))
        with self.assertRaises(EvidenceError): audit.candidates(context, anchor, bc, ppo, [dict(source, role="test")])
        metric = dict(available=True, closest_opening_distance_m=.15, start_index=1)
        trace = dict(velocity_m_s=np.asarray([[0., 0., 0.], [1., 2., 3.]]))
        source["board_normal"] = [0., 0., 1.]
        with patch("sircus.mjlab.xy650_metrics.first_flight", return_value=metric):
            y, mask, _ = model.target(source, trace, {}, {})
        _, expected, observed = audit.target_row(source, metric, trace["velocity_m_s"][1])
        np.testing.assert_allclose(y, expected, rtol=0, atol=1e-12);np.testing.assert_equal(mask, observed)

    @unittest.skipUnless(TORCH, "Pinned CPU Torch needed for independent full-fit replay")
    def test_independent_bootstrap_fit_replays_all600_steps_and_detects_weight_tampering(self):
        import torch
        from sircus.mjlab import physics_ensemble_model as model
        generator = np.random.default_rng(7);dataset = []
        for index in range(24):
            x = generator.uniform(-.5, .5, 7);positive = float(x[0] > 0)
            y = [1., 1., positive, positive, .5, np.log(2), x[0], x[1], x[2]]
            dataset.append(dict(id=f"TRAIN-{index}", role="train", x=x.tolist(), y=list(y), mask=[1.]*9, physics_invalid=False))
        with tempfile.TemporaryDirectory() as directory:
            result = model.fit(dataset, directory, "a"*64)
            checkpoint = torch.load(Path(directory)/result["path"], map_location="cpu", weights_only=True)
            audit.replay_fit(dataset, checkpoint, "a"*64, Checks())
            actual = model.Predictor(checkpoint).predict_inputs([dataset[0]["x"], dataset[1]["x"]])
            expected = audit.ensemble_forward(checkpoint, [dataset[0]["x"], dataset[1]["x"]])
            audit.same(actual, expected, Checks(), "Independent SiLU/decoder", 3e-5)
            for prediction in actual:
                audit.same(prediction, audit.decode_raw(prediction["member_raw_output"]), Checks(), "Exact fixed rank decoder", 1e-12)
            measured = model.prediction_metrics(model.Predictor(checkpoint), dataset, "synthetic")
            audit.same(measured, audit.prediction_metrics(checkpoint, dataset, "synthetic"), Checks(), "Independent requested and conditional forecast metrics", 1e-5)
            changed = copy.deepcopy(checkpoint);changed["bootstrap_indices"][0][0] = (changed["bootstrap_indices"][0][0]+1)%len(dataset)
            with self.assertRaisesRegex(EvidenceError, "bootstrap"):
                audit.replay_fit(dataset, changed, "a"*64, Checks())


if __name__ == "__main__":
    unittest.main()
