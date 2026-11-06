"""CPU-only adversarial checks for the independent fixed64 policy auditor."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sircus.mjlab import policy64_verify as audit
from sircus.mjlab.benchmark_verify import Checks, EvidenceError


TORCH = importlib.util.find_spec("torch") is not None


def source_row(ident="source", context="train-a", *, capture=True, clean=False):
    anchor = dict(id="geometry-only", bucket_xy_m=[.4, -.1], values=[.5, -.2, .3, 28., .35])
    row = dict(id=ident, role="train", context_id=context, bucket_xy_m=[.4, -.1],
               observation=audit.calibrated.expected_observation([.4, -.1], anchor).tolist(),
               normalized_residual=[.1, -.1, .2, -.2, .3], effective_source_hint=anchor,
               native_release=True, route_success=capture, route_quality=2 if clean else 1 if capture else 0,
               invalid=None, invalid_physics=False, board_bounce=True, whole_ball_entry=capture,
               max_retention_s=1. if capture else 0., closest_opening_distance_m=.01 if capture else .3,
               external_contact_steps=0, single_bounce_success=capture,
               perturbation_mixed_units=[0.]*5, contact_protocol=dict(substeps=64),
               scoring_version="sircus-calibrated-native-capture-v2", scalar_score_audit_passed=True)
    row["values"] = (np.asarray(anchor["values"])+np.asarray(row["normalized_residual"])*audit.RESIDUAL).tolist()
    row["command_values"] = row["values"]
    row["learning_reward"] = audit.calibrated.row_reward(row)
    return row


class PolicyAuditTests(unittest.TestCase):
    def test_explicit64_protocol_preserves_history_and_finite_budget(self):
        from sircus.mjlab.policy64_contract import protocol
        audit.protocol_checks(protocol(), Checks())
        for key, value in (("substeps", 128), ("noise", True), ("maximum_fresh_requests", 1283), ("test_repeats", 3)):
            changed = protocol();changed[key] = value
            with self.assertRaises(EvidenceError):
                audit.protocol_checks(changed, Checks())

    def test_teachers_prefer_clean_then_calibrated_reward_then_id_and_no_heldout(self):
        rim = source_row("a-rim")
        clean = source_row("z-clean", clean=True)
        tied = copy.deepcopy(clean);tied["id"] = "b-clean"
        miss = source_row("miss", context="train-b", capture=False)
        self.assertEqual([row["id"] for row in audit.best_teachers([rim, clean, tied, miss])], ["b-clean"])
        better = copy.deepcopy(tied);better["id"] = "zz-better";better["closest_opening_distance_m"] = 0.
        self.assertEqual(audit.best_teachers([tied, better])[0]["id"], "zz-better")
        heldout = copy.deepcopy(clean);heldout["role"] = "validation"
        with self.assertRaisesRegex(EvidenceError, "held-out"):
            audit.best_teachers([heldout])

    def test_dataset_keeps_failures_and_exact_full_row_hashes(self):
        rows = [source_row(), source_row("miss", capture=False)]
        reduced = audit.reduced_dataset(rows)
        self.assertEqual(len(reduced), 2)
        self.assertFalse(reduced[1]["route_success"])
        self.assertEqual(reduced[1]["source_record_sha256"], audit.digest(rows[1]))
        mutated = copy.deepcopy(rows[0]);mutated["unused_raw_evidence"] = "different"
        self.assertNotEqual(audit.reduced_dataset([mutated])[0]["source_record_sha256"], reduced[0]["source_record_sha256"])
        rows[1]["role"] = "test"
        with self.assertRaisesRegex(EvidenceError, "Held-out"):
            audit.reduced_dataset(rows)

    def test_split_regeneration_matches_producer_and_all_old_coordinates_excluded(self):
        from sircus.mjlab.policy64_contract import make_splits
        root = Path(__file__).resolve().parents[3]
        parent = json.loads((root/"baselines/mjlab-calibrated-benchmark-20261104/contract.json").read_text())["splits"]
        historical = json.loads((root/"baselines/mjlab-sweep-rl-benchmark-20261103/splits.json").read_text())
        actual, excluded = audit.regenerated_splits(parent, historical)
        self.assertEqual((actual, excluded), make_splits(parent, historical))
        self.assertEqual([len(actual[key]) for key in ("train", "validation", "test")], [35, 12, 50])
        fresh = np.asarray([context["bucket_xy_m"] for role in ("validation", "test") for context in actual[role]])
        self.assertTrue((np.linalg.norm(fresh[:, None]-np.asarray(excluded)[None], axis=-1) >= .005).all())
        distance = np.linalg.norm(fresh[:, None]-fresh[None], axis=-1)+np.eye(62)
        self.assertTrue((distance >= .005).all())

    def test_source_reader_retains6144_train_attempts_and_ignores_old_test_labels(self):
        contexts = [dict(id=f"train-{index:02d}", bucket_xy_m=[.4, -.1]) for index in range(35)]
        contract = dict(splits=dict(train=contexts))
        def reader(path):
            method, seed = path.parts[-2].split("-seed");seed = int(seed)
            rows = []
            for index in range(1024):
                row = source_row(f"{method}-{seed}-train-{index:05d}", context=contexts[index%35]["id"], capture=index%5 == 0)
                row.update(method=method, seed=seed, train_index=index)
                rows.append(row)
            rows.insert(5, dict(role="test", untrusted_outcome="never read"))
            return rows
        with patch.object(audit.common, "lines", side_effect=reader), patch.object(audit, "sha", return_value="a"*64):
            rows, hashes = audit.source_train_rows(Path("unused"), contract, Checks())
        self.assertEqual((len(rows), len(hashes)), (6144, 6))
        self.assertTrue(all(row["role"] == "train" for row in rows))
        with patch.object(audit.common, "lines", side_effect=lambda path: reader(path)[1:]), patch.object(audit, "sha", return_value="a"*64):
            with self.assertRaisesRegex(EvidenceError, "1024"):
                audit.source_train_rows(Path("unused"), contract, Checks())

    def test_nearest_baseline_uses_only_immutable_train_teachers_and_tie_order(self):
        a = source_row("teacher-a", "train-a");b = source_row("teacher-b", "train-b")
        a["bucket_xy_m"] = [.4, -.1];b["bucket_xy_m"] = [.6, -.1]
        self.assertEqual(audit.nearest_teacher(dict(bucket_xy_m=[.5, -.1]), [b, a])["id"], "teacher-a")
        self.assertEqual(audit.nearest_teacher(dict(bucket_xy_m=[.59, -.1]), [a, b])["id"], "teacher-b")

    def test_control_gate_requires_valid_native_releases_and_oracle(self):
        rows = []
        for capture, role in ((True, "positive_control"), (False, "negative_control")):
            for repeat in range(3):
                row = source_row(f"{role}-{repeat}", capture=capture);row["role"] = role;rows.append(row)
        self.assertEqual(audit.control_gate(rows)["status"], "passed")
        changed = copy.deepcopy(rows);changed[-1]["native_release"] = False
        self.assertEqual(audit.control_gate(changed)["status"], "failed")
        changed = copy.deepcopy(rows);changed[0]["scalar_score_audit_passed"] = False
        self.assertEqual(audit.control_gate(changed)["status"], "failed")

    @unittest.skipUnless(TORCH, "CPU Torch required for independent optimizer replay")
    def test_independent_bootstrap_reproduces_actual_full_batch_fits_and_detects_mutation(self):
        import torch
        from sircus.mjlab.policy64_model import initialized, bootstrap
        torch.set_num_threads(2)
        raw = [source_row("a", "train-a", clean=True), source_row("b", "train-b", capture=False)]
        dataset = audit.reduced_dataset(raw);teachers = [dataset[0]]
        policy = initialized(audit.SEED);initial = copy.deepcopy(policy.state_dict())
        fit = bootstrap(policy, dataset, teachers, actor_steps=3, critic_steps=2)
        trained = copy.deepcopy(policy.state_dict())
        audit.bootstrap_replay(initial, trained, dataset, teachers, fit, Checks(), actor_steps=3, critic_steps=2)
        trained["actor.4.bias"][0] += .01
        with self.assertRaisesRegex(EvidenceError, "bootstrap parameter"):
            audit.bootstrap_replay(initial, trained, dataset, teachers, fit, Checks(), actor_steps=3, critic_steps=2)

    @unittest.skipUnless(TORCH, "CPU Torch required for direct actor inference")
    def test_frozen_prediction_recomputed_and_source_labels_cannot_replace_actor_output(self):
        import torch
        from sircus.mjlab import policy64_model
        teacher = audit.reduced_dataset([source_row()])[0]
        context = dict(id="fresh", bucket_xy_m=[.6, -.3])
        anchor = teacher["effective_source_hint"]
        observation = audit.calibrated.expected_observation(context["bucket_xy_m"], anchor)
        model = policy64_model.initialized(audit.SEED)
        state = dict(policy=model.state_dict())
        mean = audit.common.network_forward(state["policy"], observation)
        record = dict(path="checkpoints/milestone-0000.pt", sha256="a"*64, samples=0)
        pred = dict(context_id="fresh", bucket_xy_m=context["bucket_xy_m"], method="bc", samples=0,
                    observation=observation.tolist(), effective_source_hint=anchor, artifact_sha256="a"*64,
                    source_training_ids=[], latent_mean=mean.tolist(), normalized_residual=np.tanh(mean.astype(float)).tolist(),
                    command_values=anchor["values"], ball_flight_search_at_inference=False, geometry_search_at_inference=True,
                    selection_or_learning=False)
        packet = dict(role="test", method="bc", samples=0, artifact=record, rows=[pred], selection_or_learning=False, one_prediction_per_layout=True)
        row = dict(pred, id="test-bc-0000-fresh", role="test", phase="test", values=anchor["values"], frozen_artifact_sha256="a"*64)
        audit.audit_prediction_packet(packet, [row], [context], record, state, [teacher], Checks())
        changed = copy.deepcopy(packet);changed["rows"][0]["normalized_residual"][0] = .1
        changed_row = dict(row);changed_row.update(changed["rows"][0])
        with self.assertRaisesRegex(EvidenceError, "frozen residual"):
            audit.audit_prediction_packet(changed, [changed_row], [context], record, state, [teacher], Checks())


if __name__ == "__main__":
    unittest.main()
