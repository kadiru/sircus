"""Explicit search proposals cannot masquerade as learned policy inference."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from sircus.mjlab.learning_contract import digest
from sircus.mjlab import learning_search as search


def cell(source="shot-001", *, confirmation=3, quality=2):
    return dict(
        id="train-00",
        split="train",
        bucket_xy_m=[0.67, -0.2],
        best=dict(
            id=source,
            phase="train-exploration",
            invalid=None,
            route_success=True,
            route_quality=quality,
            bucket_xy_m=[0.67, -0.2],
            values=[0.67, -0.04, 0.36, 25, 0.12],
        ),
        confirmation=dict(attempts=3, captures=confirmation, invalid=0),
    )


def bank(cells=None):
    return search.build_bank(
        cells or [cell()], [[0.67, -0.2]], "catalogue-sha", "contract-fingerprint"
    )


class RelativeBankTests(unittest.TestCase):
    def setUp(self):
        self.scene = json.loads((ROOT / "sircus/isaac/scene.json").read_text())

    def test_prefers_only_three_of_three_train_confirmed_teachers(self):
        strong, weak = cell(), cell("shot-002", confirmation=1)
        value = bank([weak, strong])
        self.assertEqual(len(value["templates"]), 1)
        self.assertEqual(value["templates"][0]["source_trial_id"], "shot-001")
        self.assertEqual(
            value["templates"][0]["qualification"], "train_confirmation_3_of_3"
        )
        weak_only = bank([weak])
        self.assertEqual(
            weak_only["templates"][0]["qualification"],
            "train_capture_lower_confirmation",
        )
        self.assertIn("lower", weak_only["qualification"])

    def test_uncaptured_or_invalid_teachers_are_not_in_bank(self):
        missed = cell()
        missed["best"]["route_success"] = False
        invalid = cell("shot-002")
        invalid["best"]["invalid"] = "endpoint_ik"
        self.assertEqual(bank([missed, invalid])["templates"], [])

    def test_eval_and_wrong_coordinate_sources_are_rejected(self):
        for path, value in (
            (["split"], "test"),
            (["best", "phase"], "test-selected"),
            (["bucket_xy_m"], [0.5, -0.4]),
        ):
            row = cell()
            target = row
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.assertRaises(RuntimeError):
                bank([row])
        row = cell()
        row["confirmation"]["invalid"] = 1
        with self.assertRaisesRegex(RuntimeError, "confirmation"):
            bank([row])

    def test_rotation_translation_preserve_distance_height_energy_and_tilt(self):
        source = bank()
        target = [0.67, -0.04]
        with patch.object(
            search, "feasible_mask", return_value=np.ones(16, dtype=bool)
        ):
            candidates, metadata = search.bank_candidates(source, target, self.scene)
        self.assertEqual(len(candidates), 16)
        np.testing.assert_allclose(
            np.linalg.norm(candidates[:, :2] - target, axis=1), 0.16, atol=1e-14
        )
        np.testing.assert_allclose(
            candidates[:, 2:], np.tile([0.36, 25, 0.12], (16, 1))
        )
        self.assertEqual(
            [row["rotation_deg"] for row in metadata], [22.5 * i for i in range(16)]
        )
        self.assertTrue(all(row["source_trial_id"] == "shot-001" for row in metadata))
        self.assertNotIn("success", source)
        self.assertNotIn("success", metadata[0])

    def test_real_geometric_filter_is_prior_not_ik_or_success_certificate(self):
        candidates, metadata = search.bank_candidates(bank(), [0.67, -0.2], self.scene)
        self.assertGreater(len(candidates), 0)
        self.assertEqual(len(candidates), len(metadata))
        self.assertTrue(np.isfinite(candidates).all())
        source = bank()
        source["templates"][0]["gap_m"] = 9
        with self.assertRaisesRegex(RuntimeError, "fingerprint"):
            search.bank_candidates(source, [0.67, -0.2], self.scene)
        scene = copy.deepcopy(self.scene)
        scene["bucket"]["segments"] = 32
        with self.assertRaisesRegex(RuntimeError, "64-wall"):
            search.bank_candidates(bank(), [0.67, -0.2], scene)

    def test_loader_requires_completed_study_and_frozen_catalogue_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalogue = root / "catalogue.json"
            catalogue.write_text(json.dumps([cell()]))
            contract = dict(train_layouts=[[0.67, -0.2]], version="fixture")
            contract["fingerprint"] = digest(contract)
            fp = contract["fingerprint"]
            (root / "contract.json").write_text(json.dumps(contract))
            report = dict(
                status="completed",
                frozen_evaluation=dict(captures=1),
                contract_fingerprint=fp,
            )
            (root / "report.json").write_text(json.dumps(report))
            provenance = dict(
                contract_fingerprint=fp,
                training_catalogue_sha256=search._sha256(catalogue),
            )
            (root / "artifact_provenance.json").write_text(json.dumps(provenance))
            loaded = search.load_training_bank(root, fp)
            self.assertEqual(
                loaded["training_catalogue_sha256"],
                provenance["training_catalogue_sha256"],
            )
            catalogue.write_text(catalogue.read_text() + " ")
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                search.load_training_bank(root, fp)
            provenance["training_catalogue_sha256"] = search._sha256(catalogue)
            (root / "artifact_provenance.json").write_text(json.dumps(provenance))
            report["status"] = "running"
            (root / "report.json").write_text(json.dumps(report))
            with self.assertRaisesRegex(RuntimeError, "completed"):
                search.load_training_bank(root, fp)


if __name__ == "__main__":
    unittest.main()
