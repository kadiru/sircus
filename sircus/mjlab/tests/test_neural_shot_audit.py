"""Corruption, leakage and denominator checks for the neural selector audit."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


@unittest.skipUnless(importlib.util.find_spec("numpy"), "Native audit requires NumPy")
class NeuralEvidenceTests(unittest.TestCase):
    def setUp(self):
        from sircus.mjlab import neural_shot_audit as audit
        self.audit = audit

    def splits(self):
        result = {}
        for index, split in enumerate(self.audit.SPLITS):
            identity = split + "-case"
            rows = [dict(id=identity, context_id=identity, split=split,
                         bucket_xy_m=[.1 + index * .1, -.1])]
            result[split] = dict(layouts=rows, layouts_sha256=self.audit.digest(rows))
        return result, dict.fromkeys(self.audit.SPLITS, 1)

    def check_splits(self, fixture):
        return self.audit.audit_splits(*fixture, self.audit.Checks())

    def test_valid_split_manifest_has_three_separate_positions(self):
        result = self.check_splits(self.splits())
        self.assertEqual(set(result), {"train", "validation", "test"})

    def test_different_ids_do_not_allow_shared_bucket_coordinates(self):
        archive, counts = self.splits()
        archive["test"]["layouts"][0]["bucket_xy_m"] = archive["train"]["layouts"][0]["bucket_xy_m"]
        archive["test"]["layouts_sha256"] = self.audit.digest(archive["test"]["layouts"])
        with self.assertRaisesRegex(ValueError, "separate TRAIN"):
            self.check_splits((archive, counts))

    def test_validation_position_cannot_be_relabeled_training(self):
        archive, counts = self.splits()
        archive["validation"]["layouts"][0]["split"] = "train"
        archive["validation"]["layouts_sha256"] = self.audit.digest(archive["validation"]["layouts"])
        with self.assertRaisesRegex(ValueError, "unambiguous split"):
            self.check_splits((archive, counts))

    def test_near_historical_position_is_not_fresh(self):
        archive, counts = self.splits()
        archive["test"]["provenance"] = dict(excluded_coordinates_m=[[.301, -.1]])
        with self.assertRaisesRegex(ValueError, "historical positions"):
            self.check_splits((archive, counts))

    def test_missing_test_position_is_not_silently_dropped(self):
        archive, counts = self.splits()
        archive["test"]["layouts"] = []
        with self.assertRaisesRegex(ValueError, "exact frozen position count"):
            self.check_splits((archive, counts))

    def test_nonfinite_or_boolean_coordinates_are_rejected(self):
        for value in (float("inf"), True):
            archive, counts = self.splits()
            archive["test"].pop("layouts_sha256")
            archive["test"]["layouts"][0]["bucket_xy_m"][0] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite XY"):
                self.check_splits((archive, counts))

    @staticmethod
    def native(label):
        return dict(native_release=True, invalid_physics=label == "invalid_physics",
                    route_success=label in {"direct", "table", "unclassified_capture"},
                    shot_route=label if label in {"direct", "table"} else None,
                    experimental_route=dict(robot_assisted_capture=label == "arm_assisted_capture",
                                            robot_contact_steps=int(label == "arm_assisted_capture")))

    def test_setup_only_and_invalid_physics_are_not_ordinary_flight_misses(self):
        self.assertEqual(self.audit._training_outcome(dict(setup_ok=True, observation=None)), "setup_only")
        self.assertEqual(self.audit._training_outcome(dict(setup_ok=False, observation=None)), "setup_rejected")
        for label in ("invalid_physics", "flight_miss", "direct", "table", "arm_assisted_capture"):
            with self.subTest(label=label):
                self.assertEqual(self.audit._training_outcome(dict(setup_ok=True, observation=self.native(label))), label)

    def test_capture_and_robot_assisted_labels_cannot_both_be_true(self):
        row = self.native("direct")
        row["experimental_route"]["robot_assisted_capture"] = True
        with self.assertRaisesRegex(ValueError, "labels conflict"):
            self.audit._training_outcome(dict(setup_ok=True, observation=row))

    def test_rejected_setup_cannot_acquire_a_success_label(self):
        with self.assertRaisesRegex(ValueError, "Rejected setup"):
            self.audit._training_outcome(dict(setup_ok=False, observation=self.native("direct")))

    def test_bound_checkpoint_bytes_cannot_change(self):
        with TemporaryDirectory() as root:
            directory = Path(root)
            path = directory / "neural-checkpoint.json"
            path.write_text('{"weight": 1}')
            bindings = {path.name: self.audit.sha(path)}
            self.audit.verify_bindings(directory, bindings, self.audit.Checks(), "test")
            path.write_text('{"weight": 2}')
            with self.assertRaisesRegex(ValueError, "neural-checkpoint"):
                self.audit.verify_bindings(directory, bindings, self.audit.Checks(), "test")

    def test_start_receipt_rejects_changed_commands(self):
        with TemporaryDirectory() as root:
            directory = Path(root)
            path = directory / "input.json"
            path.write_text("{}")
            bindings = {path.name: self.audit.sha(path)}
            requests = {"train": [dict(id="a")], "validation": []}
            receipt = dict(stage="collection", bindings=bindings, requests_sha256=self.audit.digest(requests))
            (directory / "native-started-collection.json").write_text(json.dumps(receipt))
            self.audit._check_receipt(directory, "collection", requests, bindings, self.audit.Checks())
            with self.assertRaisesRegex(ValueError, "commands frozen"):
                self.audit._check_receipt(directory, "collection", dict(requests, train=[]), bindings, self.audit.Checks())

    def test_test_receipt_requires_frozen_model_and_both_development_datasets(self):
        with TemporaryDirectory() as root:
            directory = Path(root)
            path = directory / "input.json"
            path.write_text("{}")
            bindings = {path.name: self.audit.sha(path)}
            requests = {"test_baseline": [], "test_neural": []}
            receipt = dict(stage="test", bindings=bindings, requests_sha256=self.audit.digest(requests))
            (directory / "native-started-test.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "model and labels frozen"):
                self.audit._check_receipt(directory, "test", requests, bindings, self.audit.Checks())


@unittest.skipUnless(importlib.util.find_spec("torch"), "Existing planner accounting imports require PyTorch")
class NeuralAccountingTests(NeuralEvidenceTests):
    def fixture(self):
        layouts, slots, observed, baseline, neural = [], [], {}, [], []
        outcomes = (("direct", "direct", "table"), ("flight_miss", "direct", None),
                    ("invalid_physics", "invalid_physics", "arm_assisted_capture"))
        for i, labels in enumerate(outcomes):
            layout = dict(id=f"case-{i}", context_id=f"case-{i}", split="test", bucket_xy_m=[.5 + .1 * i, -.4])
            layouts.append(layout)
            baseline.append(dict(layout=layout, latency_s=dict(current_planner=i + 1., clearance_expansion=2. * (i + 1))))
            neural.append(dict(layout=layout, latency_s=dict(neural_selector=.5 * (i + 1))))
            for method, label in zip(self.audit.METHODS, labels, strict=True):
                identity = layout["id"] + "-" + method
                request = None if label is None else dict(
                    id=identity, context_id=layout["id"], bucket_xy_m=layout["bucket_xy_m"], method=method,
                    intended_route="direct", values=[.4, -.4, .3, 28., .2], board_heading_offset_deg=0.,
                    board_twist_deg=0., selected_q10=[0.] * 6, selected_q14=[.1] * 6)
                slots.append(dict(id=identity, context_id=layout["id"], bucket_xy_m=layout["bucket_xy_m"],
                                  method=method, intended_route=None if request is None else "direct", request=request))
                if request is not None:
                    observed[identity] = self.native(label)
        return layouts, slots, observed, baseline, neural

    def test_all_positions_remain_in_denominator_including_refused_and_invalid(self):
        result = self.audit.summarize(*self.fixture())
        self.assertEqual([result["methods"][name]["captures"] for name in self.audit.METHODS], [1, 2, 1])
        self.assertEqual(result["methods"]["neural_selector"]["positions"], 3)
        self.assertEqual(result["methods"]["neural_selector"]["capture_rate"], 1 / 3)
        self.assertEqual(result["methods"]["neural_selector"]["unresolved"], 1)
        self.assertEqual(result["methods"]["neural_selector"]["outcomes"]["arm_assisted_capture"], 1)
        self.assertEqual(result["methods"]["current_planner"]["outcomes"]["invalid_physics"], 1)
        self.assertEqual(result["paired"]["current_planner_vs_clearance_expansion"]["gained"], 1)
        self.assertEqual(result["latency"]["neural_selector"]["median_s"], 1.)

    def test_unplanned_missing_and_duplicate_attempts_are_rejected(self):
        for fault in ("missing", "extra", "duplicate"):
            fixture = self.fixture()
            if fault == "missing":
                fixture[2].pop(next(iter(fixture[2])))
            elif fault == "extra":
                fixture[2]["unplanned"] = self.native("direct")
            else:
                fixture[1].append(deepcopy(fixture[1][0]))
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                self.audit.summarize(*fixture)

    def test_request_cannot_change_bucket_after_selection(self):
        fixture = self.fixture()
        fixture[1][0]["request"]["bucket_xy_m"] = [.9, -.2]
        with self.assertRaisesRegex(ValueError, "differs from its slot"):
            self.audit.summarize(*fixture)

    def test_nonfinite_and_negative_latency_is_rejected(self):
        for value in (-1, True, float("nan")):
            fixture = self.fixture()
            fixture[4][0]["latency_s"]["neural_selector"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                self.audit.summarize(*fixture)


if __name__ == "__main__":
    unittest.main()
