"""Development provenance and qualification never inherit old test labels."""

import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sircus.mjlab.expand_bank import (
    development_targets,
    qualified_template,
    expanded_bank,
)
from sircus.mjlab.learning_contract import digest
from sircus.mjlab.learning_search import VERSION as BANK_VERSION


class ExpansionContractTests(unittest.TestCase):
    def source(self, **change):
        row = dict(
            id="expand-shot-0000001",
            invalid=None,
            route_success=True,
            route_quality=2,
            values=[0.62, -0.1, 0.22, 24, 0.28],
            bucket_xy_m=[0.62, -0.55],
            bound_phase="old",
        )
        row.update(change)
        return row

    def confirmations(self, successes=8, count=10):
        return [
            dict(
                confirmation_source_id="expand-shot-0000001",
                invalid=None,
                route_success=i < successes,
            )
            for i in range(count)
        ]

    def bounds(self):
        return dict(
            low=[0.32, -0.3, 0.18, 12, 0.065],
            high=[1.01, 0.25, 0.53, 48, 0.52],
            version="old",
        )

    def test_old_empty_test_masks_are_explicitly_reclassified_development(self):
        targets = development_targets(
            [[0.3, -0.4], [0.8, -0.2], [1.2, 0.1]], [[1, 0], [0, 0], [0, 0]]
        )
        self.assertEqual([t["original_test_index"] for t in targets], [1, 2])
        self.assertEqual([t["role"] for t in targets], ["targeted_development"] * 2)
        self.assertEqual(targets[0]["id"], "old-test-01")
        with self.assertRaises(ValueError):
            development_targets([[0.3, -0.4]], [[0], [0]])

    def test_eight_of_ten_qualifies_with_exact_count_and_source(self):
        template = qualified_template(
            self.source(), self.confirmations(), "old-test-01", self.bounds()
        )
        self.assertEqual(template["source_phase"], "targeted_development")
        self.assertEqual(template["qualification"], "development_confirmation_8_of_10")
        self.assertEqual(
            template["confirmation"], dict(attempts=10, captures=8, invalid=0)
        )
        np.testing.assert_allclose(template["relative_board_xy_m"], [0, 0.45])

    def test_no_promotion_for_missing_or_insufficient_actual_confirmation(self):
        for rows in ([], self.confirmations(7)):
            self.assertIsNone(
                qualified_template(self.source(), rows, "dev", self.bounds())
            )

    def test_source_capture_is_required_even_when_other_labels_are_positive(self):
        self.assertIsNone(
            qualified_template(
                self.source(route_success=False),
                self.confirmations(10),
                "dev",
                self.bounds(),
            )
        )
        self.assertIsNone(
            qualified_template(
                self.source(invalid="endpoint"),
                self.confirmations(10),
                "dev",
                self.bounds(),
            )
        )

    def test_geometry_rejections_stay_in_confirmation_denominator(self):
        rows = self.confirmations(10)
        rows[-1]["invalid"] = "endpoint"
        rows[-2]["invalid"] = "endpoint"
        t = qualified_template(self.source(), rows, "dev", self.bounds())
        self.assertEqual(t["confirmation"]["captures"], 8)
        self.assertEqual(t["confirmation"]["invalid"], 2)

    def test_confirmation_identity_cannot_be_swapped(self):
        rows = self.confirmations(10)
        rows[0]["confirmation_source_id"] = "other"
        with self.assertRaises(ValueError):
            qualified_template(self.source(), rows, "dev", self.bounds())

    def test_bank_prefix_and_original_fingerprint_are_preserved(self):
        original = dict(
            version=BANK_VERSION,
            templates=[dict(source_trial_id="train-1")],
            source_scope="TRAIN",
        )
        original["sha256"] = digest(original)
        saved = copy.deepcopy(original)
        template = qualified_template(
            self.source(), self.confirmations(), "old-test-01", self.bounds()
        )
        spec = dict(
            fingerprint="expansion-contract",
            development_layouts=[dict(id="old-test-01")],
        )
        bank = expanded_bank(original, [template], spec, self.bounds())
        self.assertEqual(original, saved)
        self.assertEqual(bank["templates"][:1], saved["templates"])
        self.assertEqual(bank["parent_bank_sha256"], saved["sha256"])
        payload = dict(bank)
        fingerprint = payload.pop("sha256")
        self.assertEqual(digest(payload), fingerprint)

    def test_unlabeled_or_unconfirmed_addition_is_rejected(self):
        bank = dict(version=BANK_VERSION, templates=[])
        bank["sha256"] = digest(bank)
        spec = dict(fingerprint="expansion", development_layouts=[])
        with self.assertRaises(ValueError):
            expanded_bank(bank, [dict(source_phase="test")], spec, self.bounds())
        bad = dict(
            source_phase="targeted_development",
            confirmation=dict(attempts=0, captures=0),
        )
        with self.assertRaises(ValueError):
            expanded_bank(bank, [bad], spec, self.bounds())


if __name__ == "__main__":
    unittest.main()
