"""Robustness continuation must use actual development evidence, never labels."""

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sircus.mjlab.expand_refine import select_sources, make_promotions
from sircus.mjlab.learning_contract import digest


class RefinementTests(unittest.TestCase):
    def fixture(self):
        bounds = dict(
            low=[0.3, -0.5, 0.12, 8, 0.05],
            high=[1.04, 0.35, 0.6, 60, 0.65],
            version="wide",
        )
        target = dict(
            id="old-test-19", bucket_xy_m=[0.1, -0.5], role="targeted_development"
        )
        source = dict(
            id="expand-shot-0000001",
            phase="search",
            target_id=target["id"],
            source_phase="targeted_development",
            invalid=None,
            route_success=True,
            route_quality=2,
            minimum_mouth_clearance_mm=35,
            values=[0.5, -0.2, 0.3, 30, 0.4],
            bucket_xy_m=target["bucket_xy_m"],
            bound_phase="wide",
        )
        spec = dict(
            development_layouts=[target],
            old_bounds=bounds,
            wide_bounds=bounds,
            confirmations_per_candidate=10,
            minimum_confirmation_rate=0.8,
            prior_search_contract_fingerprint="prior",
            fingerprint="current",
        )
        return bounds, target, source, spec

    def test_only_actual_development_search_captures_selected(self):
        bounds, target, row, _ = self.fixture()
        records = [row]
        for i, change in enumerate(
            (
                dict(route_success=False),
                dict(invalid="endpoint"),
                dict(source_phase="test"),
                dict(phase="confirmation"),
                dict(target_id="other"),
            )
        ):
            records.append(dict(row, id=f"bad-{i}", **change))
        self.assertEqual(select_sources(records, [target], bounds), [row])

    def test_greatest_margin_first_and_selection_is_deterministic(self):
        bounds, target, row, _ = self.fixture()
        other = dict(
            row,
            id="second",
            minimum_mouth_clearance_mm=40,
            values=[0.6, -0.2, 0.3, 30, 0.4],
        )
        self.assertEqual(select_sources([row, other], [target], bounds, 1), [other])
        self.assertEqual(
            select_sources([row, other], [target], bounds, 2),
            select_sources([other, row], [target], bounds, 2),
        )

    def test_action_duplicates_do_not_consume_selection_slots(self):
        bounds, target, row, _ = self.fixture()
        duplicate = dict(row, id="dup", minimum_mouth_clearance_mm=20)
        selected = select_sources([row, duplicate], [target], bounds)
        self.assertEqual(selected, [row])

    def confirmations(self, source, n=10, captures=8, invalid=0):
        return [
            dict(
                confirmation_source_id=source["id"],
                route_success=i < captures,
                invalid="endpoint" if i >= n - invalid else None,
            )
            for i in range(n)
        ]

    def test_promotion_retains_original_source_and_current_confirmation_provenance(
        self,
    ):
        _, _, source, spec = self.fixture()
        rows = self.confirmations(source)
        template = make_promotions([source], {source["id"]: rows}, spec)[0]
        self.assertEqual(template["source_trial_id"], source["id"])
        self.assertEqual(template["source_record_sha256"], digest(source))
        self.assertEqual(template["source_study_contract_fingerprint"], "prior")
        self.assertEqual(template["confirmation_study_contract_fingerprint"], "current")
        self.assertEqual(template["confirmation"]["captures"], 8)

    def test_exactly_ten_and_eight_actual_captures_required(self):
        _, _, source, spec = self.fixture()
        for rows in (
            self.confirmations(source, n=9),
            self.confirmations(source, captures=7),
            self.confirmations(source, captures=8, invalid=3),
        ):
            self.assertEqual(make_promotions([source], {source["id"]: rows}, spec), [])

    def test_confirmation_count_then_margin_selects_winner(self):
        _, _, source, spec = self.fixture()
        low = dict(source, id="low", minimum_mouth_clearance_mm=2)
        confirmations = {
            source["id"]: self.confirmations(source, captures=8),
            low["id"]: self.confirmations(low, captures=9),
        }
        winner = make_promotions([source, low], confirmations, spec)[0]
        self.assertEqual(winner["source_trial_id"], "low")

    def test_selection_does_not_mutate_prior_evidence(self):
        bounds, target, row, _ = self.fixture()
        original = copy.deepcopy(row)
        select_sources([row], [target], bounds)
        self.assertEqual(row, original)


if __name__ == "__main__":
    unittest.main()
