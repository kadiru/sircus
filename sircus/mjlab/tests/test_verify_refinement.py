"""Fresh continuation evidence must remain attributable and count-complete."""

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sircus.mjlab.expand_bank import expanded_bank
from sircus.mjlab.expand_refine import make_promotions
from sircus.mjlab.learning_contract import digest, statistics
from sircus.mjlab.learning_search import VERSION as BANK_VERSION
from sircus.mjlab.verify_refinement import audit_records


class RefinementAuditTests(unittest.TestCase):
    def fixture(self):
        bounds = dict(
            low=[0.3, -0.5, 0.12, 8, 0.05],
            high=[1.04, 0.35, 0.6, 60, 0.65],
            version="wide",
        )
        targets = [
            dict(
                id=f"old-test-{i:02d}",
                bucket_xy_m=[0.1 + i * 0.01, -0.5],
                role="targeted_development",
            )
            for i in range(12)
        ]
        original = dict(
            version=BANK_VERSION, templates=[dict(source_trial_id="old-train")]
        )
        original["sha256"] = digest(original)
        s = dict(
            id="expand-shot-0000001",
            phase="search",
            target_id=targets[0]["id"],
            source_phase="targeted_development",
            invalid=None,
            route_success=True,
            route_quality=2,
            minimum_mouth_clearance_mm=35,
            values=[0.5, -0.2, 0.3, 30, 0.4],
            bucket_xy_m=targets[0]["bucket_xy_m"],
            bound_phase="wide",
            board_bounce=True,
            whole_ball_entry=True,
            max_retention_s=1.0,
            external_contact_steps=0,
            board_contact_episodes=1,
            closest_opening_distance_m=0.01,
            single_bounce_success=True,
            pose_gate=dict(
                board_position_error_mm=0.0,
                board_normal_error_deg=0.0,
                dropper_error_mm=0.0,
            ),
            endpoint_collision_pairs=[],
        )
        spec = dict(
            development_layouts=targets,
            old_bounds=bounds,
            wide_bounds=bounds,
            confirmations_per_candidate=10,
            minimum_confirmation_rate=0.8,
            old_bank_sha256=original["sha256"],
            prior_search_contract_fingerprint="prior",
            selected_source_ids=[s["id"]],
            selected_source_records_sha256=[digest(s)],
        )
        spec["fingerprint"] = digest(spec)
        rows = []
        for i in range(10):
            r = copy.deepcopy(s)
            r.update(
                id=f"refine-shot-{i:07d}",
                phase="confirmation",
                confirmation_source_id=s["id"],
                source_study_contract_fingerprint="prior",
            )
            if i >= 8:
                r.update(
                    route_success=False,
                    route_quality=0,
                    max_retention_s=0.5,
                    single_bounce_success=False,
                )
            rows.append(r)
        additions = make_promotions([s], {s["id"]: rows}, spec)
        bank = expanded_bank(original, additions, spec, bounds)
        report = dict(
            status="completed",
            phase="completed",
            contract_fingerprint=spec["fingerprint"],
            proposal_bank_sha256=bank["sha256"],
            native_search_attempts=0,
            confirmation_attempts=10,
            records=10,
            development_layouts=targets,
            additions=1,
            original_templates=1,
            expanded_templates=2,
            confirmation_statistics=statistics(rows),
        )
        return spec, report, rows, [s], bank, original, [copy.deepcopy(s)]

    def check(self, values):
        return {c["name"]: c["passed"] for c in audit_records(*values).checks}

    def test_legitimate_evidence_passes(self):
        self.assertTrue(all(self.check(self.fixture()).values()))

    def test_inherited_trials_cannot_inflate_fresh_native_count(self):
        v = self.fixture()
        v[1]["native_search_attempts"] = 6144
        self.assertFalse(self.check(v)["only_fresh_confirmations_counted"])

    def test_source_record_tampering_is_found(self):
        v = self.fixture()
        v[3][0]["values"][0] += 0.01
        checked = self.check(v)
        self.assertFalse(checked["selected_source_hashes"])
        self.assertFalse(checked["exact_original_source_records"])

    def test_shortened_confirmation_denominator_fails(self):
        v = self.fixture()
        v[2].pop()
        self.assertFalse(self.check(v)["ten_per_source_invalids_stay_in_denominator"])

    def test_geometry_invalid_does_not_count_as_capture(self):
        v = self.fixture()
        v[2][0]["invalid"] = "endpoint"
        checked = self.check(v)
        self.assertFalse(checked["old-test-00_actual_8_of_10"])
        self.assertFalse(checked["capture_rule"])


if __name__ == "__main__":
    unittest.main()
