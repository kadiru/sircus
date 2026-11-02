"""Expansion evidence checks catch role, count, score and provenance corruption."""

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sircus.mjlab.expand_bank import expanded_bank, qualified_template
from sircus.mjlab.learning_search import VERSION as BANK_VERSION
from sircus.mjlab.verify_expansion import audit_records, digest, statistics


class ExpansionAuditTests(unittest.TestCase):
    def fixture(self):
        bounds = dict(
            low=[0.32, -0.3, 0.18, 12, 0.065],
            high=[1.01, 0.25, 0.53, 48, 0.52],
            version="old",
        )
        target = dict(
            id="old-test-00",
            original_test_index=0,
            bucket_xy_m=[0.62, -0.55],
            role="targeted_development",
        )
        original = dict(
            version=BANK_VERSION, templates=[dict(source_trial_id="original-train")]
        )
        original["sha256"] = digest(original)
        spec = dict(
            old_bank_sha256=original["sha256"],
            development_layouts=[target],
            old_bounds=bounds,
            wide_bounds=dict(bounds, version="wide"),
            top_candidates=3,
            confirmations_per_candidate=10,
            minimum_confirmation_rate=0.8,
        )
        spec["fingerprint"] = digest(spec)
        source = dict(
            id="expand-shot-0000000",
            phase="search",
            bound_phase="old",
            target_id=target["id"],
            source_phase="targeted_development",
            invalid=None,
            route_success=True,
            route_quality=2,
            values=[0.62, -0.1, 0.22, 24, 0.28],
            bucket_xy_m=target["bucket_xy_m"],
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
        rows = [source]
        for i in range(10):
            row = copy.deepcopy(source)
            row.update(
                id=f"expand-shot-{i+1:07d}",
                phase="confirmation",
                confirmation_source_id=source["id"],
            )
            if i >= 8:
                row.update(
                    route_success=False,
                    route_quality=0,
                    max_retention_s=0.5,
                    single_bounce_success=False,
                )
            rows.append(row)
        template = qualified_template(source, rows[1:], target["id"], bounds)
        bank = expanded_bank(original, [template], spec, bounds)
        report = dict(
            status="completed",
            phase="completed",
            contract_fingerprint=spec["fingerprint"],
            old_bank_sha256=original["sha256"],
            proposal_bank_sha256=bank["sha256"],
            development_layouts=[target],
            records=len(rows),
            native_search_attempts=1,
            confirmation_attempts=10,
            geometry_proposals=1,
            additions=1,
            original_templates=1,
            expanded_templates=2,
            bounds=bounds,
            native_search_statistics=statistics([source]),
            confirmation_statistics=statistics(rows[1:]),
            cells=[dict(target, search=statistics([source]), promotion=template)],
        )
        return spec, report, rows, bank, original

    def checked(self, values):
        audit = audit_records(*values)
        return {c["name"]: c["passed"] for c in audit.checks}

    def test_legitimate_fixture_passes_all(self):
        self.assertTrue(all(self.checked(self.fixture()).values()))

    def test_test_role_cannot_be_silently_used_for_development(self):
        v = self.fixture()
        v[2][0]["source_phase"] = "test"
        self.assertFalse(self.checked(v)["all_trial_targets_are_declared_development"])

    def test_original_prefix_tamper_is_found_even_with_new_bank_digest(self):
        v = self.fixture()
        v[3]["templates"][0]["source_trial_id"] = "changed"
        v[3].pop("sha256")
        v[3]["sha256"] = digest(v[3])
        v[1]["proposal_bank_sha256"] = v[3]["sha256"]
        self.assertFalse(self.checked(v)["original_template_prefix_unchanged"])

    def test_shortened_confirmation_cannot_hide_in_reported_counts(self):
        v = self.fixture()
        v[2].pop()
        v[1]["records"] -= 1
        v[1]["confirmation_attempts"] -= 1
        self.assertFalse(self.checked(v)["independent_confirmation_denominators"])

    def test_swapped_source_identity_fails(self):
        v = self.fixture()
        v[2][1]["confirmation_source_id"] = "not-a-source"
        self.assertFalse(self.checked(v)["confirmation_source_association"])

    def test_capture_without_rebound_fails(self):
        v = self.fixture()
        v[2][0]["board_bounce"] = False
        self.assertFalse(self.checked(v)["capture_rule_invariants"])

    def test_wrong_native_attempt_count_fails(self):
        v = self.fixture()
        v[1]["native_search_attempts"] += 1
        self.assertFalse(self.checked(v)["native_search_count"])

    def test_actions_outside_declared_family_fail(self):
        v = self.fixture()
        v[2][0]["values"][3] = 70
        self.assertFalse(self.checked(v)["all_actions_inside_their_declared_bounds"])

    def test_template_geometry_must_come_from_actual_source(self):
        v = self.fixture()
        v[3]["templates"][-1]["height_m"] += 0.1
        self.assertFalse(self.checked(v)["old-test-00_geometry_from_measured_source"])


if __name__ == "__main__":
    unittest.main()
