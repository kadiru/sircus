"""Independent audit catches altered far-study IDs, captures and seed provenance."""

import copy
import unittest

from sircus.mjlab.expand_bank import expanded_bank, qualified_template
from sircus.mjlab.expanded_actions import OLD_BOUNDS, WIDE_BOUNDS
from sircus.mjlab.verify_far import EXPECTED_TARGETS, audit_records
from sircus.mjlab.verify_expansion import digest, statistics


class FarAuditTests(unittest.TestCase):
    def fixture(self):
        from sircus.mjlab.tests.test_verify_expansion import ExpansionAuditTests

        _, _, rows, _, original = ExpansionAuditTests().fixture()
        indices = (8, 9, 10, 12, 19, 21, 23, 26, 35, 39, 41, 48)
        targets = [
            dict(
                id=f"old-test-{i:02d}",
                original_test_index=i,
                bucket_xy_m=[0.62, -0.55],
                role="targeted_development",
            )
            for i in indices
        ]
        for i, row in enumerate(rows):
            row.update(
                id=f"far-shot-{i:07d}", target_id="old-test-08", bound_phase="wide"
            )
            if i:
                row["confirmation_source_id"] = "far-shot-0000000"
        spec = dict(
            old_bank_sha256=original["sha256"],
            development_layouts=targets,
            old_bounds=OLD_BOUNDS,
            wide_bounds=WIDE_BOUNDS,
            top_candidates=10,
            confirmations_per_candidate=10,
            minimum_confirmation_rate=0.8,
            selected_target_ids=sorted(EXPECTED_TARGETS),
            native_per_target=1536,
            selected_prior_seed_ids=[],
            selected_prior_seed_records_sha256=[],
            prior_native_search_attempts=6144,
            prior_confirmation_attempts=290,
        )
        spec["fingerprint"] = digest(spec)
        template = qualified_template(rows[0], rows[1:], "old-test-08", WIDE_BOUNDS)
        template.update(
            source_record_sha256=digest(rows[0]),
            source_study_contract_fingerprint=spec["fingerprint"],
            confirmation_study_contract_fingerprint=spec["fingerprint"],
        )
        bank = expanded_bank(original, [template], spec, WIDE_BOUNDS)
        report = dict(
            status="completed",
            phase="completed",
            contract_fingerprint=spec["fingerprint"],
            old_bank_sha256=original["sha256"],
            proposal_bank_sha256=bank["sha256"],
            development_layouts=targets,
            records=11,
            native_search_attempts=1,
            confirmation_attempts=10,
            geometry_proposals=11,
            additions=1,
            original_templates=1,
            expanded_templates=2,
            bounds=WIDE_BOUNDS,
            native_search_statistics=statistics(rows[:1]),
            confirmation_statistics=statistics(rows[1:]),
            prior_native_search_attempts=6144,
            prior_confirmation_attempts=290,
            cells=[
                dict(
                    t,
                    search=statistics(rows[:1] if t["id"] == "old-test-08" else []),
                    promotion=template if t["id"] == "old-test-08" else None,
                )
                for t in targets
            ],
        )
        return spec, report, rows, bank, original, [], []

    def checks(self, values):
        return {c["name"]: c["passed"] for c in audit_records(*values).checks}

    def test_complete_legitimate_fixture_passes(self):
        checks = self.checks(self.fixture())
        self.assertTrue(all(checks.values()), [k for k, v in checks.items() if not v])

    def test_fresh_trial_ids_and_target_scope_cannot_be_swapped(self):
        values = self.fixture()
        values[2][0]["id"] = "inherited-shot"
        self.assertFalse(self.checks(values)["sequential_far_trial_ids"])
        values = self.fixture()
        values[2][0]["target_id"] = "old-test-19"
        self.assertFalse(self.checks(values)["only_six_get_fresh_trials"])

    def test_assisted_capture_or_tampered_template_provenance_fails(self):
        values = self.fixture()
        values[2][0]["external_contact_steps"] = 1
        self.assertFalse(self.checks(values)["capture_rule_invariants"])
        values = self.fixture()
        values[3]["templates"][-1]["source_record_sha256"] = "wrong"
        self.assertFalse(self.checks(values)["old-test-08_joined_source_provenance"])

    def test_unknown_local_seed_is_not_accepted(self):
        values = self.fixture()
        values[2][0].update(
            seed_source_id="other-record", seed_source_record_sha256="unknown"
        )
        self.assertFalse(self.checks(values)["local_seed_geometry_provenance"])


if __name__ == "__main__":
    unittest.main()
