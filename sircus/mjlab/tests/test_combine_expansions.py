"""Join completed teacher evidence without weakening contracts or effort counts."""

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sircus.mjlab.combine_expansions import combine
from sircus.mjlab.expand_bank import expanded_bank, qualified_template
from sircus.mjlab.learning_contract import digest, statistics
from sircus.mjlab.learning_search import VERSION as BANK_VERSION
from sircus.mjlab.verify_combined import audit_records


class CombinationTests(unittest.TestCase):
    def fixture(self):
        targets = [
            dict(
                id=f"old-test-{i:02d}",
                bucket_xy_m=[0.1 + i * 0.01, -0.5],
                role="targeted_development",
            )
            for i in range(12)
        ]
        bounds = dict(
            low=[0.3, -0.5, 0.12, 8, 0.05],
            high=[1.04, 0.35, 0.6, 60, 0.65],
            version="wide",
        )
        original = dict(
            version=BANK_VERSION, templates=[dict(source_trial_id="old-training")]
        )
        original["sha256"] = digest(original)
        empty = statistics([])
        prior = dict(
            native_search_attempts=6144,
            confirmation_attempts=290,
            cells=[
                dict(t, search=dict(empty, attempts=512, board_bounces=512))
                for t in targets
            ],
        )
        children = {}
        for role, margin in (("refinement", 20), ("far", 30)):
            spec = dict(
                scene={"id": "same"},
                parent_contract_fingerprint="parent",
                parent_checkpoint_sha256="checkpoint",
                old_bank_study_fingerprint="old-study",
                old_bank_sha256=original["sha256"],
                old_bank_file_sha256="old-file",
                old_test_geometry_sha256="geometry",
                development_layouts=targets,
                old_bounds=bounds,
                wide_bounds=bounds,
                prior_search_contract_fingerprint="prior",
                prior_search_report_sha256="prior-report",
                prior_search_records_sha256="prior-records",
                confirmations_per_candidate=10,
                minimum_confirmation_rate=0.8,
            )
            spec["fingerprint"] = digest(spec)
            source = dict(
                id=role + "-source",
                phase="search",
                target_id=targets[0]["id"],
                source_phase="targeted_development",
                invalid=None,
                route_success=True,
                route_quality=2,
                minimum_mouth_clearance_mm=margin,
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
            )
            rows = []
            for i in range(10):
                r = dict(
                    source,
                    id=role + "-confirmation-" + str(i),
                    phase="confirmation",
                    confirmation_source_id=source["id"],
                )
                if i >= 8:
                    r.update(
                        route_success=False,
                        route_quality=0,
                        max_retention_s=0.5,
                        single_bounce_success=False,
                    )
                rows.append(r)
            template = qualified_template(source, rows, targets[0]["id"], bounds, 0.8)
            template.update(
                source_record_sha256=digest(source),
                source_study_contract_fingerprint=spec["fingerprint"],
                confirmation_study_contract_fingerprint=spec["fingerprint"],
            )
            bank = expanded_bank(original, [template], spec, bounds)
            report = dict(
                status="completed",
                phase="completed",
                contract_fingerprint=spec["fingerprint"],
                proposal_bank_sha256=bank["sha256"],
                native_search_attempts=int(role == "far"),
                confirmation_attempts=10,
                prior_native_search_attempts=6144,
                prior_confirmation_attempts=290,
                additions=1,
                cells=[
                    dict(
                        t,
                        search=(
                            statistics([source])
                            if role == "far" and t["id"] == targets[0]["id"]
                            else statistics([])
                        ),
                    )
                    for t in targets
                ],
            )
            mapping = dict(
                role=role,
                directory="/immutable/" + role,
                files_sha256={"report.json": "sha"},
                audit_dependencies=dict(
                    prior_search="/prior", old_bank_study="/old", source="/source"
                ),
            )
            children[role] = dict(
                spec=spec,
                report=report,
                bank=bank,
                rows=[source] + rows,
                mapping=mapping,
            )
        return original, children, prior

    def result(self, values):
        spec, report, bank = combine(*values)
        return spec, report, bank

    def checked(self, values, result=None):
        spec, report, bank = result or self.result(values)
        original, children, prior = values
        return {
            c["name"]: c["passed"]
            for c in audit_records(spec, report, bank, original, children, prior).checks
        }

    def test_legitimate_join_and_independent_audit_pass(self):
        values = self.fixture()
        spec, report, bank = self.result(values)
        self.assertTrue(all(self.checked(values, (spec, report, bank)).values()))
        self.assertEqual(bank["templates"][:-1], values[0]["templates"])
        self.assertEqual(bank["templates"][-1]["combined_child_role"], "far")
        self.assertEqual(report["additions"], 1)
        self.assertEqual(report["native_search_attempts"], 0)

    def test_shared_prior_is_counted_once(self):
        _, report, _ = self.result(self.fixture())
        self.assertEqual(report["effort"]["native_search_attempts_total"], 6145)
        self.assertEqual(report["effort"]["native_confirmation_attempts_total"], 310)
        self.assertEqual(report["cells"][0]["search"]["attempts"], 513)
        self.assertEqual(report["cells"][0]["fresh_confirmation"]["attempts"], 20)

    def test_running_child_blocks_join(self):
        values = self.fixture()
        values[1]["far"]["report"]["status"] = "running"
        with self.assertRaises(ValueError):
            self.result(values)

    def test_different_parent_or_scene_blocks_join(self):
        for key in ("parent_checkpoint_sha256", "scene", "development_layouts"):
            values = self.fixture()
            values[1]["far"]["spec"][key] = "changed"
            with self.assertRaises(ValueError):
                self.result(values)

    def test_original_prefix_change_blocks_join(self):
        values = self.fixture()
        values[1]["far"]["bank"]["templates"][0]["source_trial_id"] = "changed"
        with self.assertRaises(ValueError):
            self.result(values)

    def test_qualification_cannot_be_weakened(self):
        values = self.fixture()
        values[1]["far"]["bank"]["templates"][-1]["confirmation"]["captures"] = 7
        with self.assertRaises(ValueError):
            self.result(values)

    def test_no_capture_label_can_be_inherited(self):
        values = self.fixture()
        values[1]["far"]["rows"][0]["route_success"] = False
        with self.assertRaises(ValueError):
            self.result(values)

    def test_higher_confirmed_count_beats_margin(self):
        values = self.fixture()
        values[1]["refinement"]["bank"]["templates"][-1]["confirmation"]["captures"] = 9
        values[1]["refinement"]["rows"][-1].update(
            route_success=True,
            route_quality=2,
            max_retention_s=1.0,
            single_bounce_success=True,
        )
        _, _, bank = self.result(values)
        self.assertEqual(bank["templates"][-1]["combined_child_role"], "refinement")

    def test_template_counters_must_match_actual_records(self):
        values = self.fixture()
        values[1]["refinement"]["bank"]["templates"][-1]["confirmation"]["captures"] = 9
        with self.assertRaises(ValueError):
            self.result(values)

    def test_auditor_catches_effort_and_teacher_tampering(self):
        values = self.fixture()
        result = self.result(values)
        result[1]["effort"]["native_search_attempts_total"] += 6144
        self.assertFalse(
            self.checked(values, result)["distinct_effort_without_prior_double_count"]
        )
        result[2]["templates"][-1]["combined_child_role"] = "changed"
        self.assertFalse(
            self.checked(values, result)["exact_confirmed_winners_and_provenance"]
        )

    def test_input_banks_and_sources_unchanged(self):
        values = self.fixture()
        saved = copy.deepcopy(values)
        self.result(values)
        self.assertEqual(values, saved)


if __name__ == "__main__":
    unittest.main()
