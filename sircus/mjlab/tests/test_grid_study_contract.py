"""No GPU: evidence binding, denominators and frozen-validation display guards."""

import copy
import unittest

from sircus.mjlab import grid_study_contract as c


class GridStudyContractTests(unittest.TestCase):
    def setUp(self):
        self.layout = c.grid_layouts()[0]
        self.identity = dict(
            contract_fingerprint="a" * 64,
            scene_sha256="b" * 64,
            asset_sha256={"mesh": "c" * 64},
            source_sha256={"worker": "d" * 64},
            physics=copy.deepcopy(c.PHYSICS),
        )

    def attempt(self, name="mock-one", *, saved=False, success=True, fresh=19):
        request = dict(
            request_id=name,
            mode="search",
            bucket_xy=self.layout["bucket_xy_m"],
            search_budget=16,
            cpu_candidates=64,
        )
        result = dict(
            request_id=name,
            bucket_xy_m=self.layout["bucket_xy_m"],
            success=success,
            qualified=success,
            status_native=(
                "verified_shot" if success else "capture_found_confirmation_unresolved"
            ),
            method="catalog_lookup" if saved else "sweep_planner",
            configuration=dict(
                board_center=[0.6, -0.3, 0.4],
                release_center=[0.6, -0.3, 0.7],
                tilt_deg=32,
                gap_m=0.2625,
            ),
            motion=dict(passed=True),
            metrics=dict(
                cpu_candidates=84,
                geometry_rejected=53,
                search_cpu_candidates=64,
                search_geometry_rejected=52,
                search_requests=12,
                search_captures=2,
                confirmation_requests=23,
                nominal_captures=3,
                nominal_attempts=3,
                fresh_captures=fresh,
                fresh_attempts=20,
                scalar_score_audit_passed=True,
                invalid=None,
            ),
            provenance=dict(
                self.identity,
                policy_used=False,
                saved_evidence=saved,
                request_sha256="e" * 64,
            ),
            message="Synthetic test fixture; not native evidence",
        )
        return dict(
            stage="baseline",
            request=request,
            result=result,
            result_sha256="f" * 64,
            evidence_sha256={"trace.npz": "1" * 64},
            result_url="/api/planner-grid/result/" + self.layout["id"],
        )

    def report(self, attempts=None, **kwargs):
        return c.make_report(
            {self.layout["id"]: attempts or []},
            identity=self.identity,
            protocol={"fixture": True},
            **kwargs
        )

    def resign(self, report):
        raw = dict(report)
        raw.pop("fingerprint", None)
        report["fingerprint"] = c.digest(raw)

    def test_empty_snapshot_preserves_all35_unknown_targets(self):
        report = self.report()
        self.assertEqual(len(report["cells"]), 35)
        self.assertEqual(
            len({tuple(row["bucket_xy_m"]) for row in report["cells"]}), 35
        )
        self.assertEqual(report["summary"]["pending"], 35)
        self.assertEqual(report["summary"]["search_native_requests"], 0)
        self.assertIsNone(report["cells"][0]["fresh"]["rate"])
        self.assertIsNone(report["cells"][0]["selected"])
        self.assertEqual(c.validate_report(report), report)
        with self.assertRaisesRegex(ValueError, "pending"):
            self.report(status="completed")

    def test_cached_metrics_never_count_as_new_work_or_current_repeats(self):
        saved = self.attempt(saved=True)
        row = self.report([saved])["cells"][0]
        self.assertEqual(row["status"], "saved_verified")
        self.assertEqual(row["selected"]["source"], "saved_reference")
        self.assertEqual(row["search"]["native_requests"], 0)
        self.assertEqual(row["search"]["cpu_candidates"], 0)
        self.assertEqual(row["search"]["confirmation_requests"], 0)
        self.assertIsNone(row["nominal"]["captures"])
        self.assertIsNone(row["fresh"]["rate"])
        self.assertEqual(row["saved_fresh"]["captures"], 19)
        self.assertEqual(row["saved_fresh"]["attempts"], 20)

    def test_new_search_uses_search_only_counts_and_selected_cohort(self):
        first = self.attempt("first", success=False, fresh=7)
        first["result"]["metrics"]["confirmation_requests"] = 46
        chosen = self.attempt("chosen", fresh=20)
        report = self.report([first, chosen])
        row = report["cells"][0]
        self.assertEqual(row["status"], "verified_fresh")
        self.assertEqual(row["selected"]["request_id"], "chosen")
        self.assertEqual(row["selected"]["source"], "fresh_study")
        self.assertEqual(row["search"]["cpu_candidates"], 128)
        self.assertEqual(row["search"]["rejected"], 104)
        self.assertEqual(row["search"]["native_requests"], 24)
        self.assertEqual(row["search"]["confirmation_requests"], 69)
        self.assertEqual(row["fresh"]["captures"], 20)
        self.assertEqual(row["fresh"]["attempts"], 20)
        self.assertEqual(row["fresh"]["rate"], 1)
        self.assertEqual(c.validate_report(report), report)

    def test_unknown_native_capture_count_stays_unknown(self):
        attempted = self.attempt(success=False)
        attempted["result"]["metrics"].pop("search_captures")
        row = self.report([attempted])["cells"][0]
        self.assertIsNone(row["search"]["captures"])
        self.assertIsNone(row["selected"])

    def test_rejected_requested_repeats_stay_in_denominator(self):
        chosen = self.attempt(fresh=18)
        row = self.report([chosen])["cells"][0]
        self.assertEqual(row["fresh"]["rate"], 0.9)
        self.assertEqual(row["fresh"]["attempts"], 20)
        self.assertEqual(row["nominal"]["attempts"], 3)
        for captures, attempts in [(21, 20), (1, None), (True, 1), (1, -1)]:
            with self.assertRaises(ValueError):
                c.statistics(captures, attempts)

    def test_failed_retest_needs_exact_recipe_binding_before_display(self):
        saved = self.attempt("saved", saved=True)
        retry = self.attempt("retry", success=False, fresh=5)
        row = self.report([saved, retry])["cells"][0]
        self.assertIsNone(row["fresh"]["captures"])
        retry["result"]["metrics"]["confirmation_configuration_sha256"] = c.digest(
            saved["result"]["configuration"]
        )
        row = self.report([saved, retry])["cells"][0]
        self.assertEqual(row["status"], "saved_verified")
        self.assertEqual(row["fresh"]["captures"], 5)
        self.assertEqual(row["saved_fresh"]["captures"], 19)

    def test_bucket_request_result_and_source_identities_must_match(self):
        changes = [
            lambda a: a["request"].update(bucket_xy=[0.10000001, -0.65]),
            lambda a: a["result"].update(bucket_xy_m=[0.1, -0.6500001]),
            lambda a: a["result"].update(request_id="another"),
            lambda a: a["result"]["provenance"].update(scene_sha256="9" * 64),
            lambda a: a["result"]["provenance"].update(
                source_sha256={"worker": "9" * 64}
            ),
            lambda a: a["result"]["provenance"].update(asset_sha256={"mesh": "9" * 64}),
            lambda a: a["result"]["provenance"].update(policy_used=True),
            lambda a: a.update(result_url="https://external.test/evidence"),
        ]
        for change in changes:
            with self.subTest(change=change):
                attempted = self.attempt()
                change(attempted)
                with self.assertRaises(ValueError):
                    self.report([attempted])
        altered = copy.deepcopy(self.identity)
        altered["physics"]["substeps"] = 32
        with self.assertRaisesRegex(ValueError, "physics"):
            c.validate_identity(altered)

    def test_success_requires_actual_qualification_and_cannot_be_forced_by_state(self):
        for change in [
            lambda r: r.update(qualified=False),
            lambda r: r.update(motion=dict(passed=False)),
            lambda r: r["metrics"].update(fresh_captures=17),
            lambda r: r["metrics"].update(scalar_score_audit_passed=False),
        ]:
            attempted = self.attempt()
            change(attempted["result"])
            with self.assertRaisesRegex(ValueError, "qualification"):
                self.report([attempted])
        with self.assertRaisesRegex(ValueError, "fabricate"):
            self.report(states={self.layout["id"]: "verified_fresh"})

    def test_hash_tamper_duplicate_requests_and_unsafe_evidence_rejected(self):
        attempted = self.attempt()
        attempted["request_payload_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "request_payload"):
            self.report([attempted])
        attempted = self.attempt()
        attempted["result_payload_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "result_payload"):
            self.report([attempted])
        with self.assertRaisesRegex(ValueError, "double-count"):
            self.report([self.attempt(), self.attempt()])
        attempted = self.attempt()
        attempted["evidence_sha256"] = {"../outside.npz": "1" * 64}
        with self.assertRaisesRegex(ValueError, "relative"):
            self.report([attempted])
        report = self.report([self.attempt()])
        report["cells"][0]["fresh"]["captures"] = 20
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            c.validate_report(report)

    def test_self_consistent_hash_does_not_allow_false_summary_or_ready_claim(self):
        report = self.report([self.attempt()])
        report["summary"]["verified_fresh"] = 35
        self.resign(report)
        with self.assertRaisesRegex(ValueError, "summary"):
            c.validate_report(report)
        report = self.report()
        report["cells"][0]["status"] = "verified_fresh"
        self.resign(report)
        with self.assertRaisesRegex(ValueError, "selected"):
            c.validate_report(report)
        report = self.report([self.attempt()])
        report["cells"][0]["fresh"] = c.statistics(0, 20)
        self.resign(report)
        with self.assertRaisesRegex(ValueError, "qualification"):
            c.validate_report(report)

    def test_unknown_and_blocked_cells_keep_reason_and_do_not_claim_impossibility(self):
        attempted = self.attempt(success=False)
        attempted["result"].update(
            status_native="common_home_unresolved",
            message="Prescribed staging pose obstructed",
        )
        attempted["result"]["metrics"].update(search_requests=0, search_captures=0)
        report = self.report([attempted])
        row = report["cells"][0]
        self.assertEqual(row["status"], "blocked")
        self.assertIn("staging", row["reason"])
        self.assertIsNone(row["selected"])
        self.assertEqual(report["summary"]["blocked"], 1)
        self.assertEqual(report["summary"]["pending"], 34)

    def test_quality_needs_explicit_matching_repeat_cohort(self):
        attempted = self.attempt()
        attempted["quality"] = dict(clean=8, assisted=11, cohort="fresh")
        row = self.report([attempted])["cells"][0]
        self.assertEqual(row["quality"]["clean"], 8)
        self.assertEqual(row["quality"]["source"], "fresh_study")
        attempted["quality"]["clean"] = 9
        with self.assertRaisesRegex(ValueError, "quality"):
            self.report([attempted])

    def test_separate_frozen_validation_cannot_reselect_or_share_search_denominator(
        self,
    ):
        attempted = self.attempt()
        selected_hash = c.digest(attempted["result"]["configuration"])
        validation = dict(
            status="completed",
            nominal=dict(captures=2, attempts=3),
            fresh=dict(captures=12, attempts=30),
            quality=dict(clean=4, assisted=8),
            configuration_sha256=selected_hash,
            seed=321,
        )
        report = self.report(
            [attempted], validation_by_cell={self.layout["id"]: validation}
        )
        row = report["cells"][0]
        self.assertEqual(row["status"], "verified_fresh")
        self.assertEqual(row["fresh"]["captures"], 19)
        self.assertEqual(row["fresh"]["attempts"], 20)
        self.assertEqual(row["frozen_validation"]["fresh"]["rate"], 0.4)
        self.assertEqual(row["frozen_validation"]["nominal"]["captures"], 2)
        self.assertEqual(report["summary"]["frozen_validation_completed"], 1)
        self.assertEqual(report["summary"]["frozen_fresh_attempts"], 30)
        self.assertEqual(c.validate_report(report), report)
        validation["configuration_sha256"] = "7" * 64
        with self.assertRaisesRegex(ValueError, "another selected"):
            self.report([attempted], validation_by_cell={self.layout["id"]: validation})

    def test_frozen_validation_requires_full_budget_and_bound_quality(self):
        attempted = self.attempt()
        raw = dict(
            status="completed",
            configuration_sha256=c.digest(attempted["result"]["configuration"]),
            nominal=dict(captures=3, attempts=3),
            fresh=dict(captures=20, attempts=20),
        )
        with self.assertRaisesRegex(ValueError, "all requested"):
            self.report([attempted], validation_by_cell={self.layout["id"]: raw})
        raw.update(
            fresh=dict(captures=29, attempts=30), quality=dict(clean=20, assisted=10)
        )
        with self.assertRaisesRegex(ValueError, "quality"):
            self.report([attempted], validation_by_cell={self.layout["id"]: raw})
        with self.assertRaisesRegex(ValueError, "unknown grid"):
            c.make_report({"imaginary-cell": []}, identity=self.identity, protocol={})


if __name__ == "__main__":
    unittest.main()
