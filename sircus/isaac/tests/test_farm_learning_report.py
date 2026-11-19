import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from farm_learning_report import (
    CELL_IDS,
    DEFAULT_BASELINE,
    baseline_cells,
    clean_stats,
    make_handler,
    merge_transfer,
    learning_curve,
    normalize_state,
    ratio,
    read_state,
    render_page,
)


def fixture_baseline():
    return json.loads(DEFAULT_BASELINE.read_text())


def fixture_stats(captures=3):
    return dict(
        attempts=8,
        releases=7,
        invalid=1,
        captures=captures,
        clean=2,
        single_bounce=3,
        mean_learning_reward=0.4,
        mean_historical_reward=0.1,
    )


def fixture_result():
    previous = baseline_cells(fixture_baseline())
    return dict(
        version="sircus-candidate-shot-learning-v1",
        study_fingerprint="fixture-study",
        source_fingerprint=fixture_baseline()["source_fingerprint"],
        checkpoints={"candidate-policy-final.pt": "c" * 64},
        final_weight_hash="a" * 64,
        native_bank_sha256="b" * 64,
        summary={"candidate_final_stochastic": fixture_stats()},
        per_cell=[
            dict(
                cell_id=identity,
                bucket_xy_m=previous[identity]["bucket_xy_m"],
                stages={"candidate_final_stochastic": fixture_stats()},
                best_sweep_action=dict(
                    height_m=0.35,
                    gap_m=0.22,
                    tilt_deg=28,
                    radius_m=0.5,
                    azimuth_offset_deg=10,
                ),
            )
            for identity in CELL_IDS
        ],
    )


def fixture_transfer():
    main = fixture_result()
    stages = ("probe_sweep", "transfer_stochastic", "transfer_deterministic")
    one = fixture_stats()
    aggregate = {
        key: value * 35 if type(value) is int else value for key, value in one.items()
    }
    return dict(
        version="sircus-frozen-candidate-transfer-v1",
        fingerprint="d" * 64,
        parent_study_fingerprint=main["study_fingerprint"],
        source_fingerprint=main["source_fingerprint"],
        source_checkpoint_sha256=main["checkpoints"]["candidate-policy-final.pt"],
        source_policy_weight_hash=main["final_weight_hash"],
        source_native_bank_sha256=main["native_bank_sha256"],
        transfer_native_bank_sha256="e" * 64,
        policy_updates=0,
        summary={stage: copy.deepcopy(aggregate) for stage in stages},
        per_cell={
            stage: [
                dict(
                    cell_id=row["cell_id"],
                    bucket_xy_m=row["bucket_xy_m"],
                    **fixture_stats()
                )
                for row in main["per_cell"]
            ]
            for stage in stages
        },
    )


class FarmLearningReportTests(unittest.TestCase):
    def test_bound_gpu_timeout_status_reports_not_run_without_merging_payloads(self):
        main = fixture_result()
        status = dict(
            schema="sircus-transfer-probe-status-v1",
            study_fingerprint=main["study_fingerprint"],
            source_fingerprint=main["source_fingerprint"],
            source_checkpoint_sha256=main["checkpoints"]["candidate-policy-final.pt"],
            native_started=False,
            status="blocked",
            reason_code="shared_gpu_lock_timeout",
            per_cell=fixture_transfer()["per_cell"],
        )
        with tempfile.TemporaryDirectory() as name:
            output = Path(name)
            (output / "farm-study-result.json").write_text(json.dumps(main))
            (output / "transfer-probe-status.json").write_text(json.dumps(status))
            state = read_state(output)
            self.assertEqual(state["transfer"]["status"], "blocked")
            self.assertIn("did not run", state["transfer"]["reason"])
            self.assertNotIn("transfer_stochastic", state["per_cell"][0]["stages"])
            self.assertEqual(
                state["per_cell"][0]["stages"]["candidate_final_stochastic"][
                    "captures"
                ],
                3,
            )
            status["source_checkpoint_sha256"] = "f" * 64
            (output / "transfer-probe-status.json").write_text(json.dumps(status))
            self.assertEqual(read_state(output)["transfer"]["status"], "rejected")

    def test_completed_matching_transfer_overrides_an_old_status_file(self):
        with tempfile.TemporaryDirectory() as name:
            output = Path(name)
            (output / "farm-study-result.json").write_text(json.dumps(fixture_result()))
            (output / "transfer-probe-result.json").write_text(
                json.dumps(fixture_transfer())
            )
            (output / "transfer-probe-status.json").write_text(
                "obsolete-status-is-not-read"
            )
            state = read_state(output)
            self.assertEqual(state["transfer"]["status"], "verified")
            self.assertIn("transfer_stochastic", state["per_cell"][0]["stages"])

    def test_learning_curve_uses_cumulative_attempts_and_only_last16_batches(self):
        batches = [dict(attempts=35, captures=0)] * 10 + [
            dict(attempts=35, captures=35)
        ] * 10
        curve = learning_curve(dict(stages=dict(train=batches)))
        self.assertEqual(curve[-1]["throws"], 700)
        self.assertEqual(curve[-1]["rolling16rate"], 10 / 16)
        self.assertEqual(curve[0]["rolling16rate"], 0)
        self.assertEqual(learning_curve({}), [])
        with self.assertRaises(ValueError):
            learning_curve(dict(stages=dict(train=[dict(attempts=2, captures=3)])))

    def test_verified_transfer_adds_only_separate_metrics_and_preserves_main_identity(
        self,
    ):
        main = fixture_result()
        state = normalize_state(main, fixture_baseline(), complete=True)
        original = copy.deepcopy(state)
        merged = merge_transfer(state, main, fixture_transfer(), fixture_baseline())
        self.assertEqual(state, original)
        self.assertEqual(merged["identity"], original["identity"])
        self.assertEqual(
            merged["summary"]["candidate_final_stochastic"],
            original["summary"]["candidate_final_stochastic"],
        )
        self.assertEqual(
            merged["per_cell"][0]["stages"]["candidate_final_stochastic"],
            original["per_cell"][0]["stages"]["candidate_final_stochastic"],
        )
        self.assertEqual(
            merged["per_cell"][0]["stages"]["transfer_stochastic"]["captures"], 3
        )
        self.assertEqual(merged["transfer"]["status"], "verified")
        self.assertEqual(merged["transfer"]["provenance"]["policy_updates"], 0)

    def test_transfer_cannot_replace_primary_stages_through_row_or_summary_fields(self):
        main = fixture_result()
        state = normalize_state(main, fixture_baseline(), complete=True)
        transfer = fixture_transfer()
        grouped = []
        for i, row in enumerate(main["per_cell"]):
            grouped.append(
                dict(
                    cell_id=row["cell_id"],
                    bucket_xy_m=row["bucket_xy_m"],
                    stages={
                        **{stage: fixture_stats() for stage in transfer["per_cell"]},
                        "candidate_final_stochastic": fixture_stats(captures=6),
                    },
                )
            )
        transfer["per_cell"] = grouped
        transfer["summary"]["candidate_final_stochastic"] = fixture_stats(captures=6)
        merged = merge_transfer(state, main, transfer, fixture_baseline())
        self.assertEqual(
            merged["per_cell"][0]["stages"]["candidate_final_stochastic"]["captures"], 3
        )
        self.assertEqual(merged["summary"]["candidate_final_stochastic"]["captures"], 3)

    def test_transfer_requires_exact_known_parent_final_checkpoint_and_frozen_weight_identity(
        self,
    ):
        main = fixture_result()
        state = normalize_state(main, fixture_baseline(), complete=True)
        for key, wrong in (
            ("parent_study_fingerprint", "other-study"),
            ("parent_study_fingerprint", None),
            ("source_checkpoint_sha256", "f" * 64),
            ("source_policy_weight_hash", "f" * 64),
            ("source_native_bank_sha256", "f" * 64),
            ("source_fingerprint", "other-scene"),
            ("policy_updates", 1),
        ):
            transfer = fixture_transfer()
            transfer[key] = wrong
            with self.subTest(key=key, wrong=wrong), self.assertRaises(ValueError):
                merge_transfer(state, main, transfer, fixture_baseline())

    def test_optional_bad_transfer_is_excluded_without_hiding_valid_primary_results(
        self,
    ):
        with tempfile.TemporaryDirectory() as name:
            output = Path(name)
            (output / "farm-study-result.json").write_text(json.dumps(fixture_result()))
            transfer = fixture_transfer()
            transfer["source_checkpoint_sha256"] = "f" * 64
            (output / "transfer-probe-result.json").write_text(json.dumps(transfer))
            state = read_state(output)
            self.assertTrue(state["complete"])
            self.assertEqual(state["transfer"]["status"], "rejected")
            self.assertEqual(
                state["per_cell"][0]["stages"]["candidate_final_stochastic"][
                    "captures"
                ],
                3,
            )
            self.assertNotIn("transfer_stochastic", state["per_cell"][0]["stages"])

    def test_transfer_rejects_changed_buckets_missing_tables_and_wrong_totals(self):
        main = fixture_result()
        state = normalize_state(main, fixture_baseline(), complete=True)
        for change in ("bucket", "missing", "totals"):
            transfer = fixture_transfer()
            if change == "bucket":
                transfer["per_cell"]["probe_sweep"][0]["bucket_xy_m"][0] += 0.01
            elif change == "missing":
                transfer["per_cell"]["transfer_stochastic"].pop()
            else:
                transfer["summary"]["transfer_stochastic"]["captures"] -= 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                merge_transfer(state, main, transfer, fixture_baseline())

    def test_per_table_comparison_preserves_all_35_distinct_bucket_coordinates(self):
        baseline = fixture_baseline()
        state = normalize_state(fixture_result(), baseline, complete=True)
        self.assertEqual(len(state["per_cell"]), 35)
        self.assertEqual(len({tuple(c["bucket_xy_m"]) for c in state["per_cell"]}), 35)
        self.assertEqual(sum(c["baseline"]["captures"] for c in state["per_cell"]), 111)
        self.assertEqual(
            sum(c["baseline"]["captures"] == 0 for c in state["per_cell"]), 20
        )
        row = state["per_cell"][0]
        self.assertEqual(
            row["stages"]["candidate_final_stochastic"]["capture_rate"], 3 / 8
        )
        self.assertAlmostEqual(
            row["best_sweep_action"]["ball_center_release_m"], 0.6075
        )

    def test_missing_stages_and_empty_samples_remain_unknown(self):
        state = normalize_state({}, fixture_baseline())
        self.assertFalse(state["complete"])
        self.assertEqual(state["phase"], "waiting")
        self.assertTrue(all(c["stages"] == {} for c in state["per_cell"]))
        self.assertIsNone(clean_stats({"attempts": 0, "captures": 0})["capture_rate"])
        self.assertIsNone(ratio(None, 8))
        self.assertIsNone(ratio(0, 0))

    def test_unknown_or_repeated_ids_and_changed_bucket_are_rejected(self):
        original = fixture_result()
        for change in ("unknown", "duplicate", "bucket"):
            doc = copy.deepcopy(original)
            if change == "unknown":
                doc["per_cell"][0]["cell_id"] = "r6c1"
            elif change == "duplicate":
                doc["per_cell"][0]["cell_id"] = "r1c2"
            else:
                doc["per_cell"][0]["bucket_xy_m"][0] += 0.001
            with self.subTest(change=change), self.assertRaises(ValueError):
                normalize_state(doc, fixture_baseline(), complete=True)

    def test_completed_report_requires_all_tables_but_partial_progress_can_omit_them(
        self,
    ):
        doc = fixture_result()
        doc["per_cell"] = doc["per_cell"][:5]
        with self.assertRaises(ValueError):
            normalize_state(doc, fixture_baseline(), complete=True)
        self.assertEqual(len(normalize_state(doc, fixture_baseline())["per_cell"]), 35)

    def test_inconsistent_or_nonfinite_statistics_are_rejected(self):
        for data in (
            {"attempts": 4, "releases": 5},
            {"releases": 1, "captures": 2},
            {"attempts": 3, "releases": 1, "invalid": 1},
            {"captures": True},
            {"mean_learning_reward": float("nan")},
            {"clean": 3, "captures": 2},
        ):
            with self.subTest(data=data), self.assertRaises(ValueError):
                clean_stats(data)

    def test_unapproved_fields_do_not_reach_browser_state(self):
        doc = fixture_result()
        doc["secret"] = "must-not-be-served"
        doc["summary"]["candidate_final_stochastic"]["secret"] = "must-not-be-served"
        doc["per_cell"][0]["best_sweep_action"]["password"] = "must-not-be-served"
        state = normalize_state(doc, fixture_baseline(), complete=True)
        self.assertNotIn("must-not-be-served", json.dumps(state))

    def test_initial_json_cannot_terminate_script_or_inject_html(self):
        state = normalize_state(
            {"phase": "</script><img src=x onerror=alert(1)>"}, fixture_baseline()
        )
        page = render_page(state).decode()
        self.assertNotIn("</script><img", page)
        self.assertIn("\\u003c/script>", page)
        self.assertNotIn("__SIRCUS_STATE__", page)
        self.assertIn("textContent", page)
        self.assertNotIn("innerHTML", page)

    def test_only_named_aggregate_files_are_read_and_external_symlinks_are_rejected(
        self,
    ):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            output = root / "output"
            output.mkdir()
            secret = root / "secret.json"
            secret.write_text(json.dumps(fixture_result()))
            (output / "credentials.json").write_text("not-json-and-never-read")
            self.assertEqual(read_state(output)["phase"], "waiting")
            (output / "farm-study-result.json").symlink_to(secret)
            with self.assertRaisesRegex(ValueError, "outside"):
                read_state(output)

    def test_progress_then_complete_report_and_contract_identity(self):
        with tempfile.TemporaryDirectory() as name:
            output = Path(name)
            (output / "farm-study-contract.json").write_text(
                json.dumps(
                    dict(
                        version="sircus-candidate-shot-learning-v1",
                        fingerprint="fixture-study",
                        source_fingerprint=fixture_baseline()["source_fingerprint"],
                    )
                )
            )
            (output / "progress.json").write_text(json.dumps({"phase": "sweep"}))
            state = read_state(output)
            self.assertEqual(state["phase"], "sweep")
            self.assertEqual(state["identity"]["study_fingerprint"], "fixture-study")
            self.assertFalse(state["complete"])
            (output / "farm-study-result.json").write_text(json.dumps(fixture_result()))
            self.assertTrue(read_state(output)["complete"])

    def test_mismatched_scene_or_study_identity_is_rejected(self):
        doc = fixture_result()
        doc["source_fingerprint"] = "another-scene"
        with self.assertRaisesRegex(ValueError, "scene fingerprint"):
            normalize_state(doc, fixture_baseline(), complete=True)
        with tempfile.TemporaryDirectory() as name:
            output = Path(name)
            (output / "farm-study-contract.json").write_text(
                json.dumps(dict(version="other-version", fingerprint="another-study"))
            )
            (output / "progress.json").write_text(json.dumps(fixture_result()))
            with self.assertRaisesRegex(ValueError, "identity"):
                read_state(output)

    def test_routes_are_strictly_allowlisted_and_head_has_no_response_body(self):
        with tempfile.TemporaryDirectory() as name:
            handler_class = make_handler(Path(name))
            for path in (
                "/api/move",
                "/api/stop",
                "/../secret.json",
                "/progress.json",
                "/api/state/extra",
            ):
                handler = object.__new__(handler_class)
                handler.path = path
                errors = []
                handler.send_error = lambda code, *args: errors.append(code)
                handler.respond(False)
                self.assertEqual(errors, [404])
            handler = object.__new__(handler_class)
            handler.path = "/api/state?refresh=1"
            handler.wfile = io.BytesIO()
            headers = []
            handler.send_response = lambda code: headers.append(("status", code))
            handler.send_header = lambda k, v: headers.append((k, v))
            handler.end_headers = lambda: None
            handler.respond(True)
            self.assertEqual(handler.wfile.getvalue(), b"")
            self.assertIn(("status", 200), headers)
            self.assertTrue(any(k == "Content-Security-Policy" for k, v in headers))
            self.assertFalse(hasattr(handler_class, "do_POST"))


if __name__ == "__main__":
    unittest.main()
