"""Frozen-label, validation-only wall diagnostics; no GPU or native physics."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sircus.mjlab.contract import FORCE_NAMES, ScoreStream
from sircus.mjlab import benchmark_wall_diagnostic as diagnostic


def scene():
    return dict(
        simulation=dict(physics_dt_s=1 / 240),
        ball=dict(radius_m=0.0375),
        bucket=dict(
            segments=64,
            assumed_bottom_thickness_m=0.008,
            assumed_interior_depth_m=0.20,
            inner_top_diameter_m=0.22,
            inner_bottom_diameter_m=0.18,
        ),
    )


def trace(kind="wall_resets"):
    positions = np.zeros((960, 3), dtype=np.float32)
    positions[:, 2] = 0.10
    positions[:2, 2] = 0.40
    velocities = np.zeros_like(positions)
    velocities[1, 2] = 1
    forces = np.zeros((960, len(FORCE_NAMES)), dtype=np.float32)
    forces[0, 0] = 1
    if kind == "wall_resets":
        positions[np.arange(100, 960, 100), 0] = 0.20
    elif kind == "late_entry":
        positions[:840, 2] = 0.40
    elif kind == "success_then_exit":
        positions[300:, 0] = 0.40
        forces[300:, FORCE_NAMES.index("floor")] = 5
    return positions, velocities, forces


def row_for(arrays, *, identifier="sweep-7-validation-0000-layout-r0", repeat=0):
    scorer = ScoreStream(scene(), [0.0, 0.0], [0.0, 0.0, 1.0])
    for p, v, f in zip(*arrays):
        scorer.step(p, v, dict(zip(FORCE_NAMES, f)))
    measured = json.loads(
        json.dumps(scorer.result(), default=lambda value: value.item())
    )
    return dict(
        measured,
        id=identifier,
        role="validation",
        method="sweep",
        seed=7,
        samples=0,
        context_id="layout",
        repeat=repeat,
        bucket_xy_m=[0.0, 0.0],
        board_normal=[0.0, 0.0, 1.0],
        command_values=[0.6, -0.2, 0.3, 30.0, 0.2],
        trajectory_dt_s=1 / 60,
        native_release=True,
        invalid=None,
        frozen_artifact_sha256="a" * 64,
        native_cache_file="native-0000.npz",
        native_cache_row=repeat,
    )


class TraceTests(unittest.TestCase):
    def test_rim_floor_geometry_and_external_floor_contact_are_separate(self):
        arrays = trace("plain")
        arrays[0][100, 2] = 0.30
        arrays[0][200, 2] = 0.04
        arrays[2][150, FORCE_NAMES.index("floor")] = 1
        original = row_for(arrays)
        result = diagnostic.diagnose_trace(scene(), original, *arrays)
        failures = result["post_entry_constraint_failures"]
        self.assertEqual(failures["wall"]["failed_samples"], 0)
        self.assertEqual(failures["above_rim"]["failed_samples"], 1)
        self.assertEqual(failures["below_floor"]["failed_samples"], 1)
        self.assertEqual(
            result["scored_contact_observations"]["floor"][
                "samples_above_original_contact_threshold"
            ],
            1,
        )
        self.assertEqual(
            result["original_rim_and_assistance"]["external_contact_steps"], 1
        )
        self.assertFalse(original["route_success"])
        self.assertTrue(original["success"])

    def test_early_entry_wall_resets_are_distinct_from_late_entry(self):
        arrays = trace("wall_resets")
        result = diagnostic.diagnose_trace(scene(), row_for(arrays), *arrays)
        self.assertAlmostEqual(result["first_whole_entry_time_s"], 3 / 240)
        self.assertAlmostEqual(result["max_continuous_retention_s"], 99 / 240)
        self.assertTrue(result["scoring_end_whole_ball_inside"])
        self.assertTrue(result["trace_end_whole_ball_inside"])
        self.assertEqual(result["retention_resets"]["count"], 9)
        self.assertEqual(result["retention_resets"]["duration_samples"], [1] * 9)
        failures = result["post_entry_constraint_failures"]
        self.assertEqual(failures["wall"]["failed_samples"], 9)
        self.assertEqual(failures["above_rim"]["failed_samples"], 0)
        self.assertEqual(failures["below_floor"]["failed_samples"], 0)
        self.assertGreater(result["failed_wall_nominal_overlap_m"]["min"], 0)
        late_arrays = trace("late_entry")
        late = diagnostic.diagnose_trace(scene(), row_for(late_arrays), *late_arrays)
        self.assertAlmostEqual(late["first_whole_entry_time_s"], 841 / 240)
        self.assertAlmostEqual(late["max_continuous_retention_s"], 0.5)
        self.assertEqual(late["retention_resets"]["count"], 0)
        self.assertTrue(late["trace_end_whole_ball_inside"])

    def test_success_freezes_original_score_but_full_trace_end_is_separate(self):
        arrays = trace("success_then_exit")
        original = row_for(arrays)
        self.assertTrue(original["route_success"])
        result = diagnostic.diagnose_trace(scene(), original, *arrays)
        self.assertTrue(result["scoring_end_whole_ball_inside"])
        self.assertFalse(result["trace_end_whole_ball_inside"])
        self.assertLess(result["scoring_end_time_s"], result["trace_end_time_s"])
        self.assertEqual(
            result["scored_contact_observations"]["floor"][
                "samples_above_original_contact_threshold"
            ],
            0,
        )

    def test_original_score_or_cadence_change_is_rejected(self):
        arrays = trace()
        row = row_for(arrays)
        row["max_retention_s"] += 0.01
        with self.assertRaisesRegex(
            diagnostic.DiagnosticError, "Original scorer mismatch: max_retention_s"
        ):
            diagnostic.diagnose_trace(scene(), row, *arrays)
        row = row_for(arrays)
        with self.assertRaisesRegex(diagnostic.DiagnosticError, "cadence"):
            diagnostic.diagnose_trace(
                scene(), row, *arrays, native_scoring_dt_s=1 / 120
            )

    def test_display_trail_cadence_does_not_override_native_scoring_cadence(self):
        arrays = trace()
        row = row_for(arrays)
        self.assertEqual(row["trajectory_dt_s"], 1 / 60)
        result = diagnostic.diagnose_trace(scene(), row, *arrays)
        self.assertEqual(result["display_trajectory_dt_s"], 1 / 60)
        self.assertEqual(result["scoring_trace_dt_s"], 1 / 240)
        self.assertAlmostEqual(result["first_whole_entry_time_s"], 3 / 240)

    def test_legacy_npz_and_explicit_correct_timing_pass_but_wrong_dt_fails(self):
        legacy = diagnostic.native_cache_timing({})
        self.assertEqual(legacy["scoring_dt_s"], 1 / 240)
        self.assertIsNone(legacy["explicit_npz_timing"])
        correct = diagnostic.native_cache_timing(
            {"dt_s": np.asarray(1 / 240, dtype=np.float32)}
        )
        self.assertIn("dt_s", correct["explicit_npz_timing"])
        with self.assertRaisesRegex(diagnostic.DiagnosticError, "NPZ cadence differs"):
            diagnostic.native_cache_timing({"dt_s": np.asarray(1 / 60)})
        times = (np.arange(1, 961) / 240).astype(np.float32)
        diagnostic.native_cache_timing({"timestamps_s": times})
        times[250] += 1 / 240
        with self.assertRaisesRegex(
            diagnostic.DiagnosticError, "NPZ timestamps differ"
        ):
            diagnostic.native_cache_timing({"timestamps_s": times})

    def test_nonfinite_wrong_shape_or_rejected_trace_cannot_be_diagnosed(self):
        arrays = list(trace())
        row = row_for(arrays)
        changed = arrays[0].copy()
        changed[50, 0] = np.nan
        with self.assertRaisesRegex(diagnostic.DiagnosticError, "Nonfinite"):
            diagnostic.diagnose_trace(scene(), row, changed, *arrays[1:])
        with self.assertRaisesRegex(diagnostic.DiagnosticError, "4-second"):
            diagnostic.diagnose_trace(scene(), row, arrays[0][:-1], *arrays[1:])
        row["invalid"] = "geometry"
        with self.assertRaisesRegex(diagnostic.DiagnosticError, "admitted native"):
            diagnostic.diagnose_trace(scene(), row, *arrays)

    def test_trace_hash_binds_saved_dtype_and_exact_observation(self):
        arrays = trace()
        first = diagnostic.trace_sha256("request", *arrays)
        self.assertEqual(first, diagnostic.trace_sha256("request", *arrays))
        self.assertNotEqual(first, diagnostic.trace_sha256("other-request", *arrays))
        self.assertNotEqual(
            first,
            diagnostic.trace_sha256("request", *(a.astype(np.float64) for a in arrays)),
        )
        changed = arrays[0].copy()
        changed[959, 0] += 1e-6
        self.assertNotEqual(
            first, diagnostic.trace_sha256("request", changed, *arrays[1:])
        )


class RunTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.study = self.root / "study"
        self.run = self.study / "runs/sweep-seed7"
        (self.run / "native").mkdir(parents=True)
        self.reference = self.root / "reference"
        self.reference.mkdir()
        ref = dict(scene=scene(), version="test-reference")
        ref["fingerprint"] = diagnostic.digest(ref)
        self.write(self.reference / "contract.json", ref)
        contract = dict(
            identity=dict(
                scene_sha256=diagnostic.digest(scene()),
                contract_fingerprint=ref["fingerprint"],
                physics=dict(scoring_dt_s=1 / 240, horizon_s=4.0),
                source_sha256={
                    name: diagnostic.file_sha256(diagnostic.ROOT / name)
                    for name in (
                        *diagnostic.SCORER_FILES,
                        *diagnostic.TRACE_SOURCE_FILES,
                    )
                },
            ),
            protocol=dict(milestones=[0, 256], evaluation=dict(repeats=3)),
            splits=dict(validation=[dict(id="layout", bucket_xy_m=[0, 0])]),
        )
        contract["fingerprint"] = diagnostic.digest(contract)
        self.write(self.study / "contract.json", contract)
        traces = [trace("wall_resets"), trace("late_entry"), trace("success_then_exit")]
        self.rows = [
            row_for(a, identifier=f"sweep-7-validation-0000-layout-r{k}", repeat=k)
            for k, a in enumerate(traces)
        ]
        self.ledger = self.run / "rows.jsonl"
        self.write_rows()
        np.savez(
            self.run / "native/native-0000.npz",
            record_ids=np.asarray([r["id"] for r in self.rows]),
            position_m=np.stack([a[0] for a in traces], axis=1),
            velocity_m_s=np.stack([a[1] for a in traces], axis=1),
            forces_N=np.stack([a[2] for a in traces], axis=1),
        )
        counts = diagnostic._counts(self.rows)
        self.report = dict(
            version="test-run",
            method="sweep",
            seed=7,
            contract_fingerprint=contract["fingerprint"],
            curves=[
                dict(
                    method="sweep",
                    seed=7,
                    samples=0,
                    validation=counts,
                    artifact_sha256="a" * 64,
                )
            ],
            final_test=dict(private_outcome="DO_NOT_DECODE"),
        )
        self.write(self.run / "report.json", self.report)

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value) + "\n")

    def write_rows(self, suffix="THIS_MUST_NOT_BE_READ\n"):
        self.ledger.write_text(
            "".join(json.dumps(row) + "\n" for row in self.rows) + suffix
        )

    def test_complete_validation_only_diagnostic_binds_hashes_and_preserves_labels(
        self,
    ):
        original_loads = json.loads

        def guarded_loads(value, *args, **kwargs):
            self.assertNotIn("DO_NOT_DECODE", value)
            return original_loads(value, *args, **kwargs)

        with patch.object(diagnostic.json, "loads", guarded_loads):
            result = diagnostic.diagnose_run(
                self.run, self.reference, [0], request_ids=[self.rows[0]["id"]]
            )
        self.assertEqual(len(result["requests"]), 1)
        self.assertEqual(
            result["cohorts"][0]["original_counts"], diagnostic._counts(self.rows)
        )
        self.assertEqual(
            result["requests"][0]["original"]["max_retention_s"],
            self.rows[0]["max_retention_s"],
        )
        self.assertEqual(
            result["requests"][0]["native_cache_sha256"],
            diagnostic.file_sha256(self.run / "native/native-0000.npz"),
        )
        self.assertTrue(
            result["requests"][0]["diagnostic"]["original_score_reproduced"]
        )
        self.assertEqual(
            result["requests"][0]["native_cache_timing"]["scoring_dt_s"], 1 / 240
        )
        self.assertIsNone(
            result["requests"][0]["native_cache_timing"]["explicit_npz_timing"]
        )
        self.assertNotIn(str(self.root), json.dumps(result))
        self.assertEqual(
            result["run_report_sha256"],
            diagnostic.file_sha256(self.run / "report.json"),
        )

    def test_unpublished_milestone_or_incomplete_or_duplicate_cohort_is_rejected(self):
        with self.assertRaisesRegex(
            diagnostic.DiagnosticError, "not published complete"
        ):
            diagnostic.diagnose_run(self.run, self.reference, [256])
        self.rows[1] = copy.deepcopy(self.rows[0])
        self.write_rows()
        with self.assertRaisesRegex(diagnostic.DiagnosticError, "Duplicate/unexpected"):
            diagnostic.diagnose_run(self.run, self.reference, [0])

    def test_frozen_scorer_hash_and_unknown_request_selection_are_rejected(self):
        with self.assertRaisesRegex(diagnostic.DiagnosticError, "IDs must belong"):
            diagnostic.diagnose_run(
                self.run, self.reference, [0], request_ids=["test-request"]
            )
        path = self.study / "contract.json"
        contract = json.loads(path.read_text())
        contract["identity"]["source_sha256"][diagnostic.SCORER_FILES[0]] = "0" * 64
        contract.pop("fingerprint")
        contract["fingerprint"] = diagnostic.digest(contract)
        self.write(path, contract)
        with self.assertRaisesRegex(
            diagnostic.DiagnosticError, "scorer/cache producer differs from frozen"
        ):
            diagnostic.diagnose_run(self.run, self.reference, [0])

    def test_wrong_explicit_native_cache_dt_is_rejected_with_real_display_cadence(self):
        path = self.run / "native/native-0000.npz"
        with np.load(path, allow_pickle=False) as existing:
            values = {name: existing[name] for name in existing.files}
        values["dt_s"] = np.asarray(1 / 60)
        np.savez(path, **values)
        self.assertEqual(self.rows[0]["trajectory_dt_s"], 1 / 60)
        with self.assertRaisesRegex(diagnostic.DiagnosticError, "NPZ cadence differs"):
            diagnostic.diagnose_run(self.run, self.reference, [0])

    def test_missing_validation_stops_at_test_boundary_without_decoding_its_outcome(
        self,
    ):
        self.rows = self.rows[:1]
        self.write_rows(
            json.dumps(dict(role="test", samples=0, private_outcome="DO_NOT_DECODE"))
            + "\n"
        )
        with self.assertRaisesRegex(diagnostic.DiagnosticError, "Test boundary"):
            diagnostic.diagnose_run(self.run, self.reference, [0])

    def test_original_count_or_artifact_binding_change_is_rejected(self):
        self.report["curves"][0]["validation"]["captures"] += 1
        self.write(self.run / "report.json", self.report)
        with self.assertRaisesRegex(
            diagnostic.DiagnosticError, "original counts disagree"
        ):
            diagnostic.diagnose_run(self.run, self.reference, [0])
        self.report["curves"][0]["validation"]["captures"] -= 1
        self.write(self.run / "report.json", self.report)
        self.rows[0]["frozen_artifact_sha256"] = "b" * 64
        self.write_rows()
        with self.assertRaisesRegex(
            diagnostic.DiagnosticError, "frozen artifact differs"
        ):
            diagnostic.diagnose_run(self.run, self.reference, [0])

    def test_wrong_trace_id_or_foreign_test_column_is_rejected_before_arrays(self):
        path = self.run / "native/native-0000.npz"
        np.savez(path, record_ids=np.asarray(["private-test-id"]))
        with self.assertRaisesRegex(
            diagnostic.DiagnosticError, "unknown/non-selected validation IDs"
        ):
            diagnostic.diagnose_run(self.run, self.reference, [0])

    def test_output_is_external_and_never_overwrites_existing_file(self):
        result = dict(status="completed")
        with self.assertRaisesRegex(diagnostic.DiagnosticError, "outside immutable"):
            diagnostic.write_external(
                result, self.study / "diagnostic.json", self.run, self.reference
            )
        output = self.root / "diagnostic.json"
        diagnostic.write_external(result, output, self.run, self.reference)
        with self.assertRaises(FileExistsError):
            diagnostic.write_external(result, output, self.run, self.reference)
        self.assertEqual(json.loads(output.read_text()), result)

    def test_escaping_native_cache_path_is_rejected(self):
        self.rows[0]["native_cache_file"] = "../native-0000.npz"
        self.write_rows()
        with self.assertRaisesRegex(diagnostic.DiagnosticError, "cache basename"):
            diagnostic.diagnose_run(self.run, self.reference, [0])


if __name__ == "__main__":
    unittest.main()
