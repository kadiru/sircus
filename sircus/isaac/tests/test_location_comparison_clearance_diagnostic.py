"""Synthetic full-rate arithmetic cases; never launch a simulator or load weights."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
import location_comparison_clearance_diagnostic as diagnostic
import continuous_native_audit as native
from board_route import BoardRoute
from entry_quality import EntryQuality
from rl_contract import ShotScore

HELPERS = SimpleNamespace(ShotScore=ShotScore, EntryQuality=EntryQuality, BoardRoute=BoardRoute, same=native.same)


class ClearanceDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.scene = json.loads((SOURCE / "scene.json").read_text())
        self.origins = np.zeros((35, 3), dtype=np.float32)
        self.cell = dict(id="target", target_id="target", request_id="monitor:parent:0:target",
                         bucket_xy_m=[.67, -.42], physical_slot=0, split="validation", cohort="monitor")
        self.reserve = dict(batch_id="batch-00000", model="parent", phase="monitor", milestone=0,
                            cells=[self.cell])
        cells = [dict(self.cell, id=self.cell["request_id"])]
        cells += [dict(self.cell, id=f"inactive-0-{i}", request_id=f"inactive-0-{i}", target_id="inactive")
                  for i in range(1, 35)]
        self.packet = dict(cells=cells, enabled=[True] + [False] * 34, origins_m=self.origins.tolist(),
            raw_rows=[dict(cell_id=cell["id"], invalid=None if i == 0 else "inactive",
                           settled_board_normal=[0., 0., 1.]) for i, cell in enumerate(cells)],
            dt_s=1 / 240, release_step=10, score_end_step=12,
            physics=dict(file="trial-00000.npz", samples=2, sha256="a" * 64),
            proposal=dict(checkpoint=dict(sha256="b" * 64)), rewards=[-.7] * 35)
        self.data = self.arrays(2)
        self.data["position_m"][0, :, 2] = .3
        self.data["position_m"][1, :, 2] = -.3
        self.make_recorded()

    def arrays(self, count):
        return dict(time_step=np.arange(11, 11 + count),
            position_m=np.repeat(self.origins[None], count, axis=0),
            velocity_m_s=np.zeros((count, 35, 3), dtype=np.float32),
            forces_N=np.zeros((count, 35, 6), dtype=np.float32),
            joint_position_rad=np.zeros((count, 35, 2, 6), dtype=np.float32),
            joint_velocity_rad_s=np.zeros((count, 35, 2, 6), dtype=np.float32),
            board_position_m=np.zeros((count, 35, 3), dtype=np.float32),
            board_quaternion_wxyz=np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (count, 35, 1)))

    def make_recorded(self, path="auditor_float64"):
        paths = diagnostic.local_coordinates(self.data["position_m"], self.origins)[0] if self.data is not None else dict.fromkeys(diagnostic.PATHS)
        for i, row in enumerate(self.packet["raw_rows"]):
            result = diagnostic.replay_cell(self.scene, self.packet["cells"][i], row, self.data, paths[path], i, HELPERS)
            row.update(result["fields"])
        raw = self.packet["raw_rows"][0]
        label = diagnostic.labels(raw, bool(raw["invalid"]))
        reward = -1. if raw["invalid"] else 1.6 if label["clean"] else 1.5 if label["outcome"] == "capture" else -.7
        self.packet["rewards"][0] = reward
        normalized = [dict(request_id=self.cell["request_id"], target_id=self.cell["target_id"],
            **label, rejection_reason=str(raw["invalid"]) if raw["invalid"] else None, reward=reward)]
        self.packet["normalized_results"] = normalized
        self.terminal = dict(rows=copy.deepcopy(normalized))

    def run_core(self):
        return diagnostic.diagnose_packet(self.packet, self.reserve, self.terminal, self.scene, self.data, HELPERS)

    def capture(self):
        count = 253
        self.data = self.arrays(count)
        self.data["position_m"][:, 0, :2] = self.cell["bucket_xy_m"]
        self.data["position_m"][:2, 0, 2] = .3
        self.data["position_m"][2, 0, 2] = .18
        self.data["position_m"][3:, 0, 2] = .06
        self.data["forces_N"][0, 0, 0] = 1
        self.data["velocity_m_s"][1, 0, 2] = 1
        self.data["forces_N"][243:, 0, 2] = 20
        self.packet["score_end_step"] = 10 + count
        self.packet["physics"]["samples"] = count
        self.make_recorded()

    def test_paid_only_miss_and_original_unchanged_field_comparator(self):
        records = self.run_core()
        self.assertEqual(len(records), 1)
        for result in records[0]["paths"].values():
            self.assertEqual(result["outcome"], "miss")
            self.assertEqual(result["consumed_steps"], 2)
            self.assertTrue(result["freeze_rule_satisfied"])
        self.assertFalse(records[0]["paths"]["auditor_float64"]["mismatches"])
        self.assertEqual(diagnostic.TOLERANCE, 1e-7)
        self.assertIs(HELPERS.same, native.same)

    def test_capture_replays_freeze_before_later_external_contact(self):
        self.capture()
        row = self.run_core()[0]
        for result in row["paths"].values():
            self.assertEqual(result["outcome"], "capture")
            self.assertTrue(result["clean"])
            self.assertEqual(result["consumed_steps"], 243)
            self.assertEqual(result["freeze_reason"], "capture")
        a = row["paths"]["auditor_float64"]
        self.assertFalse(a["mismatches"])
        self.assertEqual(a["minimum_clearance_step"]["sample_index"], 2)
        self.assertEqual(a["minimum_clearance_step"]["physics_step"], 13)

    def test_all_returned_field_differences_include_contact_map(self):
        self.packet["raw_rows"][0]["minimum_mouth_clearance_mm"] = 9.
        self.packet["raw_rows"][0]["closest_opening_distance_m"] += 1.
        self.packet["raw_rows"][0]["contact_peaks_N"]["robot"] = 20.
        fields = {d["field"] for d in self.run_core()[0]["paths"]["auditor_float64"]["mismatches"]}
        self.assertEqual(fields, {"minimum_mouth_clearance_mm", "closest_opening_distance_m", "contact_peaks_N"})

    def test_original_absolute_tolerance_is_not_clean_margin_or_scaled(self):
        recorded = dict(minimum_mouth_clearance_mm=1.0)
        self.assertFalse(diagnostic.mismatches(recorded, dict(minimum_mouth_clearance_mm=1.0 + .5e-7), HELPERS))
        self.assertEqual(len(diagnostic.mismatches(recorded, dict(minimum_mouth_clearance_mm=1.0 + 2e-7), HELPERS)), 1)

    def test_capture_and_clean_flips_reported_separately(self):
        self.capture()
        raw = self.packet["raw_rows"][0]
        raw["route_success"] = False
        raw["route_quality"] = 0
        self.packet["normalized_results"][0].update(outcome="miss", clean=False, reward=-.7)
        self.packet["rewards"][0] = -.7
        self.terminal = dict(rows=copy.deepcopy(self.packet["normalized_results"]))
        record = self.run_core()[0]
        self.assertTrue(record["paths"]["auditor_float64"]["capture_flip"])
        self.assertTrue(record["paths"]["auditor_float64"]["clean_flip"])
        summary = diagnostic.summarize([record])
        self.assertEqual(summary["overall"]["paths"]["auditor_float64"]["capture_flips"], 1)

    def test_coordinate_identity_is_distinct_from_downstream_dtype(self):
        self.capture()
        record = self.run_core()[0]
        self.assertTrue(record["coordinate_comparison"]["coordinates_identical"])
        self.assertFalse(record["path_differences"]["auditor_vs_float32_then_float64"])
        # NumPy 2 changes some scalar arithmetic, but this test does not assume its effect on labels.
        if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
            self.assertTrue(record["path_differences"]["float32_vs_float32_then_float64"])

    def test_actual_coordinate_rounding_measured_exactly(self):
        world = np.zeros((1, 1, 3), dtype=np.float32)
        world[0, 0, 0] = 1e8
        paths, info = diagnostic.local_coordinates(world, [[1., 0., 0.]])
        self.assertFalse(info["coordinates_identical"])
        self.assertEqual(info["differing_components"], 1)
        self.assertEqual(info["maximum_absolute_difference_m"], 1.)
        self.assertEqual(paths["producer_float32_then_float64"].dtype, np.float64)

    def test_nonroundtrippable_origin_and_non_float32_world_rejected(self):
        with self.assertRaisesRegex(ValueError, "round-trip"):
            diagnostic.local_coordinates(np.zeros((1, 1, 3), np.float32), [[.1, 0., 0.]])
        with self.assertRaisesRegex(ValueError, "must be float32"):
            diagnostic.local_coordinates(np.zeros((1, 1, 3), np.float64), [[0., 0., 0.]])

    def test_packet_nonroundtrippable_origin_rejected_even_without_physics(self):
        self.packet["raw_rows"][0]["invalid"] = "unreachable"
        self.packet["physics"] = None
        self.packet["score_end_step"] = self.packet["release_step"]
        self.data = None
        self.make_recorded()
        self.packet["origins_m"][0][0] = .1
        with self.assertRaisesRegex(ValueError, "round-trip"):
            self.run_core()

    def test_rejected_no_physics_retains_rejection_and_empty_scorer_fields(self):
        self.packet["raw_rows"][0]["invalid"] = "unreachable"
        self.packet["physics"] = None
        self.packet["score_end_step"] = self.packet["release_step"]
        self.data = None
        self.make_recorded()
        row = self.run_core()[0]
        for result in row["paths"].values():
            self.assertEqual(result["outcome"], "rejected")
            self.assertEqual(result["consumed_steps"], 0)
            self.assertFalse(result["mismatches"])

    def test_missing_clock_step_rejected(self):
        self.data["time_step"][1] += 1
        with self.assertRaisesRegex(ValueError, "clock coverage"):
            self.run_core()

    def test_nonfinite_and_negative_physics_rejected(self):
        self.data["position_m"][0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Malformed physics"):
            self.run_core()
        self.data["position_m"][0, 0, 0] = 0
        self.data["forces_N"][0, 0, 0] = -1
        with self.assertRaisesRegex(ValueError, "Negative force"):
            self.run_core()

    def test_padded_count_and_mask_are_not_paid_requests(self):
        self.packet["enabled"][1] = True
        with self.assertRaisesRegex(ValueError, "Paid mask"):
            self.run_core()

    def test_request_and_target_joins_fail_closed(self):
        self.packet["cells"][0]["request_id"] = "other"
        with self.assertRaisesRegex(ValueError, "request identity"):
            self.run_core()

    def test_saved_label_join_not_repaired_in_place(self):
        self.terminal["rows"][0]["outcome"] = "capture"
        with self.assertRaisesRegex(ValueError, "normalized join"):
            self.run_core()

    def test_recording_ending_before_freeze_is_reported_not_silently_passed(self):
        self.data["position_m"][1, 0, 2] = .1
        self.make_recorded()
        row = self.run_core()[0]
        self.assertFalse(row["paths"]["auditor_float64"]["freeze_rule_satisfied"])
        summary = diagnostic.summarize([row])
        self.assertEqual(summary["overall"]["paths"]["auditor_float64"]["incomplete_freeze_rules"], 1)

    def test_exact_960_horizon_is_valid(self):
        self.data = self.arrays(960)
        self.data["position_m"][:, :, 2] = .3
        self.packet["score_end_step"] = 970
        self.packet["physics"]["samples"] = 960
        self.make_recorded()
        row = self.run_core()[0]
        self.assertEqual(row["paths"]["auditor_float64"]["freeze_reason"], "full_960_step_horizon")

    def test_final_cohorts_remain_separate_in_summary(self):
        row = self.run_core()[0]
        broad, nearby = copy.deepcopy(row), copy.deepcopy(row)
        broad["phase"], nearby["phase"] = "test_broad", "test_local"
        result = diagnostic.summarize([broad, nearby])
        self.assertEqual(result["final_tests"]["parent"]["test_broad"]["requests"], 1)
        self.assertEqual(result["final_tests"]["parent"]["test_local"]["requests"], 1)
        self.assertEqual(len(result["grouped"]), 2)
        self.assertEqual(result["grouped"][0]["checkpoint_sha256"], "b" * 64)

    def test_quaternion_check_preserves_zero_relative_tolerance(self):
        self.data["board_quaternion_wxyz"][0, 0, 0] = 1.001005
        with self.assertRaisesRegex(ValueError, "quaternion"):
            self.run_core()

    def test_native_version_gate_and_explicit_cross_version_exception(self):
        with mock.patch.object(np, "__version__", "2.5.4"):
            with self.assertRaisesRegex(ValueError, "exact NumPy 1.26.0"):
                diagnostic.runtime_info()
            report = diagnostic.runtime_info(cross_version_diagnostic=True)
            self.assertFalse(report["runtime_matches_producer"])
            self.assertFalse(report["producer_arithmetic_parity_claimed"])
            self.assertTrue(diagnostic.runtime_info(synthetic=True)["synthetic"])
        with mock.patch.object(np, "__version__", "1.26.4"):
            with self.assertRaisesRegex(ValueError, "exact NumPy 1.26.0"):
                diagnostic.runtime_info()
        with mock.patch.object(np, "__version__", "1.26.0"):
            self.assertTrue(diagnostic.runtime_info()["numpy_matches_producer"])
            with mock.patch.object(diagnostic.platform, "system", return_value="Linux"):
                self.assertFalse(diagnostic.runtime_info()["runtime_matches_producer"])
                self.assertFalse(diagnostic.runtime_info()["full_producer_runtime_verified"])

    def test_json_duplicate_and_nonfinite_rejected(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}'):
            with self.assertRaises(ValueError):
                diagnostic.read_json_bytes(raw)

    def test_sealed_bytes_and_before_after_checks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "data.json"
            path.write_bytes(b"{}")
            receipt = dict(bytes=2, sha256=hashlib.sha256(b"{}").hexdigest())
            inventory = dict(files={path.name: receipt}, file_count=1, total_bytes=2)
            files = diagnostic.SealedFiles(root, inventory)
            self.assertEqual(files.read(path.name, receipt["sha256"]), b"{}")
            result = files.finish()
            self.assertFalse(result["complete_inventory_verified"])
            self.assertTrue(result["before_after_verified"])
            path.write_bytes(b"[]")
            with self.assertRaisesRegex(ValueError, "changed during"):
                files.finish()
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                files.read(path.name)

    def test_unsafe_paths_and_links_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "target").write_text("{}")
            (root / "linked").symlink_to(root / "target")
            for name in ("../target", str(root / "target"), "linked"):
                with self.assertRaises(ValueError):
                    diagnostic.safe_file(root, name)

    def test_source_hash_pin_checked_before_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rl_contract.py"
            path.write_text("raise RuntimeError('must not import')\n")
            with self.assertRaisesRegex(ValueError, "source SHA changed"):
                diagnostic.load_original_helpers(Path(temporary), {path.name: "a" * 64})


if __name__ == "__main__":
    unittest.main()
