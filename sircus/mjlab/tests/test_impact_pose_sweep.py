"""CPU-only exact grid, validation-source and repeat-selection checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sircus.mjlab import impact_pose_sweep as sweep
from sircus.mjlab.validation_failure_audit import sha


def case(ident="validation-03-ppo"):
    return dict(id=ident, context_id="validation-03", source_method="ppo",
                source_row_id="ppo-1-validation-1024-validation-03-r0",
                bucket_xy_m=[1.01, -.60], baseline_values=[.62, -.20, .3, 30., .52])


class GridTests(unittest.TestCase):
    def test_only_gap_and_tilt_change_without_clipping(self):
        original = case(); before = copy.deepcopy(original)
        rows = sweep.grid_requests([original])
        self.assertEqual(original, before)
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({r["id"] for r in rows}), 12)
        self.assertEqual(sum(r["baseline"] for r in rows), 1)
        self.assertEqual({r["values"][3] for r in rows}, {26., 30., 34.})
        self.assertEqual({round(r["values"][4], 8) for r in rows}, {.52, .58, .64, .70})
        # The .70 request intentionally exceeds .65; admission must reject it,
        # rather than silently substituting the boundary and hiding this case.
        self.assertGreater(max(r["values"][4] for r in rows), sweep.WIDE_BOUNDS["high"][4])
        self.assertTrue(all(r["values"][:3] == original["baseline_values"][:3]
                            and r["bucket_xy_m"] == original["bucket_xy_m"] for r in rows))

    def test_selection_uses_admitted_observations_and_repeats_exact_winner(self):
        source = case(); rows = sweep.grid_requests([source])
        for i, row in enumerate(rows):
            row.update(native_release=True, route_success=False, route_quality=0,
                       max_retention_s=0., closest_opening_distance_m=.5-i*.01, invalid=None)
        rows[2].update(route_success=True, route_quality=1, max_retention_s=1.)
        rows[3].update(route_success=True, route_quality=2, max_retention_s=1.)
        rows[4].update(native_release=False, route_success=True, route_quality=2,
                       max_retention_s=100., closest_opening_distance_m=0., invalid="bounds")
        selected = sweep.choose([source], rows)
        self.assertEqual(selected[source["id"]]["id"], rows[3]["id"])
        repeats = sweep.confirmation_requests([source], selected)
        self.assertEqual(len(repeats), 3)
        self.assertEqual([r["repeat"] for r in repeats], [0, 1, 2])
        self.assertTrue(all(r["phase"] == "confirmation" and r["values"] == rows[3]["values"]
                            and r["selected_trial_id"] == rows[3]["id"] for r in repeats))

    def test_all_rejected_remains_baseline_rejection_not_fake_success(self):
        source = case(); rows = sweep.grid_requests([source])
        for row in rows: row.update(native_release=False, route_success=False, invalid="unresolved")
        selected = sweep.choose([source], rows)[source["id"]]
        self.assertTrue(selected["baseline"])
        self.assertEqual(sweep.count(rows), dict(attempts=12, native_releases=0, captures=0))
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            sweep.choose([source], rows[:-1])

    def test_counts_charge_every_requested_setup_rejection(self):
        rows = [dict(native_release=True, route_success=True, invalid=None),
                dict(native_release=True, route_success=True, invalid="invalid_physics"),
                dict(native_release=False, route_success=False, invalid="bounds")]
        self.assertEqual(sweep.count(rows), dict(attempts=3, native_releases=2, captures=1))

    def test_trial_release_uses_radius_and_centre_plane_gap(self):
        row = dict(sweep.grid_requests([case()])[0], native_release=False, invalid="bounds")
        view = sweep.trial_view(row, dict(ball=dict(radius_m=.0375)))
        self.assertAlmostEqual(view["release_ball_center_m"][2], .3+.52+.0375)
        self.assertIsNone(view["pose_summary"])
        self.assertFalse(view["success"])
        self.assertEqual(view["category"], "setup_rejected")

    def test_replay_samples_and_explicit_pose_padding(self):
        times = np.arange(257)/15360
        position = np.arange(len(times)*3).reshape(-1, 1, 3).astype(float)
        arrays = dict(time_s=times, position_m=position, pose_sample_valid=np.array([[True], [True], [False]]),
            pose_sample_time_s=np.array([[0.], [times[-1]], [0.]]),
            pose_board_center_m=np.zeros((3, 1, 3)), pose_board_quaternion_wxyz=np.tile([1., 0., 0., 0.], (3, 1, 1)),
            pose_board_normal=np.tile([0., 0., 1.], (3, 1, 1)),
            pose_link_positions10_m=np.zeros((3, 1, 6, 3)), pose_link_positions14_m=np.zeros((3, 1, 6, 3)))
        trail, history = sweep.trajectory_and_pose(arrays, 0, 1.33731)
        self.assertEqual(trail[0], dict(time=0., ball=position[0, 0].tolist()))
        self.assertEqual(trail[-1], dict(time=times[-1], ball=position[-1, 0].tolist()))
        self.assertEqual(len(history), 2)
        self.assertEqual(history[-1]["time"], times[-1])
        self.assertEqual(len(history[0]["link_positions10"]), 7)
        self.assertEqual(history[0]["link_positions14"][0], [1.33731, 0., 0.])


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.method, self.seed = "ppo", 1
        self.prefix = self.root / "runs/ppo-seed1"; (self.prefix / "native").mkdir(parents=True)
        layouts = [dict(id=f"validation-{i:02d}", bucket_xy_m=[.7+i*.01, -.6]) for i in range(12)]
        self.contract = dict(protocol=dict(milestones=[0, 256, 512, 1024]), splits=dict(validation=layouts))
        self.rows = []
        for samples in (0, 256, 512, 1024):
            for layout in layouts:
                for repeat in range(3):
                    self.rows.append(dict(id=f"ppo-1-validation-{samples:04d}-{layout['id']}-r{repeat}",
                        role="validation", method="ppo", seed=1, samples=samples, context_id=layout["id"],
                        repeat=repeat, bucket_xy_m=layout["bucket_xy_m"], values=[.6, -.2, .3, 30., .2],
                        native_release=False, route_success=False))
        self.row = next(r for r in self.rows if r["samples"] == 1024 and r["context_id"] == "validation-03" and r["repeat"] == 0)
        self.native = self.prefix / "native/native-0000.npz"
        np.savez_compressed(self.native, record_ids=np.array([self.row["id"]]))
        self.row.update(native_release=True, native_cache_file=self.native.name, native_cache_row=0, native_cache_sha256=sha(self.native))
        self.row.update(q10=[0.]*6, q14=[0.]*6, native_initial_q10=[0.]*6, native_initial_q14=[0.]*6,
            board_center_m=[.6, -.2, .3], board_normal=[0., 0., 1.], board_quaternion_wxyz=[1., 0., 0., 0.],
            native_initial_ball_center_m=[.6, -.2, .5375], release_ball_center_m=[.6, -.2, .5375])
        self.item = dict(context_id=self.row["context_id"], method="ppo", seed=1, samples=1024,
            source_row_id=self.row["id"], bucket_xy_m=self.row["bucket_xy_m"], primary_selected=True,
            recorded_replay_available=True, category="no_whole_ball_entry", miss_region=dict(region="short"))
        self.diagnostic = dict(representatives=[self.item])
        self.seal()

    def tearDown(self): self.temp.cleanup()

    def seal(self):
        # Poison non-validation metadata must never cause native trace access.
        mixed = [dict(id="poison-test", role="test", native_cache_file="forbidden-test.npz"), *self.rows]
        (self.prefix / "rows.jsonl").write_text("".join(json.dumps(r)+"\n" for r in mixed))
        (self.prefix / "selected.json").write_text(json.dumps(dict(selected_samples=1024, frozen_before_test=True)))
        self.manifest = {str(p.relative_to(self.root)): sha(p) for p in self.root.rglob("*") if p.is_file()}

    def execute(self):
        return sweep.source_cases(self.diagnostic, self.root, self.contract, self.manifest)

    def test_source_is_selected_short_validation_only(self):
        result = self.execute()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["source_row_id"], self.row["id"])
        self.assertEqual(result[0]["source_native_sha256"], sha(self.native))
        self.assertEqual(result[0]["baseline_values"], self.row["values"])
        self.assertNotIn("test", result[0]["source_row_id"])

    def test_nonselected_or_unknown_trajectory_cannot_seed_sweep(self):
        for change in (dict(primary_selected=False), dict(miss_region=dict(region="unknown")),
                       dict(category="captured_clean"), dict(recorded_replay_available=False)):
            self.diagnostic = dict(representatives=[dict(self.item, **change)])
            with self.assertRaisesRegex(ValueError, "No source-bound"):
                self.execute()

    def test_native_identity_or_bytes_change_fails_closed(self):
        np.savez_compressed(self.native, record_ids=np.array(["ppo-1-test-0000-forbidden-r0"]))
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.execute()
        self.row["native_cache_sha256"] = sha(self.native); self.seal()
        with self.assertRaisesRegex(ValueError, "Only validation"):
            self.execute()

    def test_row_cannot_claim_an_unselected_checkpoint(self):
        (self.prefix / "selected.json").write_text(json.dumps(dict(selected_samples=256, frozen_before_test=True)))
        self.manifest[str((self.prefix / "selected.json").relative_to(self.root))] = sha(self.prefix / "selected.json")
        with self.assertRaisesRegex(ValueError, "not the selected"):
            self.execute()


if __name__ == "__main__": unittest.main()
