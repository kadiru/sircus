"""Meaningful checks on split isolation, attempt denominators and read-only media."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import continuous_pilot_report as report
import continuous_pilot_server as server


def outcome(name, captured=False, invalid=None):
    return dict(cell_id=name, invalid=invalid, route_success=captured,
                board_bounce=captured, whole_ball_entry=captured,
                max_retention_s=1. if captured else 0., external_contact_steps=0,
                route_quality=2 if captured else 0, single_bounce_success=captured)


class ReportTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.run = self.root/"raw"
        (self.run/"trials").mkdir(parents=True)
        self.cells = [dict(id="a", split="train", bucket_xy_m=[.4, -.3]),
                      dict(id="b", split="test", bucket_xy_m=[.8, -.6])]
        self.write("study.json", dict(cells=self.cells, fingerprint="study", domain={}))
        self.write("feasibility.json", dict(rows=[
            dict(cell_id="a", valid=True, reason=None, action=dict(radius_m=.3)),
            dict(cell_id="a", valid=False, reason="ik", action=dict(radius_m=.4)),
            dict(cell_id="b", valid=False, reason="ik", action=dict(radius_m=.5))]))

    def write(self, name, value):
        (self.run/name).write_text(json.dumps(value))

    def training(self, batch, captured=False, invalid=None):
        return dict(mode="train", batch=batch, update_count=batch//2,
                    active_cell_ids=["a"], rows=[outcome("a", captured, invalid), outcome("b")],
                    actions=[dict(radius_m=.31), dict(radius_m=.5)], rewards=[1., 0.])

    def test_counts_keep_rejected_attempts_and_exclude_inactive_cells(self):
        self.write("trials/000.json", self.training(0, True))
        self.write("trials/001.json", self.training(1, invalid="settling"))
        value = report.summarize(self.run)
        self.assertEqual(value["training"]["attempts"], 2)
        self.assertEqual(value["training"]["captures"], 1)
        self.assertEqual(value["training"]["invalid"], 1)
        self.assertEqual(value["learning_curve"][-1]["rolling_capture_rate"], .5)
        self.assertEqual(value["learning_curve"][0]["mean_reward"], 1.)
        self.assertEqual(value["cells"][1]["training"]["attempts"], 0)
        self.assertEqual(value["feasibility"]["attempts"], 3)
        self.assertEqual(len(value["cells"][0]["feasibility"]["poses"]), 1)
        self.assertIn("no native capture label", value["cells"][0]["feasibility"]["poses"][0]["qualification"])
        self.assertEqual(value["sampled_training_action_ranges"]["radius_m"]["max"], .31)

    def test_heldout_training_and_ambiguous_active_rows_are_rejected(self):
        trial = self.training(0)
        trial["active_cell_ids"] = ["b"]
        self.write("trials/000.json", trial)
        with self.assertRaisesRegex(ValueError, "Held-out"):
            report.summarize(self.run)
        trial = self.training(0)
        trial.pop("active_cell_ids")
        self.write("trials/000.json", trial)
        with self.assertRaisesRegex(ValueError, "active-cell"):
            report.summarize(self.run)

    def test_frozen_checks_remain_separate_by_split_and_policy_stage(self):
        for i, captured in enumerate((False, True)):
            self.write(f"trials/eval-{i}.json", dict(mode="eval", update_count=i,
                policy_hash=f"frozen-{i}", active_cell_ids=["b"],
                rows=[outcome("a"), outcome("b", captured)], optimizer_samples=0))
        value = report.summarize(self.run)
        self.assertEqual(value["training"]["attempts"], 0)
        self.assertEqual([item["capture_rate"] for item in value["assessments"]], [0., 1.])
        self.assertTrue(all(item["split"] == "test" for item in value["assessments"]))
        self.assertEqual(len(value["cells"][1]["evaluation"]), 2)

    def test_geometry_coverage_is_not_optimizer_or_frozen_policy_performance(self):
        self.write("trials/coverage.json", dict(mode="coverage", batch=0,
            update_count=0, active_cell_ids=["a", "b"],
            rows=[outcome("a", True), outcome("b")], optimizer_samples=0))
        self.write("trials/train.json", self.training(0, invalid="ik"))
        value = report.summarize(self.run)
        self.assertEqual(value["coverage"]["attempts"], 2)
        self.assertEqual(value["coverage"]["captures"], 1)
        self.assertEqual(value["training"]["attempts"], 1)
        self.assertEqual(value["training"]["captures"], 0)
        self.assertEqual(value["assessments"], [])
        self.assertEqual(value["cells"][0]["coverage"]["captures"], 1)

    def test_repeated_deterministic_checks_do_not_claim_independent_layout_confidence(self):
        for i in range(3):
            self.write(f"trials/eval-{i}.json", dict(mode="eval", batch=i, update_count=0,
                policy_hash="same-weights", active_cell_ids=["b"], deterministic=True,
                rows=[outcome("b", True)], optimizer_samples=0))
        result = report.summarize(self.run)["assessments"][0]
        self.assertEqual(result["attempts"], 3)
        self.assertEqual(result["captures"], 3)
        self.assertEqual(result["unique_layouts"], 1)
        self.assertEqual(result["layouts_with_capture"], 1)
        self.assertIsNone(result["wilson95"])

    def test_eval_optimizer_feedback_and_multiple_weights_at_same_stage_rejected(self):
        trial = dict(mode="eval", update_count=0, policy_hash="weights-a",
                     active_cell_ids=["b"], rows=[outcome("b")], optimizer_samples=1)
        self.write("trials/eval-0.json", trial)
        with self.assertRaisesRegex(ValueError, "optimizer"):
            report.summarize(self.run)
        trial["optimizer_samples"] = 0
        self.write("trials/eval-0.json", trial)
        trial["policy_hash"] = "weights-b"
        self.write("trials/eval-1.json", trial)
        with self.assertRaisesRegex(ValueError, "multiple policy"):
            report.summarize(self.run)

    def test_capture_requires_whole_ball_retention_and_board_route(self):
        good = outcome("a", True)
        self.assertEqual(report.counts([good])["captures"], 1)
        for key, value in (("whole_ball_entry", False), ("board_bounce", False),
                           ("external_contact_steps", 1), ("max_retention_s", .99),
                           ("invalid", "ik")):
            bad = dict(good, **{key: value})
            with self.assertRaisesRegex(ValueError, "scoring contract"):
                report.counts([bad])

    def test_pending_cells_and_unsampled_are_not_zero_performance(self):
        value = report.summarize(self.run)
        self.assertIsNone(value["training"]["capture_rate"])
        self.assertIsNone(value["training"]["wilson95"])
        self.assertEqual(value["assessments"], [])
        self.assertEqual(value["cells"][0]["evaluation"], [])

    def test_export_does_not_overwrite_or_modify_raw_data(self):
        before = (self.run/"study.json").read_bytes()
        destination = self.root/"export"
        result = report.export(self.run, destination)
        self.assertEqual((self.run/"study.json").read_bytes(), before)
        self.assertEqual(result["schema"], report.SCHEMA)
        with self.assertRaisesRegex(ValueError, "new directory"):
            report.export(self.run, destination)
        with self.assertRaisesRegex(ValueError, "outside raw"):
            report.export(self.run, self.run/"export")

    def test_paths_and_symlinks_cannot_escape_evidence(self):
        (self.run/"alias.json").symlink_to(self.run/"study.json")
        for filename in ("../study.json", "/etc/passwd", "alias.json", "missing.json"):
            with self.assertRaises(ValueError):
                report.local_file(self.run, filename)

    def native_clips(self):
        self.write("selected-camera.json", dict(eye_m=[10., -6., 2.5],
            target_m=[7.8, -3.5, .4], up_axis="Z", resolution=[1920, 1080]))
        (self.run/"frames").mkdir(exist_ok=True)
        frames = []
        for index, phase in enumerate(("setup", "setup", "release")):
            # Byte fixtures exercise provenance logic, not image decode claims.
            (self.run/f"frames/{index:05d}.jpg").write_bytes(f"frame {index}".encode())
            joints = {name: [[.1*index]*6 for _ in range(35)] for name in ("Arm10", "Arm14")}
            frames.append(dict(frame=index, physics_step=100+index*8, phase=phase,
                               eye_m=[10., -6., 2.5], target_m=[7.8, -3.5, .4], all_joints_rad=joints))
        return [dict(frames=frames, fps=30)]

    def test_native_movie_binds_fixed_camera_steps_unique_frames_and_actual_movement(self):
        clips = self.native_clips()
        audit = report.native_frame_manifest(self.run, clips)[0]
        self.assertEqual(audit["frames"], 3)
        self.assertEqual(audit["setup_frames"], 2)
        self.assertEqual(audit["release_frames"], 1)
        self.assertAlmostEqual(audit["foreground_joint_excursion_rad"]["Arm14"], .2)
        self.assertEqual(len(audit["native_frame_sha256"]), 3)
        bad = copy.deepcopy(clips)
        bad[0]["frames"][1]["eye_m"][0] += .01
        with self.assertRaisesRegex(ValueError, "camera moved"):
            report.native_frame_manifest(self.run, bad)
        bad = copy.deepcopy(clips)
        bad[0]["frames"][1]["frame"] = 0
        with self.assertRaisesRegex(ValueError, "duplicated"):
            report.native_frame_manifest(self.run, bad)
        bad = copy.deepcopy(clips)
        bad[0]["frames"][1]["physics_step"] = 100
        with self.assertRaisesRegex(ValueError, "physics steps"):
            report.native_frame_manifest(self.run, bad)
        bad = copy.deepcopy(clips)
        bad[0]["frames"][1].pop("all_joints_rad")
        with self.assertRaisesRegex(ValueError, "readbacks"):
            report.native_frame_manifest(self.run, bad)

    def test_action_edit_preserves_motion_context_full_release_and_failed_trial(self):
        frames = []
        for i in range(100):
            q = 0. if i < 40 else .3
            frames.append(dict(phase="setup" if i < 90 else "release",
                all_joints_rad={name: [[q]*6 for _ in range(35)] for name in ("Arm10", "Arm14")},
                all_ball_world_m=[[0., 0., 0.] for _ in range(35)]))
        indices = report.action_frame_indices(frames, context_frames=2)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 99)
        self.assertTrue(set(range(37, 43)).issubset(indices))
        self.assertTrue(set(range(90, 100)).issubset(indices))
        self.assertNotIn(20, indices)
        stationary = copy.deepcopy(frames)
        for frame in stationary:
            frame["phase"] = "setup"
            frame["all_joints_rad"] = {name: [[0.]*6 for _ in range(35)] for name in ("Arm10", "Arm14")}
        indices = report.action_frame_indices(stationary, context_frames=2)
        self.assertEqual(indices, [0, 1, 2, 97, 98, 99])
        self.assertEqual(report.contiguous_spans(indices), [[0, 2], [97, 99]])
        with self.assertRaisesRegex(ValueError, "chronological"):
            report.contiguous_spans([0, 0])

    def test_action_edit_keeps_slow_drift_and_is_conservative_when_ball_data_missing(self):
        frames = [dict(phase="setup", all_joints_rad={name: [[i*.00009]*6 for _ in range(35)]
            for name in ("Arm10", "Arm14")}, all_ball_world_m=[[0., 0., 0.] for _ in range(35)]) for i in range(60)]
        indices = report.action_frame_indices(frames, context_frames=2)
        self.assertIn(30, indices)
        for frame in frames:
            frame.pop("all_ball_world_m")
        self.assertEqual(report.action_frame_indices(frames), list(range(60)))


class ReadOnlyViewerTests(unittest.TestCase):
    def test_resource_allowlist_excludes_commands_and_arbitrary_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/"report.json").write_text("{}")
            (root/"trial-001.mp4").write_bytes(b"fixture")
            self.assertEqual(server.resource(root, "/report.json")[1], "application/json")
            self.assertEqual(server.resource(root, "/trial-001.mp4")[1], "video/mp4")
            for path in ("/../report.json", "/run", "/stop", "/study.json", "/etc/passwd", "/trial-1.mp4"):
                self.assertIsNone(server.resource(root, path))
            (root/"trials.mp4").symlink_to(root/"trial-001.mp4")
            self.assertIsNone(server.resource(root, "/trials.mp4"))

    def test_ranges_enable_native_video_seeking(self):
        self.assertEqual(server.byte_range(None, 100), (0, 99, False))
        self.assertEqual(server.byte_range("bytes=10-20", 100), (10, 20, True))
        self.assertEqual(server.byte_range("bytes=80-", 100), (80, 99, True))
        self.assertEqual(server.byte_range("bytes=-20", 100), (80, 99, True))
        for value in ("bytes=100-", "bytes=30-20", "bytes=0-1,3-4", "bytes=-0", "bytes=-"):
            with self.assertRaises(ValueError):
                server.byte_range(value, 100)


if __name__ == "__main__":
    unittest.main()
