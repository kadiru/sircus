"""The parity benchmark keeps the existing complete-shot definition in both engines."""

import hashlib
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ISAAC = Path(__file__).resolve().parents[2] / "isaac"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contract import (
    ACTION_KEYS,
    FORCE_NAMES,
    ScoreStream,
    canonical,
    select_benchmark,
    verify_manifest,
)
from compare import build_report


def source_fixture(folder):
    scene = json.loads((ISAAC / "scene.json").read_text())
    spec = dict(scene=scene)
    spec["fingerprint"] = hashlib.sha256(canonical(spec)).hexdigest()
    (folder / "contract.json").write_text(json.dumps(spec))
    (folder / "result.json").write_text(
        json.dumps(dict(fingerprint=spec["fingerprint"]))
    )
    rows = []
    for i in range(27):
        kind = i % 3
        row = dict(
            id="fixture",
            azimuth_offset_deg=i,
            azimuth_deg=i,
            board_x_m=i / 100,
            board_y_m=-0.1,
            height_m=0.32,
            gap_m=0.24,
            tilt_deg=26,
            invalid=None,
            cell_id=f"fixture-{i}",
            bucket_xy_m=[0.67, -0.5],
            release_ball_center_m=[i / 100, -0.1, 0.5975],
            settled_board_center_m=[i / 100, -0.1, 0.32],
            settled_board_normal=[0, 0, 1],
            final_joint_positions_rad=dict(Arm10=[0] * 8, Arm14=[0] * 6),
            route_success=kind == 0,
            route_quality=2 if kind == 0 else 0,
            board_bounce=True,
            rim_band_contact=kind == 1,
        )
        rows.append(row)
    (folder / "clip-00.json").write_text(json.dumps(dict(rollout=rows)))
    return rows


class SharedScoreTests(unittest.TestCase):
    def setUp(self):
        self.scene = json.loads((ISAAC / "scene.json").read_text())
        self.bucket_xy = [0.67, -0.5]
        self.stream = ScoreStream(self.scene, self.bucket_xy, [0, 0, 1])
        self.zero = dict.fromkeys(FORCE_NAMES, 0.0)
        b = self.scene["bucket"]
        self.top = b["assumed_interior_depth_m"] + b["assumed_bottom_thickness_m"]

    def tick(self, pos, v=(0, 0, 0), **forces):
        self.stream.step(pos, v, dict(self.zero, **forces))

    def bounce(self):
        self.tick([0.67, -0.1, 0.36], (0, 0, -1), board=100)
        self.tick([0.67, -0.1, 0.37], (0, 0, 1))

    def mouth(self, **forces):
        self.tick([*self.bucket_xy, self.top + 0.02], (0, 0, -1), **forces)

    def inside(self, count):
        for _ in range(count):
            self.tick([*self.bucket_xy, 0.08])

    def test_board_rebound_and_one_uninterrupted_second_are_required(self):
        self.bounce()
        self.mouth()
        self.inside(239)
        self.assertFalse(self.stream.result()["complete_success"])
        self.inside(1)
        result = self.stream.result()
        self.assertTrue(result["whole_ball_entry"] and result["complete_success"])
        self.assertEqual(result["route_quality"], 2)
        self.assertEqual(result["board_contact_episodes"], 1)

    def test_rim_assistance_counts_with_lower_entry_quality(self):
        self.bounce()
        self.mouth(bucket=20)
        self.inside(240)
        result = self.stream.result()
        self.assertTrue(result["complete_success"])
        self.assertEqual(result["route_quality"], 1)
        self.assertTrue(result["rim_band_contact"])

    def test_partial_exit_resets_continuous_retention(self):
        self.bounce()
        self.mouth()
        self.inside(120)
        self.tick([*self.bucket_xy, self.top - 0.02])
        self.inside(120)
        self.assertFalse(self.stream.result()["complete_success"])
        self.assertAlmostEqual(self.stream.result()["max_retention_s"], 0.5)
        self.inside(120)
        self.assertTrue(self.stream.result()["complete_success"])

    def test_external_contact_disqualifies_an_otherwise_captured_route(self):
        self.bounce()
        self.mouth()
        self.tick([*self.bucket_xy, 0.08], table=10)
        self.inside(240)
        result = self.stream.result()
        self.assertTrue(result["success"])
        self.assertFalse(result["complete_success"])
        self.assertEqual(result["external_contact_steps"], 1)

    def test_ball_in_bucket_without_a_board_rebound_does_not_score(self):
        self.mouth()
        self.inside(300)
        self.assertFalse(self.stream.result()["complete_success"])

    def test_scoring_freezes_at_the_original_terminal_boundary(self):
        self.bounce()
        self.mouth()
        self.inside(240)
        scored = self.stream.result()
        self.tick([0, 0, -0.75], floor=100)
        self.assertEqual(scored, self.stream.result())
        self.assertTrue(self.stream.finished)

    def test_force_and_cadence_inputs_cannot_silently_change(self):
        with self.assertRaises(ValueError):
            self.stream.step([0, 0, 0], [0, 0, 0], dict(board=1))
        with self.assertRaises(ValueError):
            self.tick([0, 0, 0], floor=-1)
        scene = dict(self.scene, simulation=dict(physics_dt_s=1 / 120))
        with self.assertRaises(ValueError):
            ScoreStream(scene, self.bucket_xy, [0, 0, 1])


class BenchmarkTests(unittest.TestCase):
    def test_selection_freezes_unique_stratified_original_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source_fixture(folder)
            first = select_benchmark(folder)
            second = select_benchmark(folder)
            self.assertEqual(first, second)
            self.assertEqual(
                Counter(s["category"] for s in first["shots"]),
                dict(clean_capture=8, rim_contact_miss=8, other_miss=8),
            )
            self.assertEqual(
                len({s["action"]["board_x_m"] for s in first["shots"]}), 24
            )
            verify_manifest(first)
            first["shots"][0]["action"]["height_m"] += 0.01
            with self.assertRaises(ValueError):
                verify_manifest(first)

    def test_fresh_pairs_reject_missing_results_and_exclude_failed_initialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source_fixture(folder)
            manifest = select_benchmark(folder)
            rows = [
                dict(s["isaac_reference"], shot_id=s["shot_id"], repeat=r)
                for s in manifest["shots"]
                for r in range(3)
            ]
            first = dict(
                benchmark_fingerprint=manifest["fingerprint"],
                engine="first",
                shots=rows,
            )
            second = dict(
                benchmark_fingerprint=manifest["fingerprint"],
                engine="second",
                shots=[dict(r) for r in rows],
            )
            second["shots"][0]["invalid"] = "pose_gate"
            report = build_report(manifest, first, second)
            self.assertEqual(report["comparable_pairs"], 71)
            self.assertEqual(report["capture_agreements"], 71)
            self.assertEqual(report["second"]["initialization_failures"], 1)
            self.assertEqual(
                report["comparison"], "fresh_paired_fixed_endpoint_contact_replay"
            )
            second["physics_substeps"] = 8
            with self.assertRaisesRegex(ValueError, "same physics substeps"):
                build_report(manifest, first, second)
            second.pop("physics_substeps")
            second["shots"].pop()
            with self.assertRaises(ValueError):
                build_report(manifest, first, second)
            history = build_report(manifest, first)
            self.assertEqual(history["comparison"], "historical_reference")
            self.assertIn("historical", history["limitations"][-1])


if __name__ == "__main__":
    unittest.main()
