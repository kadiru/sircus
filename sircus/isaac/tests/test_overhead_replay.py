import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from overhead_contract import cameras, selection
from overhead_video import project, validate

ROOT = Path(__file__).resolve().parents[3]


class OverheadReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(
            (ROOT / "baselines/isaac-parallel-world-20261101/summary.json").read_text()
        )
        cls.cfg = json.loads((ROOT / "sircus/isaac/scene.json").read_text())

    def test_frozen_selection_keeps_four_distinct_targets(self):
        indices, actions = selection(self.report, self.cfg)
        self.assertEqual(indices, [0, 2, 3, 6])
        self.assertEqual(
            len({tuple(self.report["cells"][i]["bucket_xy_m"]) for i in indices}), 4
        )
        for i, action in zip(indices, actions):
            self.assertEqual(action, self.report["cells"][i]["winner"])

    def test_rejects_changed_geometry_unverified_or_unconfirmed_source(self):
        bad_scene = copy.deepcopy(self.cfg)
        bad_scene["bucket"]["inner_top_diameter_m"] += 0.001
        with self.assertRaises(ValueError):
            selection(self.report, bad_scene)
        for change in ("verification", "confirmation", "phase"):
            bad = copy.deepcopy(self.report)
            if change == "verification":
                bad["verification"]["raw_rows"] = False
            elif change == "confirmation":
                bad["cells"][0]["confirmations"][0]["single_bounce_success"] = False
            else:
                bad["phase"] = "running"
            with self.assertRaises(ValueError):
                selection(bad, self.cfg)

    def test_projection_is_orthographic_and_ros_y_points_down(self):
        camera = {"position_m": [0, 0, 0], "rotation_ros": np.eye(3).tolist()}
        view = {"width": 1000, "height": 600, "span_m": 2}
        uv = project([[0, 0, 1], [0, 0, 10], [0.5, 0.2, 1]], camera, view)
        np.testing.assert_allclose(uv, [[500, 300], [500, 300], [750, 400]])
        self.assertEqual([c["cell"] for c in cameras()], [0, 1, 2, 3, 0, 0])

    def test_encoding_rejects_missing_frames_changed_evidence_or_irregular_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "overhead-replay-test"
            run.mkdir()
            (run / "frames").mkdir()
            samples = [{"step": 10}, {"step": 12}]
            (run / "samples.json").write_text(json.dumps(samples))
            meta = {
                "run": run.name,
                "phase": "completed",
                "probes": [{"all_passed": True}],
                "rows": [{"invalid": None}],
                "frames": 2,
                "capture_stride": 2,
                "cameras": [{"name": "top"}],
                "raw_sha256": {
                    "samples.json": hashlib.sha256(
                        (run / "samples.json").read_bytes()
                    ).hexdigest()
                },
            }
            (run / "replay.json").write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "missing/extra"):
                validate(run)
            for i in range(2):
                (run / "frames" / f"top-{i:04d}.jpg").write_bytes(b"frame-fixture")
            validate(run)
            samples[1]["step"] = 13
            (run / "samples.json").write_text(json.dumps(samples))
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                validate(run)
            meta["raw_sha256"]["samples.json"] = hashlib.sha256(
                (run / "samples.json").read_bytes()
            ).hexdigest()
            (run / "replay.json").write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "Uneven"):
                validate(run)


if __name__ == "__main__":
    unittest.main()
