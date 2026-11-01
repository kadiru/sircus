import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parallel_video import encode


class ParallelVideoTests(unittest.TestCase):
    def test_rejects_incomplete_or_mismatched_recordings(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "parallel-world-20261101T000000Z-n4"
            run.mkdir()
            (run / "frames").mkdir()
            good = {
                "run": run.name,
                "phase": "completed",
                "verification": {"passed": True},
                "video_frames": 2,
            }
            for d in [{**good, "phase": "searching"}, {**good, "run": "older"}, good]:
                (run / "summary.json").write_text(json.dumps(d))
                with (
                    patch("parallel_video.subprocess.run") as ff,
                    self.assertRaises(ValueError),
                ):
                    encode(run)
                ff.assert_not_called()

    def test_keeps_unrelated_latest_pointer_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "parallel-world-20261101T000000Z-n4"
            run.mkdir()
            (run / "frames").mkdir()
            d = {
                "run": run.name,
                "phase": "completed",
                "verification": {"passed": True},
                "video_frames": 2,
            }
            (run / "summary.json").write_text(json.dumps(d))
            pointer = root / "parallel-world.json"
            pointer.write_text('{"run":"newer"}')
            for v in ["overview", "close"]:
                for i in range(2):
                    (run / "frames" / f"{v}-{i:04d}.jpg").write_bytes(b"fixture-frame")

            def encoder(*a, **k):
                (run / "demo.tmp.mp4").write_bytes(b"fixture-video")

            with patch("parallel_video.subprocess.run", side_effect=encoder):
                result = encode(run)
            self.assertEqual(result["run"], run.name)
            self.assertEqual(pointer.read_text(), '{"run":"newer"}')
            self.assertTrue((run / "demo.mp4").exists())
