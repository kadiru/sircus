import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from learning_contract import outcome_summary
from learning_report import compare, load_run


class ReportTests(unittest.TestCase):
    def fixture(self, path):
        path.mkdir()
        rows = [
            {
                "scene_id": f"test-{i}",
                "action": [20, 0, 0.15],
                "difficulty": d,
                "reward": 15 if i == 0 else -2,
                "success": i == 0,
                "invalid": None if i == 0 else "endpoint_ik",
                "invalid_component": None if i == 0 else "Arm14",
            }
            for i, d in enumerate(["near", "medium", "far"])
        ]
        summary = {
            "run": path.name,
            "training_episodes": 64,
            "unique_training_layouts": 3,
            "selected_episode": 64,
            "elapsed_s": 120,
            "learning_curve": [],
            "contract": {
                "test_pool": 3,
                "run_episodes": 64,
                "fingerprint": "same",
                "gpu": "mock",
            },
            "evaluation": {
                k: outcome_summary(rows)
                for k in ["test", "test_warm_fixed", "test_midpoint"]
            },
        }
        (path / "summary.json").write_text(json.dumps(summary))
        (path / "splits.json").write_text(
            json.dumps({"test": [{"id": r["scene_id"]} for r in rows]})
        )
        for label in ["test", "test_warm_fixed", "test_midpoint"]:
            (path / (label + ".json")).write_text(json.dumps(rows))

    def test_counts_failures_and_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            self.fixture(a)
            self.fixture(b)
            r = compare([a, b])
            self.assertEqual(r["runs"][0]["evaluation"]["test"]["captures"], 1)
            self.assertEqual(r["paired_learned"]["both_miss"], 2)
            self.assertIsNone(r["runs"][0]["checkpoint_sha256"])

    def test_rejects_incomplete_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a"
            self.fixture(p)
            f = p / "summary.json"
            s = json.loads(f.read_text())
            s["training_episodes"] = 32
            f.write_text(json.dumps(s))
            with self.assertRaises(ValueError):
                load_run(p)

    def test_rejects_summary_disagreement(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a"
            self.fixture(p)
            f = p / "summary.json"
            s = json.loads(f.read_text())
            s["evaluation"]["test"]["captures"] = 3
            f.write_text(json.dumps(s))
            with self.assertRaises(ValueError):
                load_run(p)

    def test_rejects_changed_layout_and_contract(self):
        for change in ["split", "contract"]:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                a, b = Path(tmp) / "a", Path(tmp) / "b"
                self.fixture(a)
                self.fixture(b)
                if change == "split":
                    f = b / "splits.json"
                    s = json.loads(f.read_text())
                    s["test"][0]["board_center_m"] = [1, 2, 3]
                else:
                    f = b / "summary.json"
                    s = json.loads(f.read_text())
                    s["contract"]["fingerprint"] = "other"
                f.write_text(json.dumps(s))
                with self.assertRaises(ValueError):
                    compare([a, b])
