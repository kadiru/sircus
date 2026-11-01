import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnostic_report import finalize


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name)
        self.rows = [
            {
                "distance_m": 0.2,
                "kind": "grid",
                "action": [t, 0, h],
                "success": t == 20,
                "reward": 1.0,
                "invalid": None,
            }
            for t in [12, 16, 20, 24, 28, 36, 45]
            for h in [0.06, 0.10, 0.15, 0.20, 0.25]
        ]
        self.summary = {
            "state": "completed",
            "planned": 35,
            "distances": [
                {"distance_m": 0.2, "repeats": [], "policies": [], "grid_captures": 5}
            ],
        }

    def write(self):
        (self.run / "summary.json").write_text(json.dumps(self.summary))
        (self.run / "shots.jsonl").write_text("\n".join(map(json.dumps, self.rows)))

    def test_grid_is_separate_from_policy_and_repeats_and_source_unchanged(self):
        self.write()
        before = (self.run / "summary.json").read_bytes()
        d = finalize(self.run)["distances"][0]
        self.assertEqual(len(d["grid_cells"]), 35)
        self.assertEqual(sum(c["success"] for c in d["grid_cells"]), 5)
        self.assertEqual(d["policies"], [])
        self.assertEqual(d["repeats"], [])
        self.assertEqual((self.run / "summary.json").read_bytes(), before)

    def test_running_or_truncated_sweep_rejected(self):
        self.summary["state"] = "running"
        self.write()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            finalize(self.run)
        self.summary["state"] = "completed"
        self.rows.pop()
        self.write()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            finalize(self.run)

    def test_missing_grid_cannot_be_replaced_by_repeat(self):
        self.rows[-1]["kind"] = "repeat"
        self.write()
        with self.assertRaisesRegex(ValueError, "Missing grid"):
            finalize(self.run)

    def test_duplicate_grid_cell_rejected(self):
        self.rows[-1] = self.rows[0].copy()
        self.write()
        with self.assertRaisesRegex(ValueError, "duplicated"):
            finalize(self.run)
