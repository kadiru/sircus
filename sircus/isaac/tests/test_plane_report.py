import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plane_contract import candidates
from plane_report import aggregates, verify


class PlaneReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name)
        self.raw = [
            {"candidate": c, "result": {"success": False, "invalid": "endpoint_ik"}}
            for c in candidates()
        ]
        self.summary = {
            "phase": "completed",
            "completed": 175,
            "captures": 0,
            "rows": [
                dict(c, success=False, invalid="endpoint_ik") for c in candidates()
            ],
            "repeats": [],
            "replays": [],
            "artifact_sha256": {},
            "contract": {"heights_m": [0.3, 0.35, 0.4, 0.45, 0.5]},
        }
        self.save()

    def save(self):
        (self.run / "summary.json").write_text(json.dumps(self.summary))
        (self.run / "grid.jsonl").write_text("\n".join(json.dumps(x) for x in self.raw))

    def test_zero_capture_study_is_preserved_and_export_omits_raw_rows(self):
        r = verify(self.run)
        self.assertEqual(sum(x["rejected"] for x in r["by_height"]), 175)
        self.assertEqual(r["pose_summary"]["measured_grid_shots"], 0)
        self.assertNotIn("rows", aggregates(r))

    def test_truncated_study_rejected(self):
        self.raw.pop()
        self.save()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            verify(self.run)

    def test_release_without_settling_rejected(self):
        self.raw[0]["result"] = {"success": False}
        self.save()
        with self.assertRaisesRegex(ValueError, "settling"):
            verify(self.run)
