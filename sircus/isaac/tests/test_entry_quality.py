import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catalogue_contract import grid, perturb, refinement
from entry_quality import EntryQuality


class EntryQualityTests(unittest.TestCase):
    def setUp(self):
        self.b = json.loads(
            (Path(__file__).resolve().parents[1] / "scene.json").read_text()
        )["bucket"]
        self.q = EntryQuality(self.b, 0.0375)
        self.xy = self.b["center_xy_m"]
        self.top = 0.183

    def test_clean_capture_and_miss(self):
        self.q.step([*self.xy, self.top], 0, False)
        self.q.step([*self.xy, self.top - 0.038], 0, True)
        self.assertEqual(self.q.result(True)["quality_score"], 2)
        self.assertEqual(self.q.result(False)["quality_score"], 0)

    def test_contact_capture_is_lower_tier(self):
        self.q.step([*self.xy, self.top], 5, False)
        self.q.step([*self.xy, self.top - 0.038], 0, True)
        self.assertEqual(self.q.result(True)["quality_score"], 1)

    def test_post_entry_contact_does_not_penalize_clean(self):
        self.q.step([*self.xy, self.top], 0, False)
        self.q.step([*self.xy, self.top - 0.038], 0, True)
        self.q.step([*self.xy, 0.05], 100, True)
        self.assertEqual(self.q.result(True)["quality_score"], 2)

    def test_later_partial_exit_downgrades_capture(self):
        self.q.step([*self.xy, self.top], 0, False)
        self.q.step([*self.xy, self.top - 0.038], 0, True)
        self.q.step([*self.xy, self.top - 0.02], 0, True)
        self.assertEqual(self.q.result(True)["quality_score"], 1)
        self.assertTrue(self.q.result(True)["exited_after_entry"])

    def test_later_near_rim_contact_downgrades_capture(self):
        self.q.step([*self.xy, self.top], 0, False)
        self.q.step([*self.xy, self.top - 0.038], 0, True)
        self.q.step([*self.xy, self.top - 0.038], 5, True)
        self.assertEqual(self.q.result(True)["quality_score"], 1)
        self.assertTrue(self.q.result(True)["rim_band_contact"])

    def test_unknown_or_marginal_entry_is_not_clean(self):
        self.assertEqual(self.q.result(True)["quality_score"], 1)
        self.q.step([self.xy[0] + 0.07, self.xy[1], self.top], 0, True)
        self.assertEqual(self.q.result(True)["quality_score"], 1)

    def test_search_dense_and_perturbations_bounded(self):
        rows = grid()
        self.assertEqual(len(rows), 495)
        self.assertEqual(len({r["id"] for r in rows}), 495)
        c = rows[0]
        for i in range(20):
            p = perturb(c, i)
            self.assertLessEqual(abs(c["height_m"] - p["height_m"]), 0.002001)
            self.assertLessEqual(abs(c["gap_m"] - p["gap_m"]), 0.002001)
            self.assertLessEqual(abs(c["tilt_deg"] - p["tilt_deg"]), 0.500001)
        r = dict(c, success=True, quality_score=2, minimum_mouth_clearance_mm=10)
        refined = refinement([r])
        self.assertEqual(len(refined), 8)
