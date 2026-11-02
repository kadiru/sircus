import json
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feasibility_contract import (
    ACTION_KEYS,
    PERTURBATIONS,
    cells,
    dynamic_cell,
    perturb,
    search_actions,
    select,
    specification,
)


class FeasibilityContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = json.loads(
            Path(__file__).resolve().parents[1].joinpath("scene.json").read_text()
        )

    def test_grid_is_distinct_inside_table_outer_clearance(self):
        points = [c["bucket_xy_m"] for c in cells()]
        self.assertEqual(len({tuple(p) for p in points}), 32)
        radius = self.cfg["bucket"]["outer_top_diameter_m"] / 2
        for p in points:
            for axis in range(2):
                half = self.cfg["table"]["size_m"][axis] / 2
                self.assertLessEqual(
                    abs(p[axis] - self.cfg["table"]["center_m"][axis]) + radius, half
                )

    def test_candidates_cover_xy_and_repeat_deterministically(self):
        for c in cells():
            plan = search_actions(self.cfg, c)
            self.assertEqual(plan, search_actions(self.cfg, c))
            self.assertLessEqual(len(plan), 136)
            self.assertGreaterEqual(
                len({(a["board_x_m"], a["board_y_m"]) for a in plan}), 4
            )
            for a in plan:
                dc = dynamic_cell(c, a)
                self.assertGreaterEqual(dc["distance_m"], 0.28)
                heading = math.radians(dc["heading_deg"])
                n = [math.sin(heading), -math.cos(heading)]
                vector = [b - a for a, b in zip(dc["board_xy_m"], dc["bucket_xy_m"])]
                self.assertAlmostEqual(
                    sum(a * b for a, b in zip(n, vector)), dc["distance_m"]
                )

    def test_perturbations_are_bounded_and_selection_excludes_invalid(self):
        a = dict(
            search_actions(self.cfg, cells()[0])[0],
            route_success=True,
            single_bounce_success=True,
        )
        for rep in range(30):
            p = perturb(a, 0, rep)
            self.assertEqual(p, perturb(a, 0, rep))
            for k in ACTION_KEYS:
                self.assertLessEqual(abs(p[k] - a[k]), PERTURBATIONS[k])
        self.assertIsNone(select([dict(a, invalid="ik")]))
        self.assertIsNone(select([dict(a, route_success=False)]))
        self.assertEqual(select([a]), a)

    def test_spec_preserves_geometry_and_separates_task(self):
        spec = specification(self.cfg, [0, 31])
        self.assertEqual(
            spec["scene"]["bucket"]["height_interpretation"],
            "unresolved: overall height or interior depth",
        )
        original = spec["fingerprint"]
        changed = json.loads(json.dumps(self.cfg))
        changed["ball"]["radius_m"] += 0.001
        self.assertNotEqual(original, specification(changed, [0, 31])["fingerprint"])
        with self.assertRaises(ValueError):
            specification(self.cfg, [0, 0])


if __name__ == "__main__":
    unittest.main()
