import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feasibility_report import assert_action, rate


class FeasibilityReportTests(unittest.TestCase):
    def test_invalids_stay_in_denominator(self):
        capture = {
            "route_success": True,
            "single_bounce_success": True,
            "route_quality": 2,
            "invalid": None,
            "external_contact_steps": 0,
        }
        invalid = dict(
            capture,
            route_success=False,
            single_bounce_success=False,
            route_quality=0,
            invalid="endpoint_ik_Arm10",
        )
        r = rate([capture, invalid])
        self.assertEqual((r["captures"], r["trials"], r["invalid"]), (1, 2, 1))
        self.assertEqual(r["failures"], {"endpoint_ik_Arm10": 1})

    def test_perfect_repeats_do_not_claim_certainty(self):
        capture = {
            "route_success": True,
            "single_bounce_success": True,
            "route_quality": 2,
            "invalid": None,
            "external_contact_steps": 0,
        }
        r = rate([capture] * 30)
        self.assertAlmostEqual(r["wilson95"][0], 0.8864866, places=5)
        self.assertIsNone(rate([])["wilson95"])

    def test_changed_board_position_or_repeat_seed_is_rejected(self):
        a = {
            "id": "x",
            "board_x_m": 0.67,
            "board_y_m": -0.1,
            "height_m": 0.4,
            "gap_m": 0.2,
            "tilt_deg": 24,
            "seed": 42,
        }
        assert_action(a, a)
        for key, value in [("board_x_m", 0.68), ("seed", 43)]:
            with self.subTest(key=key), self.assertRaises(AssertionError):
                assert_action(dict(a, **{key: value}), a)


if __name__ == "__main__":
    unittest.main()
