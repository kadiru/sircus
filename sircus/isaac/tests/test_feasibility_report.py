import hashlib
import json
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feasibility_contract import specification
from feasibility_report import STAGES, assert_action, compact, rate, validate_contract


class FeasibilityReportTests(unittest.TestCase):
    def test_platform_ulp_keeps_original_fingerprint(self):
        spec = json.loads(json.dumps(specification({}, [0, 7])))
        spec["cells"][1]["heading_deg"] = math.nextafter(
            spec["cells"][1]["heading_deg"], math.inf
        )
        payload = {k: v for k, v in spec.items() if k != "fingerprint"}
        spec["fingerprint"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()
        validate_contract(spec)
        spec["fingerprint"] = "changed"
        with self.assertRaises(AssertionError):
            validate_contract(spec)

    def test_different_geometry_is_not_platform_roundoff(self):
        spec = json.loads(json.dumps(specification({}, [0, 7])))
        spec["cells"][1]["heading_deg"] += 0.01
        with self.assertRaises(AssertionError):
            validate_contract(spec)

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

    def test_compact_geometry_does_not_copy_raw_trial_history(self):
        base = {
            "id": "target-00",
            "index": 0,
            "bucket_xy_m": [0.3, -0.47],
            "board_xy_m": [0.67, -0.1],
            "distance_m": 0.52,
            "heading_deg": 0,
        }
        winner = {
            "board_x_m": 0.62,
            "board_y_m": -0.15,
            "height_m": 0.4,
            "gap_m": 0.18,
            "tilt_deg": 16,
        }
        cell = dict(base, winner=winner, **{stage: [] for stage in STAGES})
        cell["baseline"] = [
            {
                "route_success": True,
                "single_bounce_success": True,
                "route_quality": 2,
                "invalid": None,
                "external_contact_steps": 0,
                "raw_note": "not exported",
            }
        ]
        data = {
            "run": "fixture",
            "phase": "completed",
            "contract": {"cells": [base]},
            "timestamp_utc": "fixture",
            "elapsed_s": 1,
            "attempts": 1,
            "max_bucket_drift_m": 0,
            "probes": [],
            "artifact_sha256": {},
            "direct_drop_audit": [],
            "cells": [cell],
        }
        exported = compact(data)["cells"][0]
        self.assertEqual(set(exported["selected_geometry"]), set(base))
        self.assertNotIn("baseline", exported)
        self.assertEqual(exported["stages"]["baseline"]["captures"], 1)


if __name__ == "__main__":
    unittest.main()
