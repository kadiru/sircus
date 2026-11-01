import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distance_catalogue_contract import DISTANCES, grid, jitter
from distance_catalogue_report import counts, validate


def fixture():
    serial = 0

    def row(c, d, success):
        nonlocal serial
        serial += 1
        return dict(
            c,
            trial_id=str(serial),
            distance_m=d,
            success=success,
            invalid=None if success else "endpoint_ik",
            quality_score=2 if success else 0,
            reward=10 if success else -2,
            board_bounce=success,
            whole_ball_entry=success,
            retention_s=1 if success else 0,
            minimum_mouth_clearance_mm=10,
            bucket_contact_before_entry=False,
            rim_band_contact=False,
            exited_after_entry=False,
            settled={
                "position_error_mm": 0.1,
                "normal_error_deg": 0.01,
                "joint_speed_rad_s": 0.001,
                "position_drift_mm": 0.01,
                "normal_drift_deg": 0.001,
            },
        )

    entries = []
    for di, d in enumerate(DISTANCES):
        rows = [row(c, d, d == 0.4) for c in grid(d)]
        e = {
            "distance_m": d,
            "rows": rows,
            "screened": [],
            "winner": None,
            "confirmation": [],
            "phase": "no_capture_found",
            "search_summary": counts(rows),
            "probes": [
                {"peak_bucket_force_N": 10},
                {"peak_bucket_force_N": 5},
                {"peak_bucket_force_N": 0},
            ],
        }
        if d == 0.4:
            c = rows[0]
            s = {"candidate": copy.deepcopy(c)}
            for mode, n in [("nominal", 5), ("perturbed", 20)]:
                s[mode] = [
                    row(c if mode == "nominal" else jitter(c, di, 0, i), d, True)
                    for i in range(n)
                ]
                s[mode + "_summary"] = counts(s[mode])
            e["screened"] = [s]
            e["winner"] = copy.deepcopy(c)
            e["confirmation"] = [
                row(jitter(c, di, 0, i, True), d, True) for i in range(50)
            ]
            e["confirmation_summary"] = counts(e["confirmation"])
            e["phase"] = "completed"
        entries.append(e)
    return {"phase": "completed", "distances": entries, "attempts": serial}


class DistanceEvidenceTests(unittest.TestCase):
    def test_complete_evidence_and_no_solution_rows(self):
        self.assertEqual(validate(fixture())["released"], 76)

    def test_incomplete_tampered_aggregates_or_seeds_rejected(self):
        d = fixture()
        d["phase"] = "confirming"
        with self.assertRaises(ValueError):
            validate(d)
        d = fixture()
        d["distances"][0]["confirmation_summary"]["captures"] = 49
        with self.assertRaises(AssertionError):
            validate(d)
        d = fixture()
        d["distances"][0]["confirmation"][0]["gap_m"] += 0.1
        with self.assertRaises(AssertionError):
            validate(d)
        d = fixture()
        d["distances"][0]["confirmation"][0]["retention_s"] = 0.5
        with self.assertRaises(AssertionError):
            validate(d)
