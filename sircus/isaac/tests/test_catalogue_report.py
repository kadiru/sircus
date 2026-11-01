import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catalogue_contract import grid, perturb, refinement
from catalogue_report import validate


def fixture():
    def row(c, trial_id, success=False):
        return dict(
            c,
            trial_id=trial_id,
            success=success,
            quality_score=1 if success else 0,
            invalid=None if success else "endpoint_ik",
            board_bounce=success,
            whole_ball_entry=success,
            retention_s=1.0 if success else 0,
            minimum_mouth_clearance_mm=0,
            settled={
                "position_error_mm": 0.1,
                "normal_error_deg": 0.01,
                "joint_speed_rad_s": 0.001,
                "position_drift_mm": 0.01,
                "normal_drift_deg": 0.001,
            },
        )

    rows = [row(c, f"grid-{i}", i == 0) for i, c in enumerate(grid())]
    rows.extend(row(c, f"refine-{i}") for i, c in enumerate(refinement(rows)))
    c = rows[0]
    entry = {
        "candidate": copy.deepcopy(c),
        "nominal": [row(c, f"nominal-{i}", True) for i in range(20)],
        "perturbed": [row(perturb(c, i), f"perturbed-{i}", True) for i in range(20)],
    }
    return {"phase": "completed", "rows": rows, "confirmed": [entry]}


class ReportTests(unittest.TestCase):
    def test_complete_fixture_and_settling(self):
        self.assertEqual(validate(fixture())["released_trials"], 41)

    def test_partial_or_tampered_evidence_rejected(self):
        d = fixture()
        d["phase"] = "confirmation"
        with self.assertRaises(ValueError):
            validate(d)
        d = fixture()
        d["confirmed"][0]["perturbed"][0]["height_m"] += 0.1
        with self.assertRaises(AssertionError):
            validate(d)
        d = fixture()
        d["confirmed"][0]["nominal"][0]["retention_s"] = 0.9
        with self.assertRaises(AssertionError):
            validate(d)
        d = fixture()
        d["confirmed"][0]["nominal"][0]["settled"]["position_error_mm"] = 20
        with self.assertRaises(AssertionError):
            validate(d)
