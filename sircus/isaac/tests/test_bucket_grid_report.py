import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from board_route import VERSION
from bucket_grid_contract import candidates, cells, contract, refine, summary
from bucket_grid_report import validate


def completed_fixture():
    cfg = json.loads((Path(__file__).resolve().parents[1] / "scene.json").read_text())
    d = {
        "phase": "completed",
        "contract": contract(cfg),
        "cells": [],
        "controls": [],
        "probes": [
            {
                "passed": True,
                "kind": k,
                "peak_N": {
                    "board": 0,
                    "bucket": 20 if k in ("bucket", "rim") else 0,
                    "table": 20 if k == "table" else 0,
                    "floor": 20 if k == "floor" else 0,
                    "robot": 20 if k == "robot" else 0,
                    "unattributed": 0,
                },
            }
            for k in ("bucket", "rim", "table", "floor", "robot")
        ],
    }
    number = 0

    def row(c, cell="control", success=False, external=0):
        nonlocal number
        number += 1
        route = success and not external
        return dict(
            c,
            trial_id=f"fixture-{number}",
            cell_id=cell,
            stage="grid",
            success=success,
            reward=10 if success else -1,
            invalid=None,
            board_bounce=True,
            whole_ball_entry=success,
            retention_s=1.0 if success else 0,
            route_version=VERSION,
            quality_score=2 if success else 0,
            route_success=route,
            route_quality=2 if route else 0,
            single_bounce_success=False,
            board_contact_episodes=2,
            external_contact_steps=external,
            contact_peaks_N={
                "board": 20,
                "bucket": 20 if success else 0,
                "table": 40 if external else 0,
                "floor": 0,
                "robot": 0,
                "unattributed": 0,
            },
            settled={
                "position_error_mm": 0,
                "normal_error_deg": 0,
                "joint_speed_rad_s": 0,
                "position_drift_mm": 0,
                "normal_drift_deg": 0,
            },
        )

    d["controls"] = [
        row(candidates()[1], success=True),
        row(candidates()[4], success=True, external=1),
    ]
    for cell in cells():
        rows = [row(c, cell["id"]) for c in candidates()]
        refined = refine(rows)
        rows += [row(c, cell["id"]) for c in refined]
        d["cells"].append(
            dict(
                cell,
                phase="no_route_setting_found",
                rows=rows,
                refinements=refined,
                search_summary=summary(rows),
                screened=[],
                winner=None,
                confirmation=[],
            )
        )
    d["attempts"] = number
    return d


class GridReportTests(unittest.TestCase):
    def test_no_success_is_valid_evidence_not_an_export_failure(self):
        d = completed_fixture()
        self.assertEqual(validate(d)["route_captures"], 1)  # control only
        self.assertEqual(validate(d)["single_bounce_captures"], 0)

    def test_rejects_incomplete_stale_and_miscounted_results(self):
        for mutation in (
            "phase",
            "contract",
            "aggregate",
            "duplicate",
            "placement",
            "probe",
        ):
            d = completed_fixture()
            if mutation == "probe":
                d["probes"][2]["peak_N"]["table"] = 0
            if mutation == "phase":
                d["phase"] = "searching"
            if mutation == "contract":
                d["contract"]["seed"] += 1
            if mutation == "aggregate":
                d["cells"][0]["search_summary"]["route_captures"] += 1
            if mutation == "duplicate":
                d["cells"][0]["rows"][0]["trial_id"] = d["controls"][0]["trial_id"]
            if mutation == "placement":
                d["cells"][0]["bucket_xy_m"] = [0, 0]
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                validate(d)

    def test_extra_contact_cannot_be_claimed_as_direct_or_single_bounce(self):
        original = completed_fixture()
        d = copy.deepcopy(original)
        d["controls"][0]["contact_peaks_N"]["table"] = 2
        with self.assertRaises(AssertionError):
            validate(d)
        d = copy.deepcopy(original)
        d["controls"][0]["single_bounce_success"] = True
        with self.assertRaises(AssertionError):
            validate(d)
