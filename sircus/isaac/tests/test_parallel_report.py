import copy
import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parallel_contract import candidates, specification
from parallel_report import validate


def fixture():
    cfg = json.loads(
        Path(__file__).resolve().parents[1].joinpath("scene.json").read_text()
    )
    spec = specification(cfg, 32, [0], 1, 1)
    spec.update(smoke=False, source_sha256={}, gpu="fixture")

    def row(a, number):
        return dict(
            a,
            trial_id=f"trial-{number}",
            cell_id="cell-00",
            success=True,
            route_success=True,
            single_bounce_success=True,
            route_quality=2,
            quality_score=2,
            reward=15,
            board_contact_episodes=1,
            external_contact_steps=0,
            board_bounce=True,
            whole_ball_entry=True,
            max_retention_s=1,
            invalid=None,
            settled={
                "position_error_mm": 0,
                "normal_error_deg": 0,
                "joint_speed_rad_s": 0,
                "position_drift_mm": 0,
                "normal_drift_deg": 0,
                "dropper_position_error_mm": 0,
            },
        )

    a = candidates()[0]
    c = copy.deepcopy(spec["cells"][0])
    c.update(rows=[row(a, 2)], winner=a, confirmations=[])
    rng = np.random.default_rng(20261102)
    perturbed = dict(a, seed=20261102)
    for key, delta in spec["perturbations"].items():
        perturbed[key] += float(rng.uniform(-delta, delta))
    c["confirmations"] = [row(perturbed, 3)]
    probes = []
    for kind, index in [
        ("bucket", 1),
        ("rim", 1),
        ("table", 2),
        ("floor", 3),
        ("robot", 4),
    ]:
        peak = [0.0] * 6
        peak[index] = 1
        probes.append(
            {
                "kind": kind,
                "all_passed": True,
                "per_cell_passed": [True],
                "peak_N": [peak],
            }
        )
    return {
        "contract": spec,
        "phase": "completed",
        "validation_passed": True,
        "batches_completed": 1,
        "max_bucket_drift_m": 0,
        "cells": [c],
        "controls": [{"rows": [row(a, 0)]}, {"rows": [row(candidates()[1], 1)]}],
        "probes": probes,
        "attempts": 4,
    }


class ParallelReportTests(unittest.TestCase):
    def test_complete_fixture_and_incomplete_evidence(self):
        self.assertEqual(len(validate(fixture())), 4)
        for key, value in [
            ("phase", "searching"),
            ("max_bucket_drift_m", 0.001),
            ("attempts", 5),
        ]:
            d = fixture()
            d[key] = value
            with self.assertRaises(AssertionError):
                validate(d)

    def test_rejects_changed_target_plan_seed_or_contact_route(self):
        for change in (
            "target",
            "action",
            "seed",
            "route",
            "retention",
            "settled",
            "probe",
        ):
            d = fixture()
            c = d["cells"][0]
            if change == "target":
                c["bucket_xy_m"][0] += 0.01
            elif change == "action":
                c["rows"][0]["height_m"] += 0.01
            elif change == "seed":
                c["confirmations"][0]["seed"] += 1
            elif change == "route":
                c["rows"][0]["external_contact_steps"] = 1
            elif change == "retention":
                c["rows"][0]["max_retention_s"] = 0.5
            elif change == "settled":
                c["rows"][0]["settled"]["dropper_position_error_mm"] = 3
            elif change == "probe":
                d["probes"][0]["peak_N"][0][1] = 0
            with self.subTest(change=change), self.assertRaises(AssertionError):
                validate(d)
