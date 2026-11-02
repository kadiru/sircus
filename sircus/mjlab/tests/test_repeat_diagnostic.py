"""Finite diagnostic protocol and score provenance tests, without GPU imports."""

from collections import Counter
import copy
import unittest

import numpy as np

from sircus.mjlab.repeat_diagnostic import (
    CAPTURED,
    EXTERNAL,
    plans,
    select_sources,
    branch_delta,
    contact_category,
)
from sircus.mjlab.expanded_actions import OLD_BOUNDS, WIDE_BOUNDS


def source(index, margin=10.0, **extra):
    return dict(
        id=f"source-{index}-{margin}",
        target_id=f"old-test-{index:02d}",
        phase="search",
        invalid=None,
        route_success=True,
        success=True,
        route_quality=2,
        max_retention_s=1.0,
        minimum_mouth_clearance_mm=margin,
        bound_phase="old",
        bucket_xy_m=[0.2, -0.5],
        values=[0.5, -0.2, 0.3, 25.0, 0.3],
        q10=[0.0] * 6,
        q14=[0.0] * 6,
        **extra,
    )


class DiagnosticProtocolTests(unittest.TestCase):
    def test_sources_are_frozen_native_search_captures_by_margin(self):
        rows = [source(i) for i in CAPTURED] + [
            source(i, margin=20.0) for i in CAPTURED
        ]
        rows.append(dict(source(CAPTURED[0], margin=100.0), phase="confirmation"))
        chosen = select_sources(rows)
        self.assertEqual([s["minimum_mouth_clearance_mm"] for s in chosen], [20.0] * 6)
        with self.assertRaises(ValueError):
            select_sources([r for r in rows if r["target_id"] != "old-test-48"])

    def test_finite_132_shot_plan_and_unchanged_nominal_cache(self):
        captured = [source(i) for i in CAPTURED]
        external = [source(i) for i in EXTERNAL]
        before = copy.deepcopy(captured)
        contract = dict(
            old_bounds=OLD_BOUNDS,
            wide_bounds=WIDE_BOUNDS,
            confirmation_noise_std=[0.001, 0.001, 0.002, 0.35, 0.002],
        )
        plan = plans(captured, external, contract)
        self.assertEqual(plan, plans(captured, external, contract))
        self.assertEqual(len(plan), 132)
        self.assertEqual(
            Counter(p["mode"] for p in plan),
            dict(
                nominal_cached=30,
                nominal_fresh_ik=30,
                jitter_fresh_ik=60,
                external_nominal_cached=12,
            ),
        )
        self.assertEqual(
            Counter(p["reset_batch"] for p in plan),
            {0: 24, 1: 24, 2: 24, 3: 24, 4: 24, 5: 6, 6: 6},
        )
        for p in plan:
            if p["mode"] != "jitter_fresh_ik":
                self.assertEqual(p["requested_values"], before[0]["values"])
            self.assertTrue(
                (
                    (np.asarray(p["requested_values"]) >= OLD_BOUNDS["low"])
                    & (np.asarray(p["requested_values"]) <= OLD_BOUNDS["high"])
                ).all()
            )
        self.assertEqual(captured, before)

    def test_wrapped_joint_branch_distance_distinguishes_2pi(self):
        d = branch_delta([0.0] * 6, [2 * np.pi, 0, 0, 0, 0, 0])
        self.assertAlmostEqual(d["max_abs_rad"], 2 * np.pi)
        self.assertLess(d["max_wrapped_rad"], 1e-12)

    def test_external_entry_is_not_promoted_to_route_capture(self):
        row = dict(
            invalid=None,
            external_contact_steps=1,
            contact_peaks_N=dict(table=900.0, robot=0.0),
            route_success=False,
            route_quality=0,
            board_bounce=True,
            whole_ball_entry=True,
        )
        self.assertEqual(contact_category(row), "external_table")
        row["invalid"] = "endpoint_ik_or_pose"
        self.assertEqual(contact_category(row), "geometry_rejected")


if __name__ == "__main__":
    unittest.main()
