"""Exercise route evidence and frozen choices without a simulator."""

import importlib.util
import json
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from sircus.mjlab import intergrid_route_labels as labels
from sircus.mjlab.intergrid_route_plan import PER_ROUTE_LIMIT, ROUTES


def trace(*, table=None, reversal=True, robot=False):
    terminal = 16
    time = np.arange(terminal + 1) * .01
    force = np.zeros((terminal, 1, len(labels.FORCE_NAMES)))
    velocity = np.zeros((terminal + 1, 1, 3))
    force[2:4, 0, labels.FORCE_NAMES.index("board")] = 2.
    velocity[5, 0, 2] = .3
    if table is not None:
        start, end = table
        force[start:end, 0, labels.FORCE_NAMES.index("table")] = 3.
        velocity[start, 0, 2] = -.4
        velocity[end, 0, 2] = .2 if reversal else -.1
    if robot:
        force[6, 0, labels.FORCE_NAMES.index("robot")] = 2.
    row = dict(id="fixture", route_success=True, scalar_score_audit_passed=True,
               route_class="direct" if table is None else "table_assisted",
               physics_steps=terminal, board_normal=[0., 0., 1.], whole_ball_entry_time_s=.11)
    return row, dict(record_ids=["fixture"], time_s=time, forces_N=force, velocity_m_s=velocity)


class RouteLabelTests(unittest.TestCase):
    def test_direct_capture_requires_no_table_contact(self):
        row, arrays = trace()
        self.assertEqual(labels.label_capture(row, arrays, 0), ("direct", 0))
        with self.assertRaisesRegex(ValueError, "contradicts direct route"):
            labels.label_capture(dict(row, route_class="table_assisted"), arrays, 0)

    def test_table_label_requires_velocity_reversal_before_whole_ball_entry(self):
        for table, reversal, expected in (
            ((7, 9), True, ("table", 1)),
            ((12, 14), True, (None, 0)),
            ((10, 13), True, (None, 0)),
            ((7, 9), False, (None, 0)),
        ):
            with self.subTest(table=table, reversal=reversal):
                row, arrays = trace(table=table, reversal=reversal)
                self.assertEqual(labels.label_capture(row, arrays, 0), expected)
                self.assertTrue(row["route_success"])

    def test_robot_contact_contradicts_verified_capture(self):
        row, arrays = trace(table=(7, 9), robot=True)
        with self.assertRaisesRegex(ValueError, "contradicts standard capture"):
            labels.label_capture(row, arrays, 0)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Study imports require PyTorch")
class RouteStudyTests(unittest.TestCase):
    def test_common_gate_uses_train_attempts_and_setup_rejections(self):
        from sircus.mjlab import intergrid_route_study as study

        train = dict(id="train-00", bucket_xy_m=[.5, -.4])
        proposals = [dict(id=f"{route}-{index}", intended_route=route)
                     for route in ROUTES for index in range(PER_ROUTE_LIMIT)]
        admitted = [dict(id=f"{name}-branch", proposal_id=name)
                    for name in ("table-0", "table-1", "direct-0")]
        train_observations = {
            "table-0-branch": dict(intended_route="table", shot_route="table", route_success=True),
            "table-1-branch": dict(intended_route="table", shot_route="table", route_success=True),
            "direct-0-branch": dict(intended_route="direct", shot_route="table", route_success=True),
        }
        packet = dict(proposals=proposals, admitted=admitted)
        gate = study.fit_gate([train], {train["id"]: packet}, train_observations)
        self.assertEqual(gate["sites"][0]["counts"], {
            "direct": dict(proposals=PER_ROUTE_LIMIT, releases=1, intended_captures=0),
            "table": dict(proposals=PER_ROUTE_LIMIT, releases=2, intended_captures=2),
        })
        self.assertEqual(gate["sites"][0]["realized"], {"table": 3})

        test = dict(id="test-00", bucket_xy_m=[.51, -.4], stratum="rectangle_center")
        menu = [
            dict(id=f"{method}-{route}-{index}", method=method, intended_route=route)
            for method in study.METHODS for route in ROUTES for index in range(2)
        ]
        feasible = [
            dict(row, proposal_id=row["id"], route_success=row["id"].endswith("-1"))
            for row in reversed(menu)
            if (row["method"] == "pooled"
                or row["method"] == "conditioned" and row["intended_route"] == "direct")
        ]
        slots = study.choose(test, menu, feasible, gate)
        self.assertEqual([slot["method"] for slot in slots], list(study.METHODS))
        self.assertEqual([slot["route_priority"] for slot in slots], [["table", "direct"]] * 3)
        self.assertEqual([slot["gate_scores"] for slot in slots],
                         [{"direct": 1 / (PER_ROUTE_LIMIT + 2), "table": 3 / (PER_ROUTE_LIMIT + 2)}] * 3)
        self.assertEqual(slots[0]["request"]["proposal_id"], "pooled-table-0")
        self.assertEqual(slots[1]["request"]["proposal_id"], "conditioned-direct-0")
        self.assertIsNone(slots[2]["request"])
        with self.assertRaisesRegex(ValueError, "only one admitted branch"):
            study.choose(test, menu, [*feasible, feasible[0]], gate)

        observed = {
            slots[0]["id"]: dict(invalid_physics=False, route_success=True, shot_route="direct"),
            slots[1]["id"]: dict(invalid_physics=True, route_success=False, shot_route=None),
        }
        report = study.summarize(slots, observed)
        self.assertEqual(report["methods"], {
            "pooled": {"direct": 1}, "conditioned": {"invalid_physics": 1},
            "specialist": {"setup_rejected": 1},
        })
        self.assertEqual(report["intent"]["pooled"]["other_route_captures"], 1)
        self.assertEqual(report["paired"], {
            f"{left}_vs_{right}": {"pooled_only" if left == "pooled" else "neither": 1}
            for left, right in combinations(study.METHODS, 2)
        })
        with self.assertRaisesRegex(ValueError, "one observation"):
            study.summarize(slots, dict(observed, **{slots[2]["id"]: observed[slots[0]["id"]]}))

    def test_test_outcomes_are_never_read_into_training_rows(self):
        from sircus.mjlab import intergrid_route_study as study

        with TemporaryDirectory() as directory:
            output = Path(directory)
            site = dict(id="train-00", bucket_xy_m=[.5, -.4])
            fixture = [
                dict(phase="test", marker="must be ignored before looking up any identity"),
                dict(phase="train", role="train", context_id=site["id"], bucket_xy_m=site["bucket_xy_m"],
                     intended_route="table", shot_route="direct", table_rebounds_before_entry=0),
            ]
            (output / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in fixture))
            with patch.object(study, "_reject", return_value=None) as reject, patch.object(
                study, "_observation", return_value=("unique", {"id": "training-sample"}),
            ) as convert:
                rows, exclusions = study.fresh_rows(output, [site], {})
        self.assertEqual(rows, [dict(id="training-sample", intended_route="table", shot_route="direct",
                                     table_rebounds_before_entry=0)])
        self.assertEqual(exclusions, {})
        reject.assert_called_once_with(fixture[1], (site["id"], site["bucket_xy_m"]))
        self.assertEqual(convert.call_count, 1)


if __name__ == "__main__":
    unittest.main()
