"""Check the shared candidate menu and matched one-drop accounting."""

import importlib.util
import unittest
from unittest.mock import Mock


@unittest.skipUnless(importlib.util.find_spec("torch"), "The route study requires PyTorch")
class FlightRankingStudyTests(unittest.TestCase):
    def setUp(self):
        from sircus.mjlab import intergrid_flight_ranking_study as study

        self.study = study
        self.layout = dict(id="case", bucket_xy_m=[.4, -.5], stratum="interior")
        self.history = [dict(id=f"neighbor-{i}", bucket_xy_m=[.41 + i * .01, -.5],
                             direct_capture=True, table_capture=False) for i in range(5)]
        self.proposals = [dict(id=name, method=route, intended_route=route)
                          for name, route in (("first", "direct"), ("second", "direct"), ("third", "table"))]
        self.admitted = [
            dict(row, id=row["id"] + "-b0", proposal_id=row["id"], bucket_xy_m=self.layout["bucket_xy_m"],
                 values=[.45, -.5, .3, 28., .2 + .01 * index], board_heading_offset_deg=0., board_twist_deg=0.,
                 selected_q10=[0.] * 6, selected_q14=[0.] * 6)
            for index, row in enumerate(self.proposals)
        ]
        self.ranker = Mock()
        self.ranker.order.side_effect = lambda admitted: sorted(admitted, key=lambda row: row["id"] != "second-b0")
        self.ranker.predict.return_value = {"rim_margin_m": .02}

    def select(self, admitted):
        return self.study.select_requests(self.layout, self.proposals, admitted, self.history, self.ranker)

    def test_ranker_can_pick_a_different_command_within_the_same_route(self):
        first, second = self.select(self.admitted)
        self.assertEqual([slot["intended_route"] for slot in (first, second)], ["direct", "direct"])
        self.assertEqual(first["request"]["proposal_id"], "first")
        self.assertEqual(second["request"]["proposal_id"], "second")
        self.assertEqual(second["ranked_candidate_ids"], ["second-b0", "first-b0", "third-b0"])
        self.assertEqual(second["admitted_distinct_commands"], 3)
        self.assertEqual(second["prediction"], {"rim_margin_m": .02})
        self.ranker.order.assert_called_once_with(self.admitted)
        self.ranker.predict.assert_called_once_with(self.admitted[1])

        miss = dict(route_success=False, invalid_physics=False, experimental_route={"robot_assisted_capture": False})
        capture = dict(route_success=True, invalid_physics=False, shot_route="table")
        report = self.study.summarize([self.layout], [first, second], {first["id"]: miss, second["id"]: capture})
        self.assertEqual(report["paired_capture"]["physics_ranker_only"], 1)
        self.assertEqual(report["different_commands"]["same_route"]["physics_ranker_only"], 1)
        self.assertEqual(report["methods"]["physics_ranker"]["attempts"], 1)
        self.assertEqual(report["methods"]["physics_ranker"]["selected_routes"]["direct"], 1)

    def test_setup_refusal_still_counts_for_both_methods(self):
        baseline, ranked = slots = self.select([])
        self.assertIsNone(baseline["request"])
        self.assertIsNone(ranked["request"])
        self.ranker.predict.assert_not_called()
        report = self.study.summarize([self.layout], slots, {})
        self.assertEqual(report["setup_refusals"], 1)
        self.assertEqual(report["paired_capture"]["neither"], 1)
        for method in self.study.METHODS:
            self.assertEqual(report["methods"][method]["attempts"], 1)
            self.assertEqual(report["methods"][method]["releases"], 0)

    def test_order_cannot_add_or_change_a_frozen_request(self):
        self.ranker.order.side_effect = lambda rows: [dict(rows[0], values=[]), *rows[1:]]
        with self.assertRaisesRegex(ValueError, "order all and only admitted"):
            self.select(self.admitted)
