"""Regression tests for the single-release selector's frozen comparison."""
import importlib.util
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("torch"), "Route study imports require PyTorch")
class RouteSelectorPlanTests(unittest.TestCase):
    def setUp(self):
        from sircus.mjlab import intergrid_route_selector_plan as plan

        self.plan = plan
        self.layout = dict(id="case", bucket_xy_m=[.4, -.5], stratum="interior")
        self.history = [dict(id=f"history-{index}", bucket_xy_m=[.4 + index / 100, -.5],
                             direct_capture=False, table_capture=index <= 5) for index in range(1, 7)]
        self.proposals = [dict(id=route, method=route, intended_route=route) for route in plan.ROUTES]
        self.admitted = [dict(row, id=row["id"] + "-b0", proposal_id=row["id"], values=[.01, .02, .3, 28., .25])
                         for row in self.proposals]

    def test_candidate_routes_use_only_the_intended_four_sources(self):
        originals, expanded, checkpoint = object(), object(), object()
        menus = {
            library: [dict(id=f"{route}-{index}", intended_route=route)
                      for route in self.plan.ROUTES for index in range(4)]
            for library in (originals, expanded)
        }

        def predict(weights, model, xy, sources, *, context_id, limit):
            self.assertIs(weights, checkpoint)
            self.assertEqual((model, xy, context_id, limit), ("specialist", [.4, -.5], "case", 4))
            self.assertEqual(len(sources), 4)
            return sources

        with (patch.object(self.plan, "source_menu", side_effect=lambda layout, library: menus[library]) as menu,
              patch.object(self.plan, "propose_route_candidates", side_effect=predict) as predictor):
            proposals = self.plan.test_proposals(self.layout, dict(original=originals, expanded=expanded), checkpoint)
        self.assertEqual([item.args[-1] for item in menu.call_args_list], [originals, expanded])
        self.assertEqual([row["method"] for row in proposals], ["direct"] * 4 + ["table"] * 4)
        self.assertEqual(len({row["id"] for row in proposals}), 8)
        for index, route in enumerate(self.plan.ROUTES):
            self.assertEqual({row["intended_route"] for row in predictor.call_args_list[index].args[-1]}, {route})

    def test_historical_rank_and_fallback_are_chosen_before_release(self):
        select = self.plan.select_requests
        direct, selector = select(self.layout, self.proposals, self.admitted, list(reversed(self.history)))
        self.assertEqual((direct["intended_route"], selector["intended_route"]), ("direct", "table"))
        self.assertEqual(selector["gate_scores"], dict(direct=1 / 7, table=6 / 7))
        self.assertEqual(selector["neighbor_ids"], [f"history-{index}" for index in range(1, 6)])
        self.assertEqual(selector["route_priority"], ["table", "direct"])

        direct, selector = select(self.layout, self.proposals, self.admitted[:1], self.history)
        self.assertEqual(selector["request"], dict(direct["request"], id=selector["id"], method=selector["method"]))
        self.assertEqual(selector["intended_route"], "direct")
        direct, selector = select(self.layout, self.proposals, self.admitted[1:], self.history)
        self.assertIsNone(direct["request"])
        self.assertEqual(selector["intended_route"], "table")

        tied = [dict(row, table_capture=False) for row in self.history]
        _, selector = select(self.layout, self.proposals, self.admitted, tied)
        self.assertEqual(selector["route_priority"], ["direct", "table"])

    def test_summary_credits_standard_wrong_route_and_preserves_disagreements(self):
        layouts = [dict(self.layout, id=f"case-{index}") for index in range(3)]
        slots = []
        records = {}
        for layout, admission, labels in zip(layouts, (self.admitted[:1], self.admitted[1:], self.admitted),
                                             (("direct", "table"), (None, "unclassified_capture"),
                                              ("arm_assisted_capture", "invalid_physics")), strict=True):
            used = self.plan.select_requests(layout, self.proposals, admission, self.history)
            slots.extend(used)
            for slot, label in zip(used, labels, strict=True):
                if label is not None:
                    records[slot["id"]] = dict(invalid_physics=label == "invalid_physics",
                                               route_success=label in self.plan.STANDARD_CAPTURES,
                                               shot_route=label if label in self.plan.ROUTES else None,
                                               experimental_route=dict(robot_assisted_capture=label == "arm_assisted_capture"))

        report = self.plan.summarize(layouts, slots, records)
        self.assertEqual(report["methods"]["direct_only"]["captures"], 1)
        self.assertEqual(report["methods"]["route_selector"]["captures"], 2)
        self.assertEqual(report["methods"]["route_selector"]["selected_routes"], dict(direct=1, table=2))
        self.assertEqual(report["methods"]["route_selector"]["requested_route_captures"], 0)
        self.assertEqual(report["paired"]["capture"], dict(both=1, direct_only=0, route_selector_only=1, neither=1))
        self.assertEqual(report["same_direct_command"], dict(attempts=1, outcome_disagreements=1))
        corrupted = dict(slots[1], request=dict(slots[1]["request"], values=[.2, .2, .3, 30., .22]))
        with self.assertRaisesRegex(ValueError, "same admitted direct command"):
            self.plan.summarize(layouts, [slots[0], corrupted, *slots[2:]], records)
