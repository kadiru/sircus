"""Exercise paired refit choices and route accounting without native simulation."""

import importlib.util
from itertools import product
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("torch"), "Route study imports require PyTorch")
class RouteRefitPlanTests(unittest.TestCase):
    def setUp(self):
        from sircus.mjlab import intergrid_route_refit_plan as plan

        self.plan = plan
        self.layouts = [
            dict(id=f"position-{index}", bucket_xy_m=[.5, -.4 + index / 100], stratum="rectangle_center")
            for index in range(4)
        ]

    def test_proposals_reuse_same_frozen_library_and_rename_only_identifiers(self):
        layout = self.layouts[0]
        library, original, refit = object(), object(), object()
        old_proposals = [
            dict(id=f"{model}-{index}", method=model, intended_route=route, source_index=index,
                 training_recipe_id=f"frozen-{index}", bucket_xy_m=layout["bucket_xy_m"],
                 values=[.4, -.3, .35, 20.0, .25])
            for model in self.plan.MODELS
            for index, route in enumerate(["direct"] * 4 + ["table"] * 4)
        ]
        new_proposals = [dict(row, values=[.41, -.3, .35, 20.0, .25]) for row in old_proposals]
        with patch.object(self.plan, "matched_proposals", side_effect=[old_proposals, new_proposals]) as propose:
            candidates = self.plan.test_proposals(layout, library, original, refit)
        self.assertEqual([c.args for c in propose.call_args_list], [(layout, library, original), (layout, library, refit)])
        self.assertEqual(len({candidate["id"] for candidate in candidates}), 48)
        self.assertEqual(list(dict.fromkeys(row["method"] for row in candidates)), list(self.plan.METHODS))
        for generation, source in zip(self.plan.GENERATIONS, (old_proposals, new_proposals), strict=True):
            self.assertEqual(
                [dict(row, id=f"{row['id']}-{generation}", method=f"{row['method']}_{generation}") for row in source],
                [row for row in candidates if row["method"].endswith(f"_{generation}")],
            )

    def test_first_admitted_respects_candidate_order_and_retains_setup_failures(self):
        layout = self.layouts[0]
        proposals = [
            dict(id=f"candidate-{index}", method="pooled_original", intended_route="table")
            for index in range(3)
        ]
        resolutions = [
            dict(proposals[index], id=f"resolution-{index}", proposal_id=proposals[index]["id"])
            for index in (2, 1)
        ]
        slots = self.plan.select_requests(layout, proposals, resolutions)
        self.assertEqual(len(slots), 12)
        chosen = [slot for slot in slots if slot["request"] is not None]
        self.assertEqual(len(chosen), 1)
        self.assertEqual(chosen[0]["id"], f"{layout['id']}-pooled_original-table")
        self.assertEqual(chosen[0]["request"]["id"], chosen[0]["id"])
        self.assertEqual(chosen[0]["request"]["proposal_id"], "candidate-1")
        with self.assertRaisesRegex(ValueError, "one admitted branch"):
            self.plan.select_requests(layout, proposals, [*resolutions, resolutions[0]])

    def test_summary_pairs_the_same_positions_and_keeps_route_specific_failures(self):
        slots = [
            dict(context_id=layout["id"], id=f"{layout['id']}-{method}-{route}",
                 method=method, intended_route=route, request=None)
            for layout, method, route in product(self.layouts, self.plan.METHODS, self.plan.ROUTES)
        ]
        observed = {
            "original": ("table", "table", "direct", "arm_assisted_capture"),
            "refit": ("table", "invalid_physics", "table", "flight_miss"),
        }
        records = {}
        for slot in slots:
            if slot["method"] not in ("pooled_original", "pooled_refit") or slot["intended_route"] != "table":
                continue
            generation = slot["method"].split("_")[-1]
            label = observed[generation][int(slot["context_id"][-1])]
            slot["request"] = {}
            records[slot["id"]] = dict(invalid_physics=label == "invalid_physics",
                                       route_success=label in ("direct", "table"), shot_route=label,
                                       experimental_route=dict(robot_assisted_capture=label == "arm_assisted_capture"))
        report = self.plan.summarize(self.layouts, slots, records)
        self.assertEqual(report["methods"]["pooled_original"]["table"], {
            "attempts": 4, "releases": 4, "captures": 2, "capture_rate": .5,
            "outcomes": {"table": 2, "direct": 1, "arm_assisted_capture": 1},
        })
        self.assertEqual(report["methods"]["pooled_refit"]["table"]["outcomes"],
                         {"table": 2, "invalid_physics": 1, "flight_miss": 1})
        self.assertEqual(report["paired"]["table"]["pooled"],
                         {"both": 1, "original_only": 1, "refit_only": 1, "neither": 1})
        self.assertEqual(report["methods"]["pooled_refit"]["direct"]["outcomes"], {"setup_rejected": 4})
        with self.assertRaisesRegex(ValueError, "one attempt"):
            self.plan.summarize(self.layouts, [*slots[:-1], slots[-2]], records)
        with self.assertRaisesRegex(ValueError, "one observation"):
            self.plan.summarize(self.layouts, slots, {})
