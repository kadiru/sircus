"""Protect the frozen library intervention and paired route denominators."""
import importlib.util
from itertools import product
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("torch"), "Route study imports require PyTorch")
class IntergridLibraryPlanTests(unittest.TestCase):
    def test_only_audited_table_captures_expand_the_original_ranked_menu(self):
        from sircus.mjlab import intergrid_library_plan as plan
        from sircus.mjlab.intergrid_route_models import ROUTE_VERSION

        original = [dict(id=f"{route}-{index}", bucket_xy_m=[0.4 + .2 * (index // 2), -.5],
                         phase="train", lineage=[index], route_quality=2, shot_route=route,
                         route_version=ROUTE_VERSION, route_success=True, native_release=True,
                         invalid=None, invalid_physics=False, label_conflict=False)
                    for route, index in product(("direct", "table"), range(4))]
        fresh = [dict(original[-1], id=f"new-{index}", bucket_xy_m=[.42, -.5]) for index in range(2)]
        exclusions = [dict(fresh[0], id="new-direct", shot_route="direct"),
                      dict(fresh[0], id="invalid", invalid_physics=True),
                      dict(fresh[0], id="miss", route_success=False),
                      dict(fresh[0], id="ambiguous", label_conflict=True)]
        library = plan.libraries(original, fresh + exclusions)
        self.assertEqual(library["original"]["direct"], library["expanded"]["direct"])
        self.assertEqual(set(library["original"]["table"]), {(.4, -.5), (.4 + .2, -.5)})
        self.assertEqual(library["expanded"]["table"][(.42, -.5)], fresh)
        with self.assertRaisesRegex(ValueError, "training captures"):
            plan.libraries(original, [dict(fresh[0], role="test")])

        def candidates(checkpoint, model, xy, menu, *, context_id, limit):
            self.assertIs(checkpoint, frozen)
            self.assertEqual((model, xy, context_id, limit), ("specialist", [.419, -.5], "test-01", 4))
            self.assertEqual(len(menu), 4)
            self.assertTrue(all(row["intended_route"] == "table" for row in menu))
            return [dict(id=f"{context_id}-{model}-{index:02d}", method=model, intended_route=row["shot_route"],
                         source_index=index, training_recipe_id=row["id"])
                    for index, row in enumerate(menu)]

        layout, frozen = dict(id="test-01", bucket_xy_m=[.419, -.5], stratum="inside"), object()
        with patch.object(plan, "propose_route_candidates", side_effect=candidates):
            proposals = plan.test_proposals(layout, library, frozen)
        self.assertEqual([row["training_recipe_id"] for row in proposals],
                         ["table-0", "table-1", "table-2", "table-3", "new-0", "new-1", "table-0", "table-1"])
        self.assertEqual([row["source_index"] for row in proposals], [0, 1, 2, 3] * 2)
        self.assertTrue(all(row["id"].endswith("-" + row["method"]) for row in proposals))

        resolved = [dict(row, id=row["id"] + "-b0", proposal_id=row["id"]) for row in (proposals[2], proposals[1])]
        slots = plan.select_requests(layout, proposals, resolved)
        self.assertEqual([slot["id"] for slot in slots], ["test-01-original-table", "test-01-expanded-table"])
        self.assertEqual(slots[0]["request"]["proposal_id"], proposals[1]["id"])
        self.assertIsNone(slots[1]["request"])

    def test_setup_and_captures_are_paired_separately(self):
        from sircus.mjlab import intergrid_library_plan as plan

        layouts = [dict(id=f"test-{index}") for index in range(5)]
        results = {"original": ("table", "direct", "arm_assisted_capture", None, None),
                   "expanded": ("table", "table", "invalid_physics", "flight_miss", None)}
        slots, records = [], {}
        for index, layout in enumerate(layouts):
            for method, outcomes in results.items():
                label, identity = outcomes[index], f"{layout['id']}-{method}-table"
                slots.append(dict(id=identity, method=method, context_id=layout["id"], intended_route="table",
                                  request=None if label is None else {}))
                if label is not None:
                    records[identity] = dict(invalid_physics=label == "invalid_physics",
                                             route_success=label in {"table", "direct"}, shot_route=label,
                                             experimental_route=dict(robot_assisted_capture=label == "arm_assisted_capture"))

        report = plan.summarize(layouts, slots, records)
        self.assertEqual(report["methods"]["original"],
                         dict(attempts=5, releases=3, setup_rate=.6, captures=1, capture_rate=.2,
                              outcomes=dict(table=1, direct=1, arm_assisted_capture=1, setup_rejected=2)))
        self.assertEqual(report["methods"]["expanded"],
                         dict(attempts=5, releases=4, setup_rate=.8, captures=2, capture_rate=.4,
                              outcomes=dict(table=2, invalid_physics=1, flight_miss=1, setup_rejected=1)))
        self.assertEqual(report["paired"], dict(setup=dict(both=3, original_only=0, expanded_only=1, neither=1),
                                               capture=dict(both=1, original_only=0, expanded_only=1, neither=3)))
        with self.assertRaisesRegex(ValueError, "one table attempt"):
            plan.summarize(layouts, slots[:-1], records)
        with self.assertRaisesRegex(ValueError, "one observation"):
            plan.summarize(layouts, slots, {})
