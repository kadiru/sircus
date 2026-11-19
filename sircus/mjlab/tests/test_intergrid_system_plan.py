"""Check the one-shot setup fallback without native releases."""
import importlib.util
import unittest
from unittest.mock import Mock, patch


@unittest.skipUnless(importlib.util.find_spec("torch"), "The route study requires PyTorch")
class SystemPlanTests(unittest.TestCase):
    def setUp(self):
        from sircus.mjlab import intergrid_system_plan as plan

        self.plan = plan
        self.layout = dict(id="case", bucket_xy_m=[.8, -.5], split="test", stratum="interior")
        self.base = self.proposal("base", "direct", .2)
        self.transferred = self.proposal("transferred", "table", .24)
        self.libraries = {"original": {"direct": [1]}, "expanded": {"table": [2]}}
        self.history = [
            dict(id=str(i), bucket_xy_m=[.41 + i * .01, -.5], direct_capture=True, table_capture=False)
            for i in range(5)
        ]
        self.ranker = Mock()
        self.ranker.order.side_effect = lambda rows: list(rows)
        self.ranker.predict.return_value = {"score_m": .02}

    def proposal(self, identity, route, gap):
        return dict(id=identity, context_id=self.layout["id"], bucket_xy_m=self.layout["bucket_xy_m"],
                    phase=self.layout["split"], role=self.layout["split"], method=route,
                    intended_route=route, values=[.4, -.5, .3, 28., gap],
                    board_heading_offset_deg=0., board_twist_deg=0.)

    @staticmethod
    def stage(proposals, admitted=None):
        return dict(proposals=proposals, records=[], admitted=[] if admitted is None else [
            dict(admitted, id=admitted["id"] + "-b0", proposal_id=admitted["id"],
                 selected_q10=[0.] * 6, selected_q14=[0.] * 6)
        ])

    def test_base_admission_keeps_the_candidate_menu_and_command(self):
        from sircus.mjlab.intergrid_candidate_replay_plan import command_key

        plan = self.plan
        base = self.stage([self.base], self.base)
        with patch.object(plan, "collection_geometry") as generate:
            self.assertEqual(plan.fallback_proposals(self.layout, self.libraries, base["proposals"],
                                                     base["admitted"]), [])
        generate.assert_not_called()
        prepared = dict(base=base, fallback=self.stage([]))
        first, second = plan.select_requests(self.layout, prepared, self.history, self.ranker)
        self.assertEqual(command_key(first["request"]), command_key(second["request"]))
        self.assertEqual(second["candidate_pool"], "base")
        self.ranker.order.assert_called_once_with(base["admitted"])

    def test_extra_preparation_removes_only_exact_cross_pool_duplicates(self):
        plan = self.plan
        duplicate = dict(self.base, id="duplicate", board_heading_offset_deg=360.)
        close = dict(self.transferred, id="close", values=[*self.base["values"][:4], .20000001])
        with patch.object(plan, "collection_geometry", return_value=[duplicate, close, self.transferred]) as generate:
            fallback = plan.fallback_proposals(self.layout, self.libraries, [self.base], [])
        generate.assert_called_once_with(self.layout, {"direct": [1], "table": [2]}, phase="test")
        self.assertEqual([row["id"] for row in fallback], ["close-setup-fallback", "transferred-setup-fallback"])

    def test_fallback_does_not_supply_a_shot_to_the_baseline(self):
        prepared = dict(base=self.stage([self.base]), fallback=self.stage([self.transferred], self.transferred))
        first, second = self.plan.select_requests(self.layout, prepared, self.history, self.ranker)
        self.assertIsNone(first["request"])
        self.assertEqual(second["candidate_pool"], "fallback")
        self.assertEqual(second["request"]["proposal_id"], "transferred")
