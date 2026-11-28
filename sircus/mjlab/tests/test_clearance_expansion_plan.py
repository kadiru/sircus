"""Bounded setup-only clearance expansion preserves the comparison control."""
import copy
import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch


@unittest.skipUnless(importlib.util.find_spec("torch"), "The route study requires PyTorch")
class ClearanceExpansionPlanTests(unittest.TestCase):
    def setUp(self):
        from sircus.mjlab import clearance_expansion_plan as plan
        self.plan = plan
        self.layout = dict(id="case", bucket_xy_m=[.8, -.5], split="test", stratum="interior")
        self.history = [dict(id=str(i), bucket_xy_m=[.41+i*.01, -.5],
                             direct_capture=True, table_capture=False) for i in range(5)]
        self.libraries = {"original": {"direct": [1]}, "expanded": {"table": [2]}}
        self.ranker = Mock()
        self.ranker.predict.side_effect = lambda row: {"score_m": row.get("score", -.02)}
        self.ranker.order.side_effect = lambda rows: sorted(rows, key=lambda row: -row.get("score", -.02))

    def row(self, identity, gap=.2, score=-.02, route="direct"):
        return dict(id=identity, context_id="case", proposal_id=identity,
                    bucket_xy_m=self.layout["bucket_xy_m"], phase="test", role="test",
                    method=route, intended_route=route, values=[.4, -.5, .3, 28., gap],
                    board_heading_offset_deg=0., board_twist_deg=0.,
                    selected_q10=[0.]*6, selected_q14=[0.]*6, score=score)

    @staticmethod
    def stage(rows=()):
        return dict(proposals=list(rows), admitted=list(rows), records=[])

    def prepared(self, **stages):
        return {name: self.stage(stages.get(name, [])) for name in self.plan.STAGES}

    def test_positive_baseline_command_is_preserved_exactly(self):
        from sircus.mjlab.intergrid_candidate_replay_plan import command_key
        row = self.row("base", score=.001)
        prepared = self.prepared(base=[row])
        before = copy.deepcopy(prepared)
        current, updated = self.plan.select_requests(self.layout, prepared, self.history, self.ranker)
        self.assertEqual(command_key(current["request"]), command_key(updated["request"]))
        self.assertEqual(current["request"]["proposal_id"], "base")
        self.assertFalse(updated["clearance_expansion_triggered"])
        self.assertFalse(updated["predicted_flight_risk"])
        self.assertEqual(prepared, before)
        prepared["expansion"] = self.stage([self.row("extra", .3, .1)])
        with self.assertRaisesRegex(ValueError, "must not trigger"):
            self.plan.select_requests(self.layout, prepared, self.history, self.ranker)

    def test_trigger_includes_zero_unavailable_and_nonfinite_predictions(self):
        for score in (None, 0., -.01, float("nan"), float("inf")):
            self.assertTrue(self.plan.needs_expansion(dict(request={}, prediction={"score_m": score})))
        self.assertTrue(self.plan.needs_expansion(dict(request=None, prediction={"score_m": 1.})))
        self.assertFalse(self.plan.needs_expansion(dict(request={}, prediction={"score_m": .0001})))

    def test_baseline_setup_fallback_is_still_the_control(self):
        row = self.row("fallback", score=-.01)
        prepared = self.prepared(fallback=[row], expansion=[self.row("expanded", .3, .02)])
        current, updated = self.plan.select_requests(self.layout, prepared, self.history, self.ranker)
        self.assertEqual(current["request"]["proposal_id"], "fallback")
        self.assertEqual(updated["request"]["proposal_id"], "expanded")
        self.assertTrue(current["predicted_flight_risk"])
        self.assertFalse(updated["predicted_flight_risk"])

    def test_all_negative_retains_best_attempt_and_no_setup_remains_in_denominator(self):
        prepared = self.prepared(base=[self.row("base")], expansion=[self.row("better", .3, -.001)])
        _, updated = self.plan.select_requests(self.layout, prepared, self.history, self.ranker)
        self.assertEqual(updated["request"]["proposal_id"], "better")
        self.assertTrue(updated["predicted_flight_risk"])
        current, updated = self.plan.select_requests(self.layout, self.prepared(), self.history, self.ranker)
        self.assertIsNone(current["request"])
        self.assertIsNone(updated["request"])
        self.assertEqual(len([current, updated]), 2)

    def test_equal_score_keeps_original_command(self):
        prepared = self.prepared(base=[self.row("z-base")], branches=[self.row("a-new", .3)])
        _, updated = self.plan.select_requests(self.layout, prepared, self.history, self.ranker)
        self.assertEqual(updated["request"]["proposal_id"], "z-base")

    def test_geometry_deduplication_and_bound(self):
        row = self.row("base")
        duplicate = dict(row, id="same", board_heading_offset_deg=360.)
        candidates = [duplicate] + [self.row(f"new-{i}", .3+i*.001) for i in range(50)]
        with patch.object(self.plan, "collection_geometry", return_value=candidates):
            result = self.plan.expansion_proposals(self.layout, self.libraries, self.prepared(base=[row]))
        self.assertEqual(len(result), self.plan.GEOMETRY_LIMIT)
        self.assertTrue(all("same" != r["id"].split("-")[0] for r in result))
        self.assertEqual(result[0]["values"], candidates[1]["values"])

    def test_branches_use_four_distinct_geometries_and_preserve_targets(self):
        rows = [self.row(f"base-{i}", .2+i*.01, -.01*i) for i in range(7)]
        rows.insert(1, dict(rows[0], id="duplicate", selected_q14=[.5]*6))
        result = self.plan.branch_proposals(self.layout, self.prepared(base=rows), self.ranker)
        self.assertEqual(len(result), self.plan.BRANCH_SETTINGS)
        for index, proposal in enumerate(result):
            source = next(r for r in rows if r["id"] == proposal["source_request_id"])
            self.assertEqual(proposal["values"], source["values"])
            self.assertEqual(proposal["seed_q10"], source["selected_q10"])
            self.assertEqual(proposal["seed_q14"], source["selected_q14"])
            self.assertEqual(proposal["branch_seed"], self.plan.BRANCH_SEED+index)
            self.assertEqual(proposal["role"], "test")

    def test_branch_dedup_never_erases_exact_dropper_and_only_accepts_changed_board(self):
        import numpy as np
        resolver = object.__new__(self.plan.BoardOnlyResolver)
        resolver.original = self.row("original")
        # Candidate1 would hide saved candidate0 in ordinary residual-first dedup.
        first = dict(q=np.array([[0.]*6, [.001]*6, [.3]*6]),
                     valid=np.array([True]*3), residual=np.array([.001, 0., .0001]))
        second = dict(q=np.array([[0.]*6, [.1]*6, [.2]*6]),
                      valid=np.array([True]*3), residual=np.array([0., .001, .002]))
        result = resolver._branches(first, second, {})
        self.assertEqual(result, [(0, 2), (0, 1)])
        first["valid"][0] = False
        self.assertEqual(resolver._branches(first, second, {}), [])

    def test_branch_worker_restores_methods_after_native_error(self):
        old = object()
        original = self.row("original")
        proposal = dict(source_request_id="original", branch_seed=1)
        arms = [SimpleNamespace(solve_candidates=Mock()), SimpleNamespace(solve_candidates=Mock())]
        methods = [arm.solve_candidates for arm in arms]
        resolver = SimpleNamespace(solver=SimpleNamespace(ik=SimpleNamespace(dropper=arms[0], board=arms[1])))
        engine = SimpleNamespace(adapter=SimpleNamespace(resolver=old),
                                 prepare=Mock(side_effect=RuntimeError("native error")))
        with patch.object(self.plan, "BoardOnlyResolver", return_value=resolver):
            with self.assertRaisesRegex(RuntimeError, "native error"):
                self.plan._prepare_branches(engine, [proposal], {"original": original}, "/tmp/assets")
        self.assertIs(engine.adapter.resolver, old)
        self.assertEqual([arm.solve_candidates for arm in arms], methods)

    def test_branch_wrapper_preserves_nested_reference_options_and_outer_budgets(self):
        import numpy as np
        old = object()
        original = self.row("original")
        proposal = dict(source_request_id="original", branch_seed=1234)
        calls = []
        def implementation(targets, initial, **kwargs):
            calls.append((targets, initial, kwargs))
            return kwargs
        arms = [SimpleNamespace(solve_candidates=implementation) for _ in range(2)]
        resolver = SimpleNamespace(solver=SimpleNamespace(ik=SimpleNamespace(dropper=arms[0], board=arms[1])))
        def prepare(rows, *, one_per_setting):
            self.assertEqual(rows, [proposal])
            self.assertFalse(one_per_setting)
            for role, arm in zip((10, 14), arms, strict=True):
                outer = arm.solve_candidates(["candidate"], "initial", starts=8,
                    iterations=77, known="saved-joints", normal_only=False)
                self.assertEqual(outer["starts"], 1 if role == 10 else self.plan.BRANCH_RULE["starts_per_arm"])
                self.assertEqual((outer["iterations"], outer["seed"]), (77, 1234))
                np.testing.assert_array_equal(outer["preserve_known"], [True])
                # Real fallback/reference solves already supply their own
                # preservation mask, random seed and solve budget. Previously
                # this call supplied preserve_known/seed twice and crashed.
                for mask in ([False], [True], None):
                    nested = arm.solve_candidates(["reference"], "reference-initial", starts=3,
                        iterations=19, known="reference-joints", normal_only=True,
                        preserve_known=mask, seed=902)
                    self.assertEqual(nested, dict(starts=3, iterations=19,
                        known="reference-joints", normal_only=True, preserve_known=mask, seed=902))
            return [original], [{"admitted": 1}]
        engine = SimpleNamespace(adapter=SimpleNamespace(resolver=old), prepare=prepare)
        with patch.object(self.plan, "BoardOnlyResolver", return_value=resolver):
            result = self.plan._prepare_branches(engine, [proposal], {"original": original}, "/tmp/assets")
        self.assertEqual(result["admitted"], [original])
        self.assertEqual(result["records"], [{"admitted": 1}])
        self.assertEqual(len(calls), 8)
        self.assertIs(engine.adapter.resolver, old)
        self.assertTrue(all(arm.solve_candidates is implementation for arm in arms))
