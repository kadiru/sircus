"""Bounded pre-IK alternatives, opt-in rank changes and unchanged exact gates."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from unittest.mock import Mock, patch
import unittest

from sircus.mjlab import approach_comparison_plan as plan
from sircus.mjlab.tests.test_neural_shot_model import checkpoint, request


class MenuTests(unittest.TestCase):
    def setUp(self):
        self.checkpoint = checkpoint()
        self.libs = dict(expanded_menu_version=plan.EXPANDED_MENU_VERSION,
                         baseline_libraries={}, expanded_sources=[])
        self.layout = dict(id="fresh", split="test", bucket_xy_m=[.8, -.3])

    def config(self):
        return plan.configure(self.libs, {}, self.checkpoint)

    def test_configuration_is_independent_idempotent_and_checkpoint_bound(self):
        original = deepcopy((self.libs, self.checkpoint))
        libs, config = self.config()
        self.assertEqual((libs, config), plan.configure(libs, config, self.checkpoint))
        self.assertEqual(config["contact_ranking_version"], plan.CONTACT_VERSION)
        self.assertEqual(config["approach_ranking_version"], plan.VERSION)
        self.assertEqual(original, (self.libs, self.checkpoint))
        changed = deepcopy(self.checkpoint)
        changed["layers"][0]["bias"][0] = .1
        with self.assertRaisesRegex(ValueError, "mismatch"):
            plan.configure(libs, config, changed)
        libs["approach_menu_checkpoint"]["layers"][0]["bias"][0] = .2
        self.assertEqual(self.checkpoint["layers"][0]["bias"][0], 0.)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            plan.configure(libs, config, self.checkpoint)

    def test_configuration_rejects_unsupported_and_recursive_inputs(self):
        for field in ("approach_ranking_version", "contact_ranking_version"):
            with self.assertRaises(ValueError):
                plan.configure(self.libs, {field: "unknown"}, self.checkpoint)
        for field in ("expanded_menu_version", "approach_menu_version"):
            bad = dict(self.libs, baseline_libraries={field: plan.VERSION})
            with self.assertRaisesRegex(ValueError, "Recursive"):
                plan.configure(bad, {}, self.checkpoint)
        with self.assertRaises(ValueError):
            plan.configure({}, {}, self.checkpoint)
        with self.assertRaises(ValueError):
            plan.configure(self.libs, {}, {"version": "route-models-not-neural"})
        for field in plan.MENU_KEYS:
            with self.assertRaises(ValueError):
                plan.configure(dict(self.libs, **{field: "partial"}), {}, self.checkpoint)

    def test_menu_preserves_originals_and_hints_with_bounded_distinct_local_settings(self):
        libs, _ = self.config()
        originals = [dict(request(f"base-{i}"), values=[.35+i*.003, -.35, .38, 28., .38],
                          source_setting_key=f"source-{i}", source_observed_route=["arm"],
                          dataset_split="test", context_id="fresh") for i in range(56)]
        before = deepcopy(originals)
        source_checkpoint = {"different": "route regressor"}
        with patch.object(plan, "expanded_menu", return_value=originals) as expand:
            rows = plan.comparison_menu(self.layout, libs, source_checkpoint)
            self.assertEqual(rows, plan.comparison_menu(self.layout, libs, source_checkpoint))
        self.assertIs(expand.call_args.args[2], source_checkpoint)
        self.assertNotIn("approach_menu_version", expand.call_args.args[1])
        self.assertEqual(originals, before)
        self.assertEqual([r for r in rows if r["id"].startswith("base-")], originals)
        self.assertEqual(len(rows), plan.MENU_LIMIT)
        self.assertEqual(len({plan.key(r) for r in rows}), len(rows))
        self.assertEqual(len({r["id"] for r in rows}), len(rows))
        self.assertEqual([r["id"].startswith("base-") for r in rows[:8]], [True, False]*4)
        local = [r for r in rows if r["id"].startswith("fresh-approach-local-")]
        self.assertEqual({r["approach_parent_proposal_id"] for r in local}, {f"base-{i}" for i in range(4)})
        for candidate in local:
            parent = next(r for r in originals if r["id"] == candidate["approach_parent_proposal_id"])
            for field in ("seed_q10", "seed_q14", "source_setting_key", "source_observed_route", "dataset_split"):
                self.assertEqual(candidate[field], parent[field])
            self.assertEqual(candidate["values"][:3], parent["values"][:3])
        self.assertTrue(any(r["approach_delta"] == dict(tilt_deg=-8., drop_gap_m=.02) for r in local))

    def test_frozen_expanded_checkpoint_chooses_seeds_not_source_regressor(self):
        libs, _ = self.config()
        originals = [request(f"base-{i}", tilt=20.+i) for i in range(6)]
        with patch.object(plan, "expanded_menu", return_value=originals), \
             patch.object(plan, "order", side_effect=lambda cp, rows: list(reversed(rows))) as rank:
            rows = plan.comparison_menu(self.layout, libs, {"version": "source-regressor"})
        self.assertEqual(rank.call_args.args[0], self.checkpoint)
        seeds = {r["approach_parent_proposal_id"] for r in rows if "approach_delta" in r}
        self.assertEqual(seeds, {f"base-{i}" for i in range(2, 6)})

    def test_bounds_skip_instead_of_clipping_and_deduplicate(self):
        libs, _ = self.config()
        original = dict(request("edge"), values=[.35, -.35, .38, 8., .05])
        with patch.object(plan, "expanded_menu", return_value=[original]):
            rows = plan.comparison_menu(self.layout, libs, {})
        self.assertEqual(len(rows), len({plan.key(r) for r in rows}))
        local = [r for r in rows if "approach_delta" in r]
        self.assertTrue(local)
        for row in local:
            self.assertGreaterEqual(row["values"][3], 8.)
            self.assertGreaterEqual(row["values"][4], .05)
            delta = row["approach_delta"]
            self.assertEqual(row["values"][3], 8.+delta["tilt_deg"])
            self.assertEqual(row["values"][4], .05+delta["drop_gap_m"])

    def test_changed_selector_binding_is_rejected_at_menu_time(self):
        libs, _ = self.config()
        libs["approach_menu_checkpoint_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "authenticated"):
            plan.comparison_menu(self.layout, libs, {})

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Pinned native planner imports require torch")
    def test_menu_dispatch_is_explicit_and_old_expanded_hook_unchanged(self):
        from sircus.mjlab import neural_shot_plan
        libs, _ = self.config()
        with patch.object(plan, "comparison_menu", return_value=[{"new": True}]) as hooked:
            self.assertEqual(neural_shot_plan.menu(self.layout, libs, {}), [{"new": True}])
            hooked.assert_called_once()
        with patch("sircus.mjlab.expanded_selector_plan.expanded_menu", return_value=[{"old": True}]) as old, \
             patch.object(plan, "comparison_menu", side_effect=AssertionError("Opt-in only")):
            self.assertEqual(neural_shot_plan.menu(self.layout, self.libs, {}), [{"old": True}])
            old.assert_called_once()
        with self.assertRaises(ValueError):
            neural_shot_plan.menu(self.layout, dict(self.libs, approach_menu_version="unknown"), {})


class RankingTests(unittest.TestCase):
    def test_prediction_preserves_diagnostics_and_only_changes_rank_not_native_scoring(self):
        original = dict(score_m=.02, predicted_route="direct", direct={"rim_clearance_m": .02},
                        table={"rim_clearance_m": .01}, contact_ranking_version=plan.CONTACT_VERSION)
        assessment = dict(original_prediction=deepcopy(original), version=plan.APPROACH_VERSION,
                          score_m=-.003, predicted_route="table", routes={"table": {"status": "sampled_overlap"}},
                          rejects_candidates=False, modifies_native_scoring=False, uses_observed_flight=False)
        ranker = object.__new__(plan.ApproachFlightRanker)
        ranker.bucket_approach = Mock()
        ranker.bucket_approach.assess.return_value = assessment
        req = request()
        with patch.object(plan.BoardFirstFlightRanker, "predict", return_value=original):
            result = ranker.predict(req)
        ranker.bucket_approach.assess.assert_called_once_with(req, ranker, prediction=original)
        self.assertEqual(original["score_m"], .02)
        self.assertEqual(result["original_flight_score_m"], .02)
        self.assertEqual(result["score_m"], -.003)
        self.assertEqual(result["predicted_route"], "table")
        self.assertNotIn("original_prediction", result["bucket_approach"])
        self.assertEqual(result["direct"], original["direct"])
        self.assertFalse(result["bucket_approach"]["rejects_candidates"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Pinned native planner imports require torch")
    def test_ranker_factory_opt_in_cannot_replace_unflagged_current(self):
        from sircus.mjlab import clearance_expansion_plan
        config = dict(scene={}, calibration={}, assets="/assets", rule={})
        with patch.dict(clearance_expansion_plan._rankers, {}, clear=True), \
             patch.object(clearance_expansion_plan, "FlightRanker") as current, \
             patch.object(plan, "ApproachFlightRanker") as updated:
            self.assertIs(clearance_expansion_plan._ranker(config), current.return_value)
            current.assert_called_once_with({}, {}, Path("/assets"), {})
            updated.assert_not_called()
            configured = dict(config, approach_ranking_version=plan.VERSION, contact_ranking_version=plan.CONTACT_VERSION)
            self.assertIs(clearance_expansion_plan._ranker(configured), updated.return_value)
            updated.assert_called_once_with({}, {}, Path("/assets"), {})
            self.assertIs(clearance_expansion_plan._ranker(config), current.return_value)
        for flags in (dict(approach_ranking_version="unknown"), dict(approach_ranking_version=plan.VERSION)):
            with self.assertRaises(ValueError):
                clearance_expansion_plan._ranker(dict(config, **flags))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Pinned native planner imports require torch")
    def test_inherited_planner_checks_setup_before_approach_and_uses_corrected_early_rule(self):
        from sircus.mjlab.tests.test_fast_planning_plan import SelectionTests
        fixture = SelectionTests()
        fixture.setUp()
        rows = [fixture.row(i, .02) for i in range(3)]
        ranker = object.__new__(plan.ApproachFlightRanker)
        ranker.bucket_approach = Mock()
        ranker.bucket_approach.assess.side_effect = lambda r, _, **kw: dict(
            score_m=-.003 if r["id"] == rows[1]["id"] else .006,
            predicted_route="direct", original_prediction=kw["prediction"])
        fixture.ranker = ranker
        def prepare(batch, _):
            return ([], [{"rejection": "IK"}]) if batch[0]["id"] == rows[0]["id"] else (batch, [])
        with patch.object(plan.BoardFirstFlightRanker, "predict", return_value=dict(score_m=.02)):
            snapshots, checked, _, _, reason = fixture.run_plan("warm_early", rows, prepare=prepare)
        self.assertEqual(checked.call_count, 3)
        self.assertEqual(reason, "early_good_shot")
        self.assertIsNone(snapshots[0]["choice"]["request"])
        self.assertIsNotNone(snapshots[1]["choice"]["request"], "Predicted contact stays eligible")
        self.assertTrue(snapshots[1]["choice"]["predicted_flight_risk"])
        self.assertEqual(snapshots[-1]["choice"]["request"]["proposal_id"], rows[2]["id"])
        assessed = [call.args[0]["id"] for call in ranker.bucket_approach.assess.call_args_list]
        self.assertEqual(assessed, [rows[1]["id"], rows[2]["id"]], "Admitted predictions cached once per decision")

    def test_wrapper_reuses_parent_deadline_and_worker_lifecycle(self):
        cp = checkpoint()
        libs = dict(expanded_menu_version=plan.EXPANDED_MENU_VERSION, baseline_libraries={}, expanded_sources=[])
        source, history = {"source": True}, []
        kwargs = dict(reference="reference", assets="assets", directory="directory", engine_kwargs={},
                      budget_s=30., startup_timeout_s=90., reference_path_budget_s=5.)
        with patch.object(plan.StalledReferencePlanner, "__init__", return_value=None) as parent:
            plan.ApproachComparisonPlanner("warm_early", libs, source, history, {}, cp, **kwargs)
        self.assertEqual(parent.call_args.kwargs, kwargs)
        args = parent.call_args.args
        self.assertEqual(args[0], "warm_early")
        self.assertIs(args[2], source)
        self.assertIs(args[3], history)
        self.assertIs(args[5], cp)
        self.assertEqual(args[1]["approach_menu_checkpoint"], cp)
        self.assertEqual(args[4]["approach_ranking_version"], plan.VERSION)
        self.assertIs(plan.ApproachComparisonPlanner.plan, plan.StalledReferencePlanner.plan)


if __name__ == "__main__":
    unittest.main()
