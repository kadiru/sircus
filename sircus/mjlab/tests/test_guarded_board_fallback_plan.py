import copy
from contextlib import nullcontext
import hashlib
import json
import math
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from sircus.mjlab import guarded_board_fallback_plan as module


def prediction(margin=.02, score=.01, route="direct"):
    return dict(score_m=score, predicted_route=route, **{route: dict(arm_clearance_m=margin)})


class ReplacementRuleTests(unittest.TestCase):
    def test_clearance_boundary_is_inclusive_and_selected_route_specific(self):
        old = prediction(score=-.03)
        self.assertTrue(module.qualifies_replacement(prediction(margin=.005, score=-.02), old))
        self.assertFalse(module.qualifies_replacement(prediction(margin=.004999, score=.1), old))
        candidate = prediction(margin=-.01, score=.1, route="table")
        candidate["direct"] = dict(arm_clearance_m=.1)
        self.assertFalse(module.qualifies_replacement(candidate, old))

    def test_negative_or_unknown_clearance_never_replaces_parent(self):
        for margin in (-.024495, 0., None, math.nan, math.inf, True):
            with self.subTest(margin=margin):
                self.assertFalse(module.qualifies_replacement(prediction(margin=margin, score=.5), prediction()))
        self.assertFalse(module.qualifies_replacement(dict(score_m=.5), prediction()))

    def test_both_scores_must_be_finite_and_improvement_strict(self):
        for new, old in ((.01,.01), (.009,.01), (None,.01), (math.inf,.01),
                         (math.nan,.01), (True,.01), (.2,None), (.2,math.nan), (.2,True)):
            with self.subTest(new=new, old=old):
                self.assertFalse(module.qualifies_replacement(prediction(score=new), prediction(score=old)))
        self.assertTrue(module.qualifies_replacement(prediction(score=.01001), prediction(score=.01)))

    def test_old_regression_is_rejected_despite_better_overall_score(self):
        old = prediction(margin=-.009162, score=-.027068)
        candidate = prediction(margin=-.024495, score=-.024495)
        decision = module.replacement_decision(candidate, old)
        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["reason"], "insufficient_arm_clearance")

    def test_protocol_keeps_geometry_menu_and_removes_soft_boundary(self):
        self.assertNotIn("primary_prefix_s", module.RULE)
        for key in ("radial_offsets_m", "tangent_offsets_m", "heading_offsets_deg",
                    "twist_offsets_deg", "tuned_proposals", "full_checks", "multistart_retry_limit"):
            self.assertEqual(module.RULE[key], module.V1_RULE[key])
        self.assertEqual(module.RULE["replacement_arm_clearance_m"], .005)


class GuardedLifecycleTests(unittest.TestCase):
    def exercise(self, *, new_score=.02, new_margin=.02, source_margin=-.02,
                 original_finish=16.64, productive_original=False, original_reason="completed",
                 setup_finish=None, ranking_finish=None, reject_replacement_publication=False,
                 admitted=True):
        import sircus.mjlab as package
        now = [0.]
        published, checked, original_deadlines = [], [], []
        layout = dict(id="development-23", split="test", stratum="test", bucket_xy_m=[1.,0.])
        request = dict(id="original-checked", values=[.6,0.,.45,24.,.2], bucket_xy_m=[1.,0.],
                       selected_q10=[.1]*6, selected_q14=[.2]*6, intended_route="direct",
                       board_heading_offset_deg=0., board_twist_deg=0.)
        base = dict(request=request, prediction=prediction(margin=source_margin, score=-.02),
                    id="development-23-warm_early", context_id=layout["id"], bucket_xy_m=[1.,0.],
                    method="warm_early", intended_route="direct", available_routes=["direct"])
        original_final = copy.deepcopy(base)
        if productive_original:
            original_final["prediction"] = prediction(margin=.0063044, score=.0063044)
        proposal = dict(request, id="geometry-0", context_id=layout["id"], group="position_heading")
        gate_call = mock.Mock(return_value=dict(passed=True))
        gate = SimpleNamespace(check=gate_call)

        def prepare(rows, **kwargs):
            checked.append((rows[0]["id"], kwargs))
            if setup_finish is not None:
                now[0] = setup_finish
            row = dict(rows[0], id=rows[0]["id"]+"-b0", proposal_id=rows[0]["id"])
            return ([row] if admitted else []), [dict(proposal_id=rows[0]["id"], admitted=int(admitted))]

        engine = SimpleNamespace(adapter=SimpleNamespace(env=None,
            resolver=SimpleNamespace(models={}, solver=SimpleNamespace(gate=gate, low=[-2.]*5, high=[90.]*5))),
            prepare=prepare)
        workers = SimpleNamespace(_engine=engine)

        def original_snapshot(value, stage):
            return dict(layout=layout, method="warm_early", active_stage=stage, choice=copy.deepcopy(value),
                completed_proposal_ids=["original-0"], ranked_proposal_ids=["original-0"], generated={"menu":[]},
                plan={k:dict(proposals=[], admitted=[], records=[]) for k in ("base","fallback","neural","branches")},
                stalled_reference_snapshot={"engine_mode":"reference"})

        def hybrid_call(layout, method, *args, publish, deadline, profile):
            original_deadlines.append(deadline)
            now[0] = 12.47 if productive_original else 1.
            publish(original_snapshot(base, "neural"))
            # Any 12-second wrapper would interrupt this productive call.
            now[0] = original_finish
            self.assertIs(gate.check, gate_call)
            gate.check("unchanged-original-gate")
            if original_reason != "deadline":
                publish(original_snapshot(original_final,
                    "early_good_shot" if productive_original else "finished"))
            return "early_good_shot" if productive_original else original_reason

        def predict(row):
            if ranking_finish is not None:
                now[0] = ranking_finish
            return prediction(margin=new_margin, score=new_score)

        ranker = SimpleNamespace(predict=predict)
        recovery = SimpleNamespace(wrist_screen=lambda *a:dict(passed=True),
            nominal_prediction=lambda *a:dict(score_m=.02), diversified=lambda rows, predictions, limit:rows[:limit])
        hybrid = SimpleNamespace(incremental_stalled_reference_plan=hybrid_call)
        clearance = SimpleNamespace(_ranker=lambda config:ranker,
            needs_expansion=lambda c:c["prediction"]["score_m"] <= 0)
        prefix = "sircus.mjlab."
        modules = {prefix+"parallel_capture_grid_workers":workers, prefix+"stalled_reference_plan":hybrid,
            prefix+"board_pose_recovery_study":recovery, prefix+"clearance_expansion_plan":clearance,
            prefix+"benchmark_verify":SimpleNamespace(digest=lambda value:hashlib.sha256(
                json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest())}

        def publish(row):
            if reject_replacement_publication and row["board_fallback"]["selected_fallback_candidate_id"]:
                now[0] = 30.
            if now[0] >= 30.:
                return False
            published.append(copy.deepcopy(row))
            return True

        profile = SimpleNamespace(started=0., stage=lambda name:nullcontext())
        with mock.patch.dict(sys.modules, modules), \
             mock.patch.object(package,"parallel_capture_grid_workers",workers,create=True), \
             mock.patch.object(package,"stalled_reference_plan",hybrid,create=True), \
             mock.patch.object(module.time,"perf_counter",lambda:now[0]), \
             mock.patch.object(module,"fallback_menu",lambda *a:[copy.deepcopy(proposal)]), \
             mock.patch.object(module,"CachedRanker",lambda ranker:SimpleNamespace(ranker=ranker,predict=ranker.predict)):
            reason = module.incremental_guarded_board_fallback_plan(layout,"warm_early",{},None,None,{"scene":{}},None,
                publish=publish,deadline=30.,profile=profile)
        self.assertEqual(original_deadlines,[30.])
        self.assertIs(gate.check,gate_call)
        return reason,published,checked,original_final

    def test_productive_original_reaches_16_6_second_good_shot_without_interruption(self):
        reason,rows,checked,original = self.exercise(productive_original=True)
        self.assertEqual(reason,"early_good_shot")
        self.assertEqual(checked,[])
        self.assertEqual(rows[-1]["choice"],original)
        state = rows[-1]["board_fallback"]
        self.assertFalse(state["triggered"])
        self.assertFalse(state["original_interrupted_by_fallback"])
        self.assertEqual(state["original_completion"]["elapsed_s"],16.64)
        self.assertTrue(state["original_completion"]["returned_naturally"])

    def test_natural_risky_completion_can_use_remaining_time(self):
        reason,rows,checked,original = self.exercise()
        self.assertEqual(reason,"early_good_shot")
        self.assertTrue(checked)
        state = rows[-1]["board_fallback"]
        self.assertEqual(state["baseline_choice"],original)
        self.assertEqual(state["trigger"]["reason"],"risky_original_natural_completion")
        self.assertEqual(state["trigger"]["elapsed_s"],16.64)
        self.assertTrue(state["replacement_decisions"][0]["accepted"])
        self.assertEqual(rows[-1]["choice"]["prediction"]["score_m"],.02)

    def test_unknown_or_negative_candidate_clearance_preserves_original(self):
        for margin in (-.024495,None,math.nan,math.inf,True):
            with self.subTest(margin=margin):
                _,rows,checked,original = self.exercise(new_score=.9,new_margin=margin)
                self.assertTrue(checked)
                self.assertEqual(rows[-1]["choice"],original)
                self.assertIsNone(rows[-1]["board_fallback"]["selected_fallback_candidate_id"])
                self.assertFalse(rows[-1]["board_fallback"]["replacement_decisions"][0]["accepted"])

    def test_tie_preserves_entire_original_choice(self):
        _,rows,_,original = self.exercise(new_score=-.02)
        self.assertEqual(rows[-1]["choice"],original)

    def test_no_admitted_geometry_preserves_original_and_records_checks(self):
        _,rows,checked,original = self.exercise(admitted=False)
        self.assertEqual(rows[-1]["choice"],original)
        self.assertEqual(len(checked),2)  # Initial check plus bounded multistart retry.
        self.assertEqual(len(rows[-1]["board_fallback"]["records"]),2)

    def test_original_deadline_never_starts_a_second_search(self):
        reason,rows,checked,original = self.exercise(original_finish=30.,original_reason="deadline")
        self.assertEqual(reason,"deadline")
        self.assertEqual(checked,[])
        self.assertEqual(rows[-1]["choice"],original)
        self.assertIsNone(rows[-1]["board_fallback"]["original_completion"])

    def test_insufficient_remaining_time_preserves_original(self):
        _,rows,checked,original = self.exercise(original_finish=29.9)
        self.assertEqual(checked,[])
        self.assertEqual(rows[-1]["choice"],original)
        self.assertFalse(rows[-1]["board_fallback"]["triggered"])

    def test_late_setup_and_ranking_cannot_replace_published_original(self):
        for lateness in (dict(setup_finish=31.),dict(ranking_finish=31.)):
            with self.subTest(lateness=lateness):
                _,rows,checked,original = self.exercise(**lateness)
                self.assertTrue(checked)
                self.assertEqual(rows[-1]["choice"],original)
                self.assertEqual(rows[-1]["active_stage"],"board_fallback_menu")
                self.assertEqual(rows[-1]["board_fallback"]["admitted"],[])
                self.assertIsNone(rows[-1]["board_fallback"]["selected_fallback_candidate_id"])

    def test_publication_deadline_race_cannot_overwrite_original(self):
        _,rows,checked,original = self.exercise(reject_replacement_publication=True)
        self.assertTrue(checked)
        self.assertEqual(rows[-1]["choice"],original)
        self.assertEqual(rows[-1]["board_fallback"]["replacement_decisions"],[])
        self.assertEqual(rows[-1]["completed_proposal_ids"],["original-0"])


if __name__ == "__main__":
    unittest.main()
