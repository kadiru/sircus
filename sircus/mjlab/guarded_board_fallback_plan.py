"""Conservative board-pose search after the original hybrid finishes naturally.

The original hybrid keeps its full shared decision deadline. This wrapper adds
no cancellation points or soft time boundary. Only unused time after its normal
return can search the unchanged v1 geometry menu. A replacement must have at
least 5 mm predicted arm clearance and a strictly better finite flight score.
All native labels remain evaluation-only; predicted clearance is not a proof of
capture. Immutable publications preserve the original checked choice on timeout.
"""
from __future__ import annotations

import copy
import math
import time

from .board_fallback_plan import (
    RULE as V1_RULE, VERSION as MENU_VERSION, arm_margin, fallback_menu,
    initialize_worker, needs_board_fallback,
)
from .fast_planning_plan import ReusablePlanner, CachedRanker, early_eligible
from .planning_budget_plan import _wrap_choice

VERSION = "sircus-guarded-board-pose-fallback-v2"
RULE = {key: copy.deepcopy(value) for key, value in V1_RULE.items()
        if key not in {"primary_prefix_s", "selection", "version"}}
RULE.update(
    version=VERSION, geometry_menu_version=MENU_VERSION,
    original_search="Unchanged hybrid natural completion or the shared absolute deadline; no soft handoff",
    replacement_arm_clearance_m=.005,
    selection="Finite selected-route arm clearance >= 5 mm and finite score strictly above the finite incumbent score; otherwise preserve incumbent",
)


def _finite(value):
    return float(value) if type(value) in (int, float) and math.isfinite(value) else None


def replacement_decision(candidate, incumbent):
    """Auditable decision based only on the unchanged predicted flight geometry."""
    candidate, incumbent = candidate or {}, incumbent or {}
    margin = arm_margin({"prediction": candidate})
    new, old = _finite(candidate.get("score_m")), _finite(incumbent.get("score_m"))
    if margin is None:
        reason = "unknown_arm_clearance"
    elif margin < RULE["replacement_arm_clearance_m"]:
        reason = "insufficient_arm_clearance"
    elif new is None or old is None:
        reason = "nonfinite_score_comparison"
    elif new <= old:
        reason = "not_strictly_better"
    else:
        reason = "clearance_qualified_improvement"
    return dict(accepted=reason == "clearance_qualified_improvement", reason=reason,
                predicted_route=candidate.get("predicted_route"), arm_clearance_m=margin,
                minimum_arm_clearance_m=RULE["replacement_arm_clearance_m"],
                candidate_score_m=new, incumbent_score_m=old)


def qualifies_replacement(candidate, incumbent):
    return replacement_decision(candidate, incumbent)["accepted"]


def incremental_guarded_board_fallback_plan(layout, method, *common_args,
                                          publish, deadline, profile):
    from . import parallel_capture_grid_workers as workers
    from . import stalled_reference_plan as hybrid
    from .board_pose_recovery_study import nominal_prediction, wrist_screen, diversified
    from .clearance_expansion_plan import _ranker, needs_expansion
    from .benchmark_verify import digest
    if method != "warm_early":
        raise ValueError("Guarded board fallback requires the warm_early selector")
    state = dict(version=VERSION, protocol=copy.deepcopy(RULE), triggered=False, trigger=None,
        original_completion=None, original_interrupted_by_fallback=False,
        baseline_choice=None, bounds=None, proposals=[], nominal_predictions={}, wrist_screen={},
        check_order=[], records=[], admitted=[], predictions={}, replacement_decisions=[],
        selected_fallback_candidate_id=None, completed=False)
    latest = None

    def original_publish(snapshot):
        nonlocal latest
        accepted = publish(dict(snapshot, board_fallback=copy.deepcopy(state)))
        if accepted:
            latest = copy.deepcopy(snapshot)
        return accepted

    # No catch, callback interruption, gate replacement, or shorter deadline.
    reason = hybrid.incremental_stalled_reference_plan(layout, method, *common_args,
        publish=original_publish, deadline=deadline, profile=profile)
    if latest is None or reason == "deadline" or time.perf_counter() >= deadline:
        return reason
    base = copy.deepcopy(latest["choice"])
    state["original_completion"] = dict(returned_naturally=True, return_reason=reason,
        elapsed_s=time.perf_counter()-profile.started, interrupted_by_fallback=False,
        choice_sha256=digest(base))
    best = copy.deepcopy(base)

    def snapshot(stage):
        value = copy.deepcopy(latest)
        value.update(active_stage=stage, choice=copy.deepcopy(best), board_fallback=copy.deepcopy(state))
        value["completed_proposal_ids"] += [r["proposal_id"] for r in state["records"]]
        return value

    if (not needs_board_fallback(base)
            or deadline-time.perf_counter() < RULE["minimum_fallback_s"]):
        state["completed"] = True
        publish(snapshot("early_good_shot" if reason == "early_good_shot" else "finished"))
        return reason
    state["baseline_choice"] = base
    state["triggered"] = True
    state["trigger"] = dict(reason="risky_original_natural_completion",
        elapsed_s=time.perf_counter()-profile.started, arm_clearance_m=arm_margin(base),
        baseline_choice_sha256=digest(base), selection_uses_flight_labels=False)
    if not publish(snapshot("board_fallback_triggered")):
        return "completed"

    engine, ranker_config = workers._engine, common_args[3]
    state["bounds"] = dict(low=list(map(float, engine.adapter.resolver.solver.low)),
                           high=list(map(float, engine.adapter.resolver.solver.high)))
    with profile.stage("physics_ranking"):
        ranker = CachedRanker(_ranker(ranker_config))
    with profile.stage("menu"):
        state["proposals"] = fallback_menu(layout, base["request"],
            state["bounds"]["low"], state["bounds"]["high"])
    eligible = []
    with profile.stage("physics_ranking"):
        for row in state["proposals"]:
            if time.perf_counter() >= deadline:
                return "completed"
            screen = wrist_screen(ranker_config["scene"], row, engine.adapter.resolver.models)
            state["wrist_screen"][row["id"]] = screen
            if screen["passed"]:
                state["nominal_predictions"][row["id"]] = nominal_prediction(
                    ranker.ranker, ranker_config["scene"], row)
                eligible.append(row)
    shortlist = diversified(eligible, state["nominal_predictions"], RULE["full_checks"])
    state["check_order"] = [r["id"] for r in shortlist]
    if not publish(snapshot("board_fallback_menu")):
        return "completed"
    rejected = []

    def prepare(proposal, multistart=False):
        nonlocal best
        if time.perf_counter() >= deadline:
            return False
        with profile.stage("arm_setup_checks"):
            added, records = engine.prepare([proposal], one_per_setting=not multistart)
        if engine.adapter.env is not None:
            raise RuntimeError("CPU board-pose search allocated a native environment")
        if time.perf_counter() >= deadline:
            return False
        decisions = []
        with profile.stage("physics_ranking"):
            predictions = {r["id"]: ranker.predict(r) for r in added}
            for request in added:
                prediction = predictions[request["id"]]
                decision = replacement_decision(prediction, best.get("prediction"))
                decisions.append(dict(request_id=request["id"], **decision))
                if decision["accepted"]:
                    best = copy.deepcopy(base)
                    best.update(prediction=prediction, candidate_pool="guarded_board_pose_fallback",
                        intended_route=request["intended_route"],
                        available_routes=sorted(set(base.get("available_routes", [])) | {request["intended_route"]}),
                        request=copy.deepcopy(request))
                    best = _wrap_choice(best, method)
                    best["predicted_flight_risk"] = needs_expansion(best)
                    state["selected_fallback_candidate_id"] = request["id"]
        if time.perf_counter() >= deadline:
            return False
        state["records"].extend(records)
        state["admitted"].extend(added)
        state["predictions"].update(predictions)
        state["replacement_decisions"].extend(decisions)
        early = early_eligible(best)
        if early:
            state["completed"] = True
        if not publish(snapshot("early_good_shot" if early else "board_fallback_check")):
            return False
        if early:
            return "early_good_shot"
        if not added and not multistart:
            rejected.append(proposal)
        return True

    for row in shortlist:
        status = prepare(row)
        if status == "early_good_shot":
            return status
        if not status:
            return "completed"
    for original in rejected[:RULE["multistart_retry_limit"]]:
        proposal = dict(copy.deepcopy(original), id=original["id"]+"-multistart",
                        retry_of=original["id"], ik_search="multistart16")
        state["proposals"].append(proposal)
        state["check_order"].append(proposal["id"])
        status = prepare(proposal, True)
        if status == "early_good_shot":
            return status
        if not status:
            return "completed"
    state["completed"] = True
    publish(snapshot("finished"))
    return "completed"


class GuardedBoardFallbackPlanner(ReusablePlanner):
    """Same worker lifecycle, full IK/path checks, and absolute decision budget."""
    def __init__(self, method, *args, **kwargs):
        if method != "warm_early":
            raise ValueError("Guarded board fallback requires warm_early")
        super().__init__(method, *args, **kwargs)
        self.worker.function = incremental_guarded_board_fallback_plan
        self.worker.initializer = initialize_worker

    def plan(self, layout):
        from .stalled_reference_plan import read_receipt
        started = time.perf_counter()
        result = super().plan(layout)
        result["stalled_reference"] = read_receipt(self.directory, result)
        result["request_wall_s"] = time.perf_counter()-started
        return result
