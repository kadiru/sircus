"""Opt-in ranking: post-board ball contacts are allowed, arm paths still checked.

The ballistic approximation models direct and one-table-rebound flight only.
It does not claim to predict robot/floor rebounds. Arm-clearance predictions
remain diagnostic; they no longer automatically lower this task's shot score.
"""
from __future__ import annotations

import copy
import math

from .intergrid_flight_ranker import FlightRanker

VERSION = "sircus-board-first-flight-rank-v1"


def permitted_contact_score(prediction, *, table_edge_margin_m=None):
    result = copy.deepcopy(prediction)
    result["historical_clearance_score_m"] = result.get("score_m")
    scores = {}
    for route in ("direct", "table"):
        row = result.get(route) or {}
        value, clearance = row.get("rim_clearance_m"), row.get("arm_clearance_m")
        # A missing arm bound means the old approximation had no supported
        # rim-time horizon. Permit a negative bound, not an unknown horizon.
        if all(type(x) in (int, float) and math.isfinite(x) for x in (value, clearance)):
            if route == "direct":
                scores[route] = value
            elif type(table_edge_margin_m) in (int, float) and math.isfinite(table_edge_margin_m):
                scores[route] = min(value, table_edge_margin_m)
    if scores:
        chosen = max(scores, key=scores.__getitem__)
        result.update(predicted_route=chosen, score_m=scores[chosen])
    else:
        result.update(predicted_route=None, score_m=None)
    result.update(contact_ranking_version=VERSION,
                  table_edge_margin_m=table_edge_margin_m,
                  prediction_scope="Direct/one-table ballistic rim heuristic; later arm contacts allowed but their rebounds are not modeled")
    return result


class BoardFirstFlightRanker(FlightRanker):
    def predict(self, request):
        prediction = super().predict(request)
        impact = prediction.get("table_contact_m")
        margin = None if impact is None else float(min(
            self.table_half[i] - abs(impact[i] - self.table_center[i]) for i in range(2)))
        return permitted_contact_score(prediction, table_edge_margin_m=margin)


def run_job(layout, variant, libraries, source_checkpoint, history, ranker_config,
            neural_checkpoint, **kwargs):
    """Same bounded pre-IK selector for both frozen checkpoints; rename receipt."""
    from .planning_budget_plan import run_budget_job, _wrap_choice

    if variant not in {"original", "retrained"}:
        raise ValueError("Unknown frozen selector variant")
    config = dict(ranker_config, contact_ranking_version=VERSION)
    result = run_budget_job(layout, "neural_selector", libraries, source_checkpoint, history,
                            config, neural_checkpoint, **kwargs)
    result["method"] = variant
    result["choice"] = _wrap_choice(result["choice"], variant)
    result["choices"] = [result["choice"]]
    return result
