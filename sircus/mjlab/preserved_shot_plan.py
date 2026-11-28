"""Preserve a current planner admission when the expanded search declines.

This is an outcome-blind portfolio rule, not a capture guarantee. Both planners
retain their complete raw evidence and independent 30-second CPU decision lanes.
"""
from __future__ import annotations

import copy
import math

from .benchmark_verify import digest
from .planning_budget_plan import _wrap_choice
from .stalled_reference_plan import request_mode

VERSION = "sircus-preserved-current-shot-v1"
RAW_METHODS = ("current", "expanded")
METHODS = ("current", "protected")


def select_pair(current, expanded, timing):
    """Choose only from authenticated, complete planner returns; never re-plan."""
    sites = dict(current=current, expanded=expanded)
    if current["layout"] != expanded["layout"]:
        raise ValueError("A preserved choice requires identical target layouts")
    for method, site in sites.items():
        if site["status"] not in {"completed", "deadline"}:
            raise ValueError("Worker errors must not be disguised as setup declines")
        choice = site["choice"]
        if choice["context_id"] != site["layout"]["id"] or choice["bucket_xy_m"] != site["layout"]["bucket_xy_m"]:
            raise ValueError("Choice does not belong to its target")
        if choice["request"] is not None:
            request_mode(choice["request"])
            for prefix in ("context_resolution", "prepared_connection"):
                if not choice["request"].get(prefix + "_sha256"):
                    raise ValueError("Checked choice requires immutable path and pose proofs")
    start, end = timing["pair_started_s"], timing["pair_finished_s"]
    if not all(type(v) in (float, int) and math.isfinite(v) for v in (start, end)) or end < start:
        raise ValueError("Invalid paired wall timing")
    for method in RAW_METHODS:
        lane = timing["lanes"][method]
        if not start <= lane["started_s"] <= lane["finished_s"] <= end:
            raise ValueError("Lane interval must be inside paired request interval")
    source = "expanded" if expanded["choice"]["request"] is not None else "current"
    return dict(version=VERSION, context_id=current["layout"]["id"],
        selected_source=source, fallback_used=source == "current",
        source_choice_sha256={m:digest(s["choice"]) for m,s in sites.items()},
        source_request_sha256={m:digest(s["choice"]["request"]) for m,s in sites.items()},
        source_has_request={m:s["choice"]["request"] is not None for m,s in sites.items()},
        timing=dict(copy.deepcopy(timing), pair_wall_s=end-start,
            reserved_worker_window_s=sum(s["decision_latency_s"] for s in sites.values()),
            scope="Two concurrent CPU lanes, each with a 30-second decision allowance. Pair wall includes initialization, decision, stopping and receipt checks. Summed lane decision windows are not measured CPU consumption."))


def slots_for_pair(current, expanded, selection, repeats=1):
    """Administrative release IDs change; all physical commands and proofs stay exact."""
    if type(repeats) is not int or repeats < 1:
        raise ValueError("Positive integral native repetition count required")
    sites = dict(current=current, expanded=expanded)
    if selection["version"] != VERSION or selection["selected_source"] != (
            "expanded" if expanded["choice"]["request"] is not None else "current"):
        raise ValueError("Preserved source differs from the frozen rule")
    rows = []
    for method in METHODS:
        source = "current" if method == "current" else selection["selected_source"]
        choice = sites[source]["choice"]
        if digest(choice) != selection["source_choice_sha256"][source] or digest(choice["request"]) != selection["source_request_sha256"][source]:
            raise ValueError("Source choice changed after selection")
        for repeat in range(repeats):
            slot = _wrap_choice(choice, method)
            identifier = f"{choice['context_id']}-{method}-r{repeat:02d}"
            slot.update(id=identifier, source_method=source, repeat_index=repeat,
                source_choice_sha256=digest(choice), source_request_sha256=digest(choice["request"]),
                fallback_used=method == "protected" and source == "current")
            if slot["request"] is not None:
                slot["request"]["id"] = identifier
            rows.append(slot)
    return rows


def description():
    return dict(version=VERSION,
        rule="Use expanded's checked request when present; otherwise use current's exact checked request. Decline if neither has one.",
        raw_methods=list(RAW_METHODS), result_methods=list(METHODS),
        uses_native_outcomes=False, capture_guarantee=False,
        physical_request_changes=False, rechecks_or_rescores_fallback=False,
        cpu_lanes=2, decision_budget_per_lane_s=30.,
        timing="Both calls complete before portfolio selection; combined request wall measured directly. Current baseline shares its CPU planning result but receives independent native drops.")
