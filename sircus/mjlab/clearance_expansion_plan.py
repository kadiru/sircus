"""Expand poor-clearance shot menus while preserving the frozen baseline planner."""
from __future__ import annotations

import copy
import math
from time import perf_counter
from pathlib import Path

from .benchmark_verify import digest
from .cell1050_heading_native import BRANCH_RULE, HeadingResolver
from .fixed35_bouncer_native import wrapped_difference
from .fixed35_broad_native import distinct_indices
from .intergrid_candidate_replay_plan import command_key
from .intergrid_flight_ranker import FlightRanker
from .intergrid_route_plan import collection_geometry
from .intergrid_system_plan import ROUTE_LIBRARIES, ROUTES, base_proposals, fallback_proposals
from .intergrid_system_plan import select_requests as baseline_select_requests
from . import parallel_capture_grid_workers as workers

VERSION = "sircus-clearance-expansion-plan-v1"
METHODS = ("current_planner", "clearance_expansion")
STAGES = ("base", "fallback", "expansion", "branches")
GEOMETRY_LIMIT = 32
BRANCH_SETTINGS = 4
MAX_BRANCHES_PER_SETTING = BRANCH_RULE["maximum_admitted_branches"]
MINIMUM_JOINT_DELTA_RAD = BRANCH_RULE["minimum_joint_difference_rad"]
BRANCH_SEED = 2026111703
_SCORE_THRESHOLD_M = 0.
_rankers = {}


def protocol():
    return dict(version=VERSION, methods=list(METHODS), score_trigger_m=_SCORE_THRESHOLD_M,
                geometry_limit=GEOMETRY_LIMIT, branch_settings=BRANCH_SETTINGS,
                branches_per_setting=MAX_BRANCHES_PER_SETTING, branch_seed=BRANCH_SEED,
                minimum_joint_delta_rad=MINIMUM_JOINT_DELTA_RAD,
                branch_starts=BRANCH_RULE["starts_per_arm"], branch_iterations=BRANCH_RULE["iterations"],
                branch_fixed="Exact dropper joints and full board/release Cartesian targets",
                selection="Maximum unchanged physics score; keep current command on ties",
                unresolved="Retain best available command with explicit nonpositive risk; no hidden refusal",
                native_attempts_per_method_per_position=1)


def _geometry(row):
    return (tuple(row["values"]), row["board_heading_offset_deg"] % 360,
            row["board_twist_deg"] % 360)


def _empty():
    return dict(proposals=[], admitted=[], records=[])


def current_selection(layout, plan, history, ranker):
    """Return the exact current interactive setup-only fallback choice."""
    _, selected = baseline_select_requests(layout, {name: plan[name] for name in STAGES[:2]}, history, ranker)
    return selected


def needs_expansion(selection):
    value = (selection.get("prediction") or {}).get("score_m")
    return (selection.get("request") is None or type(value) not in (int, float)
            or not math.isfinite(value) or value <= _SCORE_THRESHOLD_M)


def expansion_proposals(layout, libraries, plan):
    """Try the frozen wider geometry menu even when a risky setup was reachable."""
    sources = {route: libraries[name][route] for route, name in ROUTE_LIBRARIES.items()}
    seen = {_geometry(row) for name in STAGES[:2] for row in plan[name]["proposals"]}
    proposals = []
    for row in collection_geometry(layout, sources, phase=layout["split"]):
        geometry = _geometry(row)
        if geometry in seen:
            continue
        seen.add(geometry)
        proposals.append(dict(row, id=row["id"] + "-clearance-expand", method=row["intended_route"]))
        if len(proposals) == GEOMETRY_LIMIT:
            break
    return proposals


def _all_admitted(plan, stages=STAGES):
    # Equal native commands can occur under different source recipes. Keep the
    # first identity; the current stage precedes all added choices.
    unique = {}
    for name in stages:
        for row in plan[name]["admitted"]:
            unique.setdefault(command_key(row), row)
    return list(unique.values())


def branch_proposals(layout, plan, ranker):
    """Choose four promising poses before observing any native drop outcome."""
    originals = ranker.order(_all_admitted(plan, STAGES[:3]))
    seen, proposals = set(), []
    for row in originals:
        if _geometry(row) in seen:
            continue
        seen.add(_geometry(row))
        proposals.append(dict(
            id=f"{layout['id']}-clearance-board-branch-{len(proposals):02d}",
            context_id=layout["id"], bucket_xy_m=list(layout["bucket_xy_m"]),
            phase=layout["split"], role=layout["split"], method=row["intended_route"],
            intended_route=row["intended_route"], group="clearance_board_branch",
            values=list(row["values"]), board_heading_offset_deg=row["board_heading_offset_deg"],
            board_twist_deg=row["board_twist_deg"], seed_q10=list(row["selected_q10"]),
            seed_q14=list(row["selected_q14"]), source_request_id=row["id"],
            branch_seed=BRANCH_SEED + len(proposals),
        ))
        if len(proposals) == BRANCH_SETTINGS:
            break
    return proposals


class BoardOnlyResolver(HeadingResolver):
    """Use existing full-pose and path checks, varying only the board arm."""
    def __init__(self, resolver, assets, original):
        super().__init__(resolver.base, assets)
        self.original = original

    def _branches(self, first, second, hint, *, prefer_saved=False):
        import numpy as np

        fixed = np.asarray(self.original["selected_q10"])
        board = np.asarray(self.original["selected_q14"])
        # Do not deduplicate before finding the saved joints: a nearby IK
        # solution with a lower residual could otherwise erase the exact seed.
        dropper = [int(i) for i in np.flatnonzero(first["valid"])
                   if np.array_equal(first["q"][i], fixed)][:1]
        alternatives = [j for j in distinct_indices(second)
                        if float(wrapped_difference(second["q"][j], board).max()) >= MINIMUM_JOINT_DELTA_RAD]
        alternatives.sort(key=lambda j: (
            -float(wrapped_difference(second["q"][j][:3], board[:3]).max()),
            -float(wrapped_difference(second["q"][j], board).max()),
            float(second["residual"][j]), j))
        return [(i, j) for i in dropper for j in alternatives]


def _prepare_branches(engine, proposals, originals, assets):
    """Run deterministic multistart IK, restoring all worker state afterwards."""
    import numpy as np

    admitted, records = [], []
    saved_resolver = engine.adapter.resolver
    for proposal in proposals:
        original = originals[proposal["source_request_id"]]
        resolver = BoardOnlyResolver(saved_resolver, assets, original)
        engine.adapter.resolver = resolver
        arms = (resolver.solver.ik.dropper, resolver.solver.ik.board)
        solve = [arm.solve_candidates for arm in arms]
        try:
            for role, arm, implementation in zip((10, 14), arms, solve, strict=True):
                def preserved(targets, initial, *, starts, iterations,
                              role=role, implementation=implementation, **kwargs):
                    return implementation(
                        targets, initial, starts=1 if role == 10 else BRANCH_RULE["starts_per_arm"],
                        iterations=iterations, seed=proposal["branch_seed"],
                        preserve_known=np.ones(len(targets), dtype=bool), **kwargs)
                arm.solve_candidates = preserved
            added, logs = engine.prepare([proposal], one_per_setting=False)
            if any(row["selected_q10"] != original["selected_q10"] for row in added):
                raise RuntimeError("Alternate board branch changed the fixed dropper joints")
            admitted.extend(added)
            records.extend(logs)
        finally:
            for arm, implementation in zip(arms, solve, strict=True):
                arm.solve_candidates = implementation
            engine.adapter.resolver = saved_resolver
    return dict(proposals=proposals, admitted=admitted, records=records)


def _wrap(selection, method):
    result = copy.deepcopy(selection)
    identifier = f"{selection['context_id']}-{method}"
    result.update(id=identifier, method=method)
    if result["request"] is not None:
        result["request"].update(id=identifier, method=method)
    return result


def select_requests(layout, plan, history, ranker):
    """Freeze one command per planner; a nonpositive bound stays visible as risk."""
    baseline = current_selection(layout, plan, history, ranker)
    current = _wrap(baseline, METHODS[0])
    triggered = needs_expansion(baseline)
    if not triggered:
        if any(plan[name][key] for name in STAGES[2:] for key in ("proposals", "admitted", "records")):
            raise ValueError("A positive-clearance baseline must not trigger extra search")
        updated = _wrap(baseline, METHODS[1])
    else:
        pool = _all_admitted(plan)
        baseline_key = command_key(baseline["request"]) if baseline["request"] else None
        def key(row):
            score = ranker.predict(row)["score_m"]
            finite = type(score) in (int, float) and math.isfinite(score)
            return (not finite, -score if finite else 0., command_key(row) != baseline_key, row["id"])
        ordered = sorted(pool, key=key)
        chosen = ordered[0] if ordered else None
        updated = _wrap(baseline, METHODS[1])
        identifier = updated["id"]
        updated.update(
            available_routes=sorted({row["intended_route"] for row in pool}),
            admitted_distinct_commands=len(pool), ranked_candidate_ids=[row["id"] for row in ordered],
            intended_route=chosen["intended_route"] if chosen else None,
            prediction=ranker.predict(chosen) if chosen else None,
            candidate_pool=next((name for name in STAGES if chosen in plan[name]["admitted"]), None),
            request=None if chosen is None else dict(chosen, id=identifier, method=METHODS[1],
                                                     stratum=layout["stratum"]),
        )
    for choice in (current, updated):
        choice["clearance_expansion_triggered"] = triggered
        choice["predicted_flight_risk"] = needs_expansion(choice)
    return [current, updated]


def _ranker(config):
    key = digest(dict(config, assets=str(config["assets"])))
    if key not in _rankers:
        implementation = FlightRanker
        if config.get("contact_ranking_version") is not None:
            from .board_first_selector_plan import BoardFirstFlightRanker, VERSION as contact_version
            if config["contact_ranking_version"] != contact_version:
                raise ValueError("Unknown explicit contact ranking version")
            implementation = BoardFirstFlightRanker
        if config.get("approach_ranking_version") is not None:
            from .approach_comparison_plan import ApproachFlightRanker, VERSION as approach_version, CONTACT_VERSION
            if (config["approach_ranking_version"] != approach_version
                    or config.get("contact_ranking_version") != CONTACT_VERSION):
                raise ValueError("Unknown approach ranking version or contact semantics")
            implementation = ApproachFlightRanker
        _rankers[key] = implementation(config["scene"], config["calibration"], Path(config["assets"]), config["rule"])
    return _rankers[key]


def prepare_layout(layout, libraries, checkpoint, history, ranker_config):
    """Prepare one bucket in an initialized CPU worker without allocating CUDA."""
    engine = workers._engine
    if engine is None:
        raise RuntimeError("Initialize the CPU preparation worker first")
    ranker = _ranker(ranker_config)
    started = perf_counter()
    def stage(proposals):
        admitted, records = workers.prepare_proposals(proposals, True)
        return dict(proposals=proposals, admitted=admitted, records=records)
    base = stage(base_proposals(layout, libraries, checkpoint))
    fallback = stage(fallback_proposals(layout, libraries, base["proposals"], base["admitted"]))
    plan = dict(base=base, fallback=fallback, expansion=_empty(), branches=_empty())
    triggered = needs_expansion(current_selection(layout, plan, history, ranker))
    current_elapsed = perf_counter() - started
    if triggered:
        plan["expansion"] = stage(expansion_proposals(layout, libraries, plan))
        branch = branch_proposals(layout, plan, ranker)
        originals = {row["id"]: row for row in _all_admitted(plan, STAGES[:3])}
        plan["branches"] = _prepare_branches(engine, branch, originals, ranker_config["assets"])
    if engine.adapter.env is not None:
        raise RuntimeError("CPU expansion unexpectedly initialized the native simulation")
    choices = select_requests(layout, plan, history, ranker)
    elapsed = perf_counter() - started
    return dict(layout=layout, plan=plan, trigger=triggered,
                latency_s=dict(zip(METHODS, (current_elapsed, elapsed), strict=True)),
                predictions=[dict(id=row["id"], **ranker.predict(row)) for row in _all_admitted(plan)],
                choices=choices)
