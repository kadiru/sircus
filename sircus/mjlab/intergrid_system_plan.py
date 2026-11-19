"""Prepare and freeze a one-drop comparison with a setup-only fallback."""
from __future__ import annotations

from collections import Counter

from .intergrid_flight_ranking_study import select_requests as select_ranked_requests
from .intergrid_route_plan import PER_ROUTE_LIMIT as FALLBACK_PER_ROUTE
from .intergrid_route_plan import ROUTES, collection_geometry
from .intergrid_route_selector_plan import test_proposals
from .intergrid_route_study import PER_ROUTE_TEST as BASE_PER_ROUTE
from .parallel_capture_grid_workers import prepare_proposals


METHODS = ("physics_ranker", "setup_fallback")
ROUTE_LIBRARIES = dict(zip(ROUTES, ("original", "expanded"), strict=True))


def base_proposals(layout, shot_libraries, checkpoint):
    """Use the established specialist proposals, up to four per route."""
    phase = layout["split"]
    proposals = [dict(row, phase=phase, role=phase) for row in test_proposals(layout, shot_libraries, checkpoint)]
    _check_proposals(layout, proposals, BASE_PER_ROUTE)
    return proposals


def _geometry(proposal):
    return (
        *proposal["values"],
        proposal["board_heading_offset_deg"] % 360,
        proposal["board_twist_deg"] % 360,
    )


def _check_proposals(layout, proposals, limit):
    phase = layout["split"]
    counts = Counter(row["intended_route"] for row in proposals)
    if (phase not in {"train", "test"} or len({row["id"] for row in proposals}) != len(proposals)
            or counts.keys() - set(ROUTES) or any(count > limit for count in counts.values())
            or any((row["context_id"], row["bucket_xy_m"], row["phase"], row["role"], row["method"])
                   != (layout["id"], layout["bucket_xy_m"], phase, phase, row["intended_route"])
                   for row in proposals)):
        raise ValueError("Unexpected proposal identity, route budget or layout")


def fallback_proposals(layout, shot_libraries, base_proposals, base_admitted):
    """Transfer saved geometry only when the established shortlist admits no shot."""
    _check_proposals(layout, base_proposals, BASE_PER_ROUTE)
    if base_admitted:
        return []
    sources = {route: shot_libraries[library][route] for route, library in ROUTE_LIBRARIES.items()}
    seen = {_geometry(row) for row in base_proposals}
    fallback = []
    for row in collection_geometry(layout, sources, phase=layout["split"]):
        key = _geometry(row)
        if key in seen:
            continue
        seen.add(key)
        route = row["intended_route"]
        fallback.append(dict(row, id=f"{row['id']}-setup-fallback", method=route))
    _check_proposals(layout, fallback, FALLBACK_PER_ROUTE)
    return fallback


def _prepare(proposals):
    admitted, records = prepare_proposals(proposals, True)
    return dict(proposals=proposals, admitted=admitted, records=records)


def prepare_layout(layout, base, shot_libraries):
    """Run the two preparation stages in one initialized CPU worker."""
    base_stage = _prepare(base)
    fallback = fallback_proposals(layout, shot_libraries, base, base_stage["admitted"])
    return dict(base=base_stage, fallback=_prepare(fallback))


def _rank(layout, stage, history, ranker):
    _, selection = select_ranked_requests(layout, stage["proposals"], stage["admitted"], history, ranker)
    return selection


def select_requests(layout, plan, history, ranker):
    """Keep the baseline command identical whenever it is available."""
    base, fallback = plan["base"], plan["fallback"]
    baseline = _rank(layout, base, history, ranker)
    method = METHODS[1]
    identifier = f"{layout['id']}-{method}"
    if baseline["request"] is not None:
        if any(fallback.values()):
            raise ValueError("Fallback preparation is allowed only after zero base admissions")
        selected = baseline
        pool = "base"
    else:
        selected = _rank(layout, fallback, history, ranker)
        pool = "fallback"
    request = selected["request"]
    chosen = dict(selected, id=identifier, method=method, candidate_pool=pool,
                  request=None if request is None else dict(request, id=identifier, method=method))
    return [dict(baseline, candidate_pool="base"), chosen]
