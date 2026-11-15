"""Freeze one direct-only and one route-selecting attempt per bucket position."""
from __future__ import annotations

from collections import Counter
from itertools import product
import math

from .intergrid_library_plan import MODEL
from .intergrid_matched_route_study import outcome
from .intergrid_route_models import propose_route_candidates
from .intergrid_route_plan import ROUTES
from .intergrid_route_study import PER_ROUTE_TEST, source_menu


METHODS = ("direct_only", "route_selector")
GATE_NEIGHBORS = 5
STANDARD_CAPTURES = {"direct", "table", "unclassified_capture"}


def test_proposals(layout, shot_libraries, checkpoint):
    """Prepare one shared direct pool and one expanded table pool."""
    proposals = []
    for route, library in zip(ROUTES, ("original", "expanded"), strict=True):
        sources = [row for row in source_menu(layout, shot_libraries[library]) if row["intended_route"] == route]
        for row in propose_route_candidates(checkpoint, MODEL, layout["bucket_xy_m"], sources,
                                            context_id=layout["id"], limit=PER_ROUTE_TEST):
            proposals.append(dict(row, id=f"{row['id']}-{route}", method=route))
    return proposals


def select_requests(layout, proposals, admitted, route_history):
    """Choose from admitted routes using prior outcomes, before either release."""
    by_id = {row["proposal_id"]: row for row in admitted}
    proposal_ids = {row["id"] for row in proposals}
    if len(by_id) != len(admitted) or len(proposal_ids) != len(proposals) or by_id.keys() - proposal_ids:
        raise ValueError("Expected at most one admitted branch for each known proposal")
    first_admitted = {}
    for proposal in proposals:
        route = proposal["intended_route"]
        if route not in ROUTES or proposal["method"] != route:
            raise ValueError("A proposal must belong to its intended route pool")
        if proposal["id"] in by_id:
            first_admitted.setdefault(route, by_id[proposal["id"]])

    neighbors = sorted(route_history, key=lambda row: (
        math.dist(layout["bucket_xy_m"], row["bucket_xy_m"]), row["id"],
    ))[:GATE_NEIGHBORS]
    if len(neighbors) != GATE_NEIGHBORS:
        raise ValueError("Route selector needs five historically paired positions")
    scores = {route: (1 + sum(site[route + "_capture"] for site in neighbors)) / (2 + len(neighbors))
              for route in ROUTES}
    priority = sorted(ROUTES, key=lambda route: (-scores[route], ROUTES.index(route)))
    slots = []
    for method in METHODS:
        adaptive = method == "route_selector"
        order = priority if adaptive else ["direct"]
        selected = next((first_admitted[route] for route in order if route in first_admitted), None)
        identifier = f"{layout['id']}-{method}"
        slots.append(dict(
            id=identifier, method=method, context_id=layout["id"], bucket_xy_m=layout["bucket_xy_m"],
            stratum=layout["stratum"], intended_route=None if selected is None else selected["intended_route"],
            route_priority=order, neighbor_ids=[row["id"] for row in neighbors] if adaptive else [],
            gate_scores=scores if adaptive else None,
            request=None if selected is None else dict(selected, id=identifier, method=method, stratum=layout["stratum"]),
        ))
    return slots


def summarize(layouts, slots, records):
    """Pair one-shot outcomes, counting every valid standard capture."""
    layout_ids = {layout["id"] for layout in layouts}
    slot_keys = {(slot["context_id"], slot["method"]) for slot in slots}
    if (not layout_ids or len(layout_ids) != len(layouts) or len(slot_keys) != len(slots)
            or slot_keys != set(product(layout_ids, METHODS))):
        raise ValueError("Expected one attempt per layout and method")
    if (len({slot["id"] for slot in slots}) != len(slots)
            or records.keys() != {slot["id"] for slot in slots if slot["request"] is not None}):
        raise ValueError("Expected exactly one observation for each admitted slot")

    counts = {method: Counter() for method in METHODS}
    selected = {method: Counter(dict.fromkeys(ROUTES, 0)) for method in METHODS}
    aligned = Counter()
    by_layout = {}
    for slot in slots:
        method, route = slot["method"], slot["intended_route"]
        actual = outcome(records.get(slot["id"]))
        counts[method][actual] += 1
        if slot["request"] is not None:
            selected[method][route] += 1
            aligned[method] += int(actual == route)
        by_layout[slot["context_id"], method] = (slot, actual)

    methods = {}
    attempts = len(layout_ids)
    for method, labels in counts.items():
        releases = attempts - labels["setup_rejected"]
        captures = sum(labels[label] for label in STANDARD_CAPTURES)
        methods[method] = dict(attempts=attempts, releases=releases, setup_rate=releases / attempts,
                               captures=captures, capture_rate=captures / attempts, outcomes=dict(labels),
                               selected_routes=dict(selected[method]), requested_route_captures=aligned[method])

    pair_labels = {(True, True): "both", (True, False): "direct_only",
                   (False, True): "route_selector_only", (False, False): "neither"}
    paired = {stage: dict.fromkeys(pair_labels.values(), 0) for stage in ("setup", "capture")}
    shared = dict(attempts=0, outcome_disagreements=0)
    for identity in layout_ids:
        (direct, direct_label), (selector, selector_label) = (by_layout[identity, method] for method in METHODS)
        direct_request, selector_request = direct["request"], selector["request"]
        paired["setup"][pair_labels[direct_request is not None, selector_request is not None]] += 1
        paired["capture"][pair_labels[direct_label in STANDARD_CAPTURES, selector_label in STANDARD_CAPTURES]] += 1
        if selector["intended_route"] == "direct":
            if direct_request is None or dict(direct_request, id=selector["id"], method=selector["method"]) != selector_request:
                raise ValueError("Both methods must use the same admitted direct command")
            shared["attempts"] += 1
            shared["outcome_disagreements"] += int(direct_label != selector_label)
    return dict(methods=methods, paired=paired, same_direct_command=shared)
