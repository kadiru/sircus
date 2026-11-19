"""Compare original and refitted route models on the same unseen positions."""
from __future__ import annotations

from collections import Counter
from itertools import product

from .intergrid_matched_route_study import outcome
from .intergrid_route_models import MODELS
from .intergrid_route_plan import ROUTES
from .intergrid_route_study import test_proposals as matched_proposals


GENERATIONS = ("original", "refit")
METHODS = tuple(f"{model}_{generation}" for model, generation in product(MODELS, GENERATIONS))


def test_proposals(layout, library, original, refit):
    """Predict from the same original recipe library and budget in both generations."""
    grouped = {method: [] for method in METHODS}
    for generation, checkpoint in zip(GENERATIONS, (original, refit), strict=True):
        for proposal in matched_proposals(layout, library, checkpoint):
            method = f"{proposal['method']}_{generation}"
            grouped[method].append(dict(proposal, id=f"{proposal['id']}-{generation}", method=method))
    return [proposal for proposals in grouped.values() for proposal in proposals]


def select_requests(layout, proposals, resolutions):
    """Freeze one attempt per method and intended route, including setup declines."""
    by_id = {row["proposal_id"]: row for row in resolutions}
    proposal_ids = {row["id"] for row in proposals}
    if len(by_id) != len(resolutions) or len(proposal_ids) != len(proposals) or by_id.keys() - proposal_ids:
        raise ValueError("Expected at most one admitted branch for each known proposal")
    first_admitted = {}
    for proposal in proposals:
        if proposal["id"] in by_id:
            first_admitted.setdefault((proposal["method"], proposal["intended_route"]), by_id[proposal["id"]])
    slots = []
    for method, route in product(METHODS, ROUTES):
        identifier = f"{layout['id']}-{method}-{route}"
        selected = first_admitted.get((method, route))
        slots.append(dict(
            id=identifier, method=method, context_id=layout["id"], bucket_xy_m=layout["bucket_xy_m"],
            stratum=layout["stratum"], intended_route=route,
            request=None if selected is None else dict(selected, id=identifier, stratum=layout["stratum"]),
        ))
    return slots


def summarize(layouts, slots, records):
    """Count success on the requested route per position and pair each model with its refit."""
    layout_ids = {layout["id"] for layout in layouts}
    slot_keys = {(slot["context_id"], slot["method"], slot["intended_route"]) for slot in slots}
    if (not layout_ids or len(layout_ids) != len(layouts) or len(slot_keys) != len(slots)
            or slot_keys != set(product(layout_ids, METHODS, ROUTES))):
        raise ValueError("Expected exactly one attempt for each position, method and route")
    if (len({slot["id"] for slot in slots}) != len(slots)
            or records.keys() != {slot["id"] for slot in slots if slot["request"] is not None}):
        raise ValueError("Expected exactly one observation per admitted frozen choice")

    counts = {method: {route: Counter() for route in ROUTES} for method in METHODS}
    results = {}
    for slot in slots:
        method, route = slot["method"], slot["intended_route"]
        actual = outcome(records.get(slot["id"]))
        counts[method][route][actual] += 1
        results[slot["context_id"], method, route] = actual == route
    attempts = len(layout_ids)
    methods = {
        method: {
            route: dict(attempts=attempts, releases=attempts - outcomes["setup_rejected"],
                        captures=outcomes[route], capture_rate=outcomes[route] / attempts,
                        outcomes=dict(outcomes))
            for route, outcomes in by_route.items()
        }
        for method, by_route in counts.items()
    }

    pair_labels = {
        (True, True): "both",
        (True, False): "original_only",
        (False, True): "refit_only",
        (False, False): "neither",
    }
    paired = {route: {} for route in ROUTES}
    for route, model in product(ROUTES, MODELS):
        outcomes = dict.fromkeys(pair_labels.values(), 0)
        for identity in layout_ids:
            original = results[identity, f"{model}_original", route]
            refit = results[identity, f"{model}_refit", route]
            outcomes[pair_labels[original, refit]] += 1
        paired[route][model] = outcomes
    return dict(methods=methods, paired=paired)
