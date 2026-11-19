"""Compare the refitted table specialist with two frozen starting-shot libraries."""
from __future__ import annotations

from collections import Counter
from itertools import product

from .intergrid_matched_route_study import outcome
from .intergrid_route_models import _valid_release, propose_route_candidates
from .intergrid_route_study import PER_ROUTE_TEST, route_library, source_menu


METHODS = ("original", "expanded")
ROUTE = "table"
MODEL = "specialist"


def libraries(original_rows, new_rows):
    """Expand the original library only with audited table captures from collection."""
    original_rows = list(original_rows)
    additions = [row for row in new_rows if row["shot_route"] == ROUTE and _valid_release(row)]
    if any(row.get("phase") != "train" or row.get("role") in {"test", "validation"} for row in additions):
        raise ValueError("Expanded library requires training captures")
    expanded_rows = original_rows + additions
    if len({row["id"] for row in expanded_rows}) != len(expanded_rows):
        raise ValueError("Library records must have unique identities")
    return dict(original=route_library(original_rows), expanded=route_library(expanded_rows))


def test_proposals(layout, libraries, checkpoint):
    """Run the same ordered prediction screen and keep only table candidates."""
    proposals = []
    for method in METHODS:
        sources = [row for row in source_menu(layout, libraries[method]) if row["intended_route"] == ROUTE]
        for row in propose_route_candidates(checkpoint, MODEL, layout["bucket_xy_m"], sources,
                                            context_id=layout["id"], limit=PER_ROUTE_TEST):
            proposals.append(dict(row, id=f"{row['id']}-{method}", method=method))
    return proposals


def select_requests(layout, proposals, admitted):
    """Freeze the first feasible table candidate per library, or a failed setup."""
    by_id = {row["proposal_id"]: row for row in admitted}
    proposal_ids = {row["id"] for row in proposals}
    if len(by_id) != len(admitted) or len(proposal_ids) != len(proposals) or by_id.keys() - proposal_ids:
        raise ValueError("Expected at most one admitted branch for each known proposal")
    first_admitted = {}
    for proposal in proposals:
        if proposal["method"] not in METHODS or proposal["intended_route"] != ROUTE:
            raise ValueError("Expected a table proposal from one of the two libraries")
        if proposal["id"] in by_id:
            first_admitted.setdefault(proposal["method"], by_id[proposal["id"]])

    slots = []
    for method in METHODS:
        identifier = f"{layout['id']}-{method}-{ROUTE}"
        selected = first_admitted.get(method)
        slots.append(dict(
            id=identifier, method=method, context_id=layout["id"], bucket_xy_m=layout["bucket_xy_m"],
            stratum=layout["stratum"], intended_route=ROUTE,
            request=None if selected is None else dict(selected, id=identifier, stratum=layout["stratum"]),
        ))
    return slots


def summarize(layouts, slots, records):
    """Account for one table attempt per library and layout, pairing both outcomes."""
    layout_ids = {layout["id"] for layout in layouts}
    slot_keys = {(slot["context_id"], slot["method"], slot["intended_route"]) for slot in slots}
    if (not layout_ids or len(layout_ids) != len(layouts) or len(slot_keys) != len(slots)
            or slot_keys != set(product(layout_ids, METHODS, (ROUTE,)))):
        raise ValueError("Expected one table attempt per layout and library")
    if (len({slot["id"] for slot in slots}) != len(slots)
            or records.keys() != {slot["id"] for slot in slots if slot["request"] is not None}):
        raise ValueError("Expected exactly one observation for each admitted slot")

    counts = {method: Counter() for method in METHODS}
    outcomes = {}
    for slot in slots:
        actual = outcome(records.get(slot["id"]))
        counts[slot["method"]][actual] += 1
        outcomes[slot["context_id"], slot["method"]] = actual

    attempts = len(layout_ids)
    methods = {}
    for method, labels in counts.items():
        releases = attempts - labels["setup_rejected"]
        methods[method] = dict(attempts=attempts, releases=releases, setup_rate=releases / attempts,
                               captures=labels[ROUTE], capture_rate=labels[ROUTE] / attempts,
                               outcomes=dict(labels))

    pair_labels = {(True, True): "both", (True, False): "original_only",
                   (False, True): "expanded_only", (False, False): "neither"}
    paired = {stage: dict.fromkeys(pair_labels.values(), 0) for stage in ("setup", "capture")}
    for identity in layout_ids:
        original, expanded = (outcomes[identity, method] for method in METHODS)
        paired["setup"][pair_labels[original != "setup_rejected", expanded != "setup_rejected"]] += 1
        paired["capture"][pair_labels[original == ROUTE, expanded == ROUTE]] += 1
    return dict(methods=methods, paired=paired)
