"""Propose shots cheaply, learn their order, then require exact setup admission."""
from __future__ import annotations

import copy
import math
import random
from time import perf_counter

from . import parallel_capture_grid_workers as workers
from .benchmark_verify import digest
from .clearance_expansion_plan import (
    _empty, _geometry, _ranker, _prepare_branches, _all_admitted,
    branch_proposals, needs_expansion,
)
from .expanded_actions import WIDE_BOUNDS
from .intergrid_candidate_replay_plan import command_key
from .intergrid_route_plan import collection_geometry
from .intergrid_system_plan import ROUTE_LIBRARIES, base_proposals

VERSION = "sircus-neural-shot-plan-v1"
METHODS = ("current_planner", "clearance_expansion", "neural_selector")
MENU_LIMIT = 56
CHECK_BATCH = 4
CHECK_LIMIT = 16
TRAIN_RELEASE_LIMIT = 36
VALIDATION_RELEASE_LIMIT = 16


def proposal_layout(layout):
    # Legacy adapters accept train/test; the archive and learning rows retain
    # the actual validation split. Validation observations never enter fitting.
    return dict(layout, split="test" if layout["split"] == "test" else "train")


def menu(layout, libraries, checkpoint):
    """A frozen finite geometry menu; no IK, contacts or native outcomes here."""
    if libraries.get("approach_menu_version") is not None:
        from .approach_comparison_plan import VERSION, comparison_menu
        if libraries["approach_menu_version"] != VERSION:
            raise ValueError("Unknown bucket-approach menu version")
        return comparison_menu(layout, libraries, checkpoint)
    if libraries.get("expanded_menu_version") is not None:
        from .expanded_selector_plan import VERSION, expanded_menu
        if libraries["expanded_menu_version"] != VERSION:
            raise ValueError("Unknown expanded shot-menu version")
        return expanded_menu(layout, libraries, checkpoint)
    context = proposal_layout(layout)
    base = base_proposals(context, libraries, checkpoint)
    sources = {route: libraries[name][route] for route, name in ROUTE_LIBRARIES.items()}
    transferred = collection_geometry(context, sources, phase=context["split"])
    unique = {}
    for row in base + transferred:
        unique.setdefault(_geometry(row), row)
    rows = list(unique.values())
    low, high = WIDE_BOUNDS["low"], WIDE_BOUNDS["high"]
    seeds = rows[:8]
    for index in range(16):
        if not seeds:
            break
        row = copy.deepcopy(seeds[index % len(seeds)])
        sign = (-1, 1)[index // len(seeds) % 2]
        values = list(row["values"])
        # Vary height/tilt/gap jointly, preserving source seed joints as hints.
        delta = (.025 * sign, -2. * sign, .04 * sign)
        values[2:] = [min(high[i], max(low[i], values[i] + delta[i - 2])) for i in range(2, 5)]
        row.update(values=values, group="neural_height_tilt_gap_probe")
        unique.setdefault(_geometry(row), row)
    rows = list(unique.values())[:MENU_LIMIT]
    return [dict(row, id=f"{layout['id']}-neural-menu-{i:03d}", context_id=layout["id"],
                 dataset_split=layout["split"]) for i, row in enumerate(rows)]


def prepare_collection(layout, libraries, checkpoint, ranker_config):
    started = perf_counter()
    proposals = menu(layout, libraries, checkpoint)
    admitted, records = workers.prepare_proposals(proposals, True)
    ranker = _ranker(ranker_config)
    limit = TRAIN_RELEASE_LIMIT if layout["split"] == "train" else VALIDATION_RELEASE_LIMIT
    # Preserve a few strong controls within each location, then sample across
    # all feasible candidates. Do not observe outcomes while choosing drops.
    ordered = ranker.order(admitted)
    chosen = {command_key(row): row for row in ordered[:4]}
    remaining = [row for row in admitted if command_key(row) not in chosen]
    rng = random.Random(int(digest(layout)[:16], 16))
    rng.shuffle(remaining)
    for row in remaining:
        if len(chosen) >= limit:
            break
        chosen.setdefault(command_key(row), row)
    if workers._engine.adapter.env is not None:
        raise RuntimeError("Collection preparation allocated native CUDA")
    return dict(layout=layout, proposals=proposals, admitted=admitted, records=records,
                requests=list(chosen.values()), max_releases=limit,
                latency_s=perf_counter() - started)


def _choice(layout, admitted, ranker, checked, ranked_ids):
    ordered = ranker.order(admitted)
    request = ordered[0] if ordered else None
    identifier = f"{layout['id']}-neural_selector"
    return dict(id=identifier, context_id=layout["id"], bucket_xy_m=layout["bucket_xy_m"],
                stratum=layout["stratum"], method="neural_selector",
                intended_route=request["intended_route"] if request else None,
                available_routes=sorted({row["intended_route"] for row in admitted}),
                prediction=ranker.predict(request) if request else None,
                checked_proposals=checked, ranked_proposal_ids=ranked_ids,
                candidate_pool="neural_checked", request=None if request is None else
                dict(request, id=identifier, method="neural_selector", stratum=layout["stratum"]))


def prepare_neural(layout, libraries, source_checkpoint, history, ranker_config, checkpoint):
    """Independent cold preparation; rank first, solve only selected candidates."""
    from .neural_shot_model import order, predict

    started = perf_counter()
    proposals = menu(layout, libraries, source_checkpoint)
    ordered = order(checkpoint, proposals)
    ranker = _ranker(ranker_config)
    checked, admitted, records = [], [], []
    for start in range(0, min(CHECK_LIMIT, len(ordered)), CHECK_BATCH):
        batch = ordered[start:start + CHECK_BATCH]
        new, logs = workers.prepare_proposals(batch, True)
        checked.extend(batch)
        admitted.extend(new)
        records.extend(logs)
        choice = _choice(layout, admitted, ranker, len(checked), [r["id"] for r in ordered])
        if not needs_expansion(choice):
            break
    # The remaining collision risk may be due to a different elbow branch.
    # Search only checked geometry; do not secretly prepare the whole menu.
    branches = _empty()
    choice = _choice(layout, admitted, ranker, len(checked), [r["id"] for r in ordered])
    if needs_expansion(choice) and admitted:
        branch_plan = dict(base=dict(proposals=checked, admitted=admitted, records=records),
                           fallback=_empty(), expansion=_empty(), branches=_empty())
        proposed = branch_proposals(proposal_layout(layout), branch_plan, ranker)
        originals = {r["id"]: r for r in admitted}
        branches = _prepare_branches(workers._engine, proposed, originals, ranker_config["assets"])
        admitted = list({command_key(r): r for r in admitted + branches["admitted"]}.values())
        choice = _choice(layout, admitted, ranker, len(checked), [r["id"] for r in ordered])
    if workers._engine.adapter.env is not None:
        raise RuntimeError("Neural preparation allocated native CUDA")
    choice["predicted_flight_risk"] = needs_expansion(choice)
    return dict(layout=layout, plan=dict(proposals=proposals, checked=checked, admitted=admitted,
                                         records=records, branches=branches), choices=[choice],
                predictions=[dict(id=r["id"], **predict(checkpoint, r)) for r in proposals],
                latency_s=dict(neural_selector=perf_counter() - started))
