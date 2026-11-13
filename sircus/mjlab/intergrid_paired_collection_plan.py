"""Plan equal local shot searches at positions with a capture by each route."""
from __future__ import annotations

from itertools import product
import math

from .cell1050_followup_plan import geometry_key
from .expanded_actions import WIDE_BOUNDS
from .intergrid_proposer_plan import MIN_BOARD_DISTANCE_M
from .intergrid_route_plan import PER_ROUTE_LIMIT, ROUTES

VERSION = "sircus-intergrid-paired-collection-plan-v1"
MAX_SEEDS_PER_ROUTE = 3


def _finite_vector(values, count):
    vector = tuple(float(value) for value in values)
    if len(vector) != count or not all(math.isfinite(value) for value in vector):
        raise ValueError(f"Expected {count} finite shot coordinates")
    return vector


def _seed_order(seeds, route, target):
    if not seeds:
        raise ValueError(f"A verified {route} capture is required at the exact bucket position")
    identifiers = set()
    for seed in seeds:
        identifier = seed["id"]
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError(f"Invalid or duplicate {route} seed identity")
        identifiers.add(identifier)
        if (_finite_vector(seed["bucket_xy_m"], 2) != target or seed.get("shot_route", route) != route
                or seed.get("route_success", True) is not True):
            raise ValueError(f"Seed {identifier} is not a {route} capture at this bucket position")
        _finite_vector(seed["values"], 5)
        _finite_vector(seed["q10"], 6)
        _finite_vector(seed["q14"], 6)
        _finite_vector((seed.get("board_heading_offset_deg", 0), seed.get("board_twist_deg", 0)), 2)
    distinct = {}
    for seed in sorted(seeds, key=lambda row: (-row.get("route_quality", 0), row["id"])):
        distinct.setdefault(geometry_key(seed), seed)
    return list(distinct.values())[:MAX_SEEDS_PER_ROUTE]


def _variants():
    """Offset order is along, across, height, tilt, release gap, then heading."""
    nearby = [
        ("release_gap", (0., 0., 0., 0., .008, 0.)),
        ("board_tilt", (0., 0., 0., 1.2, 0., 0.)),
        ("board_height", (0., 0., .006, 0., 0., 0.)),
        ("board_forward", (.007, 0., 0., 0., 0., 0.)),
        ("board_lateral", (0., .007, 0., 0., 0., 0.)),
        ("board_heading", (0., 0., 0., 0., 0., 1.4)),
        ("board_height_at_fixed_release", (0., 0., .009, 0., -.009, 0.)),
        ("tilt_and_release_gap", (0., 0., 0., 1.6, -.012, 0.)),
    ]
    variants = [(name, tuple(sign * x for x in offsets))
                for name, offsets in nearby for sign in (-1, 1)]
    # Backfill when a seed sits near the action bounds or an old shot used the same setting.
    for scale in (1.5, .5, 2.0, .25, 3.0):
        variants.extend((name, tuple(scale * sign * x for x in offsets))
                        for name, offsets in nearby for sign in (-1, 1))
    variants.extend(("combined", (along, across, height, tilt, gap, heading))
                    for along, across, height, tilt, gap, heading in product(
                        (-.011, .011), (-.008, .008), (-.01, .01),
                        (-2., 2.), (-.016, .016), (-1.8, 1.8)))
    return variants


def _shot_variant(seed, target, offsets):
    x, y, height, tilt, gap = seed["values"]
    distance = math.dist((x, y), target)
    if distance <= MIN_BOARD_DISTANCE_M:
        raise ValueError(f"Seed {seed['id']} is too close to the bucket")
    forward_x, forward_y = (target[0] - x) / distance, (target[1] - y) / distance
    along, across, delta_height, delta_tilt, delta_gap, delta_heading = offsets
    values = [
        x + along * forward_x - across * forward_y,
        y + along * forward_y + across * forward_x,
        height + delta_height,
        tilt + delta_tilt,
        gap + delta_gap,
    ]
    if (not all(low <= value <= high for value, low, high in zip(
            values, WIDE_BOUNDS["low"], WIDE_BOUNDS["high"], strict=True))
            or math.dist(values[:2], target) <= MIN_BOARD_DISTANCE_M):
        return None
    heading = (float(seed.get("board_heading_offset_deg", 0)) + delta_heading + 180) % 360 - 180
    twist = (float(seed.get("board_twist_deg", 0)) + 180) % 360 - 180
    return values, heading, twist


def proposals_for(layout, seeds, *, budget=PER_ROUTE_LIMIT, known_geometry=()):
    """Return novel direct/table candidates at one authenticated development position."""
    if type(budget) is not int or budget <= 0 or set(seeds) != set(ROUTES):
        raise ValueError("Expected a positive budget per route and a source bank for each route")
    target = _finite_vector(layout["bucket_xy_m"], 2)
    layout_id = layout["id"]
    context_id = layout.get("context_id", layout_id)
    source_order = {route: _seed_order(seeds[route], route, target) for route in ROUTES}
    seen = set(known_geometry) | {geometry_key(seed) for route in ROUTES for seed in seeds[route]}
    variants = _variants()

    def candidates(route):
        ordered = source_order[route]
        for cycle in range(len(ordered)):
            for index, (name, offsets) in enumerate(variants):
                # Both signs of each primary probe use the same saved shot.
                seed = ordered[(index // 2 + cycle) % len(ordered)]
                action = _shot_variant(seed, target, offsets)
                if action is None:
                    continue
                values, heading, twist = action
                yield dict(
                    context_id=context_id, phase="train", role="train", method="paired_route_collection",
                    intended_route=route, variant=name, bucket_xy_m=list(target), values=values,
                    board_heading_offset_deg=heading, board_twist_deg=twist,
                    seed_id=seed["id"], training_recipe_id=seed["id"], training_bucket_xy_m=list(target),
                    source_study=seed.get("source_study"),
                    source_row_sha256=seed.get("source_row_sha256"),
                    source_trace_sha256=seed.get("source_trace_sha256"),
                    seed_q10=list(seed["q10"]), seed_q14=list(seed["q14"]),
                    offsets=dict(zip(("along_m", "across_m", "height_m", "tilt_deg", "gap_m", "heading_deg"),
                                     offsets, strict=True)),
                )

    streams = {route: candidates(route) for route in ROUTES}
    proposals = []
    for number in range(budget):
        for route in ROUTES:
            for candidate in streams[route]:
                key = geometry_key(candidate)
                if key not in seen:
                    seen.add(key)
                    proposals.append(dict(candidate, id=f"{layout_id}-{route}-{number:02d}"))
                    break
            else:
                raise ValueError(f"Insufficient novel {route} settings at {layout_id} for budget {budget}")
    return proposals
