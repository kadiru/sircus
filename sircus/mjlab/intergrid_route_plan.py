"""Freeze route-study placements and matched training-only proposal budgets."""
from __future__ import annotations

from collections import deque
from itertools import product
import math

import numpy as np

from .cell1050_followup_plan import geometry_key
from .expanded_actions import WIDE_BOUNDS
from .intergrid_proposer_plan import MIN_BOARD_DISTANCE_M
from .intergrid_proposer_plan import make_splits as _make_splits
from .intergrid_proposer_plan import sample_splits as _sample_splits
from .learning_contract import digest

VERSION = "sircus-intergrid-route-plan-v1"
SEED = 2026111701
ROUTES = ("direct", "table")
PER_ROUTE_LIMIT = 16
COLLECTION_LIMIT = PER_ROUTE_LIMIT * len(ROUTES)
LOW, HIGH = np.asarray(WIDE_BOUNDS["low"]), np.asarray(WIDE_BOUNDS["high"])


def _study_layout_ids(splits):
    for split, rows in splits.items():
        for index, row in enumerate(rows):
            identity = f"intergrid-route-{split}-{index:02d}"
            row.update(id=identity, context_id=identity)
    return splits


def sample_splits(excluded, geometry, grid=None, *, seed=SEED):
    """Replay the 24 TRAIN and 50 TEST locations from frozen exclusions and geometry."""
    return _study_layout_ids(_sample_splits(excluded, geometry, grid, seed=seed))


def make_splits(history_root, grid=None, *, scene, assets, previous_layouts=(), seed=SEED):
    """Freeze both splits before seeing route outcomes at any new placement."""
    splits, provenance = _make_splits(
        history_root, grid, scene=scene, assets=assets, previous_layouts=previous_layouts, seed=seed,
    )
    _study_layout_ids(splits)
    provenance.update(version=VERSION, splits_sha256=digest(splits))
    return splits, provenance


def _ordered_sources(library, target):
    """Use one strong seed per nearby bucket before taking more from any bucket."""
    grouped = {}
    rows = (row for group in library.values() for row in group) if isinstance(library, dict) else iter(library)
    ids = set()
    for row in rows:
        xy, values = np.asarray(row["bucket_xy_m"], float), np.asarray(row["values"], float)
        if (xy.shape != (2,) or values.shape != (5,) or not np.isfinite(xy).all()
                or not np.isfinite(values).all() or np.linalg.norm(xy - values[:2]) <= MIN_BOARD_DISTANCE_M):
            raise ValueError(f"Source geometry is invalid for {row['id']}")
        if row["id"] in ids:
            raise ValueError(f"Duplicate route source {row['id']}")
        ids.add(row["id"])
        grouped.setdefault(tuple(xy), []).append(row)
    if not grouped:
        raise ValueError("Each route needs at least one verified source")
    positions = sorted(grouped, key=lambda xy: (math.dist(xy, target), xy))[:4]
    for xy in positions:
        grouped[xy].sort(key=lambda row: (-len(row.get("lineage", [])), -row.get("route_quality", 0), row["id"]))
    return [grouped[xy][index] for index in range(4) for xy in positions if len(grouped[xy]) > index][:4]


def _source_variants(source, target, route, direction_sign):
    """Move a verified recipe while preserving or perturbing its approach geometry."""
    base = np.asarray(source["values"], float)
    direction = np.asarray(source["bucket_xy_m"], float) - base[:2]
    distance = np.linalg.norm(direction)
    angle = math.atan2(direction[1], direction[0])
    base_heading = float(source["board_heading_offset_deg"])
    base_twist = float(source["board_twist_deg"])

    def variant(name, *, shift=0., radius=1., height=0., tilt=0., gap=0., heading=0., position=None):
        new_angle = angle + math.radians(shift)
        board = target - distance * radius * np.array([math.cos(new_angle), math.sin(new_angle)])
        values = np.array([*(board if position is None else position), base[2] + height, base[3] + tilt, base[4] + gap])
        return name, np.clip(values, LOW, HIGH).tolist(), (base_heading + heading + 180.) % 360. - 180., base_twist

    translated = target - direction
    yield variant("translated")
    if route == "table":
        # Favor a lower bounce point without assigning an outcome before flight.
        yield variant("low_board", height=-.055, tilt=3., gap=.045)
    else:
        yield variant("high_board", height=.055, tilt=-3., gap=-.045)
    yield variant("azimuth_probe", shift=18. * direction_sign)
    yield variant("fixed_board", position=base[:2])
    yield variant("half_translation", position=.5 * (translated + base[:2]))
    yield variant("azimuth_probe", shift=-18. * direction_sign)
    for sign in (-1, 1):
        yield variant("heading_probe", heading=7. * sign)
        yield variant("gap_probe", gap=.035 * sign)
        yield variant("range_probe", radius=1. + .16 * sign, gap=.04 * sign)
    if route == "table":
        yield variant("longer_two_bounce_path", radius=1.23, height=-.025, gap=.075)
        yield variant("shorter_two_bounce_path", radius=.80, tilt=3., gap=-.055)
    else:
        yield variant("higher_arc", radius=1.20, tilt=-4., gap=.075)
        yield variant("shorter_flight", radius=.82, height=.025, gap=-.055)

    # A finite fallback lets a sparse route library transfer to other quadrants.
    # Every setting still passes the same geometric and IK gates before release.
    for radius, gap, shift in product((1., .75, 1.25), (0., -.04, .04),
                                      (35., -35., 65., -65., 105., -105., 145., -145.)):
        yield variant("wide_azimuth_range_gap", shift=shift, radius=radius, gap=gap,
                      height=-.025 if route == "table" else 0.)


def collection_proposals(layout, libraries):
    """Interleave an equal menu for both route intents at a TRAIN position."""
    if layout["split"] != "train":
        raise ValueError("Outcome collection is only allowed at TRAIN placements")
    return collection_geometry(layout, libraries, phase="train")


def collection_geometry(layout, libraries, *, phase):
    """Generate bounded route geometry from frozen sources without observing outcomes."""
    if phase not in {"train", "test"}:
        raise ValueError("Expected a declared training or test phase")
    if set(libraries) != set(ROUTES):
        raise ValueError("Expected verified source libraries for both route intents")
    target = np.asarray(layout["bucket_xy_m"], float)
    if target.shape != (2,) or not np.isfinite(target).all():
        raise ValueError("Expected finite training bucket XY")
    active = {
        route: deque(
            (row, iter(_source_variants(row, target, route, (-1, 1)[index % 2])))
            for index, row in enumerate(_ordered_sources(libraries[route], target))
        )
        for route in ROUTES
    }
    seen = set()
    result = []

    def next_candidate(route, number):
        while active[route]:
            source, stream = active[route].popleft()
            variant = next(stream, None)
            if variant is None:
                continue
            active[route].append((source, stream))
            group, values, heading, twist = variant
            if math.dist(target, values[:2]) <= MIN_BOARD_DISTANCE_M:
                continue
            proposal = dict(
                id=f"{layout['id']}-{route}-{number:02d}", context_id=layout["id"],
                phase=phase, role=phase, method="intergrid_route_collection",
                intended_route=route, group=group, bucket_xy_m=target.tolist(),
                values=values, board_heading_offset_deg=heading, board_twist_deg=twist,
                training_recipe_id=source["id"], training_bucket_xy_m=list(source["bucket_xy_m"]),
                training_distance_m=math.dist(source["bucket_xy_m"], target),
                source_row_sha256=source.get("source_row_sha256"),
                seed_q10=source["q10"], seed_q14=source["q14"],
            )
            key = geometry_key(proposal)
            if key in seen:
                continue
            seen.add(key)
            return proposal
        raise ValueError(f"Could not fill the {route} proposal budget for {layout['id']}")

    for number in range(PER_ROUTE_LIMIT):
        for route in ROUTES:
            result.append(next_candidate(route, number))
    return result
