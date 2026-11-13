"""Freeze spatial splits and generate geometry-guided training-only drops."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .cell1050_followup_plan import geometry_key
from .corrected_contract import grid_layouts
from .expanded_actions import WIDE_BOUNDS
from .intergrid_layouts import (
    EXCLUDED_CELLS, MIN_SEPARATION_M, NEAR_BASE_MAX_CLEARANCE_M, PERIMETER_STRIP_M,
    _historical_coordinates, _placement_geometry,
)
from .learning_contract import digest
from .table_training_contract import _coordinates

VERSION = "sircus-intergrid-proposer-plan-v1"
SEED = 2026111601
COLLECTION_LIMIT = 24
MIN_BOARD_DISTANCE_M = .15


def _grid_points(grid):
    canonical = {row["id"]: tuple(row["bucket_xy_m"]) for row in grid_layouts()}
    usable = {key: xy for key, xy in canonical.items() if key not in EXCLUDED_CELLS}
    if grid is not None:
        rows = list(grid)
        supplied = {row["id"].removeprefix("fixed35-"): tuple(row["bucket_xy_m"]) for row in rows}
        if len(supplied) != len(rows) or supplied not in (canonical, usable):
            raise ValueError("Expected the canonical fixed grid or its 33 usable locations")
    return canonical, usable


def sample_splits(excluded, geometry, grid=None, *, seed=SEED, include_train=True):
    """Rebuild 50 TEST placements and, when requested, 24 disjoint TRAIN placements."""
    canonical, usable = _grid_points(grid)
    xs, ys = (np.unique(np.asarray(list(usable.values()))[:, axis]) for axis in range(2))
    lower, upper = np.array([xs[0], ys[0]]), np.array([xs[-1], ys[-1]])
    used = np.asarray(sorted(set(map(tuple, excluded)) | set(canonical.values())), float)
    if used.ndim != 2 or used.shape[1] != 2 or not np.isfinite(used).all():
        raise ValueError("Expected finite prior bucket XY coordinates")

    fixed = [geometry["fixed_arm_bounds"][arm] for arm in ("Arm10", "Arm14")]
    bases = np.asarray([entry["center_xy_m"] for entry in fixed])
    bucket_radius = geometry["bucket_radius_m"]
    radii = np.asarray([entry["radius_m"] for entry in fixed]) + bucket_radius
    table_low, table_high = np.asarray(geometry["tabletop_xy_bounds_m"])
    splits = {"train": [], "test": []}
    rng = np.random.default_rng(seed)

    def sample(role, box_low, box_high, stratum, *, preferred=None, base_index=None):
        nonlocal used
        for attempt in range(10000):
            if preferred is not None and attempt < 1000:
                xy = rng.uniform(*preferred)
            else:
                xy = rng.uniform(box_low, box_high)
            clearance = np.linalg.norm(bases - xy, axis=1) - radii
            if clearance.min() < MIN_SEPARATION_M:
                continue
            if base_index is not None and clearance[base_index] > NEAR_BASE_MAX_CLEARANCE_M:
                continue
            margin = float(np.minimum(xy - lower, upper - xy).min())
            if stratum == "grid_perimeter" and margin > PERIMETER_STRIP_M:
                continue
            if np.minimum(xy - table_low, table_high - xy).min() < bucket_radius:
                continue
            if np.linalg.norm(used - xy, axis=1).min() < MIN_SEPARATION_M:
                continue
            nearest = min(usable, key=lambda key: math.dist(usable[key], xy))
            ix, iy = np.searchsorted(xs, xy[0]) - 1, np.searchsorted(ys, xy[1]) - 1
            identity = f"intergrid-proposer-{role}-{len(splits[role]):02d}"
            splits[role].append(dict(
                id=identity, context_id=identity, split=role, stratum=stratum,
                difficulty={"near_fixed_base": "base_proximity", "grid_perimeter": "edge"}.get(stratum, "interior"),
                bucket_xy_m=xy.tolist(), grid_cell=[int(ix), int(iy)],
                nearest_grid_id=f"fixed35-{nearest}",
                nearest_grid_distance_m=math.dist(usable[nearest], xy),
                grid_perimeter_distance_m=margin,
                fixed_base_clearance_lower_bound_m=float(clearance.min()),
            ))
            used = np.vstack((used, xy))
            return
        raise ValueError(f"Could not place {role} {stratum} sample in {box_low} to {box_high}")

    rectangles = []
    for iy in range(len(ys) - 1):
        for ix in range(len(xs) - 1):
            corner = np.array([xs[ix], ys[iy]])
            span = np.array([xs[ix + 1], ys[iy + 1]]) - corner
            box = (corner + .02 * span, corner + .98 * span)
            center = (corner + .25 * span, corner + .75 * span)
            perimeter = ix in (0, len(xs) - 2) or iy in (0, len(ys) - 2)
            rectangles.append((box, center, perimeter))
    if include_train:
        for box, center, _ in rectangles:
            sample("train", *box, "rectangle_train", preferred=center)
    for box, center, perimeter in rectangles:
        sample("test", *box, "rectangle_center", preferred=center)
        sample("test", *box, "grid_perimeter" if perimeter else "rectangle_interior")
    for index in range(len(fixed)):
        sample("test", lower + 1e-6, upper - 1e-6, "near_fixed_base", base_index=index)
    return splits


def make_splits(history_root, grid=None, *, scene, assets, previous_layouts=(), seed=SEED, include_train=True):
    """Freeze placement geometry and archived-coordinate separation before collection."""
    grid = None if grid is None else list(grid)
    root = Path(history_root).resolve()
    historical, file_hashes = _historical_coordinates(root)
    points = set(historical)
    for pattern in ("**/*layouts.json", "**/collection-plan.json", "**/choices.json"):
        for path in sorted(root.glob(pattern)):
            raw = path.read_bytes()
            name = f"output/{path.relative_to(root).as_posix()}"
            checksum = hashlib.sha256(raw).hexdigest()
            if file_hashes.setdefault(name, checksum) != checksum:
                raise ValueError(f"Historical coordinate metadata changed while reading {name}")
            _coordinates(json.loads(raw), points)
    declared = set()
    for layout in previous_layouts:
        xy = tuple(layout["bucket_xy_m"])
        if len(xy) != 2 or not all(math.isfinite(value) for value in xy):
            raise ValueError("Expected finite prior-layout bucket coordinates")
        declared.add(xy)
    canonical, usable = _grid_points(grid)
    excluded = sorted(points | declared | set(canonical.values()))
    geometry = _placement_geometry(scene, assets)
    splits = sample_splits(excluded, geometry, grid, seed=seed, include_train=include_train)
    provenance = dict(
        version=VERSION, seed=seed, minimum_separation_m=MIN_SEPARATION_M,
        grid_perimeter_strip_m=PERIMETER_STRIP_M, near_base_max_clearance_m=NEAR_BASE_MAX_CLEARANCE_M,
        usable_grid={key: list(xy) for key, xy in usable.items()},
        historical_coordinate_count=len(points), historical_files_sha256=file_hashes,
        declared_previous_coordinates_m=[list(xy) for xy in sorted(declared)],
        excluded_coordinates_m=[list(xy) for xy in excluded],
        excluded_coordinates_sha256=digest(excluded), splits_sha256=digest(splits), geometry=geometry,
        scope=("Novelty against inventoried archives, declared prior layouts and canonical coordinates. "
               + ("TEST also excludes the new TRAIN placements. " if include_train else "")
               + "Placement uses only geometry and coordinates."),
    )
    return splits, provenance


def _shot_variants(source, target, low, high, gap_direction):
    """Transfer a working source geometry, then change distance and direction."""
    base = np.asarray(source["values"], float)
    source_xy = np.asarray(source["bucket_xy_m"], float)
    delta = target - source_xy
    direction = source_xy - base[:2]
    distance = np.linalg.norm(direction)
    if distance <= MIN_BOARD_DISTANCE_M:
        return
    forward = direction / distance
    side = np.array([-forward[1], forward[0]])
    heading = float(source["board_heading_offset_deg"])

    for name, fraction, side_shift, tilt_shift, gap_shift, heading_shift in (
        ("translated", 1., 0., 0., 0., 0.),
        ("unchanged_board", 0., 0., 0., 0., 0.),
        ("partial_translation_range_gap", .5, 0., 0., None, 0.),
        ("left_offset", .75, -.025, 0., None, -8.),
        ("right_offset", .75, .025, 0., None, 8.),
        ("short_gap" if gap_direction < 0 else "long_gap", .5, 0., -2. * gap_direction, .035 * gap_direction, 0.),
        ("long_gap" if gap_direction < 0 else "short_gap", .5, 0., 2. * gap_direction, -.035 * gap_direction, 0.),
        ("left_wider", .5, -.05, 0., -.02, -14.),
        ("right_wider", .5, .05, 0., .02, 14.),
        ("translated_lower_gap", 1., 0., 1., -.025, 0.),
        ("translated_higher_gap", 1., 0., -1., .025, 0.),
    ):
        values = base.copy()
        values[:2] = np.clip(base[:2] + fraction * delta + side_shift * side, low[:2], high[:2])
        if gap_shift is None:
            range_ratio = np.linalg.norm(target - values[:2]) / distance
            gap_shift = float(np.clip(base[4] * (range_ratio - 1), -.10, .10))
        values[3:] = np.clip(base[3:] + [tilt_shift, gap_shift], low[3:], high[3:])
        angle = (heading + heading_shift + 180.) % 360. - 180.
        if ((values < low - 1e-9).any() or (values > high + 1e-9).any()
                or np.linalg.norm(target - values[:2]) <= MIN_BOARD_DISTANCE_M):
            continue
        yield name, values.tolist(), angle


def collection_proposals(layout, library):
    """Plan up to 24 distinct train-only shot settings from nearby saved captures."""
    if layout["split"] != "train":
        raise ValueError("Outcome collection is only allowed at TRAIN placements")
    target = np.asarray(layout["bucket_xy_m"], float)
    if target.shape != (2,) or not np.isfinite(target).all():
        raise ValueError("Expected finite training bucket XY")
    positions = sorted(library, key=lambda xy: (math.dist(xy, target), xy))[:4]
    if len(positions) != 4 or any(not library[xy] for xy in positions):
        raise ValueError("Collection needs working examples from four training grid positions")
    low, high = np.asarray(WIDE_BOUNDS["low"]), np.asarray(WIDE_BOUNDS["high"])
    result, seen = [], set()
    for seed_index in range(2):
        seeds = [library[xy][seed_index] for xy in positions if len(library[xy]) > seed_index]
        active = [
            (source, iter(_shot_variants(source, target, low, high, (-1, 1)[index % 2])))
            for index, source in enumerate(seeds)
        ]
        while active and len(result) < COLLECTION_LIMIT:
            next_round = []
            for source, stream in active:
                variant = next(stream, None)
                if variant is None:
                    continue
                next_round.append((source, stream))
                group, values, heading = variant
                proposal = dict(
                    id=f"{layout['id']}-collect-{len(result):02d}", context_id=layout["id"],
                    phase="train", role="train", method="intergrid_proposer_collection", group=group,
                    bucket_xy_m=target.tolist(), values=values, board_heading_offset_deg=heading,
                    board_twist_deg=float(source["board_twist_deg"]),
                    training_recipe_id=source["id"], training_bucket_xy_m=list(source["bucket_xy_m"]),
                    training_distance_m=math.dist(source["bucket_xy_m"], target),
                    seed_q10=source["q10"], seed_q14=source["q14"],
                )
                key = geometry_key(proposal)
                if key in seen:
                    continue
                seen.add(key)
                result.append(proposal)
                if len(result) == COLLECTION_LIMIT:
                    return result
            active = next_round
    if not result:
        raise ValueError(f"No in-bounds training candidates for {layout['id']}")
    return result
