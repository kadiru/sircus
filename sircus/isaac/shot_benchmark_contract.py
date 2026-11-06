"""CPU-only, versioned accounting for a fresh-layout sweep/PPO comparison.

These functions draw geometry and describe paid attempts. They do not perform
IK, simulate a shot, select a checkpoint, or certify a reachable layout.
"""

import copy
import hashlib
import json
import math

import numpy as np

VERSION = "sircus-fresh-shot-benchmark-v1"
SPLIT_VERSION = "sircus-fresh-shot-layouts-v1"
SEEDS = (20261121, 20261122, 20261123)
SPLIT_SEED = 20261120
TRAIN_BUDGET = 10000
MILESTONES = (0, 1000, 2500, 5000, 10000)
METHODS = ("sweep", "ppo")
SITE_COUNTS = dict(train=35, validation=35, test=70)
ACTION_NAMES = ("bearing_offset_rad", "radius_m", "height_m", "gap_m", "tilt_rad")
LOW = np.array([-math.pi, .28, .14, .08, math.radians(12.)])
HIGH = np.array([math.pi, 1.20, .60, .56, math.radians(52.)])
REFERENCE = np.array([.67, -.10])
BUCKET_BOUNDS = [[-.10, -.70], [1.40, .20]]
MINIMUM_SEPARATION_M = .025


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def signed(value):
    result = copy.deepcopy(value)
    result["fingerprint"] = canonical_hash(result)
    return result


def validate_fingerprint(value):
    require(isinstance(value, dict), "Expected a signed object")
    unsigned = {k: v for k, v in value.items() if k != "fingerprint"}
    require(value.get("fingerprint") == canonical_hash(unsigned), "Changed benchmark fingerprint")
    return value["fingerprint"]


def action_domain():
    return signed(dict(names=list(ACTION_NAMES), low=LOW.tolist(), high=HIGH.tolist(),
        reference_xy_m=REFERENCE.tolist(),
        gap_definition="Ball-bottom release Z minus board-center Z; ball center adds radius"))


def make_protocol():
    """The production budget is fixed; unit fixtures cannot silently lower it."""
    return signed(dict(version=VERSION, methods=list(METHODS), seeds=list(SEEDS),
        train_budget_per_method_seed=TRAIN_BUDGET, total_train_attempts=60000,
        site_counts=copy.deepcopy(SITE_COUNTS), milestones=list(MILESTONES),
        native_slots=35, logical_training_rounds=286, native_training_calls=289,
        run_order=[dict(method=method, seed=seed) for i, seed in enumerate(SEEDS)
                   for method in (METHODS if i % 2 == 0 else tuple(reversed(METHODS)))],
        domain=action_domain(),
        sweep=dict(version="sircus-prior-sobol-refinement-v1", prior_requests_per_site=1,
            signed_axis_requests_per_site=10, local_sobol_requests_per_site=32,
            axis_and_local_std_multiplier=2.0, subsequent_pattern=["global_sobol", "refine", "refine", "refine"],
            refinement_std_multiplier=4.0, refinement_budget_scales=[[0, 1.0], [2500, .5], [5000, .25]],
            sobol="Independent scrambled five-dimensional sequence per seed/TRAIN coordinate; 30-bit bin midpoints avoid infinite inverse transforms",
            prediction="Nearest TRAIN coordinate, stable ID tie-break; transfer its best observed latent residual relative to the shared geometry prior; no inference-time search"),
        initial_prior="Unchanged ContinuousPolicy geometry prior; no old weights, shot labels, map anchors or exact-context lookup",
        sample_definition="Every requested TRAIN action including IK, clearance, path and settling rejects; no free retries or replacements",
        schedule="Paired seeded permutation of all 35 TRAIN sites per logical round; exact milestone cuts split a round without spending future attempts",
        evaluation=dict(train_sites=35, validation_sites=35, test_sites=70,
            milestones=list(MILESTONES), primary_repeats=1,
            selection="Final 10000-attempt artifact only; validation is diagnostic and never selects/replaces a checkpoint",
            test="Final frozen deterministic prediction once at each of 70 unseen coordinates; no shot search, outcome query, retry or adaptation",
            perturbation_repeats=0,
            total_requested_per_method_seed=420,
            train_diagnostic_attempts=175, validation_attempts=175, test_attempts=70),
        scope="Single fixed ball; three paired seeds; privileged geometry; one terminal cooperative shot decision. Native travel/grasp and material adaptation are not learned."))


def validate_protocol(protocol):
    validate_fingerprint(protocol)
    require(protocol == make_protocol(), "Unsupported benchmark budget or protocol; version changes must be explicit")
    return protocol["fingerprint"]


def normalized_coordinates(values):
    coordinates = np.asarray(values, dtype=float)
    require(coordinates.ndim == 2 and coordinates.shape[1] == 2
            and np.isfinite(coordinates).all(), "Expected finite N by 2 bucket coordinates")
    return sorted({(float(x), float(y)) for x, y in coordinates})


def make_splits(excluded_coordinates, *, bounds=BUCKET_BOUNDS,
                minimum_separation_m=MINIMUM_SEPARATION_M, max_draws=1000000):
    """Draw fresh sites without outcome/admission conditioning or silent relaxation.

    The caller owns complete historical-coordinate discovery and its provenance.
    If 25mm fresh packing is impossible, fail rather than accept older coordinates.
    """
    excluded = normalized_coordinates(excluded_coordinates)
    boundary = np.asarray(bounds, dtype=float)
    require(boundary.shape == (2, 2) and np.isfinite(boundary).all()
            and (boundary[1] > boundary[0]).all(), "Invalid bucket-center bounds")
    require(type(minimum_separation_m) in (int, float) and np.isfinite(minimum_separation_m)
            and minimum_separation_m >= MINIMUM_SEPARATION_M,
            "Fresh layouts require at least 25mm separation")
    require(type(max_draws) is int and max_draws >= sum(SITE_COUNTS.values()), "Insufficient bounded layout draws")
    rng = np.random.default_rng(SPLIT_SEED)
    previous = np.asarray(excluded, dtype=float).reshape(-1, 2)
    groups, draws = {}, 0
    tile_size = (boundary[1] - boundary[0]) / [7, 5]
    for role, count in SITE_COUNTS.items():
        groups[role] = []
        for i in range(count):
            tile = i % 35
            row, column = divmod(tile, 7)
            tile_low = boundary[0] + tile_size * [column, row]
            while draws < max_draws:
                point = rng.uniform(tile_low, tile_low + tile_size)
                draws += 1
                if len(previous) and np.linalg.norm(previous - point, axis=1).min() < minimum_separation_m:
                    continue
                previous = np.vstack([previous, point])
                groups[role].append(dict(id=f"{role}-{i:03d}", split=role,
                    bucket_xy_m=point.tolist(), tile_index=tile, grid_row=row, grid_column=column))
                break
            require(len(groups[role]) == i + 1,
                f"Cannot fill {role} tile {tile} with {minimum_separation_m:g}m fresh separation within {max_draws} draws; no constraint was relaxed")
    return signed(dict(schema=SPLIT_VERSION, seed=SPLIT_SEED, bounds_xy_m=boundary.tolist(),
        minimum_separation_m=float(minimum_separation_m),
        historical_coordinates_sha256=canonical_hash([list(p) for p in excluded]),
        historical_coordinate_count=len(excluded), draws=draws,
        stratification=dict(columns=7, rows=5, train_per_tile=1, validation_per_tile=1,
            test_per_tile=2, test_order="First one coordinate per tile, then the second coordinate per tile"),
        selection="Seeded uniform coordinates within every tile; separation only, never reach, IK, capture or reward filtering",
        **groups))


def validate_splits(splits, *, excluded_coordinates=None):
    validate_fingerprint(splits)
    require(splits.get("schema") == SPLIT_VERSION and splits.get("seed") == SPLIT_SEED,
            "Unknown fresh-layout contract")
    separation = splits.get("minimum_separation_m")
    require(type(separation) in (int, float) and np.isfinite(separation) and separation >= MINIMUM_SEPARATION_M,
            "Fresh separation is below 25mm")
    require(type(splits.get("draws")) is int and splits["draws"] >= 140
            and type(splits.get("historical_coordinate_count")) is int and splits["historical_coordinate_count"] >= 0
            and isinstance(splits.get("historical_coordinates_sha256"), str)
            and len(splits["historical_coordinates_sha256"]) == 64
            and set(splits["historical_coordinates_sha256"]).issubset("0123456789abcdef"),
            "Invalid historical-coordinate/draw provenance")
    bounds = np.asarray(splits.get("bounds_xy_m"), dtype=float)
    require(bounds.shape == (2, 2) and np.isfinite(bounds).all() and (bounds[1] > bounds[0]).all(),
            "Invalid saved layout bounds")
    points = []
    require(splits.get("stratification") == dict(columns=7, rows=5, train_per_tile=1,
        validation_per_tile=1, test_per_tile=2,
        test_order="First one coordinate per tile, then the second coordinate per tile"),
        "Fresh spatial stratification changed")
    tile_size = (bounds[1] - bounds[0]) / [7, 5]
    for role, count in SITE_COUNTS.items():
        require(isinstance(splits.get(role), list) and len(splits[role]) == count,
                "Wrong fresh layout count")
        for i, cell in enumerate(splits[role]):
            require(set(cell) == {"id", "split", "bucket_xy_m", "tile_index", "grid_row", "grid_column"}
                    and cell["id"] == f"{role}-{i:03d}" and cell["split"] == role,
                    "Layout identity/role changed")
            tile = i % 35
            row, column = divmod(tile, 7)
            require(cell["tile_index"] == tile and cell["grid_row"] == row and cell["grid_column"] == column,
                    "Fresh layout tile identity changed")
            point = np.asarray(cell["bucket_xy_m"], dtype=float)
            require(point.shape == (2,) and np.isfinite(point).all()
                    and (point >= bounds[0]).all() and (point <= bounds[1]).all(), "Saved bucket outside bounds")
            tile_low = bounds[0] + tile_size * [column, row]
            require((point >= tile_low).all() and (point < tile_low + tile_size).all(),
                    "Fresh bucket is outside its assigned spatial tile")
            points.append(point)
    points = np.asarray(points)
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    require(distances.min() >= separation, "Fresh layout coordinates overlap")
    if excluded_coordinates is not None:
        excluded = normalized_coordinates(excluded_coordinates)
        require(splits["historical_coordinate_count"] == len(excluded)
                and splits["historical_coordinates_sha256"] == canonical_hash([list(p) for p in excluded]),
                "Historical exclusion pool changed")
        if excluded:
            require(np.linalg.norm(points[:, None] - np.asarray(excluded)[None, :], axis=2).min() >= separation,
                    "Fresh layout overlaps historical coordinates")
    return splits["fingerprint"]


def training_schedule(seed):
    require(type(seed) is int and seed in SEEDS, "Unknown benchmark seed")
    rng = np.random.default_rng(seed)
    rounds = [rng.permutation(35).tolist() for _ in range(286)]
    return [i for row in rounds for i in row][:TRAIN_BUDGET]


def batch_plan(protocol, seed):
    """No future shot is executed before an exact milestone assessment."""
    validate_protocol(protocol)
    schedule = training_schedule(seed)
    result, start = [], 0
    while start < TRAIN_BUDGET:
        milestone = next(m for m in MILESTONES if m > start)
        stop = min(start + 35 - start % 35, milestone, TRAIN_BUDGET)
        indices = schedule[start:stop]
        result.append(dict(call_index=len(result), logical_round=start // 35,
            attempt_start=start, attempt_stop=stop, attempts=stop - start,
            site_indices=indices, enabled=[i in indices for i in range(35)],
            milestone_after=stop if stop in MILESTONES else None))
        start = stop
    require(len(result) == 289 and sum(x["attempts"] for x in result) == TRAIN_BUDGET,
            "Exact milestone accounting changed")
    return result


def evaluation_cohorts(splits, role):
    validate_splits(splits)
    require(role in SITE_COUNTS, "Unknown evaluation cohort")
    cells = copy.deepcopy(splits[role])
    return [cells[i:i + 35] for i in range(0, len(cells), 35)]
