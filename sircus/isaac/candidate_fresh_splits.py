"""Outcome-blind held-out geometry for the separate seed-22 continuation.

The completed benchmark generator/validator remain unchanged. This version
draws no TRAIN positions, performs no model/IK/outcome query, and never relaxes
spacing or replaces a difficult tile. Only coordinate manifests are inputs.
"""

import copy
import math
import re

import numpy as np

from continuous_feasibility import canonical_hash
from shot_benchmark_contract import validate_splits as validate_original_splits

SCHEMA = "sircus-candidate-fresh-heldout-layouts-v1"
HISTORY_SCHEMA = "sircus-candidate-fresh-coordinate-history-v1"
AUDIT_SCHEMA = "sircus-candidate-fresh-heldout-layout-audit-v1"
SEED = 20261126
COUNTS = {"validation":35, "test":70}
BOUNDS = [[-.10, -.70], [1.40, .20]]
SEPARATION_M = .025
MAX_DRAWS = 1_000_000
CELL_FIELDS = {"id", "split", "bucket_xy_m", "tile_index", "grid_row", "grid_column"}
SELECTION = "Seeded uniform coordinates per fixed tile; separation only, never IK, reachability, capture, reward or outcome filtering"
SCOPE = "Novel relative to the supplied discovered coordinate history and all 140 original benchmark positions; not a claim about undiscovered or unpublished history. Outcome-held-out geometry preparation only."


def require(value, message):
    if not value:
        raise ValueError(message)


def signed(value):
    value = copy.deepcopy(value)
    value["fingerprint"] = canonical_hash(value)
    return value


def unsigned(value, label):
    require(type(value) is dict, "Expected signed " + label)
    require(value.get("fingerprint") == canonical_hash({k:v for k,v in value.items() if k != "fingerprint"}),
            "Changed " + label + " fingerprint")
    return {k:copy.deepcopy(v) for k,v in value.items() if k != "fingerprint"}


def coordinates(values):
    require(type(values) in (list, tuple), "Expected explicit coordinate list")
    result = []
    for point in values:
        require(type(point) in (list, tuple) and len(point) == 2
                and all(type(x) in (int, float) and math.isfinite(x) for x in point),
                "Expected finite coordinate pairs")
        result.append((float(point[0]), float(point[1])))
    return [list(point) for point in sorted(set(result))]


def build_history_receipt(values, source_sha256, *, scope):
    """Bind discovered coordinate-only files, without reading any source file."""
    points = coordinates(values)
    require(type(source_sha256) is dict and bool(source_sha256)
            and all(type(k) is str and k and type(v) is str and re.fullmatch(r"[0-9a-f]{64}", v)
                    for k,v in source_sha256.items()), "Expected coordinate-source SHA256 bindings")
    require(type(scope) is str and bool(scope.strip()), "History discovery scope is required")
    return signed(dict(schema=HISTORY_SCHEMA, coordinates=points, coordinate_count=len(points),
        coordinates_sha256=canonical_hash(points), source_sha256=dict(sorted(source_sha256.items())),
        discovery_scope=scope, outcome_files_read=0, optimizer_steps=0, native_requests=0))


def validate_history(receipt):
    value = unsigned(receipt, "coordinate history")
    require(set(value) == {"schema", "coordinates", "coordinate_count", "coordinates_sha256",
            "source_sha256", "discovery_scope", "outcome_files_read", "optimizer_steps", "native_requests"}
            and value["schema"] == HISTORY_SCHEMA, "Unknown coordinate-history schema")
    require(type(value["coordinate_count"]) is int and all(type(value[k]) is int and value[k] == 0
            for k in ("outcome_files_read", "optimizer_steps", "native_requests")),
            "Coordinate-history counters must be explicit zero integers")
    rebuilt = build_history_receipt(value["coordinates"], value["source_sha256"], scope=value["discovery_scope"])
    require(rebuilt == receipt, "History coordinates/count/bindings changed or outcomes added")
    return copy.deepcopy(value["coordinates"])


def _inputs(history_receipt, original_splits):
    history = validate_history(history_receipt)
    validate_original_splits(original_splits)
    require(original_splits["bounds_xy_m"] == BOUNDS
            and original_splits["minimum_separation_m"] == SEPARATION_M,
            "Original benchmark bounds/spacing differ from frozen contract")
    original = coordinates([cell["bucket_xy_m"] for role in ("train", "validation", "test")
                            for cell in original_splits[role]])
    require(len(original) == 140, "All 140 distinct original benchmark coordinates are required")
    excluded = coordinates(history + original)
    bindings = dict(history_receipt_fingerprint=history_receipt["fingerprint"],
        history_coordinates_sha256=canonical_hash(history), original_split_fingerprint=original_splits["fingerprint"],
        original_coordinates_sha256=canonical_hash(original), excluded_coordinates_sha256=canonical_hash(excluded))
    counts = dict(discovered_history=len(history), original_benchmark=140, excluded_union=len(excluded),
                  shared_history_original=len(history)+140-len(excluded))
    return excluded, bindings, counts


def make_fresh_splits(history_receipt, original_splits, *, max_draws=MAX_DRAWS):
    """Draw exactly 35 validation and 70 TEST sites, with a finite global cap."""
    require(type(max_draws) is int and 105 <= max_draws <= MAX_DRAWS, "Invalid bounded layout draw limit")
    excluded, bindings, counts = _inputs(history_receipt, original_splits)
    previous = np.asarray(excluded, dtype=float).reshape(-1, 2)
    boundary = np.asarray(BOUNDS, dtype=float)
    size = (boundary[1]-boundary[0])/[7, 5]
    rng = np.random.default_rng(SEED)
    groups, draws = {}, 0
    for role, count in COUNTS.items():
        groups[role] = []
        for index in range(count):
            tile = index % 35
            row, column = divmod(tile, 7)
            lower = boundary[0]+size*[column, row]
            while draws < max_draws:
                point = rng.uniform(lower, lower+size)
                draws += 1
                if np.linalg.norm(previous-point, axis=1).min() < SEPARATION_M:
                    continue
                previous = np.vstack([previous, point])
                groups[role].append(dict(id=f"fresh-{role}-{index:03d}", split=role,
                    bucket_xy_m=point.tolist(), tile_index=tile, grid_row=row, grid_column=column))
                break
            require(len(groups[role]) == index+1,
                f"Cannot fill {role} tile {tile} within {max_draws} draws; no constraint was relaxed or tile replaced")
    return signed(dict(schema=SCHEMA, seed=SEED, bounds_xy_m=copy.deepcopy(BOUNDS),
        minimum_separation_m=SEPARATION_M, draw_limit=max_draws, draws=draws,
        stratification=dict(columns=7, rows=5, validation_per_tile=1, test_per_tile=2,
            test_order="First one coordinate per tile, then the second coordinate per tile"),
        input_bindings=bindings, exclusion_counts=counts, selection=SELECTION, scope=SCOPE,
        outcome_files_read=0, optimizer_steps=0, native_requests=0,
        execution_authorized=False, release_authorized=False, **groups))


def cells_from_splits(manifest):
    """Check standalone geometry identity; history exclusion needs the full audit."""
    value = unsigned(manifest, "fresh held-out layout")
    fields = {"schema", "seed", "bounds_xy_m", "minimum_separation_m", "draw_limit", "draws",
        "stratification", "input_bindings", "exclusion_counts", "selection", "scope", "outcome_files_read",
        "optimizer_steps", "native_requests", "execution_authorized", "release_authorized", "validation", "test"}
    require(set(value) == fields and value["schema"] == SCHEMA and value["seed"] == SEED,
            "Fresh split schema, roles or seed changed")
    require(value["bounds_xy_m"] == BOUNDS and value["minimum_separation_m"] == SEPARATION_M,
            "Fresh bounds/spacing changed")
    require(type(value["draw_limit"]) is int and 105 <= value["draw_limit"] <= MAX_DRAWS
            and type(value["draws"]) is int and 105 <= value["draws"] <= value["draw_limit"],
            "Fresh draw accounting changed")
    require(value["stratification"] == dict(columns=7, rows=5, validation_per_tile=1, test_per_tile=2,
        test_order="First one coordinate per tile, then the second coordinate per tile"), "Fresh stratification changed")
    require(value["selection"] == SELECTION and value["scope"] == SCOPE
            and all(type(value[k]) is int and value[k] == 0 for k in
                ("outcome_files_read", "optimizer_steps", "native_requests"))
            and value["execution_authorized"] is value["release_authorized"] is False,
            "Fresh geometry cannot contain outcomes or authorize execution")
    require(type(value["input_bindings"]) is dict and set(value["input_bindings"]) == {
        "history_receipt_fingerprint", "history_coordinates_sha256", "original_split_fingerprint",
        "original_coordinates_sha256", "excluded_coordinates_sha256"}
        and all(type(s) is str and re.fullmatch(r"[0-9a-f]{64}",s) for s in value["input_bindings"].values()),
        "Missing fresh exclusion identities")
    count = value["exclusion_counts"]
    require(type(count) is dict and set(count) == {"discovered_history", "original_benchmark", "excluded_union", "shared_history_original"}
        and all(type(n) is int and n >= 0 for n in count.values()) and count["original_benchmark"] == 140
        and count["excluded_union"] == count["discovered_history"]+140-count["shared_history_original"]
        and count["shared_history_original"] <= min(140,count["discovered_history"]), "Invalid fresh exclusion counts")
    cells, points = [], []
    step_x = (BOUNDS[1][0]-BOUNDS[0][0])/7
    step_y = (BOUNDS[1][1]-BOUNDS[0][1])/5
    for role, number in COUNTS.items():
        require(type(value[role]) is list and len(value[role]) == number, "Fresh role count changed")
        for index, cell in enumerate(value[role]):
            tile = index % 35
            row, column = divmod(tile,7)
            require(type(cell) is dict and set(cell) == CELL_FIELDS
                    and cell["id"] == f"fresh-{role}-{index:03d}" and cell["split"] == role
                    and all(type(cell[k]) is int for k in ("tile_index", "grid_row", "grid_column"))
                    and (cell["tile_index"],cell["grid_row"],cell["grid_column"]) == (tile,row,column),
                    "Fresh position identity/tile changed or outcome fields added")
            require(type(cell["bucket_xy_m"]) is list and len(cell["bucket_xy_m"]) == 2
                    and all(type(x) in (int,float) and math.isfinite(x) for x in cell["bucket_xy_m"]),
                    "Invalid fresh coordinate")
            x,y = cell["bucket_xy_m"]
            require(BOUNDS[0][0]+column*step_x <= x < BOUNDS[0][0]+(column+1)*step_x
                    and BOUNDS[0][1]+row*step_y <= y < BOUNDS[0][1]+(row+1)*step_y,
                    "Fresh coordinate outside its assigned tile")
            require(all(math.hypot(x-a,y-b) >= SEPARATION_M for a,b in points), "Fresh coordinates overlap")
            points.append((x,y)); cells.append(copy.deepcopy(cell))
    return cells


def audit_fresh_splits(manifest, history_receipt, original_splits, *, replay=True):
    """Independently check geometry/provenance and optionally replay all draws."""
    require(type(replay) is bool, "Replay choice must be Boolean")
    cells = cells_from_splits(manifest)
    excluded, bindings, counts = _inputs(history_receipt, original_splits)
    require(manifest["input_bindings"] == bindings and manifest["exclusion_counts"] == counts,
            "Fresh exclusion pool identity changed")
    minimum = min(math.hypot(cell["bucket_xy_m"][0]-x,cell["bucket_xy_m"][1]-y)
                  for cell in cells for x,y in excluded)
    require(minimum >= SEPARATION_M, "Fresh coordinate overlaps discovered history or original benchmark")
    if replay:
        # Producer uses vectorized distances. Audit uses scalar distances and
        # saved tile metadata, without calling make_fresh_splits.
        rng = np.random.default_rng(SEED)
        accepted = [tuple(p) for p in excluded]
        draws = 0
        boundary = np.asarray(BOUNDS,float)
        size = (boundary[1]-boundary[0])/[7,5]
        for cell in cells:
            lower = boundary[0]+size*[cell["grid_column"],cell["grid_row"]]
            while draws < manifest["draw_limit"]:
                candidate = rng.uniform(lower,lower+size).tolist()
                draws += 1
                if any(math.hypot(candidate[0]-a,candidate[1]-b) < SEPARATION_M for a,b in accepted):
                    continue
                require(candidate == cell["bucket_xy_m"], "Fresh seeded coordinate/draw sequence changed")
                accepted.append(tuple(candidate))
                break
            else:
                raise ValueError("Fresh draw replay exhausted its bounded limit")
        require(draws == manifest["draws"], "Fresh accepted/rejected draw count changed")
    pair_minimum = min(math.dist(a["bucket_xy_m"],b["bucket_xy_m"])
                      for i,a in enumerate(cells) for b in cells[i+1:])
    return signed(dict(schema=AUDIT_SCHEMA, status="passed", split_fingerprint=manifest["fingerprint"],
        input_bindings=bindings, exclusion_counts=counts, validation_positions=35, test_positions=70,
        new_train_positions=0, draws=manifest["draws"], rejected_draws=manifest["draws"]-105,
        exact_draw_replay=replay, minimum_exclusion_distance_m=minimum,
        minimum_new_pair_distance_m=pair_minimum, outcome_files_read=0, optimizer_steps=0,
        native_requests=0, scope=SCOPE, execution_authorized=False, release_authorized=False))


def validate_fresh_splits(manifest, history_receipt, original_splits):
    audit_fresh_splits(manifest,history_receipt,original_splits,replay=True)
    return manifest["fingerprint"]
