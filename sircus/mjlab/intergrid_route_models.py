"""Fit matched pooled, route-conditioned and per-route shot proposers."""
from __future__ import annotations

from collections import defaultdict
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .benchmark_verify import digest
from .cell1050_followup_plan import geometry_key
from .gpu_slot import GPU_UUID
from .intergrid_proposer import (
    DIMENSIONS, FEATURE_COUNT, HIGH, LOW, MAX_PAIR_DISTANCE, NEIGHBORS,
    RECIPE_SCALE, RIDGE, _features, _matches, _source, _vector, _wrap_degrees,
)
from .table_assisted_score import ROUTE_VERSION

VERSION = "sircus-intergrid-route-models-v1"
MODELS = ("pooled", "conditioned", "specialist")
CONDITIONED_RESIDUAL_RIDGE = 4 * RIDGE
UNMODELED_ROUTES = {"not_captured", "standard_capture", "robot_assisted_capture", "uncategorized_capture",
                   "other", "ambiguous", "other_or_ambiguous"}
MAX_CANDIDATES = 8


def _valid_release(row):
    return (
        row.get("route_success") is True
        and row.get("route_version") == ROUTE_VERSION and row.get("native_release") is True
        and not row.get("invalid") and row.get("invalid_physics") is False
        and not row.get("label_conflict")
    )


def _groups(rows, routes):
    rows = list(rows)
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Route training requires nonempty rows with unique identities")
    requested = None if routes is None else set(routes)
    if requested is not None and (
        len(requested) < 2 or not all(isinstance(route, str) and route for route in requested)
        or requested & UNMODELED_ROUTES
    ):
        raise ValueError("Compare at least two explicitly supported route labels")
    groups = defaultdict(lambda: defaultdict(list))
    captures = []
    for row in sorted(rows, key=lambda item: item["id"]):
        route = row.get("shot_route")
        if (
            not isinstance(route, str) or not route or route in UNMODELED_ROUTES
            or (requested is not None and route not in requested)
        ):
            continue
        if row.get("phase") in {"test", "validation"} or row.get("role") in {"test", "validation"}:
            raise ValueError(f"Route fit received a held-out row: {row['id']}")
        if not _valid_release(row):
            continue
        source = dict(_source(row), route=route)
        key = tuple(round(float(value), 10) for value in source["xy"])
        groups[route][key].append(source)
        captures.append([source["id"], route, source["xy"].tolist(), source["values"].tolist(),
                         source["joints"].tolist()])
    if requested is not None and set(groups) != requested:
        raise ValueError(f"No valid captures for requested routes: {sorted(requested - groups.keys())}")
    return groups, captures, len(rows)


def _location_pairs(xy, target, buckets):
    """Use compatible examples from the nearest positions supporting this route."""
    neighbors = sorted((other for other in buckets if other != xy),
                       key=lambda other: (math.dist(xy, other), other))[:NEIGHBORS]
    pairs = []
    for other in neighbors:
        references = buckets[other]
        for start in range(0, len(target), 64):
            block = target[start:start + 64]
            chosen, distances = _matches(block, references)
            pairs.extend((row, references[index], float(distance))
                         for row, index, distance in zip(block, chosen, distances, strict=True)
                         if distance <= MAX_PAIR_DISTANCE)
    return pairs


def _training_pairs(groups, *, require_all):
    eligible = {}
    unsupported = {}
    for route, buckets in sorted(groups.items()):
        locations = [
            (xy, len(rows), _location_pairs(xy, rows, buckets))
            for xy, rows in sorted(buckets.items())
        ]
        paired_locations = [item for item in locations if item[2]]
        if len(paired_locations) < 2:
            unsupported[route] = dict(captures=sum(map(len, buckets.values())),
                                      capture_locations=len(buckets), paired_locations=len(paired_locations))
        else:
            eligible[route] = paired_locations
    if unsupported and require_all:
        raise ValueError(f"Requested route lacks matched recipes across two locations: {sorted(unsupported)}")
    if len(eligible) < 2:
        raise ValueError("Route comparison requires at least two routes with cross-location training pairs")

    blocks, reports, lineage = {}, {}, []
    for route, locations in eligible.items():
        x, y, weights, report = [], [], [], []
        location_weight = 1 / len(locations)
        for xy, count, pairs in locations:
            weight = location_weight / len(pairs)
            report.append(dict(bucket_xy_m=list(xy), captures=count, pairs=len(pairs),
                               conditional_loss_weight=location_weight,
                               joint_loss_weight=location_weight / len(eligible)))
            for target, source, distance in pairs:
                delta = target["values"] - source["values"]
                delta[5:] = _wrap_degrees(delta[5:])
                x.append(_features(target["xy"], source))
                y.append(delta / RECIPE_SCALE)
                weights.append(weight)
                lineage.append([route, target["id"], source["id"], distance, weight / len(eligible)])
        blocks[route] = np.asarray(x), np.asarray(y), np.asarray(weights)
        reports[route] = dict(
            captures=sum(map(len, groups[route].values())), capture_locations=len(groups[route]),
            paired_locations=len(locations), pairs=len(weights), loss_weight=1 / len(eligible),
            locations=report,
        )
    return blocks, reports, unsupported, digest(lineage)


def _solve(normal, rhs, device):
    if device == "cpu":
        return np.linalg.solve(normal, rhs)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_UUID:
        raise ValueError("CUDA route model fit requires only the authorized 4090 to be visible")
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected the single selected CUDA device")
    return torch.linalg.solve(torch.as_tensor(normal, device="cuda:0"),
                              torch.as_tensor(rhs, device="cuda:0")).cpu().numpy()


def _fit_models(blocks, device):
    """Use the same observations and loss weights for every formulation."""
    width = FEATURE_COUNT
    route_count = len(blocks)
    identity = np.eye(width)
    pooled_normal = RIDGE * identity
    pooled_rhs = np.zeros((width, len(DIMENSIONS)))
    conditioned_normal = np.zeros((width * (route_count + 1),) * 2)
    conditioned_rhs = np.zeros((width * (route_count + 1), len(DIMENSIONS)))
    shared = slice(0, width)
    conditioned_normal[shared, shared] = RIDGE * identity
    specialists = {}
    for index, (route, (x, y, weights)) in enumerate(blocks.items(), 1):
        normal = x.T @ (weights[:, None] * x)
        rhs = x.T @ (weights[:, None] * y)
        specialists[route] = _solve(normal + RIDGE * identity, rhs, device)
        normal /= route_count
        rhs /= route_count
        pooled_normal += normal
        pooled_rhs += rhs
        residual = slice(index * width, (index + 1) * width)
        conditioned_normal[shared, shared] += normal
        conditioned_normal[shared, residual] = conditioned_normal[residual, shared] = normal
        conditioned_normal[residual, residual] = normal + CONDITIONED_RESIDUAL_RIDGE / route_count * identity
        conditioned_rhs[shared] += rhs
        conditioned_rhs[residual] = rhs
    pooled = _solve(pooled_normal, pooled_rhs, device)
    conditioned = _solve(conditioned_normal, conditioned_rhs, device).reshape(route_count + 1, width, -1)
    return pooled, conditioned[0], dict(zip(blocks, conditioned[1:], strict=True)), specialists


def fit_route_models(rows, *, seed, device="cpu", routes=None):
    """Fit all three models from common matched positive TRAIN pairs."""
    if type(seed) is not int or seed < 0 or device not in {"cpu", "cuda", "cuda:0"}:
        raise ValueError("Specify a nonnegative run seed and cpu or the authorized cuda:0")
    groups, captures, training_rows = _groups(rows, routes)
    blocks, route_support, unsupported, pair_sha256 = _training_pairs(groups, require_all=routes is not None)
    pooled, shared, residuals, specialists = _fit_models(blocks, device)
    errors = {}
    for route, (x, y, weights) in blocks.items():
        matrices = (pooled, shared + residuals[route], specialists[route])
        errors[route] = {model: np.sqrt(weights @ ((x @ weights_matrix - y) ** 2)).tolist()
                         for model, weights_matrix in zip(MODELS, matrices, strict=True)}
    checkpoint = dict(
        version=VERSION, seed=seed, device="cpu" if device == "cpu" else "cuda:0",
        dimensions=list(DIMENSIONS), model_names=list(MODELS), routes=list(blocks),
        route_version=ROUTE_VERSION, training_rows=training_rows,
        training_captures=sum(row[1] in blocks for row in captures),
        training_pairs=sum(len(block[0]) for block in blocks.values()),
        training_positions=len({tuple(row[2]) for row in captures if row[1] in blocks}),
        dataset_sha256=digest(captures), pair_sha256=pair_sha256,
        route_support=route_support, unsupported_routes=unsupported, training_rmse=errors,
        loss="Uniform over route labels and uniform over paired target positions within each route.",
        ridge=RIDGE, conditioned_residual_ridge=CONDITIONED_RESIDUAL_RIDGE,
        neighbor_limit=NEIGHBORS, maximum_pair_distance=MAX_PAIR_DISTANCE,
        coefficients=dict(pooled=pooled.tolist(), conditioned_shared=shared.tolist(),
                          conditioned_residual={route: beta.tolist() for route, beta in residuals.items()},
                          specialist={route: beta.tolist() for route, beta in specialists.items()}),
    )
    checkpoint["fingerprint"] = digest(checkpoint)
    _coefficients(checkpoint)
    return checkpoint


def _matrix(value):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (FEATURE_COUNT, len(DIMENSIONS)) or not np.isfinite(matrix).all():
        raise ValueError("Route model coefficient shape or values differ")
    return matrix


def _coefficients(checkpoint):
    unsigned = {key: value for key, value in checkpoint.items() if key != "fingerprint"}
    routes = checkpoint.get("routes")
    if (
        checkpoint.get("version") != VERSION or checkpoint.get("model_names") != list(MODELS)
        or checkpoint.get("route_version") != ROUTE_VERSION or checkpoint.get("dimensions") != list(DIMENSIONS)
        or checkpoint.get("fingerprint") != digest(unsigned)
        or not isinstance(routes, list) or len(routes) < 2 or routes != sorted(set(routes))
    ):
        raise ValueError("Route model checkpoint schema or fingerprint differs")
    coefficients = checkpoint["coefficients"]
    residuals, specialists = coefficients["conditioned_residual"], coefficients["specialist"]
    if set(residuals) != set(routes) or set(specialists) != set(routes):
        raise ValueError("Route model coefficients omit a supported route")
    shared = _matrix(coefficients["conditioned_shared"])
    pooled = _matrix(coefficients["pooled"])
    return {
        "pooled": {route: pooled for route in routes},
        "conditioned": {route: shared + _matrix(residuals[route]) for route in routes},
        "specialist": {route: _matrix(specialists[route]) for route in routes},
    }


def propose_route_candidates(checkpoint, model, bucket_xy_m, sources, *, context_id="intergrid", limit=MAX_CANDIDATES):
    """Apply each frozen model to the same ordered saved recipe menu."""
    if model not in MODELS or type(limit) is not int or not 1 <= limit <= MAX_CANDIDATES:
        raise ValueError("Choose a supported model and a candidate limit from one to eight")
    target_xy = _vector(bucket_xy_m, 2, "target bucket")
    matrices = _coefficients(checkpoint)[model]
    seen, proposals = set(), []
    for source_index, row in enumerate(sources):
        route = row.get("intended_route", row.get("shot_route"))
        if route not in matrices:
            continue
        source = _source(row)
        change = (_features(target_xy, source) @ matrices[route]) * RECIPE_SCALE
        values = source["values"] + change
        values[5:] = _wrap_degrees(values[5:])
        if (
            not np.isfinite(values).all() or not ((LOW <= values[:5]) & (values[:5] <= HIGH)).all()
            or np.linalg.norm(target_xy - values[:2]) <= .15
        ):
            continue
        original = geometry_key(dict(values=source["values"][:5],
                                     board_heading_offset_deg=source["values"][5],
                                     board_twist_deg=source["values"][6]))
        proposal = dict(
            id=f"{context_id}-{model}-{source_index:02d}", context_id=context_id,
            phase="test", role="test", method=model, intended_route=route,
            bucket_xy_m=target_xy.tolist(), values=values[:5].tolist(),
            board_heading_offset_deg=float(values[5]), board_twist_deg=float(values[6]),
            seed_q10=source["joints"][:6].tolist(), seed_q14=source["joints"][6:].tolist(),
            source_index=source_index, training_recipe_id=source["id"],
            training_bucket_xy_m=source["xy"].tolist(),
            training_distance_m=float(np.linalg.norm(target_xy - source["xy"])),
            source_values=source["values"].tolist(), predicted_change=change.tolist(),
            route_models_fingerprint=checkpoint["fingerprint"],
        )
        key = geometry_key(proposal)
        if key in seen or key == original:
            continue
        seen.add(key)
        proposals.append(proposal)
        if len(proposals) == limit:
            break
    return proposals


def save_checkpoint(path, checkpoint):
    _coefficients(checkpoint)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(checkpoint, allow_nan=False, indent=2) + "\n"
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent) as stream:
        stream.write(data)
        stream.flush()
        os.link(stream.name, path)
    return checkpoint["fingerprint"]


def load_checkpoint(path):
    checkpoint = json.loads(Path(path).read_text(encoding="utf-8"))
    _coefficients(checkpoint)
    return checkpoint
