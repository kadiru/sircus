"""Predict continuous shot changes from a specific known capture and new target."""
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
from .expanded_actions import WIDE_BOUNDS
from .gpu_slot import GPU_UUID
from .intergrid_ranker import TRAIN_POSITIONS
from .table_assisted_score import ROUTE_VERSION

VERSION = "sircus-intergrid-proposer-v1"
NEIGHBORS = 4
RIDGE = 3e-3
MAX_PAIR_DISTANCE = 2.0
SPATIAL_SCALE_M = np.array([.20, .225])
RECIPE_CENTER = np.array([.67, -.075, .36, 32., .35, 0., 0.])
RECIPE_SCALE = np.array([.10, .10, .08, 8., .09, 16., 30.])
POSITION_SCALE_M = np.array([.57, .45])
DIMENSIONS = ("board_x_m", "board_y_m", "board_height_m", "board_tilt_deg", "drop_gap_m",
              "board_heading_offset_deg", "board_twist_deg")
LOW, HIGH = np.asarray(WIDE_BOUNDS["low"]), np.asarray(WIDE_BOUNDS["high"])


def _vector(value, width, label):
    array = np.asarray(value, dtype=float)
    if array.shape != (width,) or not np.isfinite(array).all():
        raise ValueError(f"Expected {width} finite values for {label}")
    return array


def _wrap_degrees(value):
    return (value + 180.) % 360. - 180.


def _recipe(row):
    return np.r_[
        _vector(row["values"], 5, "shot settings"),
        _vector([row.get("board_heading_offset_deg", 0.), row.get("board_twist_deg", 0.)],
                2, "orientation offsets"),
    ]


def _source(row):
    """Use either a saved recipe row or a proposal copied from a known recipe."""
    identity = row.get("training_recipe_id", row["id"])
    if not isinstance(identity, str) or not identity:
        raise ValueError("Expected a nonempty source recipe identity")
    xy = row.get("training_bucket_xy_m", row["bucket_xy_m"])
    q10 = row.get("seed_q10", row.get("q10"))
    q14 = row.get("seed_q14", row.get("q14"))
    return dict(
        id=identity, xy=_vector(xy, 2, "source bucket"), values=_recipe(row),
        joints=np.r_[_vector(q10, 6, "dropper seed"), _vector(q14, 6, "board-arm seed")],
        route=row.get("route_class"),
    )


def _features(target_xy, source):
    """Model spatial derivatives conditioned on the particular source shot."""
    delta = (target_xy - source["xy"]) / SPATIAL_SCALE_M
    position = ((target_xy + source["xy"]) / 2 - [.67, -.20]) / POSITION_SCALE_M
    recipe = (source["values"] - RECIPE_CENTER) / RECIPE_SCALE
    geometry = np.r_[1., position, recipe[:5], np.sin(np.deg2rad(source["values"][5:])),
                    np.cos(np.deg2rad(source["values"][5:])), np.sin(source["joints"]),
                    np.cos(source["joints"])]
    return np.r_[delta[0] * geometry, delta[1] * geometry, delta ** 2, delta.prod()]


FEATURE_COUNT = 75


def _successful_sources(rows):
    rows = list(rows)
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Training rows require unique identities")
    groups = defaultdict(list)
    source_records = []
    excluded_conflicts = 0
    for row in sorted(rows, key=lambda row: row["id"]):
        if row.get("label_conflict"):
            excluded_conflicts += 1
            continue
        if row.get("route_success") is not True:
            continue
        if (
            row.get("route_version") != ROUTE_VERSION or row.get("native_release") is not True
            or row.get("invalid") or row.get("invalid_physics") is not False
        ):
            raise ValueError(f"Successful recipe lacks valid standard-v3 native evidence: {row['id']}")
        source = _source(row)
        key = tuple(round(float(value), 10) for value in source["xy"])
        groups[key].append(source)
        source_records.append([source["id"], source["xy"].tolist(), source["values"].tolist(),
                               source["joints"].tolist(), source["route"]])
    if not TRAIN_POSITIONS.issubset(groups):
        missing = sorted(TRAIN_POSITIONS - groups.keys())
        raise ValueError(f"Successful examples required at all 33 original positions; missing {missing}")
    return groups, digest(source_records), len(rows), excluded_conflicts


def _matches(target, references):
    """Prefer a nearby matching shot and arm configuration over a new family."""
    target_values = np.asarray([row["values"] for row in target])
    reference_values = np.asarray([row["values"] for row in references])
    offsets = target_values[:, None, :] - reference_values[None, :, :]
    offsets[:, :, 5:] = _wrap_degrees(offsets[:, :, 5:])
    distances = ((offsets / RECIPE_SCALE) ** 2).mean(axis=2)
    target_joints = np.asarray([row["joints"] for row in target])
    reference_joints = np.asarray([row["joints"] for row in references])
    joint_error = target_joints[:, None, :] - reference_joints[None, :, :]
    distances += .35 * (1. - np.cos(joint_error)).mean(axis=2)
    target_routes = np.asarray([row["route"] for row in target], dtype=object)
    reference_routes = np.asarray([row["route"] for row in references], dtype=object)
    distances += .45 * (target_routes[:, None] != reference_routes[None, :])
    chosen = distances.argmin(axis=1)
    return chosen, distances[np.arange(len(target)), chosen]


def _pairs(groups):
    """Pair successful examples across positions and balance the target locations."""
    features, targets, weights, report, lineage = [], [], [], [], []
    for xy, target in sorted(groups.items()):
        neighbor_xy = sorted(
            (other for other in groups if other != xy),
            key=lambda other: (math.dist(other, xy), other),
        )[:NEIGHBORS]
        pairs = []
        for other in neighbor_xy:
            references = groups[other]
            # Keep temporary arrays bounded when a bucket has many saved captures.
            for begin in range(0, len(target), 64):
                block = target[begin:begin + 64]
                matches, distances = _matches(block, references)
                pairs.extend(
                    (shot, references[index], float(distance))
                    for shot, index, distance in zip(block, matches, distances, strict=True)
                    if distance <= MAX_PAIR_DISTANCE
                )
        if not pairs:
            raise ValueError(f"No compatible cross-position recipe pairs for training bucket {xy}")
        location_weight = 1 / len(groups)
        pair_weight = location_weight / len(pairs)
        for shot, reference, distance in pairs:
            change = shot["values"] - reference["values"]
            change[5:] = _wrap_degrees(change[5:])
            features.append(_features(shot["xy"], reference))
            targets.append(change / RECIPE_SCALE)
            weights.append(pair_weight)
            lineage.append([shot["id"], reference["id"], distance, pair_weight])
        report.append(dict(bucket_xy_m=target[0]["xy"].tolist(), captures=len(target),
                           pairs=len(pairs), loss_weight=location_weight))
    return np.asarray(features), np.asarray(targets), np.asarray(weights), report, digest(lineage)


def _fit(x, y, weights, device):
    normal = x.T @ (weights[:, None] * x) + RIDGE * np.eye(x.shape[1])
    rhs = x.T @ (weights[:, None] * y)
    if device == "cpu":
        return np.linalg.solve(normal, rhs)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_UUID:
        raise ValueError("CUDA proposer fit requires only the authorized 4090 to be visible")
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected the single selected CUDA device")
    return torch.linalg.solve(torch.as_tensor(normal, device="cuda:0"),
                              torch.as_tensor(rhs, device="cuda:0")).cpu().numpy()


def fit_proposer(rows, *, seed, device="cpu"):
    """Fit source-conditioned recipe changes using equally weighted target buckets."""
    if type(seed) is not int or seed < 0 or device not in ("cpu", "cuda", "cuda:0"):
        raise ValueError("Specify a nonnegative run seed and cpu or the authorized cuda:0")
    groups, dataset_sha256, training_rows, conflicts = _successful_sources(rows)
    x, y, weights, bucket_groups, pair_sha256 = _pairs(groups)
    coefficients = _fit(x, y, weights, device)
    checkpoint = dict(
        version=VERSION, dimensions=list(DIMENSIONS), route_version=ROUTE_VERSION,
        seed=seed, device="cpu" if device == "cpu" else "cuda:0",
        fit="weighted ridge on neighboring successful source/target recipe pairs",
        ridge=RIDGE, neighbor_limit=NEIGHBORS, maximum_pair_distance=MAX_PAIR_DISTANCE,
        training_rows=training_rows, training_captures=sum(map(len, groups.values())),
        excluded_conflicts=conflicts, base_positions=len(TRAIN_POSITIONS),
        additional_positions=len(groups) - len(TRAIN_POSITIONS), training_pairs=len(weights),
        dataset_sha256=dataset_sha256, pair_sha256=pair_sha256, bucket_groups=bucket_groups,
        normalized_training_rmse=np.sqrt(weights @ ((x @ coefficients - y) ** 2)).tolist(),
        coefficients=coefficients.tolist(),
    )
    checkpoint["fingerprint"] = digest(checkpoint)
    _coefficients(checkpoint)
    return checkpoint


def _coefficients(checkpoint):
    unsigned = {key: value for key, value in checkpoint.items() if key != "fingerprint"}
    if (
        checkpoint.get("version") != VERSION or checkpoint.get("dimensions") != list(DIMENSIONS)
        or checkpoint.get("route_version") != ROUTE_VERSION
        or checkpoint.get("fingerprint") != digest(unsigned)
    ):
        raise ValueError("Proposer checkpoint schema or fingerprint differs")
    coefficients = np.asarray(checkpoint["coefficients"], dtype=float)
    if coefficients.shape != (FEATURE_COUNT, len(DIMENSIONS)) or not np.isfinite(coefficients).all():
        raise ValueError("Proposer coefficient shape or values differ")
    return coefficients


def propose_candidates(checkpoint, bucket_xy_m, sources, *, context_id="intergrid", limit=None):
    """Predict one continuous variant per source, preserving the source order."""
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("A candidate limit must be a positive integer")
    target_xy = _vector(bucket_xy_m, 2, "target bucket")
    coefficients = _coefficients(checkpoint)
    seen, proposals = set(), []
    for row in sources:
        source = _source(row)
        change = (_features(target_xy, source) @ coefficients) * RECIPE_SCALE
        values = source["values"] + change
        values[5:] = _wrap_degrees(values[5:])
        original = geometry_key(dict(values=source["values"][:5],
                                     board_heading_offset_deg=source["values"][5],
                                     board_twist_deg=source["values"][6]))
        if (
            not np.isfinite(values).all()
            or not ((LOW <= values[:5]) & (values[:5] <= HIGH)).all()
            or np.linalg.norm(target_xy - values[:2]) <= .15
        ):
            continue
        proposal = dict(
            id=f"{context_id}-predicted-{len(proposals):02d}", context_id=context_id,
            phase="test", role="test", method="continuous_proposer",
            bucket_xy_m=target_xy.tolist(), values=values[:5].tolist(),
            board_heading_offset_deg=float(values[5]), board_twist_deg=float(values[6]),
            seed_q10=source["joints"][:6].tolist(), seed_q14=source["joints"][6:].tolist(),
            training_recipe_id=source["id"], training_bucket_xy_m=source["xy"].tolist(),
            training_distance_m=float(np.linalg.norm(target_xy - source["xy"])),
            source_values=source["values"].tolist(), predicted_change=change.tolist(),
            proposer_fingerprint=checkpoint["fingerprint"],
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
