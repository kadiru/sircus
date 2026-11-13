"""Train a frozen binary capture ranker on the 33 available fixed-grid positions."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .benchmark_verify import digest
from .corrected_contract import grid_layouts
from .gpu_slot import GPU_UUID
from .table_assisted_score import ROUTE_VERSION

VERSION = "sircus-intergrid-ranker-v1"
HIDDEN = 32
LEARNING_RATE = .003
WEIGHT_DECAY = 1e-4
BLOCKED = {(3, 0), (3, 6)}
TRAIN_POSITIONS = frozenset(
    tuple(round(value, 10) for value in row["bucket_xy_m"])
    for row in grid_layouts() if (row["row"], row["column"]) not in BLOCKED
)
ANGLES = ("heading", "twist") + tuple(f"arm{arm}_joint_{j}" for arm in (10, 14) for j in range(1, 7))
FEATURES = (
    "bucket_x_m", "bucket_y_m", "board_x_m", "board_y_m", "board_height_m",
    "board_tilt_deg", "drop_gap_m", "board_to_bucket_x_m", "board_to_bucket_y_m",
    "board_to_bucket_distance_m", "release_height_without_ball_radius_m",
) + tuple(f"{function}_{angle}" for function in ("sin", "cos") for angle in ANGLES)


def _vector(value, width, label):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (width,) or not np.isfinite(array).all():
        raise ValueError(f"Expected {width} finite values for {label}")
    return array


def _joints(row, name):
    value = row.get("selected_" + name)
    return _vector(row.get(name) if value is None else value, 6, name)


def features(row):
    """Encode pre-release observations and periodic joint/orientation angles."""
    bucket = _vector(row["bucket_xy_m"], 2, "bucket position")
    values = _vector(row["values"], 5, "shot parameters")
    board_angles = _vector(
        [row.get("board_heading_offset_deg", 0.0), row.get("board_twist_deg", 0.0)],
        2, "board orientation offsets",
    )
    angles = np.r_[np.deg2rad(board_angles), _joints(row, "q10"), _joints(row, "q14")]
    delta = bucket - values[:2]
    return np.r_[bucket, values, delta, np.linalg.norm(delta), values[2] + values[4],
                 np.sin(angles), np.cos(angles)]


def _training_data(rows):
    rows = list(rows)
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Training examples must have distinct nonempty identities")
    by_position = {}
    for row in sorted(rows, key=lambda item: item["id"]):
        if not isinstance(row["id"], str) or not row["id"]:
            raise ValueError("Training examples must have nonempty string identities")
        if (
            row.get("native_release") is not True or row.get("invalid")
            or row.get("invalid_physics") is not False or row.get("route_version") != ROUTE_VERSION
            or type(row.get("route_success")) is not bool
        ):
            raise ValueError(f"Expected a valid standard-v3 native release: {row['id']}")
        xy = tuple(round(float(value), 10) for value in _vector(row["bucket_xy_m"], 2, "training bucket"))
        if xy not in TRAIN_POSITIONS:
            raise ValueError(f"Training row falls outside the 33 allowed positions: {row['id']}")
        if row.get("label_conflict") is True:
            continue
        by_position.setdefault(xy, []).append((row["id"], features(row), int(row["route_success"])))
    if set(by_position) != TRAIN_POSITIONS:
        missing = sorted(TRAIN_POSITIONS - by_position.keys())
        raise ValueError(f"Every one of the 33 training positions requires an eligible release; missing {missing}")

    matrix, labels, weights, groups, source = [], [], [], [], []
    for xy, examples in sorted(by_position.items()):
        per_row_weight = 1 / (len(by_position) * len(examples))
        groups.append(dict(bucket_xy_m=list(xy), count=len(examples),
                           captures=sum(label for _, _, label in examples), loss_weight=1 / len(by_position)))
        for identifier, vector, label in examples:
            matrix.append(vector)
            labels.append(label)
            weights.append(per_row_weight)
            source.append([identifier, vector.tolist(), label])
    if set(labels) != {0, 1}:
        raise ValueError("Binary training requires both native captures and misses")
    return (np.asarray(matrix), np.asarray(labels, dtype=float), np.asarray(weights), groups,
            digest(source), len(rows) - len(labels))


def _initial_parameters(width, seed):
    rng = np.random.default_rng(seed)
    return (
        rng.normal(0.0, 1 / math.sqrt(width), size=(width, HIDDEN)),
        np.zeros(HIDDEN),
        rng.normal(0.0, 1 / math.sqrt(HIDDEN), size=HIDDEN),
        np.zeros(1),
    )


def _sigmoid(logits):
    return 1 / (1 + np.exp(-np.clip(logits, -50, 50)))


def _numpy_fit(x, labels, weights, initial, epochs):
    params = list(initial)
    first = [np.zeros_like(p) for p in params]
    second = [np.zeros_like(p) for p in params]
    curve = []
    for step in range(epochs + 1):
        w1, b1, w2, b2 = params
        hidden = np.tanh(x @ w1 + b1)
        logits = hidden @ w2 + b2
        if step in (0, epochs) or step % 20 == 0:
            losses = np.logaddexp(0, logits) - labels * logits
            curve.append(dict(step=step, weighted_bce=float(losses @ weights)))
        if step == epochs:
            break
        derivative = (_sigmoid(logits) - labels) * weights
        first_layer = derivative[:, None] * w2 * (1 - hidden ** 2)
        gradients = [x.T @ first_layer + WEIGHT_DECAY * w1, first_layer.sum(axis=0),
                     hidden.T @ derivative + WEIGHT_DECAY * w2, np.array([derivative.sum()])]
        for i, gradient in enumerate(gradients):
            first[i] = .9 * first[i] + .1 * gradient
            second[i] = .999 * second[i] + .001 * gradient ** 2
            mean = first[i] / (1 - .9 ** (step + 1))
            square = second[i] / (1 - .999 ** (step + 1))
            params[i] -= LEARNING_RATE * mean / (np.sqrt(square) + 1e-8)
    return params, curve


def _cuda_fit(x, labels, weights, initial, epochs):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_UUID:
        raise ValueError("CUDA ranker fit requires only the authorized 4090 to be visible")
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected the single selected CUDA device")
    device = "cuda:0"
    params = [torch.tensor(p, dtype=torch.float64, device=device, requires_grad=True) for p in initial]
    x, labels, weights = (torch.as_tensor(value, device=device) for value in (x, labels, weights))
    w1, b1, w2, b2 = params
    optimizer = torch.optim.Adam([
        dict(params=[w1, w2], weight_decay=WEIGHT_DECAY),
        dict(params=[b1, b2], weight_decay=0),
    ], lr=LEARNING_RATE)
    curve = []
    for step in range(epochs + 1):
        logits = torch.tanh(x @ w1 + b1) @ w2 + b2
        losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        loss = losses @ weights
        if step in (0, epochs) or step % 20 == 0:
            curve.append(dict(step=step, weighted_bce=float(loss.detach().cpu())))
        if step == epochs:
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return [p.detach().cpu().numpy() for p in params], curve


def fit_ranker(rows, *, seed, device="cpu", epochs=120):
    """Fit equal-position weighted binary cross-entropy and freeze the parameters."""
    if type(seed) is not int or seed < 0 or type(epochs) is not int or not 1 <= epochs <= 1000:
        raise ValueError("Specify a nonnegative seed and 1–1,000 training steps")
    if device not in ("cpu", "cuda", "cuda:0"):
        raise ValueError("Ranker device must be cpu or the authorized cuda:0")
    x, labels, weights, groups, dataset_sha256, excluded = _training_data(rows)
    center = weights @ x
    scale = np.sqrt(weights @ (x - center) ** 2)
    scale = np.maximum(scale, 1e-6)
    normalized = (x - center) / scale
    initial = _initial_parameters(normalized.shape[1], seed)
    fit = _numpy_fit if device == "cpu" else _cuda_fit
    parameters, curve = fit(normalized, labels, weights, initial, epochs)
    checkpoint = dict(
        version=VERSION, features=list(FEATURES), route_version=ROUTE_VERSION,
        seed=seed, device="cpu" if device == "cpu" else "cuda:0", steps=epochs,
        loss="standard v3 binary cross-entropy, each training bucket contributes 1/33",
        learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        dataset_sha256=dataset_sha256, training_rows=len(labels), excluded_conflicts=excluded,
        bucket_groups=groups, center=center.tolist(), scale=scale.tolist(),
        parameters=[p.tolist() for p in parameters], curve=curve,
    )
    checkpoint["fingerprint"] = digest(checkpoint)
    _validated_parameters(checkpoint)
    return checkpoint


def _validated_parameters(checkpoint):
    unsigned = {key: value for key, value in checkpoint.items() if key != "fingerprint"}
    if (
        checkpoint.get("version") != VERSION or checkpoint.get("route_version") != ROUTE_VERSION
        or checkpoint.get("features") != list(FEATURES)
        or checkpoint.get("fingerprint") != digest(unsigned)
    ):
        raise ValueError("Ranker checkpoint schema or fingerprint differs")
    size = len(FEATURES)
    center = _vector(checkpoint["center"], size, "feature center")
    scale = _vector(checkpoint["scale"], size, "feature scale")
    if (scale <= 0).any():
        raise ValueError("Ranker checkpoint requires positive feature scales")
    parameters = [np.asarray(value, dtype=float) for value in checkpoint["parameters"]]
    if [p.shape for p in parameters] != [(size, HIDDEN), (HIDDEN,), (HIDDEN,), (1,)] or not all(
        np.isfinite(p).all() for p in parameters
    ):
        raise ValueError("Ranker parameter shapes or values differ")
    return center, scale, parameters


def predict_scores(checkpoint, candidates):
    """Return frozen capture scores using only geometry and feasible joint commands."""
    center, scale, (w1, b1, w2, b2) = _validated_parameters(checkpoint)
    values = [features(candidate) for candidate in candidates]
    if not values:
        return np.empty(0)
    x = (np.asarray(values) - center) / scale
    return _sigmoid(np.tanh(x @ w1 + b1) @ w2 + b2)


def rank_candidates(checkpoint, candidates):
    """Return candidate indices in descending score, preserving exact ties."""
    return np.argsort(-predict_scores(checkpoint, candidates), kind="stable").tolist()


def save_checkpoint(path, checkpoint):
    """Write a verified checkpoint exactly once and return its model fingerprint."""
    _validated_parameters(checkpoint)
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
    _validated_parameters(checkpoint)
    return checkpoint
