"""Position-balanced supervised ranking of shot proposals before arm IK.

Inference uses only proposed geometry and optional *source* joint seeds. It
never reads solved joints, flight predictions, or observations from the query.
The probabilities rank proposals; they do not authorize a release or replace
the exact setup/path checks. Training is explicitly CPU-only and exports plain
JSON weights so inference does not import torch or any simulator dependency.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
import math


VERSION = "sircus-neural-pre-ik-shot-selector-v1"
FEATURE_VERSION = "sircus-proposed-geometry-source-seeds-v1"
DEFAULT_EPOCHS = (100, 250, 500)
ARCHITECTURE = (64, 32, 2)
LEARNING_RATE = .003
WEIGHT_DECAY = .0001
HEADS = ("setup", "capture_given_setup")
FEATURE_NAMES = (
    "bucket_x", "bucket_y", "board_x", "board_y", "bucket_minus_board_x",
    "bucket_minus_board_y", "board_bucket_distance", "board_height",
    "board_tilt", "release_gap", "board_height_plus_gap", "table_intent",
    "sin_bucket_bearing", "cos_bucket_bearing", "sin_heading_offset",
    "cos_heading_offset", "sin_board_twist", "cos_board_twist",
) + tuple(
    name for arm in (10, 14) for name in (
        f"source_seed_arm{arm}_missing",
        *(f"{function}_source_seed_arm{arm}_joint{joint}"
          for joint in range(1, 7) for function in ("sin", "cos")),
    )
)


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                             separators=(",", ":")).encode()).hexdigest()


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Shot features require finite numeric values")
    return float(value)


def _vector(value, length):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"Expected a {length}-value geometry vector")
    return [_number(number) for number in value]


def feature(request):
    """Encode an original proposal; solved/native fields are deliberately ignored."""
    x, y, height, tilt, gap = _vector(request["values"], 5)
    bx, by = _vector(request["bucket_xy_m"], 2)
    route = request["intended_route"]
    if route not in {"direct", "table"}:
        raise ValueError("Expected a direct or table route proposal")
    heading = math.radians(_number(request["board_heading_offset_deg"]))
    twist = math.radians(_number(request["board_twist_deg"]))
    dx, dy = bx - x, by - y
    bearing = math.atan2(dy, dx)
    values = [bx / .75, by / .75, x / .75, y / .75, dx / .6, dy / .6,
              math.hypot(dx, dy) / .6, height / .5, tilt / 45., gap / .5,
              (height + gap) / 1., float(route == "table"),
              math.sin(bearing), math.cos(bearing), math.sin(heading), math.cos(heading),
              math.sin(twist), math.cos(twist)]
    for arm in (10, 14):
        seed = request.get(f"seed_q{arm}")
        values.append(float(seed is None))
        if seed is None:
            values.extend([0.] * 12)
        else:
            for angle in _vector(seed, 6):
                values.extend((math.sin(angle), math.cos(angle)))
    if len(values) != len(FEATURE_NAMES) or not all(map(math.isfinite, values)):
        raise ValueError("Invalid proposed geometry encoding")
    return values


def _position(request):
    # Match the historical study precision, not row order or arbitrary case IDs.
    return tuple(round(value, 10) for value in _vector(request["bucket_xy_m"], 2))


def outcome(record):
    """Portable form of intergrid_matched_route_study.outcome with schema checks."""
    setup = record["setup_ok"]
    if type(setup) is not bool:
        raise ValueError("setup_ok must be boolean")
    row = record.get("observation")
    if not setup:
        if row is not None:
            raise ValueError("A rejected setup cannot have a native observation")
        return "setup_rejected"
    if row is None:
        # Admitted proposals beyond the collection release cap still provide
        # positive setup evidence, but no flight/capture label.
        return "setup_only"
    if not isinstance(row, dict) or row.get("native_release") is not True:
        raise ValueError("An admitted setup requires its scored native release")
    for key in ("invalid_physics", "route_success"):
        if type(row.get(key)) is not bool:
            raise ValueError("Native validity and capture labels must be boolean")
    robot_capture = row.get("experimental_route", {}).get("robot_assisted_capture")
    if type(robot_capture) is not bool:
        raise ValueError("Native observation must retain robot-assisted capture provenance")
    if row["invalid_physics"]:
        return "invalid_physics"
    if row["route_success"]:
        if robot_capture:
            raise ValueError("Standard capture and robot-assisted capture labels conflict")
        route = row.get("shot_route")
        if route not in {"direct", "table", None}:
            raise ValueError("Unexpected successful shot route")
        return route or "unclassified_capture"
    return "arm_assisted_capture" if robot_capture else "flight_miss"


def _encoded(rows, split, *, labeler=outcome):
    if not rows:
        raise ValueError(f"{split} data cannot be empty")
    data, identifiers = [], set()
    for record in rows:
        if record.get("split") != split:
            raise ValueError(f"Only explicit {split} rows are allowed in this split")
        request = record["request"]
        identifier = request.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("Proposal IDs must be nonempty and unique within each split")
        identifiers.add(identifier)
        label = labeler(record)
        setup = record["setup_ok"]
        capture_mask = setup and label not in {"invalid_physics", "setup_only"}
        capture = None if label == "setup_only" else label in {"direct", "table", "unclassified_capture"}
        data.append(dict(id=identifier, position=_position(request), feature=feature(request),
                         labels=[float(setup), float(capture is True)], masks=[1., float(capture_mask)],
                         outcome=label, setup_ok=setup, capture=capture))
    return data


def _weights(data, head):
    """Every position with a known label contributes equal weight to this head."""
    counts = Counter(row["position"] for row in data if row["masks"][head])
    sites = len(counts)
    if not sites:
        return [0.] * len(data)
    return [(1. / (sites * counts[row["position"]])) if row["masks"][head] else 0.
            for row in data]


def _validate_checkpoint(checkpoint):
    if (checkpoint.get("version") != VERSION or checkpoint.get("feature_version") != FEATURE_VERSION
            or checkpoint.get("feature_names") != list(FEATURE_NAMES)
            or checkpoint.get("heads") != list(HEADS)
            or checkpoint.get("activation") != "relu"
            or checkpoint.get("architecture") != [len(FEATURE_NAMES), *ARCHITECTURE]):
        raise ValueError("Unsupported neural selector checkpoint")
    layers = checkpoint.get("layers")
    if not isinstance(layers, list) or len(layers) != len(ARCHITECTURE):
        raise ValueError("Checkpoint requires three dense layers")
    inputs = len(FEATURE_NAMES)
    for layer, outputs in zip(layers, ARCHITECTURE, strict=True):
        if (not isinstance(layer, dict) or not isinstance(layer.get("weight"), list)
                or len(layer["weight"]) != outputs):
            raise ValueError("Incorrect checkpoint weight shape")
        for row in layer["weight"]:
            _vector(row, inputs)
        _vector(layer.get("bias"), outputs)
        inputs = outputs


def _logits(checkpoint, encoded):
    values = encoded
    for index, layer in enumerate(checkpoint["layers"]):
        values = [bias + math.fsum(weight * value for weight, value in zip(row, values, strict=True))
                  for row, bias in zip(layer["weight"], layer["bias"], strict=True)]
        if index < len(ARCHITECTURE) - 1:
            values = [max(0., value) for value in values]
    return values


def _sigmoid(value):
    exponential = math.exp(-abs(value))
    return 1. / (1. + exponential) if value >= 0 else exponential / (1. + exponential)


def predict(checkpoint, request):
    """Return ranking probabilities, not a guarantee or permission to execute."""
    _validate_checkpoint(checkpoint)
    setup, capture = map(_sigmoid, _logits(checkpoint, feature(request)))
    return dict(setup_probability=setup, capture_given_setup_probability=capture,
                capture_probability=setup * capture)


def order(checkpoint, proposals):
    """Stable ranking of original pre-IK proposals; exact admission comes later."""
    _validate_checkpoint(checkpoint)
    scored = []
    seen = set()
    for index, request in enumerate(proposals):
        identifier = request.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("Ranked proposals need distinct nonempty IDs")
        seen.add(identifier)
        setup, capture = map(_sigmoid, _logits(checkpoint, feature(request)))
        scored.append((-setup * capture, index, request))
    return [row for _, _, row in sorted(scored, key=lambda item: item[:2])]


def _evaluate(checkpoint, data):
    logits = [_logits(checkpoint, row["feature"]) for row in data]
    losses = []
    for head in range(2):
        weights = _weights(data, head)
        losses.append(math.fsum(weight * (max(z[head], 0.) - z[head] * row["labels"][head]
                                         + math.log1p(math.exp(-abs(z[head]))))
                                for row, z, weight in zip(data, logits, weights, strict=True)))
    grouped = defaultdict(list)
    for index, (row, z) in enumerate(zip(data, logits, strict=True)):
        score = _sigmoid(z[0]) * _sigmoid(z[1])
        grouped[row["position"]].append((-score, index, row))
    cases = []
    for xy, menu in sorted(grouped.items()):
        ordered = sorted(menu, key=lambda entry: entry[:2])
        chosen = ordered[0][2]
        feasible = next((entry[2] for entry in ordered if entry[2]["setup_ok"]), None)
        cases.append(dict(bucket_xy_m=list(xy), top_id=chosen["id"],
                          top_setup=chosen["setup_ok"], top_capture=chosen["capture"],
                          first_feasible_id=None if feasible is None else feasible["id"],
                          first_feasible_capture=False if feasible is None else feasible["capture"]))
    return dict(loss=sum(losses), setup_loss=losses[0], conditional_capture_loss=losses[1],
                positions=len(cases), top_proposal_setups=sum(row["top_setup"] for row in cases),
                top_proposal_captures=sum(row["top_capture"] is True for row in cases),
                top_proposal_capture_unknown=sum(row["top_capture"] is None for row in cases),
                first_feasible_captures=sum(row["first_feasible_capture"] is True for row in cases),
                first_feasible_capture_unknown=sum(row["first_feasible_capture"] is None for row in cases), cases=cases)


def fit(training_rows, validation_rows, *, seed=2026111707, epochs=DEFAULT_EPOCHS,
        labeler=None, label_version=None):
    """Fit two binary heads on TRAIN only; select an epoch by validation loss.

    Each record must contain request/original proposal, setup_ok, observation
    (None after rejection or an admitted but unreleased proposal), and explicit
    split ('train' or 'validation'). Invalid physics and unreleased admitted
    proposals contribute only to the setup head. Robot-assisted capture
    remains a noncapture for this standard direct/table task. No TEST rows are
    accepted. Validation selects an epoch; its labels never enter gradients.
    """
    # An explicit versioned labeler allows a new task definition without
    # rewriting archived observations or changing this historical default.
    if (labeler is None) != (label_version is None):
        raise ValueError("A custom labeler requires its explicit label version")
    if label_version is not None and (not callable(labeler) or not isinstance(label_version, str) or not label_version):
        raise ValueError("Expected a callable versioned labeler")
    training = _encoded(training_rows, "train", labeler=labeler or outcome)
    validation = _encoded(validation_rows, "validation", labeler=labeler or outcome)
    positions = [{row["position"] for row in dataset} for dataset in (training, validation)]
    identifiers = [{row["id"] for row in dataset} for dataset in (training, validation)]
    if positions[0] & positions[1] or identifiers[0] & identifiers[1]:
        raise ValueError("Training and validation positions and IDs must be disjoint")
    if not all(any(row["masks"][head] for row in dataset) for dataset in (training, validation)
               for head in range(2)):
        raise ValueError("Both splits need valid native capture labels")
    if (not isinstance(epochs, (tuple, list)) or not epochs
            or any(type(epoch) is not int or epoch < 1 for epoch in epochs)
            or list(epochs) != sorted(set(epochs)) or max(epochs) > 5000):
        raise ValueError("Epoch checkpoints must be unique increasing positive integers, at most 5000")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Expected a nonnegative integer training seed")

    import torch  # No CUDA device is queried or initialized by this module.
    from torch import nn

    metadata = dict(version=VERSION, feature_version=FEATURE_VERSION, feature_names=list(FEATURE_NAMES),
                    heads=list(HEADS), architecture=[len(FEATURE_NAMES), *ARCHITECTURE],
                    activation="relu", seed=seed, device="cpu", learning_rate=LEARNING_RATE,
                    weight_decay=WEIGHT_DECAY, checkpoint_epochs=list(epochs),
                    training_sha256=_digest(training_rows), validation_sha256=_digest(validation_rows),
                    training_positions=len(positions[0]), validation_positions=len(positions[1]),
                    training_rows=len(training), validation_rows=len(validation),
                    training_outcomes=dict(sorted(Counter(row["outcome"] for row in training).items())),
                    validation_outcomes=dict(sorted(Counter(row["outcome"] for row in validation).items())),
                    selection="Lowest sum of position-balanced validation head losses; earlier epoch on ties",
                    input_scope="Proposed geometry and optional source joint seeds; no solved joints or rollout data")
    if label_version is not None:
        metadata["label_version"] = label_version
    # Local CPU generator initialization avoids altering a caller's random state.
    threads = torch.get_num_threads()
    history, best = [], None
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            network = nn.Sequential(nn.Linear(len(FEATURE_NAMES), 64), nn.ReLU(),
                                    nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2)).cpu()
            optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            x = torch.tensor([row["feature"] for row in training], dtype=torch.float32, device="cpu")
            target = torch.tensor([row["labels"] for row in training], dtype=torch.float32, device="cpu")
            weight = torch.tensor(list(zip(_weights(training, 0), _weights(training, 1), strict=True)),
                                  dtype=torch.float32, device="cpu")
            for epoch in range(1, epochs[-1] + 1):
                optimizer.zero_grad(set_to_none=True)
                loss = (nn.functional.binary_cross_entropy_with_logits(network(x), target, reduction="none")
                        * weight).sum()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                optimizer.step()
                if epoch not in epochs:
                    continue
                layers = [dict(weight=layer.weight.detach().tolist(), bias=layer.bias.detach().tolist())
                          for layer in network if isinstance(layer, nn.Linear)]
                checkpoint = dict(metadata, epoch=epoch, layers=layers)
                metrics = _evaluate(checkpoint, validation)
                history.append(dict(epoch=epoch, train_loss=float(loss.detach()), **metrics))
                if best is None or metrics["loss"] < best[0]:
                    best = (metrics["loss"], checkpoint, metrics)
    finally:
        torch.set_num_threads(threads)
    checkpoint = best[1]
    checkpoint["fingerprint"] = _digest(checkpoint)
    return dict(checkpoint=checkpoint, validation=dict(selected_epoch=checkpoint["epoch"],
                selected=best[2], history=history, checkpoint_sha256=_digest(checkpoint),
                note="Validation diagnostics select training epoch; fresh native TEST remains required"))
