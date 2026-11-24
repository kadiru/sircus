"""Position-balanced supervised selector with one row per distinct setting.

Repeated observations supply a fractional capture target, never extra weight.
The network ranks proposals; exact arm setup/path checks still authorize them.
"""
from __future__ import annotations

import math
from . import neural_shot_model as base

VERSION = "sircus-expanded-deduplicated-selector-v1"


def probability(value):
    if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("A finite probability in [0, 1] is required")
    return float(value)


def encoded(rows, split):
    if not rows:
        raise ValueError("A nonempty declared split is required")
    data, seen = [], set()
    for row in rows:
        if row.get("split") != split:
            raise ValueError("Only the declared training/validation split is allowed")
        identifier = row["request"]["id"]
        key = row["dedup_key"]
        if not identifier or key in seen:
            raise ValueError("Settings must be deduplicated before fitting")
        seen.add(key)
        setup = probability(row["setup_probability"])
        capture = row.get("capture_probability")
        valid = row["valid_native_count"]
        if type(valid) is not int or valid < 0 or (capture is None) != (valid == 0):
            raise ValueError("Capture target must use only known valid native outcomes")
        if valid and setup != 1:
            raise ValueError("Native outcome requires an admitted setup")
        data.append(dict(id=identifier, position=base._position(row["request"]),
                         feature=base.feature(row["request"]),
                         labels=[setup, 0. if capture is None else probability(capture)],
                         masks=[1., float(valid > 0)]))
    return data


def metrics(checkpoint, data):
    losses = []
    for head in range(2):
        terms = []
        for row, weight in zip(data, base._weights(data, head), strict=True):
            z = base._logits(checkpoint, row["feature"])[head]
            terms.append(weight * (max(z, 0.) - z * row["labels"][head] + math.log1p(math.exp(-abs(z)))))
        losses.append(math.fsum(terms))
    return dict(loss=sum(losses), setup_loss=losses[0], conditional_capture_loss=losses[1])


def fit(training_rows, validation_rows, *, seed=2026112411, epochs=(100, 250, 500)):
    training, validation = encoded(training_rows, "train"), encoded(validation_rows, "validation")
    positions = [{row["position"] for row in data} for data in (training, validation)]
    if positions[0] & positions[1]:
        raise ValueError("All settings and repeats from a position must stay in one split")
    if any(not any(row["masks"][head] for row in data) for head in range(2) for data in (training, validation)):
        raise ValueError("Both splits require valid capture observations")
    if (not epochs or list(epochs) != sorted(set(epochs))
            or any(type(v) is not int or not 0 < v <= 5000 for v in epochs)):
        raise ValueError("Expected increasing positive epoch checkpoints <=5000")
    import torch
    from torch import nn
    metadata = dict(version=base.VERSION, feature_version=base.FEATURE_VERSION,
        feature_names=list(base.FEATURE_NAMES), heads=list(base.HEADS),
        architecture=[len(base.FEATURE_NAMES), *base.ARCHITECTURE], activation="relu",
        label_version=VERSION, seed=seed, device="cpu", learning_rate=base.LEARNING_RATE,
        weight_decay=base.WEIGHT_DECAY, checkpoint_epochs=list(epochs),
        training_sha256=base._digest(training_rows), validation_sha256=base._digest(validation_rows),
        training_positions=len(positions[0]), validation_positions=len(positions[1]),
        training_rows=len(training), validation_rows=len(validation),
        weighting="Equal weight per bucket position per head; equal weight per distinct setting within position; repeats estimate fractional targets only",
        selection="Lowest position-balanced validation BCE; earlier epoch on ties",
        input_scope="Proposed geometry and original pre-IK source hints; no solved query joints, sensor input or native test outcomes")
    threads = torch.get_num_threads()
    history, best = [], None
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            net = nn.Sequential(nn.Linear(len(base.FEATURE_NAMES), 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2)).cpu()
            optimizer = torch.optim.Adam(net.parameters(), lr=base.LEARNING_RATE, weight_decay=base.WEIGHT_DECAY)
            x = torch.tensor([r["feature"] for r in training], dtype=torch.float32)
            y = torch.tensor([r["labels"] for r in training], dtype=torch.float32)
            weight = torch.tensor(list(zip(base._weights(training, 0), base._weights(training, 1), strict=True)), dtype=torch.float32)
            vx = torch.tensor([r["feature"] for r in validation], dtype=torch.float32)
            vy = torch.tensor([r["labels"] for r in validation], dtype=torch.float32)
            vw = torch.tensor(list(zip(base._weights(validation, 0), base._weights(validation, 1), strict=True)), dtype=torch.float32)
            for epoch in range(1, epochs[-1]+1):
                optimizer.zero_grad(set_to_none=True)
                loss = (nn.functional.binary_cross_entropy_with_logits(net(x), y, reduction="none") * weight).sum()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                optimizer.step()
                if epoch not in epochs:
                    continue
                with torch.no_grad():
                    vl = (nn.functional.binary_cross_entropy_with_logits(net(vx), vy, reduction="none") * vw).sum(dim=0).tolist()
                metric = dict(epoch=epoch, train_loss=float(loss.detach()), loss=sum(vl), setup_loss=vl[0], conditional_capture_loss=vl[1])
                history.append(metric)
                if best is None or metric["loss"] < best[0]:
                    layers = [dict(weight=l.weight.detach().tolist(), bias=l.bias.detach().tolist()) for l in net if isinstance(l, nn.Linear)]
                    best = (metric["loss"], dict(metadata, epoch=epoch, layers=layers))
    finally:
        torch.set_num_threads(threads)
    checkpoint = best[1]
    checkpoint["fingerprint"] = base._digest(checkpoint)
    return dict(checkpoint=checkpoint, validation=dict(selected_epoch=checkpoint["epoch"], history=history))
