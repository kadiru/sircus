"""Bounded cooperative shot-learning study; privileged geometry, no hardware."""

import hashlib
import json
import numpy as np

VERSION = "sircus-mjlab-cooperative-bandit-v1"
NAMES = ["board_x_m", "board_y_m", "board_height_m", "tilt_deg", "gap_m"]
LOW = np.array([.32, -.30, .18, 12., .065])
HIGH = np.array([1.01, .25, .53, 48., .52])
WARM = np.array([.65, -.04, .36, 25., .24])
SEED = 20261103


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def splits():
    train = [[float(x), float(y)] for y in np.linspace(-.65, .25, 5)
             for x in np.linspace(.10, 1.24, 7)]
    rng = np.random.default_rng(SEED + 1)
    validation = rng.uniform([.1, -.65], [1.24, .25], (20, 2)).tolist()
    rng = np.random.default_rng(SEED + 2)
    test = rng.uniform([.1, -.65], [1.24, .25], (50, 2)).tolist()
    assert len({tuple(x) for x in train + validation + test}) == 105
    return dict(train=train, validation=validation, test=test)


def constraints(n):
    return dict(board=np.full((n, 3), np.nan), dropper=np.full((n, 3), np.nan))


def observations(bucket, locks):
    bucket = np.asarray(bucket)
    validate_locks(locks, len(bucket))
    b, d = np.asarray(locks["board"]), np.asarray(locks["dropper"])
    bm, dm = np.isfinite(b).all(1), np.isfinite(d).all(1)
    # Masks distinguish a real zero coordinate from an absent constraint.
    return np.column_stack([(bucket - [.67, -.20]) / [.6, .5], bm, dm,
                            np.nan_to_num(b) / [1., .7, .6],
                            np.nan_to_num(d) / [1., .7, 1.2]]).astype(np.float32)


def actions(latents):
    return LOW + (np.tanh(np.asarray(latents)) + 1) * (HIGH - LOW) / 2


def latents(values):
    return np.arctanh(np.clip(2 * (np.asarray(values) - LOW) / (HIGH - LOW) - 1, -.999, .999))


def project(values, locks, radius=.0375):
    """Exact positional locks. No solver silently relaxes an operator constraint."""
    a = np.asarray(values, dtype=float).copy()
    validate_locks(locks,len(a))
    b, d = np.asarray(locks["board"]), np.asarray(locks["dropper"])
    bm, dm = np.isfinite(b).all(1), np.isfinite(d).all(1)
    a[bm, :3] = b[bm]
    # Restricted center-impact family: release XY and board XY coincide.
    a[dm & ~bm, :2] = d[dm & ~bm, :2]
    a[dm, 4] = d[dm, 2] - a[dm, 2] - radius
    conflict = bm & dm & (np.linalg.norm(b[:, :2] - d[:, :2], axis=1) > 1e-9)
    valid = np.isfinite(a).all(1) & ~conflict
    valid &= (a >= LOW - 1e-9).all(1) & (a <= HIGH + 1e-9).all(1)
    return a, valid


def validate_locks(locks,n):
    if set(locks)!={"board","dropper"}:
        raise ValueError("Expected board and dropper lock arrays")
    for values in locks.values():
        v=np.asarray(values)
        if v.shape!=(n,3) or not (np.isnan(v).all(1)|np.isfinite(v).all(1)).all():
            raise ValueError("Each positional lock must be all finite or all NaN (unlocked)")


def proposals(bucket, rng):
    """Exploration seed: varied geometry and approximate ballistic energy.

    This analytic proposal is not used as a learned policy or a capture label.
    Only native mjlab trajectories determine rewards and success.
    """
    n = len(bucket)
    a = rng.uniform(LOW, HIGH, (n, 5))
    a[:, 0] = rng.uniform(.43, .89, n)
    a[:, 1] = rng.uniform(-.20, .19, n)
    a[:, 2] = rng.uniform(.22, .47, n)
    a[:, 3] = rng.uniform(16, 42, n)
    distance = np.linalg.norm(np.asarray(bucket) - a[:, :2], axis=1)
    theta = np.deg2rad(a[:, 3])
    s = 1.65 * np.sin(theta) * np.cos(theta)
    c = 1.65 * np.cos(theta) ** 2 - 1
    den = 4 * s**2 * (a[:, 2] - .1455 + c / s * distance)
    gap = distance**2 / np.maximum(den, .05)
    a[:, 4] = np.clip(gap * rng.uniform(.70, 1.30, n), LOW[4], HIGH[4])
    return a


def reward(row):
    if row["invalid"]:
        return -1.
    # Proximity is retained for misses, but external contacts never receive
    # capture/retention bonuses. The terminal capture definition is unchanged.
    shaped = .15 * float(row["board_bounce"]) + .5 * np.exp(-row["closest_opening_distance_m"] / .12)
    if row["external_contact_steps"]:
        return float(shaped - .5)
    return float(shaped + .5 * min(row["max_retention_s"], 1.)
                 + 3 * row["route_success"] + .3 * (row["route_quality"] == 2)
                 + .1 * row["single_bounce_success"])


def statistics(rows):
    return dict(attempts=len(rows), captures=sum(bool(r["route_success"]) and not r["invalid"] for r in rows),
                clean=sum(r["route_quality"] == 2 and not r["invalid"] for r in rows),
                invalid=sum(bool(r["invalid"]) for r in rows),
                board_bounces=sum(bool(r["board_bounce"]) and not r["invalid"] for r in rows),
                mean_reward=float(np.mean([reward(r) for r in rows])) if rows else 0.)
