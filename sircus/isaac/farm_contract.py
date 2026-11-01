"""Versioned one-decision PPO shot task for 32 fixed, distinct bucket targets."""

import copy
import hashlib
import json
import math

import numpy as np
from parallel_contract import cells

VERSION = "sircus-parallel-shot-ppo-v1"
LOW = np.array([0.30, 0.08, 12.0])
HIGH = np.array([0.45, 0.25, 40.0])
WARM = np.array([0.35, 0.22, 24.0])
FIXED = np.array([0.35, 0.24, 24.0])
SEED = 20261103


def observation(cell):
    dx, dy = np.array(cell["bucket_xy_m"]) - cell["board_xy_m"]
    return np.array(
        [
            dx / 0.25,
            dy / 0.70,
            cell["distance_m"] / 0.70,
            math.sin(math.radians(cell["heading_deg"])),
            math.cos(math.radians(cell["heading_deg"])),
        ],
        dtype=np.float32,
    )


def actions(latents):
    values = LOW + (np.tanh(np.asarray(latents)) + 1) * (HIGH - LOW) / 2
    return [
        dict(id="ppo", height_m=float(h), gap_m=float(g), tilt_deg=float(t))
        for h, g, t in np.atleast_2d(values)
    ]


def reward(row):
    """No capture reward for table-assisted/other external routes; legacy unchanged."""
    if row["invalid"]:
        return -1.0
    if row["external_contact_steps"]:
        return -0.5
    if not row["board_bounce"]:
        return -0.25
    return float(
        0.10
        + 0.25 * math.exp(-row["closest_opening_distance_m"] / 0.12)
        + 0.25 * min(row["max_retention_s"], 1.0)
        + float(row["route_success"])
        + 0.10 * (row["route_quality"] == 2)
        + 0.05 * row["single_bounce_success"]
    )


def splits(count=32):
    base = cells(32)[:count]
    result = {"train": copy.deepcopy(base)}
    for label, seed in (("validation", SEED + 1), ("test", SEED + 2)):
        rng = np.random.default_rng(seed)
        group = copy.deepcopy(base)
        for c in group:
            delta = rng.uniform(0.018, 0.036, 2) * rng.choice([-1, 1], 2)
            c["bucket_xy_m"] = (np.array(c["bucket_xy_m"]) + delta).tolist()
            dx, dy = np.array(c["bucket_xy_m"]) - c["board_xy_m"]
            c["distance_m"] = float(np.hypot(dx, dy))
            c["heading_deg"] = math.degrees(math.atan2(dy, dx) + math.pi / 2)
            c["id"] = label + "-" + c["id"]
        result[label] = group
    xy = [tuple(c["bucket_xy_m"]) for group in result.values() for c in group]
    if len(set(xy)) != len(xy):
        raise ValueError("Scene split overlap")
    return result


def statistics(rows):
    return dict(
        attempts=len(rows),
        captures=sum(r["route_success"] for r in rows),
        clean=sum(r["route_quality"] == 2 for r in rows),
        single_bounce=sum(r["single_bounce_success"] for r in rows),
        invalid=sum(bool(r["invalid"]) for r in rows),
        mean_reward=float(np.mean([reward(r) for r in rows])),
    )


def contract(cfg, updates, batches, count=32):
    if not 1 <= updates <= 32 or not 1 <= batches <= 4 or count not in (4, 32):
        raise ValueError("Unbounded farm experiment")
    value = dict(
        version=VERSION,
        scene=cfg,
        seed=SEED,
        num_envs=count,
        updates=updates,
        batches_per_update=batches,
        training_throws=updates * batches * count,
        observation_dim=5,
        action_names=["board_height_m", "drop_gap_m", "board_tilt_deg"],
        action_low=LOW.tolist(),
        action_high=HIGH.tolist(),
        initial_action=WARM.tolist(),
        fixed_baseline=FIXED.tolist(),
        lr=0.0003,
        clip=0.2,
        epochs=4,
        entropy_coef=0.001,
        value_coef=0.5,
        grad_norm=1.0,
        latent_logstd=-1.2,
        terminal=True,
        bootstrap=False,
        curriculum=False,
        split_seed_validation=SEED + 1,
        split_seed_test=SEED + 2,
        capture_rule="Board bounce + whole-ball entry + one second continuously inside, with no detected external contact; rim-assisted captures count.",
        policy_inputs="Privileged bucket/board geometry; no camera or point-cloud input.",
        scope="PPO high-level shot selection with endpoint initialization; no pickup, gripper actuation or arm-transit learning.",
    )
    value["fingerprint"] = hashlib.sha256(
        json.dumps(value, sort_keys=True).encode()
    ).hexdigest()
    return value
