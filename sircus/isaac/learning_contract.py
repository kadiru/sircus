"""Privileged scene-conditioned, one-shot RL study specification (not trajectories)."""

import hashlib
import json

import numpy as np
from rl_contract import HIGH, LOW, fingerprint

VERSION = "sircus-scene-conditioned-shot-v3"


def scenes(split, count):
    seeds = {"train": 71031, "validation": 81031, "test": 91031, "development": 101031}
    rng = np.random.default_rng(seeds[split])
    result = []
    for i in range(count):
        board = rng.uniform([0.64, -0.14, 0.36], [0.70, -0.08, 0.42])
        distance = float(rng.uniform(0.20, 0.40))
        heading = float(rng.uniform(-0.25, 0.25))
        bucket = board[:2] + distance * np.array([np.sin(heading), -np.cos(heading)])
        result.append(
            {
                "id": f"{split}-{i:05d}",
                "board_center_m": board.tolist(),
                "bucket_xy_m": bucket.tolist(),
                "distance_m": distance,
                "difficulty": "near"
                if distance < 0.267
                else "medium"
                if distance < 0.333
                else "far",
            }
        )
    return result


def observation(scene, cfg):
    c = np.array(scene["board_center_m"])
    b = np.array(scene["bucket_xy_m"])
    bucket = cfg["bucket"]
    return np.array(
        [
            c[0] / 0.8,
            c[1] / 0.4,
            c[2] / 0.5,
            *((b - c[:2]) / 0.4),
            (bucket["assumed_interior_depth_m"] + bucket["assumed_bottom_thickness_m"])
            / 0.3,
            bucket["inner_top_diameter_m"] / 0.2,
            bucket["inner_bottom_diameter_m"] / 0.2,
            *(np.array(cfg["board"]["size_m"][:2]) / 0.3),
            cfg["ball"]["radius_m"] / 0.05,
        ],
        dtype=np.float32,
    )


def contract(cfg):
    data = {
        "version": VERSION,
        "bucket_body": "kinematic compound; walls and floor move through PhysX root state",
        "physics_scoring_fingerprint": fingerprint(cfg),
        "observations": "11 privileged normalized geometry values; no camera input",
        "actions": ["board_tilt_deg", "bucket_heading_offset_deg", "release_gap_m"],
        "low": LOW.tolist(),
        "high": HIGH.tolist(),
        "arm_solver": "board center+normal IK (free in-plane spin), dropper full-pose IK; endpoints only",
        "algorithm": "one-step contextual PPO, terminal returns; centralized separate actor/value networks shared by both arms",
        "network": [11, 64, 64, 3],
        "optimizer": "Adam",
        "lr": 0.0007,
        "clip": 0.2,
        "epochs": 4,
        "batch_shots": 64,
        "entropy": 0.001,
        "value_coefficient": 0.5,
        "gradient_clip": 1.0,
        "gamma": 1.0,
        "reward_scale": 15.0,
        "logstd_bounds": [-2.8, -0.2],
        "initial_logstd": -1.0,
        "warm_start_action": [21.3590135, -0.9408402, 0.16769218],
        "train_pool": 1000,
        "validation_pool": 128,
        "test_pool": 256,
        "validation_selection": "every512 shots: first32 validation twice (contact repeatability); select success count then mean reward; test used once after selection",
        "success": "filtered board contact+rebound, whole-ball entry, continuous1s retention",
        "limits": [
            "no grasp or collision-checked path learning",
            "no board compliance or calibrated ball properties",
            "static geometry, no camera input",
            "no U/V action; release above board center",
        ],
    }
    data["fingerprint"] = hashlib.sha256(
        json.dumps(data, sort_keys=True).encode()
    ).hexdigest()
    return data


def outcome_summary(rows):
    valid = [r for r in rows if not r.get("invalid")]
    n = len(rows)
    unique = len({r.get("scene_id", i) for i, r in enumerate(rows)})
    wins = sum(bool(r["success"]) for r in rows)
    p = wins / n if n else 0.0
    z = 1.96
    center = (p + z * z / (2 * n)) / (1 + z * z / n) if n else 0
    half = (
        z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n) if n else 0
    )
    return {
        "trials": n,
        "unique_layouts": unique,
        "captures": wins,
        "success_rate": p,
        "wilson95": [max(0.0, float(center - half)), min(1.0, float(center + half))]
        if unique == n
        else None,
        "valid_endpoints": len(valid),
        "invalid_endpoints": n - len(valid),
        "mean_reward": float(np.mean([r["reward"] for r in rows])) if rows else None,
        "by_distance": {
            k: {
                "trials": sum(r["difficulty"] == k for r in rows),
                "captures": sum(r["success"] and r["difficulty"] == k for r in rows),
            }
            for k in ["near", "medium", "far"]
        },
    }
