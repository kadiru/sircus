"""Short balanced study contract, disjoint seeded layouts, no simulator imports."""

import hashlib
import json

import numpy as np
from learning_contract import contract as legacy_contract
from robust_ik import VERSION as IK_VERSION

VERSION = "sircus-balanced-settled-pilot-v1"
BINS = [(0.20, 0.267, "near"), (0.267, 0.333, "medium"), (0.333, 0.40, "far")]
SEEDS = {"train": 161101, "validation": 171101, "test": 181101, "development": 191101}


def layouts(split, per_bin):
    rng = np.random.default_rng(SEEDS[split])
    result = []
    for lo, hi, label in BINS:
        for i in range(per_bin):
            c = rng.uniform([0.64, -0.14, 0.36], [0.70, -0.08, 0.42])
            d, heading = rng.uniform(lo, hi), rng.uniform(-0.25, 0.25)
            b = c[:2] + d * np.array([np.sin(heading), -np.cos(heading)])
            result.append(
                {
                    "id": f"pilot-{split}-{label}-{i:03d}",
                    "board_center_m": c.tolist(),
                    "bucket_xy_m": b.tolist(),
                    "distance_m": float(d),
                    "difficulty": label,
                }
            )
    return result


def balanced_indices(pool, per_bin, rng):
    chosen = []
    for _, _, name in BINS:
        ids = [i for i, s in enumerate(pool) if s["difficulty"] == name]
        chosen.extend(rng.choice(ids, per_bin, replace=False).tolist())
    rng.shuffle(chosen)
    return chosen


def contract(cfg, episodes=384):
    data = {
        "version": VERSION,
        "legacy_reference_fingerprint": legacy_contract(cfg)["fingerprint"],
        "ik_version": IK_VERSION,
        "ik_acceptance": 1e-4,
        "settling": {
            "position_mm": 2,
            "normal_deg": 0.5,
            "joint_speed_rad_s": 0.005,
            "max_corrections": 3,
            "ticks_per_correction": 120,
            "stability_window_ticks": 60,
            "max_position_drift_mm": 0.1,
            "max_normal_drift_deg": 0.05,
            "gravity_compensation": "PhysX model feed-forward on both arms each step; gravity and original drive limits unchanged",
            "no_release_if_unsettled": True,
        },
        "development_gate": {
            "trials": 12,
            "min_valid_per_distance_bin": 3,
            "require_all_ik_valid_poses_settled": True,
            "retain_endpoint_rejections_in_report": True,
        },
        "scoring_normal": "measured pre-release board normal; same contact/whole-ball/1s scorer",
        "splits": {
            "seeds": SEEDS,
            "train_per_bin": 32,
            "validation_per_bin": 6,
            "test_per_bin": 12,
        },
        "training": {
            "algorithm": "contextual PPO",
            "episodes": episodes,
            "batch": 48,
            "per_bin_per_batch": 16,
            "validation_every": 192,
            "seed": 43,
            "warm_start": "selected 100-layout checkpoint",
            "initial_logstd": -0.7,
            "lr": 0.0007,
            "epochs": 4,
            "clip": 0.2,
            "entropy": 0.001,
            "reward_scale": 15,
            "gamma": 1,
            "value_coefficient": 0.5,
            "gradient_clip": 1,
            "logstd_bounds": [-2.8, -0.2],
        },
        "comparison": "all four policies on identical fresh test scenes and new solver/settle contract; no optimization at inference",
        "selection": "validation captures then mean reward; initial checkpoint eligible",
        "scope": "privileged-state simulation pilot; endpoint reset, no grasp/path learning or quasi-perception",
    }
    data["fingerprint"] = hashlib.sha256(
        json.dumps(data, sort_keys=True).encode()
    ).hexdigest()
    return data
