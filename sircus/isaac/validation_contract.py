"""Predeclared seeded perception sweep and evaluation-only visibility labels."""

import numpy as np

VERSION = "sircus-perception-validation-v3"
CONDITIONS = [
    {
        "name": "clean",
        "intensity": 500.0,
        "color": [0.85, 0.9, 1.0],
        "sigma_m": 0.0,
        "dropout": 0.0,
    },
    {
        "name": "dim_warm",
        "intensity": 150.0,
        "color": [1.0, 0.7, 0.45],
        "sigma_m": 0.0,
        "dropout": 0.0,
    },
    {
        "name": "bright_cool",
        "intensity": 900.0,
        "color": [0.6, 0.8, 1.0],
        "sigma_m": 0.0,
        "dropout": 0.0,
    },
    {
        "name": "noise_2mm",
        "intensity": 500.0,
        "color": [0.85, 0.9, 1.0],
        "sigma_m": 0.002,
        "dropout": 0.01,
    },
    {
        "name": "noise_8mm",
        "intensity": 500.0,
        "color": [0.85, 0.9, 1.0],
        "sigma_m": 0.008,
        "dropout": 0.10,
    },
    {
        "name": "combined",
        "intensity": 150.0,
        "color": [1.0, 0.7, 0.45],
        "sigma_m": 0.008,
        "dropout": 0.10,
    },
]


def layouts(seed, count):
    rng = np.random.default_rng(seed)
    for i in range(count):
        joint_delta = rng.uniform(-0.08, 0.08, 6)
        # Resolved through manufacturer FK before capture; no visual feedback.
        board = np.array([0.67, -0.1, 0.4])
        bucket = rng.uniform([0.45, -0.64], [0.87, -0.34])
        mode = ["above_board", "between_board_bucket", "table", "inside_bucket"][i % 4]
        if mode == "above_board":
            ball = board + np.array([0, 0, 0.16])
        elif mode == "between_board_bucket":
            ball = np.r_[(board[:2] + bucket) / 2, 0.27]
        elif mode == "table":
            ball = np.r_[rng.uniform([0.1, -0.68], [0.4, -0.35]), 0.0375]
        else:
            ball = np.r_[bucket, 0.008 + 0.0375]
        yield {
            "layout": i,
            "board_joint_delta_rad": joint_delta.tolist(),
            "board_center_m": board.tolist(),
            "bucket_xy_m": bucket.tolist(),
            "ball_position_m": ball.tolist(),
            "ball_mode": mode,
        }


def paired_source(condition):
    return {"noise_2mm": "clean", "noise_8mm": "clean", "combined": "dim_warm"}.get(
        condition
    )


def noisy_depth(depth, sigma, dropout, seed):
    rng = np.random.default_rng(seed)
    out = np.array(depth, copy=True)
    valid = np.isfinite(out) & (out > 0.05) & (out < 5)
    out[valid] += rng.normal(0, sigma, int(valid.sum()))
    out[valid & (rng.random(out.shape) < dropout)] = np.nan
    out[(out <= 0.05) | (out >= 5)] = np.nan
    return out


def visible_sphere_pixels(depth, k, camera_to_world, center, radius):
    """Evaluation ONLY: ideal ray/sphere surface agreement with clean depth.

    Does not use RGB or estimator outputs. At least20 pixels within5mm defines
    visible-enough for the prespecified ball localization metric, not human labels.
    """
    t = np.linalg.inv(camera_to_world)
    c = t[:3, :3] @ np.asarray(center) + t[:3, 3]
    v, u = np.indices(depth.shape)
    rays = np.stack(
        [(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], np.ones_like(u)], -1
    )
    a = (rays * rays).sum(-1)
    b = (rays * c).sum(-1)
    disc = b * b - a * (c @ c - radius * radius)
    z = (b - np.sqrt(np.maximum(disc, 0))) / a
    return int(
        ((disc > 0) & (z > 0) & np.isfinite(depth) & (np.abs(depth - z) < 0.005)).sum()
    )


def summarize(records, planned, excluded):
    groups = {}
    for condition in CONDITIONS:
        rows = [r for r in records if r["condition"] == condition["name"]]
        objects = {}
        for name in ["table", "board", "bucket", "ball"]:
            values = [r["errors"][name] for r in rows]
            ok = [v for v in values if v["status"] == "estimated"]
            errors = [
                v.get("center_error_mm", v.get("plane_distance_error_mm")) for v in ok
            ]
            objects[name] = {
                "estimated": len(ok),
                "total": len(rows),
                "mean_mm": float(np.mean(errors)) if errors else None,
                "p95_mm": float(np.percentile(errors, 95)) if errors else None,
                "max_mm": max(errors) if errors else None,
            }
        visible = [r for r in rows if r["ball_visible_pixels"] >= 20]
        accurate = [
            r
            for r in visible
            if r["errors"]["ball"]["status"] == "estimated"
            and r["errors"]["ball"]["center_error_mm"] <= 10
        ]
        groups[condition["name"]] = {
            "objects": objects,
            "ball_visible": len(visible),
            "ball_localized_10mm": len(accurate),
            "ball_detection_when_visible": sum(
                r["errors"]["ball"]["status"] == "estimated" for r in visible
            ),
        }
    return {
        "contract": VERSION,
        "planned_observations": planned,
        "completed_observations": len(records),
        "excluded_layouts": excluded,
        "conditions": groups,
        "scope": "Static randomized perception only; not policy performance or physical accuracy. Conditional errors exclude missing detections. Ball visibility is evaluation-only analytic depth agreement.",
    }


def verify_pair(noisy, base, truth, baseline_truth):
    """Evaluation-only guard against lighting/pose confounds in noise comparisons."""
    for field in ["rgb", "intrinsics", "camera_to_world"]:
        if not np.array_equal(noisy[field], base[field]):
            raise ValueError(f"Paired noise case changed {field}")
    if (
        truth["objects"] != baseline_truth["objects"]
        or truth["source_render_frame_id"] != baseline_truth["frame_id"]
    ):
        raise ValueError("Paired noise case changed evaluation geometry/source")
