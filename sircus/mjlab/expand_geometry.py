"""Dense geometry proposals for the unresolved-target development search.

Original bounds are the default; an explicitly versioned wider family is optional.
Native joint, collision, transform and scoring gates are retained. This proposal prior
does not transfer source success labels or prove that rejected scenes are
impossible.  Ball outcomes must come from fresh native simulation.
"""

from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from .learning_contract import HIGH, LOW
from .bank_geometry import REASONS, _bucket_array

VERSION = "sircus-unresolved-shot-proposals-v1"


def maximum_gap(board_xyz, scene, wrist_radius=0.842):
    xyz = np.asarray(board_xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("Expected boardXYZ[N,3]")
    horizontal = (xyz[:, :2] ** 2).sum(1)
    vertical = np.sqrt(np.maximum(wrist_radius**2 - horizontal, 0))
    fixed = scene["ball"]["radius_m"] + 0.095 + 0.09
    cap = 0.364 + vertical - xyz[:, 2] - fixed
    return np.where(horizontal <= wrist_radius**2, cap, -np.inf)


OLD_BOUNDS = {
    "low": LOW.tolist(),
    "high": HIGH.tolist(),
    "version": "original-shot-bounds-v1",
}
WIDE_BOUNDS = {
    "low": [0.30, -0.50, 0.12, 8.0, 0.05],
    "high": [1.04, 0.35, 0.60, 60.0, 0.65],
    "version": "sircus-expanded-shot-actions-v1",
}


def validate_bounds(bounds=None):
    bounds = OLD_BOUNDS if bounds is None else bounds
    if not {"low", "high"}.issubset(bounds) or set(bounds) - {"low", "high", "version"}:
        raise ValueError("Expected explicit low/high action bounds")
    low, high = np.asarray(bounds["low"], dtype=float), np.asarray(
        bounds["high"], dtype=float
    )
    if (
        low.shape != (5,)
        or high.shape != (5,)
        or not np.isfinite([low, high]).all()
        or not (high > low).all()
    ):
        raise ValueError("Finite increasing five-dimensional bounds required")
    if low[2] <= 0 or low[3] < 0 or high[3] >= 90 or low[4] <= 0:
        raise ValueError("Positive board height/gap and tilt from0to90degrees required")
    return low, high


def outer_geometry_mask(scene, bucket_xy, values, *, bounds=None):
    xy = _bucket_array(bucket_xy)
    values = np.asarray(values, dtype=float)
    low, high = validate_bounds(bounds)
    if values.ndim != 3 or values.shape[0] != len(xy) or values.shape[2] != 5:
        raise ValueError("Expected values[N,K,5]")
    delta = xy[:, None] - values[..., :2]
    distance = np.linalg.norm(delta, axis=-1)
    angle = np.deg2rad(values[..., 3])
    normal = np.concatenate(
        (
            delta / np.maximum(distance[..., None], 1e-12) * np.sin(angle)[..., None],
            np.cos(angle)[..., None],
        ),
        axis=-1,
    )
    wrist14 = values[..., :3] - (scene["board"]["tool_offset_m"] + 0.09) * normal
    wrist10 = values[..., :3].copy()
    wrist10[..., 2] += scene["ball"]["radius_m"] + values[..., 4] + 0.095 + 0.09
    within = (
        np.isfinite(values).all(-1)
        & (values >= low - 1e-9).all(-1)
        & (values <= high + 1e-9).all(-1)
        & (distance > 0.15)
    )
    reach = (np.linalg.norm(wrist10 - [0, 0, 0.364], axis=-1) <= 0.842) & (
        np.linalg.norm(wrist14 - [scene["arms"]["base_spacing_m"], 0, 0.364], axis=-1)
        <= 0.842
    )
    return within, reach


def _broad(bucket, rng, scene, bounds=None):
    n = len(bucket)
    low, high = validate_bounds(bounds)
    result = rng.uniform(low, high, (n, 5))
    wide = (low < LOW).any() or (high > HIGH).any()
    result[:, 0] = rng.uniform(
        low[0] if wide else max(low[0], 0.46), min(high[0], 0.86 if wide else 0.83), n
    )
    result[:, 1] = rng.uniform(low[1], high[1], n)
    # Low boards paired with a larger fall gap were missing from the old bank.
    result[:, 2] = rng.uniform(low[2], min(high[2], 0.40), n)
    lower = rng.random(n) < 0.65
    result[lower, 2] = rng.uniform(low[2], min(high[2], 0.285), lower.sum())
    result[:, 3] = rng.uniform(low[3], high[3], n)
    distance = np.linalg.norm(bucket - result[:, :2], axis=1)
    angle = np.deg2rad(result[:, 3])
    # Approximate ballistic seed only: measured contact response decides reward.
    restitution = rng.uniform(0.60, 0.80, n)
    horizontal = (1 + restitution) * np.sin(angle) * np.cos(angle)
    vertical = (1 + restitution) * np.cos(angle) ** 2 - 1
    den = 4 * horizontal**2 * (result[:, 2] - 0.1455 + vertical / horizontal * distance)
    ballistic = distance**2 / np.maximum(den, 0.015)
    cap = np.minimum(high[4], maximum_gap(result[:, :3], scene))
    result[:, 4] = np.minimum(ballistic * rng.uniform(0.75, 1.3, n), cap)
    mode = rng.random(n)
    near_cap = mode < 0.25
    uniform = (mode >= 0.25) & (mode < 0.4)
    result[near_cap, 4] = cap[near_cap] * rng.uniform(0.88, 1.0, near_cap.sum())
    result[uniform, 4] = low[4] + rng.random(uniform.sum()) * np.maximum(
        cap[uniform] - low[4], 0
    )
    return result


def proposals(bucket, rng, scene, *, stage="broad", best=None, bounds=None):
    """Return finite values[N,5] under explicit bounds, with dense reach-filtered seeds.

    ``best`` may be a vector[5] or values[N,5].  Refinement perturbs geometry
    locally but cannot silently leave the supplied bounds. Wider bounds require
    the matching versioned native environment, not the historical runner.
    If the finite resampling budget is exhausted, the last bounded candidate is
    returned for the caller's strict native audit to reject; never call that
    target impossible.  Use ``proposal_mask`` to inspect the cheap prior mask.
    """
    bucket = np.asarray(bucket, dtype=float)
    if bucket.ndim != 2 or bucket.shape[1] != 2 or not np.isfinite(bucket).all():
        raise ValueError("Expected finite bucketXY[N,2]")
    if stage not in ("broad", "refine") or (stage == "refine" and best is None):
        raise ValueError("Use broad or refine with a supplied best action")
    low, high = validate_bounds(bounds)
    n = len(bucket)
    values = np.zeros((n, 5))
    pending = np.ones(n, dtype=bool)
    if best is not None:
        best = np.broadcast_to(np.asarray(best, dtype=float), (n, 5))
        if not np.isfinite(best).all():
            raise ValueError("Best action must be finite")
    for _ in range(96):
        ids = np.flatnonzero(pending)
        if not len(ids):
            break
        candidate = _broad(bucket[ids], rng, scene, bounds)
        if stage == "refine":
            local = rng.random(len(ids)) < 0.85
            candidate[local] = best[ids[local]] + rng.normal(
                0, [0.018, 0.018, 0.020, 2.5, 0.024], (local.sum(), 5)
            )
        candidate = np.nan_to_num(
            np.clip(candidate, low, high), neginf=low[4], posinf=high[4]
        )
        within, reach = outer_geometry_mask(
            scene, bucket[ids], candidate[:, None], bounds=bounds
        )
        passed = (within & reach)[:, 0]
        values[ids] = candidate
        pending[ids[passed]] = False
    return values


def proposal_mask(bucket, values, scene, bounds=None):
    within, reach = outer_geometry_mask(
        scene, bucket, np.asarray(values)[:, None], bounds=bounds
    )
    return (within & reach)[:, 0]


def audit_values(
    scene,
    assets,
    bucket_xy,
    values,
    *,
    bounds=None,
    chunk_size=512,
    starts=4,
    iterations=90,
    progress=None,
):
    """Return ``mask[N,K], details`` using CPU IK and the complete native model.

    ``details`` holds stable-slot NumPy q10/q14/error/reason arrays, a compact
    summary and only the failed native contact pairs. Non-audited q slots are
    zero; only mask-true q slots may be used. No GPU context is initialized.
    """
    import mujoco
    from .learning_ik import ShotIK
    from .model import build_scene

    start = time.monotonic()
    xy = _bucket_array(bucket_xy)
    values = np.asarray(values, dtype=float)
    within, reach = outer_geometry_mask(scene, xy, values, bounds=bounds)
    shape = values.shape[:2]
    flat = values.reshape(-1, 5)
    buckets = np.broadcast_to(xy[:, None], (*shape, 2)).reshape(-1, 2)
    reasons = np.ones(shape, dtype=np.uint8)
    reasons[within] = 2
    eligible = within & reach
    reasons[eligible] = 3
    q10 = np.zeros((*shape, 6))
    q14 = np.zeros((*shape, 6))
    position_error = np.full((*shape, 2), -1.0)
    normal_error = np.full(shape, -1.0)
    collision_pairs = []
    if not 1 <= chunk_size <= 4096:
        raise ValueError("Finite CPU IK chunk_size must be 1..4096")
    assets = Path(assets)
    seeds = json.loads((assets / "poses.json").read_text())
    ik = ShotIK(scene, assets, device="cpu")
    model = mujoco.MjModel.from_xml_string(
        build_scene(scene, assets, solref=(0.004, 0.15), contact_friction=0.35)
    )
    data = mujoco.MjData(model)
    joint = lambda name: model.joint(name).id
    qi10 = np.array([model.jnt_qposadr[joint(f"Arm10/joint{i}")] for i in range(1, 7)])
    qi14 = np.array([model.jnt_qposadr[joint(f"Arm14/joint{i}")] for i in range(1, 7)])
    fingers = [
        model.jnt_qposadr[joint(f"Arm10/finger_{side}_joint")]
        for side in ("left", "right")
    ]
    ball = int(model.jnt_qposadr[joint("ball/free")])
    board = model.body("Arm14/impact_board").id
    flange = model.body("Arm10/link6").id
    bucket = model.body("bucket").id
    mocap = int(model.body_mocapid[bucket])
    bucket_geoms = np.array(
        [
            i
            for i in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith(
                "bucket/"
            )
        ]
    )
    all_joint10 = np.array([joint(f"Arm10/joint{i}") for i in range(1, 7)])
    all_joint14 = np.array([joint(f"Arm14/joint{i}") for i in range(1, 7)])
    indices = np.flatnonzero(eligible.reshape(-1))
    for offset in range(0, len(indices), chunk_size):
        selected = indices[offset : offset + chunk_size]
        action = flat[selected]
        result = ik.solve(
            buckets[selected],
            action[:, :3],
            action[:, 3],
            action[:, 4],
            initial10=seeds["dropper_joint_rad"],
            initial14=seeds["board_joint_rad"],
            starts=starts,
            iterations=iterations,
        )
        first = result["q10"].numpy()
        second = result["q14"].numpy()
        q10.reshape(-1, 6)[selected] = first
        q14.reshape(-1, 6)[selected] = second
        valid = result["valid"].numpy()
        for j, index in enumerate(selected):
            if not valid[j]:
                continue
            limits10 = model.jnt_range[all_joint10]
            limits14 = model.jnt_range[all_joint14]
            if not (
                (first[j] >= limits10[:, 0] - 1e-9).all()
                and (first[j] <= limits10[:, 1] + 1e-9).all()
                and (second[j] >= limits14[:, 0] - 1e-9).all()
                and (second[j] <= limits14[:, 1] + 1e-9).all()
            ):
                continue
            data.qpos[:] = model.qpos0
            data.qpos[qi10] = first[j]
            data.qpos[qi14] = second[j]
            data.qpos[fingers] = 0.025
            data.qpos[ball : ball + 3] = [0, 0, 2]
            data.qpos[ball + 3 : ball + 7] = [1, 0, 0, 0]
            data.qvel[:] = 0
            data.mocap_pos[mocap] = [*buckets[index], 0]
            data.mocap_quat[mocap] = [1, 0, 0, 0]
            mujoco.mj_forward(model, data)
            expected_bucket = np.r_[buckets[index], 0]
            collider_error = np.linalg.norm(
                data.geom_xpos[bucket_geoms]
                - (expected_bucket + model.geom_pos[bucket_geoms]),
                axis=1,
            ).max()
            if collider_error > 1e-8:
                reasons.reshape(-1)[index] = 6
                continue
            bp = np.linalg.norm(data.xpos[board] - action[j, :3])
            dp = np.linalg.norm(
                data.xpos[flange]
                - (result["targets"]["release_center_m"][j] + [0, 0, 0.095])
            )
            bn = data.xmat[board].reshape(3, 3)[:, 2]
            angle = np.degrees(
                np.arccos(
                    np.clip(bn @ result["targets"]["board_normal_world"][j], -1, 1)
                )
            )
            position_error.reshape(-1, 2)[index] = [bp, dp]
            normal_error.reshape(-1)[index] = angle
            if bp >= 0.002 or dp >= 0.002 or angle >= 0.5:
                reasons.reshape(-1)[index] = 5
                continue
            bad = []
            for contact in data.contact[: data.ncon]:
                names = [
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or ""
                    for g in contact.geom
                ]
                if contact.dist < -0.002 and not any(
                    name.startswith("ball/") for name in names
                ):
                    bad.append(names)
            if bad:
                reasons.reshape(-1)[index] = 4
                collision_pairs.append(
                    dict(
                        layout_index=int(index // shape[1]),
                        candidate_index=int(index % shape[1]),
                        pairs=bad,
                    )
                )
                continue
            reasons.reshape(-1)[index] = 0
        if progress is not None:
            progress(
                dict(
                    processed=min(offset + chunk_size, len(indices)),
                    eligible=len(indices),
                    endpoint_qualified=int((reasons == 0).sum()),
                )
            )
    mask = reasons == 0
    summary = dict(
        version=VERSION,
        action_bounds_version=(OLD_BOUNDS if bounds is None else bounds).get(
            "version", "explicit-unversioned"
        ),
        layouts=shape[0],
        candidates_per_layout=shape[1],
        total_slots=int(np.prod(shape)),
        outer_prior_eligible=len(indices),
        endpoint_qualified=int(mask.sum()),
        layouts_with_endpoint=int(mask.any(1).sum()),
        reasons={
            REASONS[int(k)]: int(v) for k, v in Counter(reasons.reshape(-1)).items()
        },
        elapsed_s=time.monotonic() - start,
        scope="CPU native endpoint/clearance feasibility only; no ball rollout, success label or collision-free transit claim",
    )
    return mask, dict(
        summary=summary,
        reasons=reasons,
        q10=q10,
        q14=q14,
        position_error_m=position_error,
        board_normal_error_deg=normal_error,
        collision_pairs=collision_pairs,
    )


# Explicit aliases for the search/training drivers.
audit_expanded_candidates = audit_values
audit_candidates = audit_values
