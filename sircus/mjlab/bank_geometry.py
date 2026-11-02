"""Stable finite candidate geometry with CPU endpoint/contact feasibility masks.

Materialization never drops slots or transfers a source success label. A masked
candidate remains at its original ID. Feasibility is only an endpoint gate, not
proof of a successful bounce, collision-free transit or arbitrary-table coverage.
"""

from collections import Counter
from contextlib import contextmanager
import json
import math
from pathlib import Path
import time

import numpy as np

from .learning_contract import HIGH, LOW, digest
from .learning_search import VERSION as BANK_VERSION

VERSION = "sircus-fixed-bank-endpoint-geometry-v1"
ROTATIONS = 16
ROTATION_STEP_DEG = 22.5
OUTER_WRIST_RADIUS_M = 0.842
FEATURE_NAMES = (
    "bucket_x",
    "bucket_y",
    "board_x",
    "board_y",
    "board_height",
    "drop_gap",
    "board_tilt",
    "relative_board_x",
    "relative_board_y",
    "board_bucket_distance",
    "heading_sin",
    "heading_cos",
)
REASONS = {
    0: "endpoint_qualified",
    1: "action_bounds_or_distance",
    2: "outer_wrist_bound",
    3: "endpoint_ik_unresolved",
    4: "endpoint_collision",
    5: "native_endpoint_pose",
    6: "bucket_collider_transform",
}


def _bucket_array(bucket_xy):
    xy = np.asarray(bucket_xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("Expected finite bucketXY [N,2]")
    return xy


def build_candidates(bank, bucket_xy, scene):
    """Return fixed ``values[N,K,5], metadata[K]``; K = 16 × teacher count.

    IDs are turn-major then teacher-minor, stable across all query positions and
    masks. The unchanged 64-wall bucket has exact 22.5-degree rotational periods.
    """
    xy = _bucket_array(bucket_xy)
    payload = dict(bank)
    fingerprint = payload.pop("sha256")
    if bank.get("version") != BANK_VERSION or digest(payload) != fingerprint:
        raise ValueError("Training bank fingerprint changed")
    if scene["bucket"]["segments"] != 64:
        raise ValueError("Bank rotations require the frozen 64-wall bucket")
    templates = bank["templates"]
    metadata, values = [], []
    for turn in range(ROTATIONS):
        angle = math.radians(turn * ROTATION_STEP_DEG)
        rotation = np.array(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        )
        for index, template in enumerate(templates):
            relative = np.asarray(template["relative_board_xy_m"], dtype=float)
            other = np.array(
                [template["height_m"], template["tilt_deg"], template["gap_m"]],
                dtype=float,
            )
            if (
                relative.shape != (2,)
                or not np.isfinite(relative).all()
                or not np.isfinite(other).all()
            ):
                raise ValueError("Malformed training template geometry")
            position = xy + rotation @ relative
            values.append(
                np.column_stack((position, np.broadcast_to(other, (len(xy), 3))))
            )
            metadata.append(
                dict(
                    candidate_id=f"template-{index:03d}-rotation-{turn:02d}",
                    template_index=index,
                    rotation_index=turn,
                    rotation_deg=turn * ROTATION_STEP_DEG,
                    source_trial_id=template["source_trial_id"],
                    source_cell_id=template.get("source_cell_id"),
                    qualification=template.get("qualification", "unknown"),
                )
            )
    array = (
        np.stack(values, axis=1) if values else np.empty((len(xy), 0, 5), dtype=float)
    )
    return array, metadata


def candidate_features(bucket_xy, values):
    """Twelve normalized geometry-only features; no reward or capture labels."""
    xy = _bucket_array(bucket_xy)
    values = np.asarray(values, dtype=float)
    if values.ndim != 3 or values.shape[0] != len(xy) or values.shape[2] != 5:
        raise ValueError("Expected candidate values [N,K,5]")
    bucket = np.broadcast_to(xy[:, None], (*values.shape[:2], 2))
    relative = values[..., :2] - bucket
    distance = np.linalg.norm(relative, axis=-1)
    direction = -relative / np.maximum(distance[..., None], 1e-12)
    return np.concatenate(
        (
            (bucket - [0.67, -0.20]) / [0.6, 0.5],
            (values[..., :2] - [0.67, -0.20]) / [0.6, 0.5],
            values[..., 2:3] / 0.6,
            values[..., 4:5] / 0.6,
            values[..., 3:4] / 90,
            relative / 0.9,
            distance[..., None] / 1.2,
            direction[..., 1:2],
            direction[..., 0:1],
        ),
        axis=-1,
    ).astype(np.float32)


def outer_geometry_mask(scene, bucket_xy, values):
    """Loose 842 mm wrist outer bound, preserving candidates beyond v2's 810 mm prior.

    The nominal two-link length sum is 841.64 mm. The extra 0.36 mm keeps this
    cheap prefilter from rejecting floating-point boundary poses. Actual IK,
    native FK and contact checks remain authoritative.
    """
    xy = _bucket_array(bucket_xy)
    values = np.asarray(values, dtype=float)
    if values.ndim != 3 or values.shape[0] != len(xy) or values.shape[2] != 5:
        raise ValueError("Expected candidate values [N,K,5]")
    bucket = np.broadcast_to(xy[:, None], (*values.shape[:2], 2))
    delta = bucket - values[..., :2]
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
        & (values >= LOW - 1e-9).all(-1)
        & (values <= HIGH + 1e-9).all(-1)
        & (distance > 0.15)
    )
    reach = (
        np.linalg.norm(wrist10 - [0, 0, 0.364], axis=-1) <= OUTER_WRIST_RADIUS_M
    ) & (
        np.linalg.norm(wrist14 - [scene["arms"]["base_spacing_m"], 0, 0.364], axis=-1)
        <= OUTER_WRIST_RADIUS_M
    )
    return within, reach


def audit_candidates(
    scene,
    assets,
    bucket_xy,
    values,
    *,
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
    within, reach = outer_geometry_mask(scene, xy, values)
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


class FixedEndpointIK:
    """Read-only cached endpoint provider, with requested-geometry checks.

    The original native environment still verifies FK, bucket transforms and
    contacts. This provider only prevents a second optimizer from choosing a
    different joint branch than the branch whose feasibility was audited.
    """

    def __init__(self, scene, bucket_xy, values, q10, q14, valid, *, device="cpu"):
        self.scene = scene
        self.bucket = _bucket_array(bucket_xy)
        self.values = np.asarray(values, dtype=float)
        self.q10, self.q14 = np.asarray(q10, dtype=float), np.asarray(q14, dtype=float)
        self.valid = np.asarray(valid, dtype=bool)
        self.device = device
        n = len(self.bucket)
        if (
            not n
            or self.values.shape != (n, 5)
            or self.q10.shape != (n, 6)
            or self.q14.shape != (n, 6)
            or self.valid.shape != (n,)
            or not all(np.isfinite(x).all() for x in (self.values, self.q10, self.q14))
        ):
            raise ValueError(
                "Cached endpoint arrays have inconsistent shapes or nonfinite values"
            )

    def solve(self, bucket_xy, board_xyz, tilt_deg, gap_m, **unused):
        import torch
        from .learning_ik import shot_targets

        bucket = _bucket_array(bucket_xy)
        n = len(bucket)
        if n < len(self.bucket):
            raise ValueError("Cached endpoint provider cannot truncate its batch")
        pad = lambda array: np.concatenate(
            (array, np.repeat(array[-1:], n - len(array), axis=0))
        )
        expected_bucket, expected_values = pad(self.bucket), pad(self.values)
        valid = pad(self.valid)
        actual = np.column_stack(
            (board_xyz, np.broadcast_to(tilt_deg, (n,)), np.broadcast_to(gap_m, (n,)))
        )
        errors = (
            np.abs(actual - expected_values) > [1e-9, 1e-9, 1e-9, 1e-7, 1e-9]
        ).any(1) | (np.abs(bucket - expected_bucket) > 1e-9).any(1)
        if (errors & valid).any():
            raise RuntimeError(
                "Cached joint branch requested for different valid geometry"
            )
        targets = shot_targets(
            self.scene, bucket, actual[:, :3], actual[:, 3], actual[:, 4]
        )
        return dict(
            q10=torch.as_tensor(pad(self.q10), device=self.device, dtype=torch.float64),
            q14=torch.as_tensor(pad(self.q14), device=self.device, dtype=torch.float64),
            valid=torch.as_tensor(valid, device=self.device),
            targets=targets,
            version="sircus-cached-endpoint-provider-v1",
        )


@contextmanager
def fixed_endpoints(environment, bucket_xy, values, q10, q14, valid):
    """Temporarily install the audited branch without changing the v2 environment."""
    previous = environment.ik
    environment.ik = FixedEndpointIK(
        environment.scene, bucket_xy, values, q10, q14, valid, device=environment.device
    )
    try:
        yield environment
    finally:
        environment.ik = previous
