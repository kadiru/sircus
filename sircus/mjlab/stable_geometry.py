"""Deterministic, continuation-aware native endpoint qualification.

Every mathematical branch pair is subjected to the historical limit, native FK,
bucket-transform and endpoint collision gates. An alternative is not discarded
because a lower-residual branch collides. This is not ball-flight feasibility or
collision-free transit planning. Original geometry/solver/scoring files remain
unchanged.
"""

from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from .bank_geometry import REASONS, _bucket_array
from .expand_geometry import OLD_BOUNDS, outer_geometry_mask
from .learning_ik import shot_targets
from .stable_ik import StableShotIK, VERSION as IK_VERSION

VERSION = "sircus-query-stable-native-geometry-v1"


def known_array(value, shape):
    if value is None:
        return np.full((*shape, 6), np.nan)
    a = np.asarray(value, dtype=float)
    if a.shape == (shape[0], 6) and shape[1] == 1:
        a = a[:, None]
    if a.shape != (*shape, 6) or not (
        np.isfinite(a).all(-1) | np.isnan(a).all(-1)
    ).all():
        raise ValueError("Known joints must be finite [N,K,6] or all-NaN rows")
    return a.copy()


def known_request_match(bucket, values, known_values=None, known_bucket_xy=None):
    """Only exact stored/current request equality permits old nominal q reuse."""
    shape = values.shape[:2]
    if known_values is None and known_bucket_xy is None:
        return np.zeros(shape, dtype=bool)
    if known_values is None or known_bucket_xy is None:
        raise ValueError("Nominal preservation needs both known_values and known_bucket_xy")
    actions = np.asarray(known_values, dtype=float)
    xy = np.asarray(known_bucket_xy, dtype=float)
    if actions.shape == (shape[0], 5) and shape[1] == 1:
        actions = actions[:, None]
    if xy.shape == (shape[0], 2):
        xy = np.broadcast_to(xy[:, None], (*shape, 2))
    if actions.shape != (*shape, 5) or xy.shape != (*shape, 2):
        raise ValueError("Known requests require actions[N,K,5] and bucket[N,K,2]")
    if not ((np.isfinite(actions).all(-1) | np.isnan(actions).all(-1)).all()
        and (np.isfinite(xy).all(-1) | np.isnan(xy).all(-1)).all()):
        raise ValueError("Known request rows must be finite or all NaN")
    return (actions == values).all(-1) & (xy == bucket[:, None]).all(-1)


def ordered_pairs(first, second, *, known10=None, known14=None):
    """Stable preference: exact known pair, known board/dropper, then start IDs.

    Residual does not outrank an already valid continuation. All returned pairs
    still need the full native gate. Joint-period-equivalent starts are retained:
    joint limits and exact saved branches, not only wrapped angles, are evidence.
    """
    ids10, ids14 = np.flatnonzero(first["valid"]), np.flatnonzero(second["valid"])
    exact10 = {int(i): known10 is not None and np.isfinite(known10).all() and np.array_equal(first["q"][i], known10) for i in ids10}
    exact14 = {int(i): known14 is not None and np.isfinite(known14).all() and np.array_equal(second["q"][i], known14) for i in ids14}
    pairs = [(int(i), int(j)) for i in ids10 for j in ids14]
    return sorted(pairs, key=lambda ij: (
        not (exact10[ij[0]] and exact14[ij[1]]), not exact14[ij[1]],
        not exact10[ij[0]], ij[1], ij[0]))


def select_pair(first, second, gate, *, known10=None, known14=None):
    """Test branches through an injected native gate; useful in CPU mock tests."""
    attempts = []
    for i, j in ordered_pairs(first, second, known10=known10, known14=known14):
        checked = gate(first["q"][i], second["q"][j])
        attempts.append(dict(branch10=i, branch14=j, **checked))
        if checked["reason"] == 0:
            return i, j, checked, attempts
    return None, None, None, attempts


class NativeEndpointGate:
    """The same endpoint limits/tolerances/model as expand_geometry.audit_values."""
    def __init__(self, scene, assets):
        import mujoco
        from .model import build_scene

        self.mujoco = mujoco
        self.model = m = mujoco.MjModel.from_xml_string(build_scene(
            scene, Path(assets), solref=(0.004, 0.15), contact_friction=0.35))
        self.data = mujoco.MjData(m)
        joints = lambda role: np.asarray([m.joint(f"{role}/joint{i}").id for i in range(1, 7)])
        ji10, ji14 = joints("Arm10"), joints("Arm14")
        self.qi10, self.qi14 = m.jnt_qposadr[ji10], m.jnt_qposadr[ji14]
        self.limits10, self.limits14 = m.jnt_range[ji10], m.jnt_range[ji14]
        self.fingers = [m.jnt_qposadr[m.joint(f"Arm10/finger_{s}_joint").id] for s in ("left", "right")]
        self.ball = int(m.jnt_qposadr[m.joint("ball/free").id])
        self.board, self.flange = m.body("Arm14/impact_board").id, m.body("Arm10/link6").id
        self.mocap = int(m.body_mocapid[m.body("bucket").id])
        self.bucket_geoms = np.asarray([i for i in range(m.ngeom) if
            (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("bucket/")])

    def check(self, q10, q14, bucket_xy, values, targets):
        mj, m, d = self.mujoco, self.model, self.data
        result = dict(reason=3, board_position_error_m=-1., dropper_position_error_m=-1.,
            board_normal_error_deg=-1., bucket_collider_error_m=-1., pairs=[])
        q10, q14 = np.asarray(q10), np.asarray(q14)
        if not (np.isfinite(q10).all() and np.isfinite(q14).all() and
            (q10 >= self.limits10[:, 0] - 1e-9).all() and (q10 <= self.limits10[:, 1] + 1e-9).all() and
            (q14 >= self.limits14[:, 0] - 1e-9).all() and (q14 <= self.limits14[:, 1] + 1e-9).all()):
            return result
        d.qpos[:] = m.qpos0
        d.qpos[self.qi10], d.qpos[self.qi14] = q10, q14
        d.qpos[self.fingers] = .025
        d.qpos[self.ball:self.ball + 3] = [0, 0, 2]
        d.qpos[self.ball + 3:self.ball + 7] = [1, 0, 0, 0]
        d.qvel[:] = 0
        d.mocap_pos[self.mocap], d.mocap_quat[self.mocap] = [*bucket_xy, 0], [1, 0, 0, 0]
        mj.mj_forward(m, d)
        collider = float(np.linalg.norm(d.geom_xpos[self.bucket_geoms] -
            (np.r_[bucket_xy, 0] + m.geom_pos[self.bucket_geoms]), axis=1).max())
        result["bucket_collider_error_m"] = collider
        if collider > 1e-8:
            result["reason"] = 6
            return result
        bp = float(np.linalg.norm(d.xpos[self.board] - values[:3]))
        dp = float(np.linalg.norm(d.xpos[self.flange] - (targets["release_center_m"] + [0, 0, .095])))
        normal = d.xmat[self.board].reshape(3, 3)[:, 2]
        angle = float(np.degrees(np.arccos(np.clip(normal @ targets["board_normal_world"], -1, 1))))
        result.update(board_position_error_m=bp, dropper_position_error_m=dp, board_normal_error_deg=angle)
        if bp >= .002 or dp >= .002 or angle >= .5:
            result["reason"] = 5
            return result
        bad = []
        for c in d.contact[:d.ncon]:
            names = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, int(g)) or "" for g in c.geom]
            if c.dist < -.002 and not any(s.startswith("ball/") for s in names):
                bad.append(names)
        result["pairs"] = bad
        result["reason"] = 4 if bad else 0
        return result


def audit_values(scene, assets, bucket_xy, values, *, bounds=None, known_q10=None,
    known_q14=None, known_values=None, known_bucket_xy=None,
    starts=8, iterations=120, chunk_size=128, progress=None):
    """Return stable mask and exact selected native branches, preserving slots.

    Known branches are optional local seeds, not success labels. Even an exact
    previously accepted branch must pass every current native endpoint gate.
    All absent known rows are NaN; partial or infinite hints are rejected.
    """
    begin = time.monotonic()
    xy = _bucket_array(bucket_xy)
    values = np.asarray(values, dtype=float)
    within, reach = outer_geometry_mask(scene, xy, values, bounds=bounds)
    if not 1 <= chunk_size <= 4096:
        raise ValueError("Finite CPU chunk size1..4096 required")
    shape = values.shape[:2]
    known10, known14 = known_array(known_q10, shape), known_array(known_q14, shape)
    nominal_match = known_request_match(xy, values, known_values, known_bucket_xy)
    flat = values.reshape(-1, 5)
    bucket = np.broadcast_to(xy[:, None], (*shape, 2)).reshape(-1, 2)
    hints10, hints14 = known10.reshape(-1, 6), known14.reshape(-1, 6)
    reasons = np.ones(shape, dtype=np.uint8)
    reasons[within] = 2
    eligible = within & reach
    reasons[eligible] = 3
    q10, q14 = np.zeros((*shape, 6)), np.zeros((*shape, 6))
    branch10, branch14 = np.full(shape, -1, dtype=np.int16), np.full(shape, -1, dtype=np.int16)
    preserved = np.zeros(shape, dtype=bool)
    refined = np.zeros(shape, dtype=bool)
    residual10, residual14 = np.full(shape, -1.), np.full(shape, -1.)
    pair_count = np.zeros(shape, dtype=np.int16)
    valid10, valid14 = np.zeros(shape, dtype=np.int16), np.zeros(shape, dtype=np.int16)
    position = np.full((*shape, 2), -1.)
    angle = np.full(shape, -1.)
    query10, query14 = np.full(shape, "", dtype="<U64"), np.full(shape, "", dtype="<U64")
    collisions, diagnostics = [], []
    assets = Path(assets)
    seeds = json.loads((assets / "poses.json").read_text())
    ik, gate = StableShotIK(scene, assets, device="cpu"), NativeEndpointGate(scene, assets)
    selected = np.flatnonzero(eligible.reshape(-1))
    for offset in range(0, len(selected), chunk_size):
        ids = selected[offset:offset + chunk_size]
        action = flat[ids]
        pool = ik.candidates(bucket[ids], action[:, :3], action[:, 3], action[:, 4],
            initial10=seeds["dropper_joint_rad"], initial14=seeds["board_joint_rad"],
            known_q10=hints10[ids], known_q14=hints14[ids], starts=starts, iterations=iterations,
            preserve_known=nominal_match.reshape(-1)[ids])
        p10 = {k: pool["dropper"][k].cpu().numpy() for k in ("q", "valid", "residual")}
        p14 = {k: pool["board"][k].cpu().numpy() for k in ("q", "valid", "residual")}
        for j, index in enumerate(ids):
            first = {k: a[j] for k, a in p10.items()}
            second = {k: a[j] for k, a in p14.items()}
            valid10.reshape(-1)[index], valid14.reshape(-1)[index] = first["valid"].sum(), second["valid"].sum()
            query10.reshape(-1)[index] = pool["dropper"]["query_sha256"][j]
            query14.reshape(-1)[index] = pool["board"]["query_sha256"][j]
            targets = {k: v[j] for k, v in pool["targets"].items()}
            check = lambda a, b: gate.check(a, b, bucket[index], flat[index], targets)
            i10, i14, checked, attempts = select_pair(first, second, check,
                known10=hints10[index], known14=hints14[index])
            pair_count.reshape(-1)[index] = len(attempts)
            failed = [a for a in attempts if a["reason"]]
            if failed:
                diagnostics.append(dict(layout_index=int(index // shape[1]),
                    candidate_index=int(index % shape[1]), tested_pairs=len(attempts),
                    rejected_pair_reasons={str(k): int(v) for k, v in Counter(a["reason"] for a in failed).items()}))
            if checked is None:
                if attempts:
                    # First deterministic native rejection describes the slot;
                    # every candidate pair was tested before rejecting it.
                    checked = attempts[0]
                    reasons.reshape(-1)[index] = checked["reason"]
                    if checked["pairs"]:
                        collisions.append(dict(layout_index=int(index // shape[1]),
                            candidate_index=int(index % shape[1]), pairs=checked["pairs"]))
                continue
            reasons.reshape(-1)[index] = 0
            q10.reshape(-1, 6)[index], q14.reshape(-1, 6)[index] = first["q"][i10], second["q"][i14]
            branch10.reshape(-1)[index], branch14.reshape(-1)[index] = i10, i14
            residual10.reshape(-1)[index], residual14.reshape(-1)[index] = first["residual"][i10], second["residual"][i14]
            position.reshape(-1, 2)[index] = [checked["board_position_error_m"], checked["dropper_position_error_m"]]
            angle.reshape(-1)[index] = checked["board_normal_error_deg"]
            preserved.reshape(-1)[index] = np.array_equal(q10.reshape(-1, 6)[index], hints10[index]) and np.array_equal(q14.reshape(-1, 6)[index], hints14[index])
            refined.reshape(-1)[index] = (i10 == 0 and i14 == 0 and
                np.isfinite(hints10[index]).all() and np.isfinite(hints14[index]).all() and
                not preserved.reshape(-1)[index])
        if progress:
            progress(dict(processed=min(offset + chunk_size, len(selected)), eligible=len(selected),
                endpoint_qualified=int((reasons == 0).sum())))
    mask = reasons == 0
    summary = dict(version=VERSION, ik_version=IK_VERSION,
        action_bounds_version=(OLD_BOUNDS if bounds is None else bounds).get("version"),
        layouts=shape[0], candidates_per_layout=shape[1], total_slots=int(np.prod(shape)),
        outer_prior_eligible=len(selected), endpoint_qualified=int(mask.sum()),
        layouts_with_endpoint=int(mask.any(1).sum()), known_pairs_preserved=int(preserved.sum()),
        preserved_exact_nominal=int((preserved & nominal_match).sum()),
        refined_continuation=int(refined.sum()), polishing_tolerance=1e-8,
        native_pairs_tested=int(pair_count.sum()), starts=starts, iterations=iterations,
        reasons={REASONS[int(k)]: int(v) for k, v in Counter(reasons.reshape(-1)).items()},
        elapsed_s=time.monotonic() - begin,
        scope="Query-stable CPU full-pose/joint/native endpoint collision qualification, not ball-flight or transit validation")
    return mask, dict(summary=summary, reasons=reasons, q10=q10, q14=q14,
        branch10=branch10, branch14=branch14, known_pair_preserved=preserved,
        known_request_match=nominal_match, preserved_exact_nominal=preserved & nominal_match,
        refined_continuation=refined, residual10=residual10, residual14=residual14,
        native_pairs_tested=pair_count, valid_branches10=valid10, valid_branches14=valid14,
        query10_sha256=query10, query14_sha256=query14,
        position_error_m=position, board_normal_error_deg=angle,
        collision_pairs=collisions, diagnostics=diagnostics)


audit_candidates = audit_values
