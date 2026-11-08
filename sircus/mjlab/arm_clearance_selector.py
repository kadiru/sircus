"""Frozen CPU-only arm clearance gate for nominal post-bounce flight.

This filters a pre-existing candidate order; it neither changes the learned
model nor probes native ball outcomes. Convex collision geometry is placed at
the exact admitted joint pair. A conservative bound covers the space between
samples of each of the five predicted ballistic paths. Model error and later
bucket contact are deliberately outside this geometric statement.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .consistent_flight_selector import MEMBERS, nominal_origin, trajectory_for_member


VERSION = "sircus-post-bounce-arm-clearance-v1"
MARGIN_M = .002
MAX_SPACING_S = .002
MAX_HORIZON_S = 1.5
SCOPE = ("All five frozen nominal post-board rebound paths, exact admitted held "
    "arm joints, both arms' collision geometry, excluding only the intended board. "
    "Conservative geometric clearance is conditional on the predicted flight; "
    "not native contact, success, incoming drop, driven transit, or retention proof.")
RULE = dict(version=VERSION, extra_sphere_margin_m=MARGIN_M,
    maximum_sample_spacing_s=MAX_SPACING_S, maximum_horizon_s=MAX_HORIZON_S,
    start="requested_sphere_plane_tangent", end="first_descending_rim_band_start",
    members_required=MEMBERS, missing_or_overlong_horizon="reject",
    between_sample_bound="max_endpoint_speed_times_half_actual_spacing",
    collision_geometry="compiled_convex_mesh_hulls_and_enclosing_primitive_boxes",
    excluded_collision_geometry="Arm14/impact_board/collision/0",
    incoming_drop_checked=False, native_flight_queries=0,
    changes_candidate_bank=False, maximum_candidate_geometry_visits=4)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def vector(value, size, name):
    data = np.asarray(value, dtype=float)
    if data.shape != (size,) or not np.isfinite(data).all():
        raise ValueError("Finite "+name+" required")
    return data


def box_planes(center, half_size):
    """Outward normalized planes for an enclosing local primitive box."""
    center, size = vector(center, 3, "box center"), vector(half_size, 3, "box half size")
    if np.any(size <= 0): raise ValueError("Positive enclosing box half sizes required")
    normals = np.r_[np.eye(3), -np.eye(3)]
    offsets = np.r_[-center-size, center-size]
    return np.column_stack((normals, offsets))


def mesh_planes(vertices):
    """The native mesh collider uses the convex hull, not visible triangles."""
    from scipy.spatial import ConvexHull
    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4 or not np.isfinite(points).all():
        raise ValueError("Finite nondegenerate mesh vertices required")
    planes = ConvexHull(points).equations
    lengths = np.linalg.norm(planes[:, :3], axis=1)
    if np.any(lengths <= 0): raise ValueError("Degenerate mesh planes")
    return np.unique(np.round(planes/lengths[:, None], 12), axis=0)


def world_planes(planes, rotation, translation):
    planes = np.asarray(planes, dtype=float)
    rotation, translation = np.asarray(rotation, dtype=float), vector(translation, 3, "translation")
    if (planes.ndim != 2 or planes.shape[1] != 4 or len(planes) < 4
        or not np.isfinite(planes).all() or rotation.shape != (3, 3)
        or not np.isfinite(rotation).all() or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-9, rtol=0)
        or not np.isclose(np.linalg.det(rotation), 1., atol=1e-9, rtol=0)):
        raise ValueError("Finite normalized polytope planes and proper rigid rotation required")
    norms = np.linalg.norm(planes[:, :3], axis=1)
    if not np.allclose(norms, 1., atol=1e-9, rtol=0): raise ValueError("Unit face normals required")
    normals = planes[:, :3]@rotation.T
    return np.column_stack((normals, planes[:, 3]-normals@translation))


def path_bound(origin, velocity, horizon, gravity, colliders, ball_radius):
    """Conservative surface clearance over a continuous predicted parabola.

    Maximum outward plane distance is a lower bound on distance to a convex
    collider. Distance is 1-Lipschitz, so subtracting maximum speed times half
    the sample spacing covers every point between the included endpoints.
    Enclosing primitive boxes can reject a clear path but cannot prove a false
    pass. This numerical bound does not account for learned velocity error.
    """
    p, v, g = vector(origin, 3, "origin"), vector(velocity, 3, "velocity"), vector(gravity, 3, "gravity")
    h, radius = float(horizon), float(ball_radius)
    if not math.isfinite(h) or not 0 < h <= MAX_HORIZON_S:
        raise ValueError("Complete horizon within frozen limit required")
    if not math.isfinite(radius) or radius <= 0: raise ValueError("Positive ball radius required")
    if not colliders: raise ValueError("Complete arm collision geometry required")
    segments = max(1, int(math.ceil(h/MAX_SPACING_S)))
    times = np.linspace(0., h, segments+1)
    positions = np.asarray(trajectory_for_member(p, v, times, gravity=g))
    speed = max(float(np.linalg.norm(v)), float(np.linalg.norm(v+g*h)))
    error = speed*(h/segments)/2
    best, closest = math.inf, None
    for collider in colliders:
        planes = np.asarray(collider["planes"], dtype=float)
        if (planes.ndim != 2 or planes.shape[1] != 4 or len(planes) < 4
            or not np.isfinite(planes).all()
            or not np.allclose(np.linalg.norm(planes[:, :3], axis=1), 1., atol=1e-9, rtol=0)):
            raise ValueError("Complete finite collider planes required")
        # Roundoff slack is included instead of rounding a near-zero gap up.
        lower = (positions@planes[:, :3].T+planes[:, 3]).max(axis=1)-radius-error-1e-10
        index = int(np.argmin(lower)); measured = float(lower[index])
        if measured < best:
            best, closest = measured, dict(geom=collider["name"], time_s=float(times[index]),
                center_m=positions[index].tolist(), surface_clearance_lower_bound_m=measured)
    return dict(passed=bool(best >= MARGIN_M), minimum_clearance_lower_bound_m=best,
        closest=closest, horizon_s=h, samples=len(times), actual_spacing_s=h/segments,
        maximum_speed_m_s=speed, between_sample_allowance_m=error,
        required_margin_m=MARGIN_M, collision_geometry_count=len(colliders))


class ArmClearanceGate:
    """Compile once; reuse CPU FK for exact endpoint branches without stepping."""
    def __init__(self, scene, assets):
        import mujoco as mj
        from .model import build_scene
        self.mj = mj
        xml = build_scene(scene, Path(assets), solref=(.004, .15), contact_friction=.35)
        self.model_sha256 = hashlib.sha256(xml.encode()).hexdigest()
        self.m = m = mj.MjModel.from_xml_string(xml); self.d = mj.MjData(m)
        self.qi = {role: np.asarray([m.jnt_qposadr[m.joint(f"{role}/joint{i}").id] for i in range(1, 7)])
            for role in ("Arm10", "Arm14")}
        self.limits = {role: np.asarray([m.jnt_range[m.joint(f"{role}/joint{i}").id] for i in range(1, 7)])
            for role in ("Arm10", "Arm14")}
        self.fingers = [m.jnt_qposadr[m.joint(f"Arm10/finger_{side}_joint").id] for side in ("left", "right")]
        self.ball = m.jnt_qposadr[m.joint("ball/free").id]
        self.mocap = m.body_mocapid[m.body("bucket").id]
        self.radius = float(scene["ball"]["radius_m"])
        self.nominal_geometry = dict(ball_radius_m=self.radius,
            board_size_m=scene['board']['size_m'], mount_yaw_rad=scene['board']['mount_yaw_rad'])
        self.local, self.excluded = [], []
        for index in range(m.ngeom):
            name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, index) or ""
            if not name.startswith(("Arm10/", "Arm14/")): continue
            if name == RULE["excluded_collision_geometry"]:
                self.excluded.append(name); continue
            if m.geom_contype[index] == 0 and m.geom_conaffinity[index] == 0: continue
            if int(m.geom_type[index]) == int(mj.mjtGeom.mjGEOM_MESH):
                mesh = int(m.geom_dataid[index]); start, count = int(m.mesh_vertadr[mesh]), int(m.mesh_vertnum[mesh])
                planes = mesh_planes(m.mesh_vert[start:start+count]); kind = "native_convex_mesh"
            else:
                planes = box_planes(m.geom_aabb[index, :3], m.geom_aabb[index, 3:]); kind = "enclosing_primitive_box"
            self.local.append(dict(index=index, name=name, planes=planes, kind=kind))
        if self.excluded != [RULE["excluded_collision_geometry"]] or not self.local:
            raise ValueError("Expected complete robot model and one intended impact board")
        self.geometry_sha256 = fingerprint([dict(name=c["name"], kind=c["kind"], planes=c["planes"].tolist()) for c in self.local])

    def colliders(self, q10, q14, bucket_xy):
        m, d = self.m, self.d
        q = {"Arm10": vector(q10, 6, "dropper joints"), "Arm14": vector(q14, 6, "board joints")}
        for role, values in q.items():
            limits = self.limits[role]
            if np.any(values < limits[:, 0]-1e-9) or np.any(values > limits[:, 1]+1e-9):
                raise ValueError("Exact admitted joints must satisfy native limits")
        d.qpos[:] = m.qpos0; d.qvel[:] = 0
        for role, values in q.items(): d.qpos[self.qi[role]] = values
        d.qpos[self.fingers] = .025
        d.qpos[self.ball:self.ball+7] = [0, 0, 2, 1, 0, 0, 0]
        d.mocap_pos[self.mocap] = [*vector(bucket_xy, 2, "bucket XY"), 0]
        d.mocap_quat[self.mocap] = [1, 0, 0, 0]
        self.mj.mj_forward(m, d)
        return [dict(name=c["name"], kind=c["kind"], planes=world_planes(c["planes"],
            d.geom_xmat[c["index"]].reshape(3, 3), d.geom_xpos[c["index"]])) for c in self.local]

    def check(self, q10, q14, bucket_xy, values, flight_diagnostic, *, gravity=(0., 0., -9.81)):
        action = vector(values, 5, "five shot values")
        bucket = vector(bucket_xy, 2, "bucket XY")
        result = dict(version=VERSION, rule_sha256=fingerprint(RULE), passed=False,
            reason="nominal_flight_unavailable", required_margin_m=MARGIN_M,
            minimum_clearance_lower_bound_m=None, members=[], scope=SCOPE,
            model_sha256=self.model_sha256, geometry_sha256=self.geometry_sha256,
            endpoint_sha256=fingerprint(dict(q10=vector(q10, 6, "dropper joints").tolist(),
                q14=vector(q14, 6, "board joints").tolist(), bucket_xy_m=bucket.tolist(), values=action.tolist())),
            excluded_collision_geometry=list(self.excluded), incoming_drop_checked=False,
            native_queries=0, uses_observed_rebound=False)
        launch, members = flight_diagnostic.get("launch", {}), flight_diagnostic.get("members", [])
        if launch.get("supported_planar_contact") is not True or len(members) != MEMBERS:
            return result
        expected = nominal_origin(action, bucket, self.nominal_geometry)
        if expected.get('supported_planar_contact') is not True or any(
            launch.get(key) != expected[key] for key in ('origin_m', 'board_normal')):
            raise ValueError("Nominal rebound geometry differs from exact requested action")
        if [member.get("member") for member in members] != list(range(MEMBERS)):
            raise ValueError("Exactly five identity-matched frozen member flights required")
        colliders = self.colliders(q10, q14, bucket)
        for member in members:
            horizon = member.get("rim_band_start_s")
            if (member.get("full_downward_rim_band_available") is not True
                or not isinstance(horizon, (float, int)) or not math.isfinite(horizon)
                or not 0 < horizon <= MAX_HORIZON_S):
                result.update(reason="missing_or_overlong_complete_flight_horizon"); return result
            if member.get("origin_m") != launch.get("origin_m"):
                raise ValueError("Predicted flight origin identity mismatch")
            result["members"].append(dict(member=member["member"], **path_bound(
                launch["origin_m"], member["velocity_world_m_s"], horizon, gravity, colliders, self.radius)))
        result.update(passed=all(r["passed"] for r in result["members"]),
            minimum_clearance_lower_bound_m=min(r["minimum_clearance_lower_bound_m"] for r in result["members"]))
        result["reason"] = None if result["passed"] else "predicted_arm_clearance_unproven"
        return result
