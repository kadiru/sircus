"""Tighter, still conservative support-plane clearance for the same flight.

CPU collision-distance queries only propose separating directions. Acceptance
uses a fresh support maximum over every compiled hull/enclosing-box vertex,
not the distance returned by the collision library. The original margin,
sample spacing, five predictions, arm geometry and time horizon are unchanged.
"""
from __future__ import annotations

import copy
import itertools
import math

import numpy as np

from . import arm_clearance_selector as legacy

VERSION = "sircus-post-bounce-support-clearance-v1"
RULE = dict(legacy.RULE, version=VERSION,
    distance_bound="max_original_face_and_witness_direction_support_plane",
    direction_proposal="cpu_mujoco_geomDistance_nearest_geometry_point",
    support="maximum_over_all_compiled_mesh_or_enclosing_box_vertices",
    refinement="all_samples_of_colliders_with_original_bound_below_margin",
    certificate="unit_normals_with_independently_recomputable_vertex_support")
SCOPE = legacy.SCOPE + " One correction: refine conservative face-only distance bounds; horizon unchanged."
fingerprint = legacy.fingerprint


def box_vertices(center, half_size):
    center = legacy.vector(center, 3, "box center")
    size = legacy.vector(half_size, 3, "box size")
    if np.any(size <= 0):
        raise ValueError("Positive box size required")
    return center + np.asarray(list(itertools.product((-1., 1.), repeat=3))) * size


def support_bound(points, vertices, normals):
    """Signed center-to-support-plane lower bound; each normal is unit length."""
    points, vertices, normals = map(lambda x: np.asarray(x, dtype=float), (points, vertices, normals))
    if (points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape
            or vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4
            or not all(np.isfinite(x).all() for x in (points, vertices, normals))
            or not np.allclose(np.linalg.norm(normals, axis=1), 1., atol=1e-9, rtol=0)):
        raise ValueError("Finite points, full vertices and unit normals required")
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    return np.einsum("ij,ij->i", points, normals) - (normals @ vertices.T).max(axis=1)


def certify_path(origin, velocity, horizon, gravity, colliders, radius, direction_provider):
    """Reconstruct the whole path; nearest-point directions never decide a pass."""
    old = legacy.path_bound(origin, velocity, horizon, gravity, colliders, radius)
    times = np.linspace(0., horizon, old["samples"])
    points = np.asarray(legacy.trajectory_for_member(origin, velocity, times, gravity=gravity))
    allowance = old["between_sample_allowance_m"]
    best, closest, refinements = math.inf, None, []
    for collider in colliders:
        planes = np.asarray(collider["planes"], dtype=float)
        face = (points @ planes[:, :3].T + planes[:, 3]).max(axis=1)
        bound = face - radius - allowance - 1e-10
        if float(bound.min()) < legacy.MARGIN_M:
            normals = np.asarray(direction_provider(collider, points), dtype=float)
            support = support_bound(points, collider["vertices"], normals)
            bound = np.maximum(face, support) - radius - allowance - 1e-10
            refinements.append(dict(geom=collider["name"], support_normals=normals.tolist()))
        index = int(bound.argmin())
        if float(bound[index]) < best:
            best = float(bound[index])
            closest = dict(geom=collider["name"], time_s=float(times[index]),
                center_m=points[index].tolist(), surface_clearance_lower_bound_m=best)
    return dict(old, passed=bool(best >= legacy.MARGIN_M),
        minimum_clearance_lower_bound_m=best, closest=closest,
        legacy_minimum_clearance_lower_bound_m=old["minimum_clearance_lower_bound_m"],
        refinements=refinements)


class PreciseArmClearanceGate(legacy.ArmClearanceGate):
    def __init__(self, scene, assets):
        super().__init__(scene, assets)
        self.vertices = {}
        for collider in self.local:
            index = collider["index"]
            if collider["kind"] == "native_convex_mesh":
                mesh = int(self.m.geom_dataid[index])
                start, count = int(self.m.mesh_vertadr[mesh]), int(self.m.mesh_vertnum[mesh])
                vertices = self.m.mesh_vert[start:start+count].copy()
            else:
                vertices = box_vertices(self.m.geom_aabb[index, :3], self.m.geom_aabb[index, 3:])
            self.vertices[collider["name"]] = vertices
        self.vertex_geometry_sha256 = fingerprint({name: vertices.tolist() for name, vertices in self.vertices.items()})
        self.ball_geom = self.m.geom("ball/geom").id

    def colliders(self, q10, q14, bucket_xy):
        result = super().colliders(q10, q14, bucket_xy)
        for collider in result:
            index = self.m.geom(collider["name"]).id
            collider["vertices"] = self.vertices[collider["name"]] @ self.d.geom_xmat[index].reshape(3, 3).T + self.d.geom_xpos[index]
        return result

    def _directions(self, collider, points):
        geom = self.m.geom(collider["name"]).id
        planes = np.asarray(collider["planes"])
        result = []
        for point in points:
            self.d.qpos[self.ball:self.ball+3] = point
            self.mj.mj_kinematics(self.m, self.d)
            witness = np.zeros(6, dtype=float)
            distance = float(self.mj.mj_geomDistance(self.m, self.d, self.ball_geom, geom, 10., witness))
            normal = point - witness[3:]
            length = float(np.linalg.norm(normal))
            if not math.isfinite(distance) or distance >= 10. or not np.isfinite(normal).all() or length < 1e-12:
                normal = planes[int(np.argmax(point @ planes[:, :3].T + planes[:, 3])), :3]
            else:
                normal = normal / length
            result.append(normal)
        return np.asarray(result)

    def check(self, q10, q14, bucket_xy, values, flight_diagnostic, *, gravity=(0., 0., -9.81)):
        # Reuse all unchanged input/nominal-geometry/horizon checks before refining.
        old = super().check(q10, q14, bucket_xy, values, flight_diagnostic, gravity=gravity)
        result = copy.deepcopy(old)
        result.update(version=VERSION, rule_sha256=fingerprint(RULE), scope=SCOPE,
                      vertex_geometry_sha256=self.vertex_geometry_sha256)
        if len(old["members"]) != legacy.MEMBERS:
            return result
        colliders = self.colliders(q10, q14, bucket_xy)
        result["members"] = [dict(member=member["member"], **certify_path(
            flight_diagnostic["launch"]["origin_m"], member["velocity_world_m_s"],
            saved["horizon_s"], gravity, colliders, self.radius, self._directions))
            for member, saved in zip(flight_diagnostic["members"], old["members"])]
        result.update(passed=all(member["passed"] for member in result["members"]),
            minimum_clearance_lower_bound_m=min(member["minimum_clearance_lower_bound_m"] for member in result["members"]))
        result["reason"] = None if result["passed"] else "predicted_arm_clearance_unproven"
        return result
