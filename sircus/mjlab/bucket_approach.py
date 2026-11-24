"""Opt-in whole-approach bucket geometry diagnostic; never an outcome scorer.

The existing frozen effective-rebound model supplies the same predicted flight.
This module additionally checks the sphere against every convex bucket wall and
its base before the end of the predicted whole-sphere opening passage. Positive
clearance certifies only that nominal path, not the accuracy of its rebound
model. A warning lowers a candidate's heuristic rank; it never forbids a valid
rim, table or robot-assisted native capture.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from .arm_clearance_selector import fingerprint, vector
from .consistent_flight_selector import _descending_time, member_flight
from .intergrid_flight_ranker import rebound

VERSION = "sircus-full-bucket-approach-v1"
MAX_HORIZON_S = 1.5
DEFAULT_SPACING_S = .001
SCOPE = ("Conditional nominal flight from release through one board rebound and, "
    "when forecast, one table rebound, ending at the descending whole-sphere rim "
    "passage. Convex native bucket walls and base are checked continuously using "
    "sampled exact sphere distances with a speed/spacing bound. No bucket response, "
    "second board rebound, robot rebound or retention dynamics are predicted; "
    "native scoring is unchanged and all candidates remain eligible.")


def _hull(name, vertices):
    from scipy.spatial import ConvexHull
    vertices = np.asarray(vertices, dtype=float)
    hull = ConvexHull(vertices)
    return dict(name=name, vertices=vertices, planes=hull.equations,
                triangles=vertices[hull.simplices])


def _segment_distance_squared(points, a, b):
    edge = b-a
    amount = np.clip(np.einsum("ntj,tj->nt", points[:, None, :]-a, edge)
                     / np.einsum("tj,tj->t", edge, edge), 0., 1.)
    delta = points[:, None, :]-a-amount[:, :, None]*edge
    return np.einsum("ntj,ntj->nt", delta, delta)


def convex_signed_distance(points, collider):
    """Exact Euclidean exterior distance to a convex triangulated hull.

    Interior distance is negative distance to the nearest supporting face.
    Triangle edges are included, so an inflated-halfplane corner approximation
    does not incorrectly report a sphere collision at a clear rim corner.
    """
    p = np.asarray(points, dtype=float)
    triangles = collider["triangles"]
    a, b, c = (triangles[:, i, :] for i in range(3))
    ab, ac = b-a, c-a
    normal = np.cross(ab, ac)
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    delta = p[:, None, :]-a
    signed = np.einsum("ntj,tj->nt", delta, normal)
    projection = delta-signed[:, :, None]*normal
    aa, cc, cross = (np.einsum("tj,tj->t", x, y) for x, y in ((ab, ab), (ac, ac), (ab, ac)))
    u = np.einsum("ntj,tj->nt", projection, ab)
    v = np.einsum("ntj,tj->nt", projection, ac)
    denominator = aa*cc-cross*cross
    s = (cc*u-cross*v)/denominator
    t = (aa*v-cross*u)/denominator
    face_sq = np.where((s >= 0) & (t >= 0) & (s+t <= 1), signed*signed, np.inf)
    sq = np.minimum.reduce((face_sq, _segment_distance_squared(p, a, b),
                            _segment_distance_squared(p, a, c), _segment_distance_squared(p, b, c)))
    outside = np.sqrt(sq.min(axis=1))
    planes = collider["planes"]
    face = (p@planes[:, :3].T+planes[:, 3]).max(axis=1)
    return np.where(face <= 0, face, outside)


def cylinder_signed_distance(points, center, radius, half_height):
    p = np.asarray(points, dtype=float)-center
    q = np.column_stack((np.linalg.norm(p[:, :2], axis=1)-radius, np.abs(p[:, 2])-half_height))
    return np.linalg.norm(np.maximum(q, 0.), axis=1)+np.minimum(q.max(axis=1), 0.)


class BucketApproach:
    def __init__(self, scene, *, step_s=DEFAULT_SPACING_S):
        from sircus.isaac.geometry import bucket_wedge
        self.bucket = copy.deepcopy(scene["bucket"])
        self.radius = float(scene["ball"]["radius_m"])
        self.step_s = float(step_s)
        if (not math.isfinite(self.step_s) or not 0 < self.step_s <= .01
                or not math.isfinite(self.radius) or self.radius <= 0):
            raise ValueError("Positive sphere radius and sample spacing within 10 ms required")
        self.walls = [_hull(f"bucket/wall{i:02}", bucket_wedge(self.bucket, i))
                      for i in range(self.bucket["segments"])]
        self.base = dict(name="bucket/base", center=np.array([0., 0., self.bucket["assumed_bottom_thickness_m"]/2]),
                         radius=self.bucket["outer_bottom_diameter_m"]/2,
                         half_height=self.bucket["assumed_bottom_thickness_m"]/2)
        self.geometry_source = "native_model_bucket_wedge_input_vertices"
        self._compiled_model = None
        self._fingerprint()

    def _fingerprint(self):
        self.geometry_sha256 = fingerprint(dict(
            walls=[dict(name=c["name"], vertices=c["vertices"].tolist()) for c in self.walls],
            base=dict(name=self.base["name"], center=self.base["center"].tolist(),
                      radius=self.base["radius"], half_height=self.base["half_height"]),
            ball_radius_m=self.radius, source=self.geometry_source))

    def _bind_compiled(self, ranker, bucket_xy):
        """Bind once to the same compiled model used by the endpoint ranker.

        MuJoCo may recenter/rotate mesh vertices while compiling. Apply its
        compiled geom-local transform in the bucket body. Avoid world-coordinate
        subtraction so the geometry hash is independent of bucket placement.
        """
        gate = getattr(ranker, "clearance", None)
        if gate is None:
            return
        model, data = gate.m, gate.d
        if self._compiled_model is model:
            return
        body = model.body("bucket").id
        walls = []
        for i in range(self.bucket["segments"]):
            name = f"bucket/wall{i:02}"; index = model.geom(name).id
            mesh = int(model.geom_dataid[index])
            start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            if int(model.geom_bodyid[index]) != body:
                raise ValueError("Expected native bucket wall attached directly to bucket body")
            rotation = np.zeros(9)
            gate.mj.mju_quat2Mat(rotation, model.geom_quat[index])
            vertices = model.mesh_vert[start:start+count]@rotation.reshape(3, 3).T
            vertices += model.geom_pos[index]
            walls.append(_hull(name, vertices))
        index = model.geom("bucket/base").id
        rotation = np.zeros(9)
        gate.mj.mju_quat2Mat(rotation, model.geom_quat[index])
        if int(model.geom_bodyid[index]) != body or not np.allclose(rotation.reshape(3, 3), np.eye(3), atol=1e-9, rtol=0):
            raise ValueError("This study requires the native upright bucket base")
        radius = float(model.geom_size[gate.ball_geom, 0])
        if not math.isclose(radius, self.radius, abs_tol=1e-12, rel_tol=0):
            raise ValueError("Native sphere radius differs from approach geometry")
        self.walls = walls
        self.base = dict(name="bucket/base", center=model.geom_pos[index].copy(),
                         radius=float(model.geom_size[index, 0]), half_height=float(model.geom_size[index, 1]))
        self.geometry_source = "compiled_mujoco_convex_mesh_vertices_and_cylinder"
        self._compiled_model = model
        self._fingerprint()

    def distances(self, points, bucket_xy):
        """Sphere surface distance to each native wall/base (metres)."""
        p = np.asarray(points, dtype=float)
        xy = vector(bucket_xy, 2, "bucket XY")
        if p.ndim != 2 or p.shape[1] != 3 or not len(p) or not np.isfinite(p).all():
            raise ValueError("Finite N by 3 ball centers required")
        p = p-np.array([*xy, 0.])
        distances = [convex_signed_distance(p, wall) for wall in self.walls]
        distances.append(cylinder_signed_distance(p, self.base["center"], self.base["radius"], self.base["half_height"]))
        return np.column_stack(distances)-self.radius

    def assess_flight(self, origin, velocity, horizon_s, bucket_xy, *, gravity=(0., 0., -9.81),
                      start_time_s=0., label="flight"):
        """Distance over the complete continuous nominal parabola, not just rim."""
        p, v, g = (vector(value, 3, name) for value, name in
                   ((origin, "origin"), (velocity, "velocity"), (gravity, "gravity")))
        h, start = float(horizon_s), float(start_time_s)
        if (not math.isfinite(h) or not 0 < h <= MAX_HORIZON_S
                or not math.isfinite(start) or start < 0):
            raise ValueError("Finite positive horizon <= 1.5 s and nonnegative start time required")
        intervals = max(1, int(math.ceil(h/self.step_s)))
        times = np.linspace(0., h, intervals+1)
        points = p+times[:, None]*v+.5*times[:, None]**2*g
        distances = self.distances(points, bucket_xy)
        minimum = distances.min(axis=1)
        at = int(minimum.argmin()); geom_index = int(distances[at].argmin())
        speed = max(float(np.linalg.norm(v)), float(np.linalg.norm(v+g*h)))
        allowance = speed*h/intervals/2+1e-10
        lower = float(minimum[at]-allowance)
        overlap_indices = np.flatnonzero(minimum <= 0.)
        names = [wall["name"] for wall in self.walls]+[self.base["name"]]
        first = None
        if len(overlap_indices):
            j = int(overlap_indices[0]); k = int(distances[j].argmin())
            first = dict(time_s=start+float(times[j]), segment_time_s=float(times[j]),
                time_bracket_s=[start+float(times[max(0, j-1)]), start+float(times[j])],
                center_m=points[j].tolist(), geom=names[k],
                vertical_velocity_m_s=float(v[2]+g[2]*times[j]),
                rising=bool(v[2]+g[2]*times[j] > 0), sampled_surface_clearance_m=float(minimum[j]))
        status = "sampled_overlap" if first else "clear" if lower > 0 else "between_samples_unproven"
        return dict(label=label, start_time_s=start, horizon_s=h, samples=len(times), actual_spacing_s=h/intervals,
            maximum_speed_m_s=speed, between_sample_allowance_m=allowance,
            minimum_sampled_clearance_m=float(minimum[at]), minimum_clearance_lower_bound_m=lower,
            status=status, first_predicted_contact=first,
            closest=dict(time_s=start+float(times[at]), center_m=points[at].tolist(), geom=names[geom_index]),
            native_contact=False, predicts_contact_response=False)

    def assess(self, request, ranker, prediction=None):
        """Wrap an existing FlightRanker/BoardFirstFlightRanker without changing it.

        Every route's score is the minimum of its existing heuristic score and
        predicted bucket clearance. Negative warnings remain rankable. Neither
        native scoring nor arm setup/path acceptance is altered.
        """
        supplied_prediction = prediction is not None
        prediction = copy.deepcopy(ranker.predict(request) if prediction is None else prediction)
        bucket = request["bucket_xy_m"]
        # A supplied cached prediction may belong to a different last FK call.
        # Reposition the compiled bucket before deriving bucket-local vertices.
        if supplied_prediction and getattr(ranker, "clearance", None) is not None:
            ranker.clearance.colliders(request["selected_q10"], request["selected_q14"], bucket)
        self._bind_compiled(ranker, bucket)
        result = dict(version=VERSION, scope=SCOPE, geometry_sha256=self.geometry_sha256,
            geometry_source=self.geometry_source, ball_radius_m=self.radius, sample_spacing_s=self.step_s,
            original_prediction=prediction, routes={}, predicted_route=None, predicted_score_m=None, score_m=None,
            modifies_native_scoring=False, rejects_candidates=False, uses_observed_flight=False)
        if prediction.get("board_launch_m") is None:
            result["missing_reason"] = "unsupported_board_launch"
            return result
        p = vector(prediction["board_launch_m"], 3, "board launch")
        v = vector(prediction["board_velocity_m_s"], 3, "board velocity")
        g = np.asarray(ranker.gravity)
        release = np.array(request["values"][:3], dtype=float)
        release[2] += request["values"][4]+self.radius
        fall = float(release[2]-p[2])
        fall_time = math.sqrt(-2*fall/g[2]) if fall > 0 else 0.
        if not 0 < fall_time <= MAX_HORIZON_S:
            result["missing_reason"] = "unsupported_complete_incoming_horizon"
            return result
        incoming = [self.assess_flight(release, [0., 0., 0.], fall_time, bucket,
                    gravity=g, label="release_to_board")]
        table_z = float(ranker.table_center[2]+ranker.table_half[2]+self.radius)
        table_time = _descending_time(p[2], v[2], table_z, g[2])
        for route in ("direct", "table"):
            old = prediction.get(route)
            if not old:
                continue
            if route == "direct":
                origin, velocity, start, segments = p, v, fall_time, list(incoming)
            else:
                if table_time is None or table_time > MAX_HORIZON_S or prediction.get("table_contact_m") is None:
                    continue
                origin = vector(prediction["table_contact_m"], 3, "table contact")
                velocity = rebound(v+g*table_time, np.array([0., 0., 1.]), *ranker.coefficients[2:])
                start = fall_time+table_time
                segments = incoming+[self.assess_flight(p, v, table_time, bucket,
                    gravity=g, start_time_s=fall_time, label="board_to_table")]
            flight = member_flight(origin, velocity, bucket, ranker.geom)
            horizon = flight["rim_band_end_s"]
            if horizon is None or horizon > MAX_HORIZON_S:
                result["routes"][route] = dict(status="unsupported_horizon", score_m=None,
                    original_rim_margin_m=old.get("rim_clearance_m"), segments=segments,
                    minimum_clearance_lower_bound_m=None, first_predicted_contact=None)
                continue
            segments.append(self.assess_flight(origin, velocity, horizon, bucket, gravity=g,
                start_time_s=start, label="board_to_rim" if route == "direct" else "table_to_rim"))
            margin = min(segment["minimum_clearance_lower_bound_m"] for segment in segments)
            contacts = [segment["first_predicted_contact"] for segment in segments if segment["first_predicted_contact"]]
            original_score = old.get("rim_clearance_m")
            if prediction.get("contact_ranking_version") is None and old.get("arm_clearance_m") is not None:
                original_score = min(original_score, old["arm_clearance_m"]) if original_score is not None else None
            if route == "table" and prediction.get("table_edge_margin_m") is not None and original_score is not None:
                original_score = min(original_score, prediction["table_edge_margin_m"])
            score = min(original_score, margin) if original_score is not None and old.get("arm_clearance_m") is not None else None
            result["routes"][route] = dict(status="sampled_overlap" if contacts else "clear" if margin > 0 else "between_samples_unproven",
                score_m=score, original_score_m=original_score, original_rim_margin_m=old.get("rim_clearance_m"),
                minimum_clearance_lower_bound_m=margin, first_predicted_contact=min(contacts,key=lambda item:item["time_s"]) if contacts else None,
                segments=segments, horizon_scope="through_descending_sphere_rim_band_before_retention")
        scores = {route: value["score_m"] for route,value in result["routes"].items() if value["score_m"] is not None}
        if scores:
            route = max(scores, key=scores.__getitem__)
            result.update(predicted_route=route, predicted_score_m=scores[route], score_m=scores[route])
        return result
