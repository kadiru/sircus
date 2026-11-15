"""Rank admitted shots by predicted rim and arm clearance before release."""

from __future__ import annotations

import math

import numpy as np

from .arm_clearance_distance import PreciseArmClearanceGate, certify_path
from .arm_clearance_selector import MAX_HORIZON_S
from .consistent_flight_selector import _descending_time, geometry, member_flight


RULE_VERSION = "sircus-intergrid-ballistic-rank-v1"
COEFFICIENTS = (
    "board_normal_restitution", "board_tangent_retention",
    "table_normal_restitution", "table_tangent_retention",
)


def rebound(velocity, normal, restitution, tangent_retention):
    """Apply one effective normal and tangential speed ratio at contact."""
    normal_velocity = np.dot(velocity, normal) * normal
    return tangent_retention * (velocity - normal_velocity) - restitution * normal_velocity


def board_launch(board, board_size, release, radius, gravity_z, restitution, tangent):
    """Find the ball's first top-face tangent on the vertical release ray."""
    center, normal = board[:3, 3], board[:3, 2]
    if normal[2] <= 0:
        return None
    origin = release.copy()
    origin[2] = center[2] + (
        radius + board_size[2] / 2 - np.dot(normal[:2], release[:2] - center[:2])
    ) / normal[2]
    contact = board[:3, :3].T @ (origin - radius * normal - center)
    fall = release[2] - origin[2]
    if fall <= 0 or np.any(np.abs(contact[:2]) > board_size[:2] / 2):
        return None
    incoming = np.array([0., 0., -math.sqrt(-2 * gravity_z * fall)])
    return origin, rebound(incoming, normal, restitution, tangent)


class FlightRanker:
    def __init__(self, scene, calibration, assets, rule):
        if rule["version"] != RULE_VERSION:
            raise ValueError("Unknown flight ranking rule")
        self.coefficients = tuple(float(rule["coefficients"][name]) for name in COEFFICIENTS)
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in self.coefficients):
            raise ValueError("Effective rebound coefficients must be within [0, 1]")
        self.geom = geometry(scene, calibration)
        self.radius = self.geom["ball_radius_m"]
        self.gravity = np.asarray(self.geom["gravity_m_s2"])
        self.board_size = np.asarray(self.geom["board_size_m"])
        self.table_center = np.asarray(scene["table"]["center_m"], dtype=float)
        self.table_half = np.asarray(scene["table"]["size_m"], dtype=float) / 2
        self.clearance = PreciseArmClearanceGate(scene, assets)
        self.board_geom = self.clearance.m.geom(self.clearance.excluded[0]).id

    def _clearance(self, origin, velocity, horizon, colliders):
        return certify_path(
            origin, velocity, horizon, self.gravity, colliders, self.radius,
            self.clearance._directions,
        )["minimum_clearance_lower_bound_m"]

    def _rim(self, origin, velocity, bucket, colliders):
        flight = member_flight(origin, velocity, bucket, self.geom)
        time = flight["rim_band_end_s"]
        return dict(
            rim_xy_m=(flight["first_descending_center_rim"] or {}).get("xy_m"),
            rim_clearance_m=flight["conservative_rim_band_clearance_m"],
            arm_clearance_m=self._clearance(origin, velocity, time, colliders)
            if time is not None and time <= MAX_HORIZON_S else None,
        )

    def predict(self, request):
        """Use only the requested release, held joint pair, and compiled geometry."""
        bucket = request["bucket_xy_m"]
        colliders = self.clearance.colliders(request["selected_q10"], request["selected_q14"], bucket)
        data = self.clearance.d
        board = np.eye(4)
        board[:3, :3] = data.geom_xmat[self.board_geom].reshape(3, 3)
        board[:3, 3] = data.geom_xpos[self.board_geom]
        release = np.array(request["values"][:3], dtype=float)
        release[2] += request["values"][4] + self.radius
        board_e, board_tangent, table_e, table_tangent = self.coefficients
        launch = board_launch(board, self.board_size, release, self.radius, self.gravity[2],
                              board_e, board_tangent)
        result = dict(board_launch_m=None, board_velocity_m_s=None, table_contact_m=None,
                      direct=None, table=None, predicted_route=None, score_m=None)
        if launch is None:
            return result
        origin, velocity = launch
        result.update(board_launch_m=origin.tolist(), board_velocity_m_s=velocity.tolist())
        direct = self._rim(origin, velocity, bucket, colliders)
        result["direct"] = direct
        scores = {}
        if direct["arm_clearance_m"] is not None:
            scores["direct"] = min(direct["arm_clearance_m"], direct["rim_clearance_m"])

        # Continue the flight through one tabletop rebound when the first
        # arc cannot make a clear aperture passage.
        table_z = self.table_center[2] + self.table_half[2] + self.radius
        time = _descending_time(origin[2], velocity[2], table_z, self.gravity[2])
        if (direct["rim_clearance_m"] is None or direct["rim_clearance_m"] < 0) and (
            time is not None and time <= MAX_HORIZON_S
        ):
            impact = origin + velocity * time + .5 * self.gravity * time ** 2
            edge = float(min(self.table_half[:2] - np.abs(impact[:2] - self.table_center[:2])))
            if edge >= 0:
                result["table_contact_m"] = impact.tolist()
                table_velocity = rebound(
                    velocity + self.gravity * time, np.array([0., 0., 1.]), table_e, table_tangent,
                )
                table = self._rim(impact, table_velocity, bucket, colliders)
                if table["arm_clearance_m"] is not None:
                    first_clearance = self._clearance(origin, velocity, time, colliders)
                    table["arm_clearance_m"] = min(table["arm_clearance_m"], first_clearance)
                    scores["table"] = min(table["arm_clearance_m"], table["rim_clearance_m"], edge)
                result["table"] = table
        if scores:
            route = max(scores, key=scores.__getitem__)
            result.update(predicted_route=route, score_m=scores[route])
        return result

    def order(self, admitted):
        ranked = [(request, self.predict(request)["score_m"]) for request in admitted]
        return [request for request, _ in sorted(
            ranked, key=lambda item: rank_key(item[0], item[1]),
        )]


def rank_key(request, score):
    return (score is None, -score if score is not None else 0., request["id"])
