"""Extend the frozen support certificate to a descending table-tangent plane.

Only the conditional, post-board-rebound flight window changes. Predictions,
endpoint geometry, ranking, support bounds and margins retain their old meaning.
This free-flight projection does not simulate table/bucket deflections or prove
capture. A refusal is a failure to certify clearance, not a predicted collision.
"""
from __future__ import annotations

import copy
import math

from . import arm_clearance_distance as precise
from . import arm_clearance_selector as legacy

VERSION = "sircus-post-bounce-table-horizon-clearance-v1"
RULE = dict(precise.RULE, version=VERSION,
    end="first_descending_tabletop_sphere_tangent",
    original_end=precise.RULE["end"],
    target_center_z="scene_table_center_z_plus_half_height_plus_ball_radius",
    crossing="positive_descending_root_of_unchanged_member_ballistic_z",
    original_window_must_pass=True,
    shortened_extension="reject",
    missing_or_overlong_horizon="reject_without_clamping")
SCOPE = (legacy.SCOPE + " Support-plane refinement is unchanged. Only the end "
    "of each of the same five predicted free-flight paths extends from descending "
    "rim-band start to descending ball-center Z=tabletop+ball radius. This is a "
    "conditional projection, not a simulation of table/bucket rebounds, incoming "
    "drop, retention or driven-arm transit.")
fingerprint = legacy.fingerprint


def descending_table_horizon(origin, velocity, gravity, *, tabletop_z, ball_radius):
    """Return the untruncated descending crossing; callers enforce the old cap.

    With a=-gravity_z and target=tabletop_z+radius, the positive descending root
    is (velocity_z + sqrt(velocity_z**2 + 2*a*(origin_z-target))) / a.
    """
    p = legacy.vector(origin, 3, "flight origin")
    v = legacy.vector(velocity, 3, "flight velocity")
    g = legacy.vector(gravity, 3, "gravity")
    if (not isinstance(tabletop_z, (int, float)) or not math.isfinite(tabletop_z)
            or not isinstance(ball_radius, (int, float)) or not math.isfinite(ball_radius)
            or ball_radius <= 0 or g[2] >= 0):
        raise ValueError("Finite tabletop, positive radius and downward gravity required")
    target = float(tabletop_z + ball_radius)
    if p[2] < target:
        raise ValueError("Rebound origin is below the table-tangent plane")
    a = -float(g[2])
    root = (float(v[2]) + math.sqrt(float(v[2])**2 + 2*a*(float(p[2])-target))) / a
    if not math.isfinite(root) or root <= 0 or v[2] + g[2]*root >= 0:
        raise ValueError("No positive descending table-tangent crossing")
    return root


class HorizonArmClearanceGate(precise.PreciseArmClearanceGate):
    def __init__(self, scene, assets):
        super().__init__(scene, assets)
        table_center = legacy.vector(scene["table"]["center_m"], 3, "table center")
        table_size = legacy.vector(scene["table"]["size_m"], 3, "table size")
        if any(table_size <= 0):
            raise ValueError("Positive table dimensions required")
        self.tabletop_z = float(table_center[2] + table_size[2]/2)

    def check(self, q10, q14, bucket_xy, values, flight_diagnostic, *, gravity=(0., 0., -9.81)):
        # Preserve the previous origin, member-ID, geometry and endpoint checks.
        old = super().check(q10, q14, bucket_xy, values, flight_diagnostic, gravity=gravity)
        result = copy.deepcopy(old)
        result.update(version=VERSION, rule_sha256=fingerprint(RULE), scope=SCOPE,
            original_rule_sha256=old["rule_sha256"],
            original_horizon_end=precise.RULE["end"], horizon_end=RULE["end"],
            tabletop_z_m=self.tabletop_z, target_ball_center_z_m=self.tabletop_z+self.radius,
            original_passed=old["passed"],
            original_minimum_clearance_lower_bound_m=old["minimum_clearance_lower_bound_m"],
            horizon_metadata=[], passed=False)
        if len(old["members"]) != legacy.MEMBERS:
            return result
        horizons = []
        for member, saved in zip(flight_diagnostic["members"], old["members"]):
            info = dict(member=member["member"], original_horizon_s=saved["horizon_s"],
                new_horizon_s=None, maximum_horizon_s=legacy.MAX_HORIZON_S)
            result["horizon_metadata"].append(info)
            try:
                horizon = descending_table_horizon(flight_diagnostic["launch"]["origin_m"],
                    member["velocity_world_m_s"], gravity,
                    tabletop_z=self.tabletop_z, ball_radius=self.radius)
            except ValueError:
                result.update(members=[], minimum_clearance_lower_bound_m=None,
                    reason="missing_descending_table_tangent_horizon")
                return result
            info["new_horizon_s"] = horizon
            if horizon > legacy.MAX_HORIZON_S or horizon < saved["horizon_s"]:
                result.update(members=[], minimum_clearance_lower_bound_m=None,
                    reason="overlong_or_shortened_table_tangent_horizon")
                return result
            horizons.append(horizon)
        colliders = self.colliders(q10, q14, bucket_xy)
        result["members"] = [dict(member=member["member"],
            original_horizon_s=saved["horizon_s"], **precise.certify_path(
                flight_diagnostic["launch"]["origin_m"], member["velocity_world_m_s"],
                horizon, gravity, colliders, self.radius, self._directions))
            for member, saved, horizon in zip(flight_diagnostic["members"], old["members"], horizons)]
        longer_passed = all(m["passed"] for m in result["members"])
        result.update(passed=bool(old["passed"] and longer_passed), extended_window_passed=longer_passed,
            minimum_clearance_lower_bound_m=min(m["minimum_clearance_lower_bound_m"] for m in result["members"]))
        result["reason"] = (None if result["passed"] else "original_window_clearance_unproven"
            if not old["passed"] else "predicted_arm_clearance_unproven")
        return result
