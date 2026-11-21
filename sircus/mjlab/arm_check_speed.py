"""Opt-in broad-phase batching for the frozen geometric arm checker.

Only independent AABB distances are batched. Native collision queries, convex
fallbacks, contacts, joint/path sampling and corridor rules retain their original
order and semantics. Install on one existing gate; no global monkeypatching or
simulation stepping is performed. The historical checker remains unchanged.
"""
from types import MethodType

import numpy as np

from .transit_audit import GeometricGate

VERSION = "sircus-vectorized-arm-broad-phase-v1"
_ROUNDING_GUARD = 32 * np.finfo(float).eps


def _indices(gate):
    # Pairs are normally fixed by construction. Rebuild if a caller deliberately
    # changes them, so the optimization cannot silently retain obsolete filters.
    pairs = tuple(gate.pairs)
    if pairs != gate._sircus_vectorized_pairs:
        gate._sircus_vectorized_pairs = pairs
        gate._sircus_vectorized_indices = np.asarray(pairs, dtype=np.intp).reshape(-1, 2)
    return gate._sircus_vectorized_indices


def install_vectorized_gate(gate):
    """Install on a GeometricGate instance and return that same instance.

    Existing model, data, bucket position and all references to the gate survive.
    Refuse an unknown overridden checker rather than bypassing additional rules.
    Repeated installation is harmless.
    """
    method = getattr(gate.check, "__func__", None)
    if method is _check:
        return gate
    if method is not GeometricGate.check:
        raise TypeError("Expected the unmodified GeometricGate.check method")
    gate._sircus_vectorized_pairs = None
    _indices(gate)
    gate.check = MethodType(_check, gate)
    gate.arm_check_implementation = VERSION
    return gate


def _check(self, q, *, corridor=None):
    q = np.asarray(q, float)
    lower = np.r_[self.native.limits10[:, 0], self.native.limits14[:, 0]]
    upper = np.r_[self.native.limits10[:, 1], self.native.limits14[:, 1]]
    if q.shape != (12,) or not np.isfinite(q).all() or (q < lower).any() or (q > upper).any():
        return dict(passed=False, reason="joint_limits_or_nonfinite")
    self.state(q)
    self.world_points = [p @ self.d.geom_xmat[i].reshape(3, 3).T + self.d.geom_xpos[i]
        for i, p in enumerate(self.local_points)]
    boxes = np.asarray([self.aabb(i) for i in range(self.m.ngeom)])
    indices = _indices(self)
    left, right = indices[:, 0], indices[:, 1]
    gaps = np.maximum(np.maximum(boxes[left, 0] - boxes[right, 1],
        boxes[right, 0] - boxes[left, 1]), 0)
    distances = np.linalg.norm(gaps, axis=1)
    exact = np.zeros(len(indices), dtype=bool)
    unresolved, unreliable = [], 0

    # A vector reduction can round differently from scalar np.linalg.norm.
    # Recheck threshold-near pairs with the original scalar operation before
    # deciding whether to call MuJoCo. The small guard changes no threshold.
    guard = _ROUNDING_GUARD * max(1., abs(self.clearance))
    for index in np.flatnonzero(distances <= self.clearance + guard):
        i, j = self.pairs[index]
        conservative = float(np.linalg.norm(gaps[index]))
        distance = conservative
        exact[index] = True
        if conservative < self.clearance:
            witness = np.zeros(6)
            distance = float(self.mj.mj_geomDistance(self.m, self.d, i, j, .25, witness))
            if distance < .249 and abs(np.linalg.norm(witness[:3] - witness[3:]) - abs(distance)) > 1e-5:
                unreliable += 1
                distance = conservative
            else:
                distance = max(distance, conservative)
            if distance < self.clearance:
                distance = max(distance, self.separation_lower_bound(i, j))
        if distance < self.clearance:
            unresolved.append(dict(pair=[self.names[i], self.names[j]], lower_bound_m=distance))
        distances[index] = distance

    minimum, pair = 1., None
    # Recompute every contender for the reported minimum with scalar arithmetic.
    # Original pair order breaks exact ties, including the initial 1 m sentinel.
    eligible = np.flatnonzero(distances <= 1. + _ROUNDING_GUARD)
    if len(eligible):
        approximate_minimum = min(1., float(distances[eligible].min()))
        for index in eligible[distances[eligible] <= approximate_minimum + _ROUNDING_GUARD]:
            distance = float(distances[index]) if exact[index] else float(np.linalg.norm(gaps[index]))
            if distance < minimum:
                i, j = self.pairs[index]
                minimum, pair = distance, [self.names[i], self.names[j]]

    bad = []
    for c in self.d.contact[:self.d.ncon]:
        a, b = map(int, c.geom)
        if not (a in self.moving_geoms or b in self.moving_geoms):
            continue
        bodies = {self.body_names[a], self.body_names[b]}
        if any(set(pair) == bodies for pair in self.carried_exclusions):
            continue
        if c.dist < self.clearance:
            bad.append(dict(pair=[self.names[a], self.names[b]], distance_m=float(c.dist)))
    board = self.d.xpos[self.native.board].copy()
    rot = self.d.xmat[self.native.board].reshape(3, 3)
    size = np.asarray(self.m.geom_size[self.m.geom("Arm14/impact_board/collision/0").id])
    board_top = float(board[2] + np.abs(rot[2]) @ size)
    drop_h = float(self.d.xpos[self.native.flange, 2])
    height_ok = not corridor or ((corridor.get("board_top_max_m") is None or board_top <= corridor["board_top_max_m"])
        and (corridor.get("dropper_flange_min_m") is None or drop_h >= corridor["dropper_flange_min_m"]))
    return dict(passed=not bad and not unresolved and height_ok,
        reason="clearance" if bad else ("unproven_clearance" if unresolved else ("corridor" if not height_ok else None)),
        minimum_clearance_m=minimum, closest_pair=pair, board_top_m=board_top, dropper_flange_z_m=drop_h,
        margin_contacts=bad, inconsistent_distance_queries=unreliable, unresolved_clearance_pairs=unresolved)
