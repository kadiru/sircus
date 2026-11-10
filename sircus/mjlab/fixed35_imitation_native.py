"""Shared bucket-facing release orientation for both fixed-grid policies.

The actor still outputs five continuous shot coordinates. The deterministic yaw
behavior changes only the dropper target rotation; full-pose IK and the existing
sampled setup-path checks remain required. Geometry hints contain no flight labels.
"""
from __future__ import annotations

import copy
import time
import math
import numpy as np
from .benchmark_verify import digest
from .calibrated_admission import (VERSION, CONNECTION_POLICY_VERSION, verify_resolution,
    verify_connection, source_geometry, compose_anchor_prefix, anchor_prefix_identity,
    _vector, outer_geometry_mask, ordered_pairs)
from .admission import native_pose_error, base14
from .gripper_yaw_geometry import DOWNWARD
from .learning_ik import shot_targets
from scipy.spatial.transform import Rotation
from .yaw_tilt_gap_geometry import rotation_evidence
from .table_training_native import Engine

BEHAVIOR_VERSION = "sircus-continuous-bucket-facing-release-v1"


def yaw_target(scene, bucket, values):
    # Build the requested orientation even for an inadmissible close target;
    # the unchanged task-distance guard below records a decline, never aborts
    # an entire policy evaluation because one network proposal is invalid.
    bucket = _vector(bucket, 2, "bucket")
    values = _vector(values, 5, "values")
    delta = bucket-values[:2]
    yaw = float(math.atan2(delta[1], delta[0]))
    targets = {k:v[0] for k,v in shot_targets(scene,bucket[None],values[None,:3],[values[3]],[values[4]]).items()}
    targets["dropper_flange_world"][:3,:3] = Rotation.from_euler("z",yaw).as_matrix() @ DOWNWARD
    return yaw, targets


class FixedYawResolver:
    def __init__(self, base):
        self.base = base
        self._connected = {}

    def __getattr__(self, name):
        return getattr(self.base, name)

    def connect(self, anchor, requested_values):
        verify_resolution(anchor,self.sources,self.home)
        self.base._check_resolution_bytes(anchor)
        values = _vector(requested_values, 5, "requested values").tolist()
        bucket = _vector(anchor["bucket_xy_m"], 2, "bucket").tolist()
        delta = np.asarray(bucket)-np.asarray(values[:2])
        within_bounds = bool((np.asarray(values)>=self.base.solver.low-1e-9).all() and (np.asarray(values)<=self.base.solver.high+1e-9).all())
        if np.linalg.norm(delta) > .15 and within_bounds:
            yaw, targets = yaw_target(self.scene, bucket, values)
        else:
            yaw, targets = float(math.atan2(delta[1],delta[0])), None
        rotation = Rotation.from_euler("z",yaw).as_matrix() @ DOWNWARD
        behavior = dict(version=BEHAVIOR_VERSION, yaw_rad=yaw,
                        rotation_matrix=rotation.tolist(),
                        target_dropper_flange_world=targets["dropper_flange_world"].tolist() if targets is not None else None,
                        geometry_hint_only=True)
        key = digest(dict(version=BEHAVIOR_VERSION, connection_policy=CONNECTION_POLICY_VERSION,
            resolution_sha256=anchor["sha256"], values=values, bucket_xy_m=bucket, release_behavior=behavior))
        if key in self._connected: return dict(copy.deepcopy(self._connected[key]), cache_hit=True)
        begin = time.monotonic(); timing = dict(home_s=0., endpoint_s=0., path_s=0., wall_s=0.)
        record = dict(version=VERSION, bucket_xy_m=bucket, values=values, resolution_file=anchor["file"],
            resolution_sha256=anchor["sha256"], home_sha256=self.home["sha256"],
            passed=False, geometry_eligible=False, endpoint_eligible=None, reason=None,
            q10=[0.]*6, q14=[0.]*6, proof=None, endpoint_attempts=None, path_attempts=[], timing=timing,
            action_substituted=False, connection_policy=CONNECTION_POLICY_VERSION, anchor_values=anchor["anchor_values"],
            anchor_derivation_sha256=digest(anchor["anchor_derivation"]), release_behavior=behavior, behavior_key=key)
        if not anchor["home_passed"]:
            record["reason"] = "fixed_base_clearance_rejected" if anchor.get("hard_reject") else "canonical_home_unresolved_in_layout"
        else:
            hint = source_geometry(anchor["effective_source_hint"])
            if anchor.get("resolved"):
                hint.update({k: anchor["resolved"][k] for k in ("q10", "q14", "values", "bucket_xy_m")})
            start = time.monotonic(); endpoint = self.solve(bucket, values, hint); timing["endpoint_s"] += time.monotonic()-start
            record["endpoint_attempts"] = endpoint; record["endpoint_eligible"] = endpoint["passed"]
            record["reason"] = endpoint["reason"]
            for row in endpoint["candidates"]:
                target = np.r_[row["q10"], row["q14"]]
                start = time.monotonic()
                if anchor["passed"]:
                    aq = np.r_[anchor["resolved"]["q10"], anchor["resolved"]["q14"]]
                    if np.array_equal(aq, target):
                        proof = compose_anchor_prefix(anchor, self.home, target)
                    else:
                        suffix = self.planner.plan(target, bucket, start=aq)
                        if suffix["passed"]:
                            proof = compose_anchor_prefix(anchor, self.home, target, suffix)
                        else:
                            rejected = dict(passed=False, reason="finite_anchor_suffix_unresolved",
                                connection_route=dict(kind="anchor_suffix_rejected", prefix=anchor_prefix_identity(anchor),
                                    target_q=target.tolist(), suffix_proof=suffix))
                            suffix_index = len(record["path_attempts"])
                            record["path_attempts"].append(rejected)
                            proof = copy.deepcopy(self.planner.plan(target, bucket))
                            proof["connection_route"] = dict(kind="full_home_after_anchor_suffix", prefix=anchor_prefix_identity(anchor),
                                suffix_attempt_index=suffix_index)
                else:
                    # An unresolved nominal anchor never vetoes an actual action.
                    proof = self.planner.plan(target, bucket)
                timing["path_s"] += time.monotonic()-start
                record["path_attempts"].append(proof)
                if proof["passed"]:
                    record.update(passed=True, geometry_eligible=True, q10=row["q10"], q14=row["q14"],
                        proof=proof, endpoint=row["endpoint"], reason=None); break
            if endpoint["passed"] and not record["passed"]: record["reason"] = "finite_requested_setup_path_unresolved"
        timing["wall_s"] = time.monotonic()-begin
        record.update(self.base._save("fixed-yaw-connections", key, record))
        verify_connection(record, anchor, self.home, values)
        self._connected[key] = copy.deepcopy(record)
        return dict(record, cache_hit=False)

    def solve(self, bucket_xy, values, hint):
        bucket, values = _vector(bucket_xy, 2, "bucket"), _vector(values, 5, "values")
        info = dict(passed=False, reason=None, candidates=[], pair_attempts=[])
        within, reach = outer_geometry_mask(self.base.solver.scene, bucket[None], values[None, None], bounds=self.base.solver.bounds)
        info["analytic_hint"] = dict(old_within=bool(within[0, 0]), old_workspace=bool(reach[0, 0]), hard_veto=False)
        # Preserve explicit five-parameter bounds and the existing >150mm
        # board-to-bucket task constraint. Only the wrist-sphere heuristic is
        # diagnostic; it cannot suppress actual IK/collision checks.
        if (values < self.base.solver.low - 1e-9).any() or (values > self.base.solver.high + 1e-9).any():
            info["reason"] = "action_bounds_rejected"; return info
        if np.linalg.norm(bucket - values[:2]) <= .15:
            info["reason"] = "board_bucket_distance_constraint_rejected"; return info
        hint = source_geometry(hint)
        _, targets = yaw_target(self.scene, bucket, values)
        solver = self.base.solver
        pools = []
        for arm, target, initial, known in (
            (solver.ik.dropper, targets["dropper_flange_world"], solver.initial["dropper_joint_rad"], hint["q10"]),
            (solver.ik.board, targets["board_flange_local"], solver.initial["board_joint_rad"], hint["q14"])):
            pool = arm.solve_candidates(target[None], initial, known=np.asarray(known)[None],
                starts=16, iterations=180, normal_only=False, preserve_known=np.asarray([False]))
            pools.append({k:pool[k][0].cpu().numpy() for k in ("q", "valid", "residual")})
        first, second = pools
        info.update(targets={k:v.tolist() for k,v in targets.items()}, starts=16,
                    iterations=180, normal_only=False, selection_uses_flight_labels=False,
                    geometry_hint_sha256=digest(hint))
        info["valid_ik_branches"] = [int(first["valid"].sum()), int(second["valid"].sum())]
        seen, bad10, bad14 = set(), set(), set()
        for i, j in ordered_pairs(first, second, known10=np.asarray(hint["q10"]), known14=np.asarray(hint["q14"])):
            if i in bad10 or j in bad14: continue
            q10, q14 = first["q"][i], second["q"][j]
            identity = tuple(np.round(np.r_[q10, q14], 5))
            if identity in seen: continue
            seen.add(identity)
            checked = self.base.solver.endpoint.check(q10, q14, bucket, values, targets)
            attempt = dict(branch10=int(i), branch14=int(j), checked=checked)
            info["pair_attempts"].append(attempt)
            if checked["reason"]:
                for role in checked.get("independent_reject_arms", []):
                    (bad10 if role == 10 else bad14).add(i if role == 10 else j)
                continue
            q = np.r_[q10, q14]
            errors = [native_pose_error(self.base.solver.gate, role, q, target) for role, target in (
                (10, targets["dropper_flange_world"]),
                (14, base14(self.base.solver.scene) @ targets["board_flange_local"]))]
            attempt["native_pose_errors"] = errors
            if any(e["position_m"] >= 1e-4 or .25 * e["rotation_rad"] >= 1e-4 for e in errors):
                attempt["pose_rejected"] = True; continue
            info["candidates"].append(dict(bucket_xy_m=bucket.tolist(), values=values.tolist(),
                q10=q10.tolist(), q14=q14.tolist(), geometry_eligible=True,
                branch10=int(i), branch14=int(j), endpoint=checked))
            if len(info["candidates"]) >= self.base.solver.max_pairs: break
        info["passed"] = bool(info["candidates"])
        info["reason"] = None if info["passed"] else ("finite_ik_unresolved" if not first["valid"].any() or not second["valid"].any() else "strict_endpoint_clearance_or_pose_rejected")
        return info


class FixedYawEngine(Engine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adapter.resolver = FixedYawResolver(self.adapter.resolver)

    def evaluate(self, requests, phase):
        rows, timing = super().evaluate(requests, phase)
        for row in rows:
            row["adapter_directory"] = "native-yaw"
            row["release_behavior_version"] = BEHAVIOR_VERSION
            values = np.asarray(row["values"])
            solver = self.adapter.resolver.solver
            target_defined = (np.linalg.norm(np.asarray(row["bucket_xy_m"])-values[:2]) > .15
                              and (values>=solver.low-1e-9).all() and (values<=solver.high+1e-9).all())
            if target_defined:
                row["yaw_evidence"] = rotation_evidence(self.adapter.ref["scene"], self.adapter.resolver.solver.gate, row)
            else:
                if row.get("native_release"):
                    raise RuntimeError("An action outside the valid target domain cannot release")
                delta = np.asarray(row["bucket_xy_m"])-np.asarray(row["values"][:2])
                yaw = float(math.atan2(delta[1],delta[0]))
                row["yaw_evidence"] = dict(requested_yaw_rad=yaw,requested_rotation_matrix=(Rotation.from_euler("z",yaw).as_matrix() @ DOWNWARD).tolist(),
                    basis="requested orientation only; inadmissible action did not release")
            if not row["connection_file"].startswith("fixed-yaw-connections/"):
                raise RuntimeError("Every fixed-yaw prediction requires a distinct orientation proof")
            if row.get("native_release"):
                for name in ("q10", "q14"):
                    if row["native_initial_"+name] != np.asarray(row[name],np.float32).astype(float).tolist():
                        raise RuntimeError("Native initial joints differ from admitted yaw endpoint")
        return rows, timing
