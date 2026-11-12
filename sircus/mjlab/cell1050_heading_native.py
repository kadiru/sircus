"""Prepare and replay exact full-pose branches with a variable board heading."""
from __future__ import annotations

from collections import deque
import copy
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .admission import base14, board_flange_world, native_pose_error
from .bank_geometry import FixedEndpointIK
from .benchmark_verify import contained, digest, read, sha
from .calibrated_adapter import CalibratedAdapter, vector
from .calibrated_admission import (
    CONNECTION_POLICY_VERSION, VERSION, anchor_prefix_identity, anchor_search_spec,
    compose_anchor_prefix, verify_connection, verify_resolution,
)
from .fixed35_bouncer_native import wrapped_difference
from .fixed35_broad_native import bouncer_order, distinct_indices, reach_models, wrist_bound
from .fixed35_imitation_native import yaw_target
from .learning_contract import constraints
from .table_correction_native import copy_bound
from .table_training_native import Engine


HEADING_VERSION = "sircus-cell1050-variable-board-heading-v1"
BRANCH_RULE = dict(
    starts_per_arm=16,
    iterations=180,
    maximum_pairs=24,
    maximum_admitted_branches=4,
    minimum_joint_difference_rad=.05,
    position_tolerance_m=1e-4,
    rotation_tolerance_rad=4e-4,
    path="Both serialized joint orders from the authenticated anchor, then from home",
)


def heading_targets(scene, bucket, values, offset_deg, twist_deg=0.):
    """Rotate the board about its center while keeping the release point fixed."""
    if not np.isfinite([offset_deg, twist_deg]).all() or max(abs(offset_deg), abs(twist_deg)) > 180:
        raise ValueError("Finite board heading and twist offsets within ±180 degrees required")
    _, targets = yaw_target(scene, bucket, values)
    if offset_deg or twist_deg:
        world = targets["board_world"]
        azimuth = Rotation.from_euler("z", offset_deg, degrees=True).as_matrix()
        twist = Rotation.from_euler("z", twist_deg, degrees=True).as_matrix()
        world[:3, :3] = azimuth @ world[:3, :3] @ twist
        targets["board_flange_local"] = np.linalg.inv(base14(scene)) @ board_flange_world(scene, world)
        targets["board_normal_world"] = world[:3, 2]
    return targets


def _hint(proposal, anchor):
    explicit = ("seed_q10" in proposal, "seed_q14" in proposal)
    if any(explicit):
        if not all(explicit):
            raise ValueError("A warm start requires both joint vectors")
        return {key: vector(proposal["seed_" + key], 6) for key in ("q10", "q14")}
    source = anchor["resolved"] if anchor["passed"] else anchor["effective_source_hint"]
    return {key: vector(source[key], 6) for key in ("q10", "q14")}


def _path(planner, start, target):
    """Use the existing sampled edge implementation with a finite route menu."""
    for order in ((14, 10), (10, 14)):
        proof = planner._audit_path(planner._serial_points(start, target, order))
        if proof["passed"]:
            return dict(proof, reason=None)
    return dict(passed=False, reason="both_serialized_joint_orders_rejected")


class HeadingResolver:
    def __init__(self, base, assets):
        self.base = base
        self.models = reach_models(str(Path(assets).resolve()))
        self.pending = deque()
        self.connections = {}

    def __getattr__(self, name):
        return getattr(self.base, name)

    def _paths(self, anchor, target):
        home = np.r_[self.home["q10"], self.home["q14"]]
        if not anchor["passed"]:
            return [_path(self.planner, home, target)]
        prefix = anchor_prefix_identity(anchor)
        start = np.asarray(prefix["anchor_q"])
        if np.array_equal(start, target):
            return [compose_anchor_prefix(anchor, self.home, target)]
        suffix = _path(self.planner, start, target)
        if suffix["passed"]:
            return [compose_anchor_prefix(anchor, self.home, target, suffix)]
        failed = dict(
            passed=False, reason="finite_anchor_suffix_unresolved",
            connection_route=dict(
                kind="anchor_suffix_rejected", prefix=prefix, target_q=target.tolist(),
                suffix_proof=suffix,
            ),
        )
        fallback = _path(self.planner, home, target)
        fallback["connection_route"] = dict(
            kind="full_home_after_anchor_suffix", prefix=prefix, suffix_attempt_index=0,
        )
        return [failed, fallback]

    def _branches(self, first, second, hint):
        board_order = bouncer_order(second, hint["q14"])
        dropper_order = sorted(
            distinct_indices(first),
            key=lambda i: (float(wrapped_difference(first["q"][i], hint["q10"]).max()),
                           float(first["residual"][i]), i),
        )
        return [(i, j) for i in dropper_order for j in board_order]

    def _save_branch(self, proposal, anchor, targets, candidate, paths):
        q10, q14 = candidate["q10"], candidate["q14"]
        heading = proposal["board_heading_offset_deg"]
        twist = proposal["board_twist_deg"]
        key = digest(dict(
            version=HEADING_VERSION, resolution_sha256=anchor["sha256"],
            values=proposal["values"], board_heading_offset_deg=heading, board_twist_deg=twist,
            q10=q10, q14=q14,
        ))
        if key in self.connections:
            return self.connections[key]
        record = dict(
            version=VERSION, heading_rule_version=HEADING_VERSION,
            board_heading_offset_deg=heading, board_twist_deg=twist,
            heading_key=key, targets={name: value.tolist() for name, value in targets.items()},
            bucket_xy_m=anchor["bucket_xy_m"], values=proposal["values"],
            resolution_file=anchor["file"], resolution_sha256=anchor["sha256"],
            home_sha256=self.home["sha256"], passed=True, geometry_eligible=True,
            endpoint_eligible=True, reason=None, q10=q10, q14=q14,
            proof=paths[-1], endpoint=candidate["endpoint"],
            endpoint_attempts=dict(candidates=[candidate]), path_attempts=paths,
            action_substituted=False, connection_policy=CONNECTION_POLICY_VERSION,
            anchor_values=anchor["anchor_values"], anchor_derivation_sha256=digest(anchor["anchor_derivation"]),
        )
        record.update(self.base._save("heading-connections", key, record))
        verify_connection(record, anchor, self.home, proposal["values"])
        self.connections[key] = record
        return record

    def prepare_batch(self, proposals, anchor):
        """Run IK in batches, then certify only a finite number of branch paths."""
        verify_resolution(anchor, self.sources, self.home)
        self.base._check_resolution_bytes(anchor)
        if not anchor["home_passed"]:
            return [([], "common_home_not_admitted") for _ in proposals]
        bucket = np.asarray(anchor["bucket_xy_m"])
        solver = self.solver
        prepared = [([], "proven_full_pose_wrist_reach_rejected") for _ in proposals]
        admitted_to_ik = []
        for index, proposal in enumerate(proposals):
            values = np.asarray(proposal["values"])
            if ((values < solver.low - 1e-9).any() or (values > solver.high + 1e-9).any()
                    or np.linalg.norm(bucket - values[:2]) <= .15):
                prepared[index] = ([], "action_bounds_or_minimum_distance_rejected")
                continue
            targets = heading_targets(
                self.scene, bucket, values, proposal["board_heading_offset_deg"], proposal["board_twist_deg"],
            )
            if (wrist_bound(self.models["dropper"], targets["dropper_flange_world"])["passed"]
                    and wrist_bound(self.models["bouncer"], targets["board_flange_local"])["passed"]):
                admitted_to_ik.append((index, targets, _hint(proposal, anchor)))
        if not admitted_to_ik:
            return prepared
        pools = []
        for arm, role, key, initial in (
            (solver.ik.dropper, "q10", "dropper_flange_world", "dropper_joint_rad"),
            (solver.ik.board, "q14", "board_flange_local", "board_joint_rad"),
        ):
            result = arm.solve_candidates(
                np.asarray([targets[key] for _, targets, _ in admitted_to_ik]), solver.initial[initial],
                known=np.asarray([hint[role] for _, _, hint in admitted_to_ik]),
                starts=BRANCH_RULE["starts_per_arm"], iterations=BRANCH_RULE["iterations"],
                normal_only=False,
            )
            pools.append({name: result[name].cpu().numpy() for name in ("q", "valid", "residual")})

        self.planner.gate.bucket_xy = bucket
        for row_index, (index, targets, hint) in enumerate(admitted_to_ik):
            proposal = proposals[index]
            first, second = ({key: value[row_index] for key, value in pool.items()} for pool in pools)
            pairs = self._branches(first, second, hint)
            connections = []
            reason = "strict_endpoint_clearance_or_pose_rejected" if pairs else "finite_ik_unresolved"
            rejected = {10: set(), 14: set()}
            attempts = 0
            for i, j in pairs:
                if i in rejected[10] or j in rejected[14]:
                    continue
                if attempts >= BRANCH_RULE["maximum_pairs"]:
                    break
                q10, q14 = first["q"][i], second["q"][j]
                if any(
                    max(np.max(wrapped_difference(q10, previous["q10"])),
                        np.max(wrapped_difference(q14, previous["q14"])))
                    < BRANCH_RULE["minimum_joint_difference_rad"]
                    for previous in connections
                ):
                    continue
                attempts += 1
                endpoint = solver.endpoint.check(q10, q14, bucket, proposal["values"], targets)
                if endpoint["reason"]:
                    for role in endpoint.get("independent_reject_arms", ()):
                        rejected[role].add(i if role == 10 else j)
                    continue
                q = np.r_[q10, q14]
                errors = [
                    native_pose_error(solver.gate, role, q, target)
                    for role, target in (
                        (10, targets["dropper_flange_world"]),
                        (14, base14(self.scene) @ targets["board_flange_local"]),
                    )
                ]
                if any(
                    e["position_m"] >= BRANCH_RULE["position_tolerance_m"]
                    or e["rotation_rad"] >= BRANCH_RULE["rotation_tolerance_rad"]
                    for e in errors
                ):
                    continue
                paths = self._paths(anchor, q)
                if not paths[-1]["passed"]:
                    reason = "finite_requested_setup_path_unresolved"
                    continue
                candidate = dict(
                    values=proposal["values"], q10=q10.tolist(), q14=q14.tolist(),
                    branch10=i, branch14=j, endpoint=endpoint, native_pose_errors=errors,
                )
                connections.append(self._save_branch(proposal, anchor, targets, candidate, paths))
                if len(connections) >= BRANCH_RULE["maximum_admitted_branches"]:
                    break
            prepared[index] = (connections, None if connections else reason)
        return prepared

    def connect(self, anchor, values):
        if not self.pending:
            raise ValueError("An explicit frozen heading request is required")
        request = self.pending.popleft()
        path = contained(self.directory, request["prepared_connection_file"])
        fingerprint = request["prepared_connection_sha256"]
        if sha(path) != fingerprint:
            raise ValueError("Frozen heading connection bytes changed")
        record = dict(read(path), file=request["prepared_connection_file"], sha256=fingerprint)
        verify_connection(record, anchor, self.home, values)
        if (not record["passed"] or record["heading_rule_version"] != HEADING_VERSION
                or request["bucket_xy_m"] != anchor["bucket_xy_m"] or request["values"] != list(values)):
            raise ValueError("Prepared heading request lost its action or admission binding")
        for key in ("heading_key", "board_heading_offset_deg", "board_twist_deg"):
            if request[key] != record[key]:
                raise ValueError("Prepared board orientation changed: " + key)
        for key in ("q10", "q14"):
            if request["selected_" + key] != record[key]:
                raise ValueError("Prepared joint branch changed: " + key)
        return dict(record, cache_hit=True)


class HeadingEndpoints(FixedEndpointIK):
    def __init__(self, scene, bucket, values, q10, q14, offsets, twists, device):
        super().__init__(scene, bucket, values, q10, q14, np.ones(len(bucket), dtype=bool), device=device)
        targets = [
            heading_targets(scene, xy, v, offset, twist)
            for xy, v, offset, twist in zip(bucket, values, offsets, twists)
        ]
        self.targets = {key: np.stack([target[key] for target in targets]) for key in targets[0]}

    def solve(self, bucket_xy, board_xyz, tilt_deg, gap_m, **kwargs):
        result = super().solve(bucket_xy, board_xyz, tilt_deg, gap_m, **kwargs)
        count = len(bucket_xy) - len(self.bucket)
        result["targets"] = {
            key: np.concatenate((values, np.repeat(values[-1:], count, axis=0)))
            for key, values in self.targets.items()
        }
        return result


class HeadingAdapter(CalibratedAdapter):
    def _native(self, candidates, timing):
        env = self._environment(timing)
        bucket = np.asarray([row["bucket_xy_m"] for row in candidates])
        values = np.asarray([row["values"] for row in candidates])
        q10 = np.asarray([row["q10"] for row in candidates])
        q14 = np.asarray([row["q14"] for row in candidates])
        offsets, twists = zip(*(self.orientation_by_id[row["id"]] for row in candidates))
        previous = env.ik
        try:
            env.ik = HeadingEndpoints(env.scene, bucket, values, q10, q14, offsets, twists, env.device)
            expected = env.ik.targets["board_world"][:, :3, :3]
            rows, metadata = env.run(bucket, values, constraints(len(candidates)), record=True)
        finally:
            env.ik = previous
        for row, command10, command14, target in zip(rows, q10, q14, expected):
            for key, command in (("native_initial_q10", command10), ("native_initial_q14", command14)):
                if row[key] != command.astype(np.float32).astype(float).tolist():
                    raise RuntimeError("Native initial joints changed from the admitted heading branch")
            actual = Rotation.from_quat(np.roll(row["board_quaternion_wxyz"], -1)).as_matrix()
            if Rotation.from_matrix(actual @ target.T).magnitude() >= BRANCH_RULE["rotation_tolerance_rad"]:
                raise RuntimeError("Native initial board orientation changed from the admitted heading")
        return rows, env.last_arrays, metadata


class HeadingEngine(Engine):
    """The native64 scorer and engine are shared with the historical studies."""

    def __init__(self, ref, assets, directory, *, anchor_source=None, anchor_directory=None, **kwargs):
        super().__init__(ref, assets, directory, **kwargs)
        original = self.adapter
        resolver = HeadingResolver(original.resolver, assets)
        if (anchor_source is None) != (anchor_directory is None):
            raise ValueError("Supply both the historical anchor source and its proof directory")
        if anchor_source is not None:
            anchor = copy_bound(
                Path(anchor_directory), self.directory, anchor_source["context_resolution_file"],
                anchor_source["context_resolution_sha256"],
            )
            verify_resolution(anchor, resolver.sources, resolver.home)
            key = digest(dict(
                version=VERSION, bucket_xy_m=anchor_source["bucket_xy_m"],
                home_sha256=resolver.home["sha256"], sources=resolver.sources,
                bounds=resolver.bounds, anchor_search=anchor_search_spec(),
            ))
            if anchor["key"] != key or anchor["bucket_xy_m"] != anchor_source["bucket_xy_m"]:
                raise ValueError("Historical setup anchor does not match the current geometry")
            resolver.base._resolved[key] = anchor
        self.adapter = HeadingAdapter(
            ref, assets, directory, prior_records=original.sources, calibration=original.calibration,
            home=original.home, batch=original.batch, resolver=resolver,
            environment_factory=original.environment_factory,
        )
        self.prepared_ids = set()

    def prepare(self, proposals):
        if not proposals:
            return [], []
        if self.adapter.resolver.pending:
            raise RuntimeError("Cannot prepare during native evaluation")
        start = time.monotonic()
        normalized = []
        for proposal in proposals:
            item = copy.deepcopy(proposal)
            identifier = item["id"]
            if not isinstance(identifier, str) or not identifier or identifier in self.prepared_ids:
                raise ValueError("A fresh nonempty heading proposal ID is required")
            self.prepared_ids.add(identifier)
            item["values"] = vector(item["values"], 5)
            item["bucket_xy_m"] = vector(item["bucket_xy_m"], 2)
            item["board_heading_offset_deg"] = float(item["board_heading_offset_deg"])
            item["board_twist_deg"] = float(item.get("board_twist_deg", 0.))
            normalized.append(item)
        anchor, _ = self.adapter.resolve_context(normalized[0])
        if any(item["bucket_xy_m"] != anchor["bucket_xy_m"] for item in normalized):
            raise ValueError("A heading preparation batch must use one fixed bucket")
        results = self.adapter.resolver.prepare_batch(normalized, anchor)
        requests, preparations = [], []
        for item, (connections, reason) in zip(normalized, results):
            admitted_ids = []
            for index, connection in enumerate(connections):
                request = dict(
                    item, id=f"{item['id']}-b{index}", proposal_id=item["id"],
                    command_values=item["values"], native_prepared=True,
                    prepared_connection_file=connection["file"], prepared_connection_sha256=connection["sha256"],
                    heading_key=connection["heading_key"], selected_q10=connection["q10"],
                    selected_q14=connection["q14"],
                    source_hint=anchor["source_hint"], effective_source_hint=anchor["effective_source_hint"],
                    source_teacher_id=anchor["source_hint"]["id"],
                    context_resolution_file=anchor["file"], context_resolution_sha256=anchor["sha256"],
                    context_home_sha256=anchor["home_sha256"], context_layout_eligible=anchor["layout_eligible"],
                    context_anchor_admitted=anchor["anchor_admitted"], context_hard_reject=anchor["hard_reject"],
                )
                requests.append(request)
                admitted_ids.append(request["id"])
            preparations.append(dict(
                proposal_id=item["id"], proposal_sha256=digest(item), admitted=len(admitted_ids),
                admitted_request_ids=admitted_ids, reason=reason,
                timing=dict(wall_s=(time.monotonic() - start) / len(normalized)),
            ))
        return requests, preparations

    def evaluate(self, requests, phase):
        resolver = self.adapter.resolver
        if resolver.pending or any(request.get("native_prepared") is not True for request in requests):
            raise ValueError("Only frozen heading requests can reach the native engine")
        resolver.pending = deque(requests)
        self.adapter.orientation_by_id = {
            request["id"]: (request["board_heading_offset_deg"], request["board_twist_deg"])
            for request in requests
        }
        try:
            rows, timing = super().evaluate(requests, phase)
            if resolver.pending or any(row.get("native_release") is not True for row in rows):
                raise RuntimeError("Prepared heading request did not execute as admitted")
        finally:
            resolver.pending.clear()
            self.adapter.orientation_by_id.clear()
        return rows, timing
