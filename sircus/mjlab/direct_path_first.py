"""Opt-in candidate path before reference-path search, with separate evidence.

The historical resolvers and native physics are unchanged. The deferred context
only certifies canonical home and chooses a geometry hint: it NEVER claims that
reference search failed or completed. Candidate endpoints retain the historical
IK/pose/clearance checks. Every accepted setup retains the full forward/reverse
0.5-degree sampled certificate and 5 mm clearance. Reference search runs only
after both serialized direct paths fail, and its historical evidence is retained.
"""
from __future__ import annotations

from collections import deque
import copy
import json
from pathlib import Path
import time

import numpy as np

from . import calibrated_admission as legacy
from .calibrated_admission import source_geometry
from .admission import base14, native_pose_error
from .benchmark_verify import contained, digest, read, sha
from .calibrated_adapter import (vector, timing_record, add_geometry_timing,
                                 VERSION as HISTORICAL_ADAPTER_VERSION)
from .cell1050_heading_native import (
    HeadingResolver, HeadingAdapter, HeadingEngine, HEADING_VERSION, BRANCH_RULE,
    heading_targets, _hint, wrist_bound,
)
from .fixed35_bouncer_native import wrapped_difference
from .arm_check_speed import install_vectorized_gate
from .arm_path_speed import install_path_short_circuit

VERSION = "sircus-candidate-direct-path-first-v1"
ADAPTER_VERSION = "sircus-candidate-direct-path-first-adapter-v1"
ORDERS = ((14, 10), (10, 14))
FEATURES = ["sircus-vectorized-arm-broad-phase-v1",
            "sircus-arm-path-rejection-short-circuit-v1", VERSION]


def context_key(resolver, bucket):
    return digest(dict(version=VERSION, bucket_xy_m=vector(bucket, 2),
        home_sha256=resolver.home["sha256"], sources=resolver.sources,
        bounds=resolver.bounds, anchor_search=legacy.anchor_search_spec()))


def verify_resolution(record, sources, home, bounds=None):
    """Validate deferred home-only context; no fake exhaustive-search claim."""
    if record.get("version") != VERSION:
        return legacy.verify_resolution(record, sources, home)
    bucket = vector(record.get("bucket_xy_m"), 2)
    first = legacy.anchor_candidates(sources, bucket)[0]
    expected = dict(
        home_sha256=home["sha256"], reference_search_deferred=True,
        passed=False, anchor_admitted=False, attempts=[], resolved=None, proof=None,
        source_hint=first["source"], anchor_values=first["derivation"]["values"],
        anchor_derivation=first["derivation"],
        effective_source_hint=legacy.effective_source(first["source"], bucket, first["derivation"]),
        anchor_search=legacy.anchor_search_spec(), selection_uses_flight_labels=False,
    )
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("Deferred reference context identity/provenance changed")
    checked = record.get("home_geometry", {})
    if (type(record.get("home_passed")) is not bool
            or record["home_passed"] is not bool(checked.get("passed"))
            or record.get("layout_eligible") is not record["home_passed"]
            or record.get("hard_reject") is not bool(checked.get("hard_reject"))
            or record.get("reason") != (None if checked.get("passed") else checked.get("reason"))):
        raise ValueError("Deferred context lost its canonical home check")
    if bounds is not None:
        key = digest(dict(version=VERSION, bucket_xy_m=bucket, home_sha256=home["sha256"],
            sources=[legacy.source_geometry(s) for s in sources], bounds=bounds,
            anchor_search=legacy.anchor_search_spec()))
        if record.get("key") != key:
            raise ValueError("Deferred context source/bounds identity changed")
    return True


def legacy_connection(record, anchor):
    """Construct the exact old-format binding for an actual fallback proof."""
    result = dict(record)
    result.update(version=legacy.VERSION, connection_policy=legacy.CONNECTION_POLICY_VERSION,
        resolution_file=anchor["file"], resolution_sha256=anchor["sha256"],
        anchor_values=anchor["anchor_values"],
        anchor_derivation_sha256=digest(anchor["anchor_derivation"]))
    return result


def _serial_points(start, target, order):
    points = [np.asarray(start, float).copy()]
    for role in order:
        q = points[-1].copy()
        selected = slice(0, 6) if role == 10 else slice(6, 12)
        q[selected] = np.asarray(target)[selected]
        points.append(q)
    return [q.tolist() for q in points]


def verify_connection(record, anchor, home, requested_values):
    if record.get("version") != VERSION:
        return legacy.verify_connection(record, anchor, home, requested_values)
    expected = dict(home_sha256=home["sha256"], resolution_sha256=anchor["sha256"],
        resolution_file=anchor["file"], values=vector(requested_values, 5),
        bucket_xy_m=anchor["bucket_xy_m"], action_substituted=False,
        connection_policy=VERSION, anchor_values=anchor["anchor_values"],
        anchor_derivation_sha256=digest(anchor["anchor_derivation"]))
    if anchor.get("version") != VERSION or any(record.get(k) != v for k,v in expected.items()):
        raise ValueError("Direct-first action/context identity changed")
    if (not record.get("passed") or not record.get("endpoint_eligible")
            or not record.get("geometry_eligible") or not anchor["home_passed"]):
        raise ValueError("Direct-first connection requires home/endpoint admission")
    start = np.r_[home["q10"], home["q14"]]
    target = np.r_[vector(record.get("q10"), 6), vector(record.get("q14"), 6)]
    attempts = record.get("direct_path_attempts", [])
    if not 1 <= len(attempts) <= 2:
        raise ValueError("Direct-first requires its ordered direct attempts")
    for index, attempt in enumerate(attempts):
        if (attempt.get("order") != list(ORDERS[index])
                or attempt.get("proof", {}).get("waypoints") != _serial_points(start, target, ORDERS[index])):
            raise ValueError("Direct path order/endpoints changed")
        if index < len(attempts)-1 and attempt["proof"].get("passed") is not False:
            raise ValueError("Direct path skipped an accepted order")
    fallback = record.get("fallback_resolution")
    if fallback is None:
        if attempts[-1]["proof"].get("passed") is not True or record["proof"] != attempts[-1]["proof"]:
            raise ValueError("Direct proof must be the first accepted direct order")
        if record["path_attempts"] != [a["proof"] for a in attempts]:
            raise ValueError("Direct path attempts lost their certificates")
    else:
        if len(attempts) != 2 or any(a["proof"].get("passed") is not False for a in attempts):
            raise ValueError("Reference search requires both direct orders to fail first")
        if fallback.get("version") != legacy.VERSION:
            raise ValueError("Reference fallback must retain its historical evidence version")
        legacy.verify_connection(legacy_connection(record, fallback), fallback, home, requested_values)
    candidates = record.get("endpoint_attempts", {}).get("candidates", [])
    if not any(c.get("values") == record["values"] and c.get("q10") == record["q10"]
               and c.get("q14") == record["q14"] for c in candidates):
        raise ValueError("Candidate endpoint branch association changed")
    seed = record.get("ik_seed", {})
    if seed.get("kind") not in ("explicit_proposal_joints", "deferred_source_geometry"):
        raise ValueError("Direct-first connection must retain its IK seed provenance")
    for role in ("q10", "q14"):
        vector(seed.get(role), 6)
        if seed["kind"] == "deferred_source_geometry" and seed[role] != anchor["effective_source_hint"][role]:
            raise ValueError("Deferred source IK hint changed")
    if seed.get("source_id") != anchor["source_hint"]["id"]:
        raise ValueError("Direct-first seed source association changed")
    fingerprint = seed.get("proposal_sha256")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint):
        raise ValueError("Direct-first proof requires the exact normalized proposal digest")
    key = digest(dict(version=VERSION, resolution_sha256=anchor["sha256"],
        values=record["values"], board_heading_offset_deg=record["board_heading_offset_deg"],
        board_twist_deg=record["board_twist_deg"], q10=record["q10"], q14=record["q14"],
        proposal_sha256=fingerprint))
    if record.get("heading_key") != key:
        raise ValueError("Direct-first proposal/endpoint proof key changed")
    if record["proof"] not in record["path_attempts"]:
        raise ValueError("Selected path proof missing from attempts")
    legacy.verify_path(record["proof"], start, target)
    return True


class DirectPathResolver(HeadingResolver):
    def __init__(self, base, assets):
        super().__init__(base, assets)
        # The existing worker reset clears _connected: no per-query context
        # escapes the existing worker lifecycle or contributes hidden warm data.
        self.contexts = base._connected
        self.last_direct_attempts = []
        self.last_fallback = None

    def resolve(self, bucket_xy):
        bucket = vector(bucket_xy, 2)
        key = context_key(self, bucket)
        if key in self.contexts:
            record = self.contexts[key]
            self.base._check_resolution_bytes(record)
            return dict(copy.deepcopy(record), cache_hit=True)
        relative = f"direct-contexts/{key}.json"
        path = contained(self.directory, relative)
        if path.exists():
            record = dict(read(path), file=relative, sha256=sha(path))
            verify_resolution(record, self.sources, self.home, self.bounds)
            self.base._check_resolution_bytes(record)
            self.contexts[key] = record
            return dict(record, cache_hit=True)
        begin = time.monotonic()
        checked = self.planner.check_home(bucket)
        first = legacy.anchor_candidates(self.sources, bucket)[0]
        record = dict(version=VERSION, key=key, bucket_xy_m=bucket,
            home_sha256=self.home["sha256"], home_geometry=checked,
            home_passed=bool(checked["passed"]), layout_eligible=bool(checked["passed"]),
            hard_reject=bool(checked.get("hard_reject")), passed=False, anchor_admitted=False,
            resolved=None, proof=None, reference_search_deferred=True,
            source_hint=first["source"], anchor_values=first["derivation"]["values"],
            anchor_derivation=first["derivation"],
            effective_source_hint=legacy.effective_source(first["source"], bucket, first["derivation"]),
            anchor_search=legacy.anchor_search_spec(), attempts=[],
            reason=None if checked["passed"] else checked.get("reason"),
            timing=dict(home_s=time.monotonic()-begin, endpoint_s=0., path_s=0., wall_s=time.monotonic()-begin),
            selection_uses_flight_labels=False)
        verify_resolution(record, self.sources, self.home, self.bounds)
        record.update(self.base._save("direct-contexts", key, record))
        self.contexts[key] = copy.deepcopy(record)
        return dict(record, cache_hit=False)

    def _paths(self, anchor, target):
        home = np.r_[self.home["q10"], self.home["q14"]]
        self.last_direct_attempts = []
        self.last_fallback = None
        for order in ORDERS:
            proof = self.planner._audit_path(self.planner._serial_points(home, target, order))
            self.last_direct_attempts.append(dict(order=list(order), proof=proof))
            if proof["passed"]:
                return [row["proof"] for row in self.last_direct_attempts]
        fallback = self.base.resolve(anchor["bucket_xy_m"])
        legacy.verify_resolution(fallback, self.sources, self.home)
        self.base._check_resolution_bytes(fallback)
        self.last_fallback = fallback
        return super()._paths(fallback, target)

    def _save_branch(self, proposal, anchor, targets, candidate, paths):
        q10, q14 = candidate["q10"], candidate["q14"]
        heading, twist = proposal["board_heading_offset_deg"], proposal["board_twist_deg"]
        key = digest(dict(version=VERSION, resolution_sha256=anchor["sha256"],
            values=proposal["values"], board_heading_offset_deg=heading, board_twist_deg=twist,
            q10=q10, q14=q14, proposal_sha256=digest(proposal)))
        if key in self.connections:
            return self.connections[key]
        record = dict(version=VERSION, heading_rule_version=HEADING_VERSION,
            board_heading_offset_deg=heading, board_twist_deg=twist, heading_key=key,
            targets={name:value.tolist() for name,value in targets.items()},
            bucket_xy_m=anchor["bucket_xy_m"], values=proposal["values"],
            resolution_file=anchor["file"], resolution_sha256=anchor["sha256"],
            home_sha256=self.home["sha256"], passed=True, geometry_eligible=True,
            endpoint_eligible=True, reason=None, q10=q10, q14=q14,
            proof=paths[-1], endpoint=candidate["endpoint"],
            endpoint_attempts=dict(candidates=[candidate]), path_attempts=paths,
            action_substituted=False, connection_policy=VERSION,
            anchor_values=anchor["anchor_values"], anchor_derivation_sha256=digest(anchor["anchor_derivation"]),
            direct_path_attempts=copy.deepcopy(self.last_direct_attempts),
            fallback_resolution=copy.deepcopy(self.last_fallback),
            ik_seed=dict(kind="explicit_proposal_joints" if "seed_q10" in proposal else "deferred_source_geometry",
                q10=vector(_hint(proposal, anchor)["q10"], 6),
                q14=vector(_hint(proposal, anchor)["q14"], 6),
                proposal_sha256=digest(proposal),
                source_id=anchor["source_hint"]["id"]))
        verify_connection(record, anchor, self.home, proposal["values"])
        record.update(self.base._save("direct-connections", key, record))
        self.connections[key] = record
        return record

    def prepare_batch(self, proposals, anchor, *, one_per_setting=False):
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
                starts=1 if one_per_setting else BRANCH_RULE["starts_per_arm"],
                iterations=80 if one_per_setting else BRANCH_RULE["iterations"],
                normal_only=False,
            )
            pools.append({name: result[name].cpu().numpy() for name in ("q", "valid", "residual")})

        self.planner.gate.bucket_xy = bucket
        for row_index, (index, targets, hint) in enumerate(admitted_to_ik):
            proposal = proposals[index]
            first, second = ({key: value[row_index] for key, value in pool.items()} for pool in pools)
            pairs = self._branches(first, second, hint, prefer_saved=one_per_setting)
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
                if one_per_setting or len(connections) >= BRANCH_RULE["maximum_admitted_branches"]:
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
        fallback = record.get("fallback_resolution")
        if fallback is not None:
            legacy.verify_resolution(fallback, self.sources, self.home)
            self.base._check_resolution_bytes(fallback)
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


class DirectPathAdapter(HeadingAdapter):
    def resolve_context(self, context):
        begin = time.monotonic()
        record = self.resolver.resolve(vector(context["bucket_xy_m"], 2))
        verify_resolution(record, self.sources, self.home, self.resolver.bounds)
        measured = timing_record()
        add_geometry_timing(measured, record)
        measured["wall_s"] = time.monotonic()-begin
        return record, measured

    def evaluate(self, requests, phase):
        begin, timing = time.monotonic(), timing_record()
        if not isinstance(phase, str) or not phase:
            raise ValueError("Nonempty calibrated trial phase required")
        normalized = []
        for request in requests:
            if not isinstance(request.get("id"), str) or not request["id"] or request["id"] in self._ids or any(r["id"] == request["id"] for r in normalized):
                raise ValueError("Unique nonempty requested trial ID required")
            value = copy.deepcopy(request)
            value["bucket_xy_m"] = vector(request.get("bucket_xy_m"), 2)
            value["values"] = vector(request.get("values"), 5)
            normalized.append(value)
        self._ids.update(r["id"] for r in normalized)
        self.storage_check()
        candidates = []
        for request in normalized:
            resolution = self.resolver.resolve(request["bucket_xy_m"])
            verify_resolution(resolution, self.sources, self.home)
            # Normally a cache hit, since the worker resolved before predicting.
            # A direct engineering call must still pay any newly performed work.
            add_geometry_timing(timing, resolution)
            for key, expected in (("context_resolution_file", resolution["file"]),
                                  ("context_resolution_sha256", resolution["sha256"]),
                                  ("context_home_sha256", self.home["sha256"])):
                if request.get(key) != expected:
                    raise ValueError("Requested action lost its pre-policy setup identity")
            if source_geometry(request.get("source_hint")) != source_geometry(resolution["source_hint"]):
                raise ValueError("Requested source differs from pre-policy resolver")
            if request.get("effective_source_hint") != resolution["effective_source_hint"]:
                raise ValueError("Requested effective anchor changed after prediction")
            if request.get("source_teacher_id") != resolution["source_hint"]["id"]:
                raise ValueError("Requested source teacher identity changed before prediction")
            for key, expected in (("context_layout_eligible", resolution["layout_eligible"]),
                                  ("context_anchor_admitted", resolution["anchor_admitted"]),
                                  ("context_hard_reject", resolution["hard_reject"])):
                if request.get(key) is not expected:
                    raise ValueError("Requested action changed its pre-policy setup status")
            connection = self.resolver.connect(resolution, request["values"])
            verify_connection(connection, resolution, self.home, request["values"])
            add_geometry_timing(timing, connection)
            row = dict(
                id=request["id"], bucket_xy_m=request["bucket_xy_m"], values=request["values"], phase=phase,
                native_release=False, invalid=connection.get("reason") or "requested_setup_not_admitted",
                success=False, complete_success=False, route_success=False, route_quality=0,
                single_bounce_success=False, board_bounce=False, whole_ball_entry=False,
                max_retention_s=0., closest_opening_distance_m=2., external_contact_steps=0,
                board_contact_episodes=0, invalid_physics=False,
                geometry_eligible=bool(connection["passed"]), motion_admitted=bool(connection["passed"]),
                endpoint_eligible=connection["endpoint_eligible"],
                geometry_reason=connection.get("reason"), q10=connection["q10"], q14=connection["q14"],
                connection_file=connection["file"], connection_sha256=connection["sha256"],
                context_resolution_file=resolution["file"], context_resolution_sha256=resolution["sha256"],
                context_home_sha256=self.home["sha256"], context_layout_eligible=resolution["layout_eligible"],
                context_anchor_admitted=resolution["anchor_admitted"], context_hard_reject=resolution["hard_reject"],
                source_hint=copy.deepcopy(request["source_hint"]), effective_source_hint=copy.deepcopy(request["effective_source_hint"]),
                source_teacher_id=request["source_teacher_id"],
                action_substituted=False, adapter_version=ADAPTER_VERSION,
            )
            candidates.append(row)
        for offset in range(0, len(candidates), self.batch):
            group = candidates[offset:offset+self.batch]
            active = [i for i, row in enumerate(group) if row["geometry_eligible"]]
            if active:
                self.storage_check()
                native, arrays, metadata = self._native([group[i] for i in active], timing)
                if len(native) != len(active):
                    raise RuntimeError("Native engine changed requested-action count")
                timing["gpu_s"] += float(metadata.get("native_rollout_s", metadata.get("seconds", 0.)))
                post_start = time.monotonic()
                filename = f"native-{self.packet:04d}.npz"
                self.packet += 1
                native_path = self.directory/filename
                if native_path.exists():
                    raise ValueError("Refusing to replace native evidence")
                np.savez_compressed(native_path, **arrays, record_ids=np.asarray([group[i]["id"] for i in active]))
                native_sha256 = sha(native_path)
                for column, (index, measured) in enumerate(zip(active, native)):
                    row = group[index]
                    if measured.get("native_release") is not True or measured.get("values") != row["values"] or measured.get("bucket_xy_m") != row["bucket_xy_m"]:
                        raise RuntimeError("Native observation/action association changed")
                    for key in ("q10", "q14"):
                        if not np.allclose(measured[key], row[key], atol=1e-12, rtol=0):
                            raise RuntimeError("Native joint branch differs from exact path proof")
                    row.update(copy.deepcopy(measured), native_cache_file=filename, native_cache_row=column,
                               native_cache_sha256=native_sha256)
                    # The native row's invalid value denotes numerical-domain
                    # failures; geometry admission was already proven above.
                    if row.get("invalid_physics"):
                        row.update(invalid="native_physics_outside_calibrated_domain", route_success=False,
                                   success=False, complete_success=False, route_quality=0, single_bounce_success=False)
                    else:
                        row["invalid"] = None
                    timing["native_releases"] += 1
                timing["postprocess_s"] += time.monotonic()-post_start
                timing["postprocess_s"] += float(metadata.get("score_and_oracle_s", 0.))+float(metadata.get("raw_transfer_s", 0.))
        with (self.directory/"shots.jsonl").open("a") as stream:
            for row in candidates:
                stream.write(json.dumps({k: v for k, v in row.items() if k != "trajectory_m"}, allow_nan=False)+"\n")
        timing.update(requested=len(candidates), wall_s=time.monotonic()-begin)
        return candidates, timing


class DirectPathEngine(HeadingEngine):
    """Versioned admission bridge; inherits unchanged native64 drop execution."""
    def __init__(self, ref, assets, directory, **kwargs):
        if kwargs.get("anchor_source") is not None or kwargs.get("anchor_directory") is not None:
            raise ValueError("Direct-first uses its explicit deferred context, not a supplied anchor")
        super().__init__(ref, assets, directory, **kwargs)
        old = self.adapter
        resolver = DirectPathResolver(old.resolver.base, assets)
        install_vectorized_gate(resolver.solver.gate)
        install_path_short_circuit(resolver.planner)
        self.adapter = DirectPathAdapter(ref, assets, directory,
            prior_records=old.sources, calibration=old.calibration, home=old.home,
            batch=old.batch, resolver=resolver, environment_factory=old.environment_factory)

    def prepare(self, proposals, *, one_per_setting=False):
        # The original expansion temporarily installs BoardOnlyResolver. Keep
        # its exact branch ordering/fixed-dropper rule, but use this explicit
        # admission bridge instead of silently reverting to eager references.
        original = self.adapter.resolver
        if isinstance(original, DirectPathResolver):
            return super().prepare(proposals, one_per_setting=one_per_setting)
        from .clearance_expansion_plan import BoardOnlyResolver
        if not isinstance(original, BoardOnlyResolver):
            raise ValueError("Unsupported resolver in direct-first candidate preparation")
        replacement = DirectPathResolver(original.base, self.adapter.assets)
        replacement._branches = original._branches
        self.adapter.resolver = replacement
        try:
            return super().prepare(proposals, one_per_setting=one_per_setting)
        finally:
            self.adapter.resolver = original


def hydrate_direct_resolution(engine, bucket):
    """Load the frozen deferred record; native replay may not perform planning."""
    resolver = engine.adapter.resolver
    key = context_key(resolver, bucket)
    path = contained(resolver.directory, f"direct-contexts/{key}.json")
    if not path.is_file():
        raise ValueError("Frozen deferred context is missing; refusing native replanning")
    return resolver.resolve(bucket)


def initialize_direct_first(ref, assets, directory, kwargs):
    from . import parallel_capture_grid_workers as workers
    from .sweep_worker import atomic
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.nice(10)
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("Preparation workers require an uninitialized CUDA runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    workers._engine = DirectPathEngine(ref, assets, directory, **kwargs)
    atomic(Path(directory)/"arm-check-features.json", dict(variant="direct_first", features=FEATURES))

# Explicit spelling used by the paired runner.
initialize_worker = initialize_direct_first
