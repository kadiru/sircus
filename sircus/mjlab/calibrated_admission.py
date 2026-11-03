"""Geometry-only source resolution and sampled setup admission, version 2.

No physics rollout, score, training label, hardware or service is used here.
The cheap wrist sphere is diagnostic only. Exact requested parameters never get
projected to a substitute action; only IK branches and setup paths may vary.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .admission import (
    VERIFIED_PARK_Q10, REFERENCE_Q14, TransitAdmission, base14,
    board_flange_world, body_transform, native_pose_error,
)
from .corrected_geometry import StrictEndpointGate, independent_rejections
from .expand_geometry import WIDE_BOUNDS, outer_geometry_mask, validate_bounds
from .learning_contract import digest
from .learning_ik import shot_targets
from .stable_geometry import ordered_pairs
from .stable_ik import StableShotIK
from .transit_audit import audit_segment

VERSION = "sircus-calibrated-source-and-setup-admission-v2"
HOME_VERSION = "sircus-calibrated-canonical-initial-park-v2"
CLEARANCE_M = .005
SAMPLE_STEP_RAD = float(np.deg2rad(.5))
HOME_CENTERS = (
    (1.02, .15, .65), (.95, .25, .60), (1.10, .20, .65),
    (.95, -.25, .60), (1.05, .15, .75), (.85, .25, .65),
)
SCOPE = "prescribed canonical initial park to exact requested endpoint and reverse; finite sampled geometry, not actuator/grasp validation"

ANCHOR_ATTEMPT_BUDGET = 64
DERIVED_OFFSETS = (
    ("height25_tilt-minus4", (0., 0., .025, -4., 0.)),
    ("height50_tilt-minus8", (0., 0., .05, -8., 0.)),
    ("height-minus25_tilt4", (0., 0., -.025, 4., 0.)),
    ("height-minus50_tilt8", (0., 0., -.05, 8., 0.)),
    ("height25_gap-minus60", (0., 0., .025, 0., -.06)),
    ("height50_gap-minus120", (0., 0., .05, 0., -.12)),
    ("height-minus25_gap60", (0., 0., -.025, 0., .06)),
    ("height-minus50_gap120", (0., 0., -.05, 0., .12)),
    ("x40", (.04, 0., 0., 0., 0.)), ("x-minus40", (-.04, 0., 0., 0., 0.)),
    ("y40", (0., .04, 0., 0., 0.)), ("y-minus40", (0., -.04, 0., 0., 0.)),
    ("x40_height25_tilt-minus4", (.04, 0., .025, -4., 0.)),
    ("x-minus40_height25_tilt-minus4", (-.04, 0., .025, -4., 0.)),
    ("y40_height25_tilt-minus4", (0., .04, .025, -4., 0.)),
    ("y-minus40_height25_tilt-minus4", (0., -.04, .025, -4., 0.)),
)


def _plain(value):
    if isinstance(value, np.ndarray): return _plain(value.tolist())
    if isinstance(value, np.generic): return _plain(value.item())
    if isinstance(value, dict): return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def _vector(value, n, name):
    x = np.asarray(value, float)
    if x.shape != (n,) or not np.isfinite(x).all():
        raise ValueError(f"Finite {name}[{n}] required")
    return x


def _signed(value):
    out = _plain(value)
    out["sha256"] = digest(out)
    return out


def validate_home(home, scene):
    out = copy.deepcopy(home)
    fingerprint = out.pop("sha256", None)
    if (home.get("version") != HOME_VERSION or digest(out) != fingerprint
            or home.get("scene_sha256") != digest(scene)):
        raise ValueError("Canonical park identity/scene changed")
    _vector(home.get("q10"), 6, "home q10")
    _vector(home.get("q14"), 6, "home q14")
    if not home.get("passed"):
        raise ValueError("A verified canonical initial park is required")
    return copy.deepcopy(home)


def source_geometry(source):
    """Only geometric/joint provenance is eligible for source continuation."""
    result = {k: _vector(source.get(k), n, k).tolist()
              for k, n in (("values", 5), ("bucket_xy_m", 2), ("q10", 6), ("q14", 6))}
    for key in ("id", "source_id", "source_record_sha256", "source_namespace",
                "configuration_sha256", "source_study"):
        if key in source:
            if not isinstance(source[key], str) or not source[key]:
                raise ValueError("Nonempty source provenance required")
            result[key] = source[key]
    if "id" not in result: raise ValueError("Stable source id required")
    return result


def rank_sources(sources, bucket_xy):
    bucket = _vector(bucket_xy, 2, "bucket")
    return sorted((source_geometry(s) for s in sources),
                  key=lambda s: (float(np.linalg.norm(np.asarray(s["bucket_xy_m"]) - bucket)), s["id"]))



def anchor_search_spec():
    return dict(version="sircus-shared-geometry-anchor-search-v1", max_attempts=ANCHOR_ATTEMPT_BUDGET,
        source_order="nearest original source bucket XY then source ID",
        attempt_order="all nominal sources, then declared offsets per ranked source; truncate total budget",
        offsets=[dict(id=name, values=list(values)) for name, values in DERIVED_OFFSETS],
        minimum_board_bucket_xy_distance_m=.15, distance_comparison="strictly_greater",
        flight_labels_used=False, clips_or_projects_action=False)


def anchor_candidates(sources, bucket):
    ranked = rank_sources(sources, bucket)
    plan = []
    def add(source, rank, offset_id, offset):
        values = (np.asarray(source["values"], float) + np.asarray(offset)).tolist()
        plan.append(dict(source=source, derivation=dict(
            kind="nominal_source" if offset_id == "nominal" else "bounded_geometry_offset",
            source_id=source["id"], source_rank=rank,
            original_source_geometry_sha256=digest(source_geometry(source)),
            original_values=source["values"], offset_id=offset_id, offset=list(offset), values=values,
            geometry_only=True)))
    for rank,source in enumerate(ranked): add(source,rank,"nominal",[0.]*5)
    for rank,source in enumerate(ranked):
        for offset_id,offset in DERIVED_OFFSETS: add(source,rank,offset_id,offset)
    return plan[:ANCHOR_ATTEMPT_BUDGET]


def effective_source(source, bucket, derivation, resolved=None):
    result = source_geometry(source)
    result.update(values=list(derivation["values"]), original_source_id=source["id"],
        original_source_geometry_sha256=derivation["original_source_geometry_sha256"],
        derivation_id=derivation["offset_id"], derivation_offset=list(derivation["offset"]),
        geometry_bucket_xy_m=list(bucket), geometry_resolved=resolved is not None)
    if resolved is not None:
        result.update(q10=resolved["q10"],q14=resolved["q14"])
    return result


def _coverage_points(layouts=None):
    if layouts is not None:
        points = [r["bucket_xy_m"] if isinstance(r, dict) else r for r in layouts]
    else:
        # Independent geometry verification grid, not training flight outcomes.
        points = [[float(x), float(y)] for y in np.linspace(-.65, .25, 7)
                  for x in np.linspace(.10, 1.24, 9)]
    points = [_vector(p, 2, "coverage bucket").tolist() for p in points]
    result = [list(p) for p in sorted(set(tuple(p) for p in points))]
    if not result: raise ValueError("Nonempty independent home coverage required")
    return result


class StrictSolver:
    """Stable actual IK plus native and independent 5mm endpoint checks."""
    def __init__(self, scene, assets, *, ik=None, gate=None, bounds=None, max_pairs=6):
        self.scene, self.assets = copy.deepcopy(scene), Path(assets)
        self.ik = ik or StableShotIK(scene, assets, device="cpu")
        self.endpoint = gate or StrictEndpointGate(scene, assets, CLEARANCE_M)
        self.gate = self.endpoint.strict
        self.bounds = copy.deepcopy(bounds or WIDE_BOUNDS)
        self.low, self.high = validate_bounds(self.bounds)
        self.initial = json.loads((self.assets / "poses.json").read_text())
        self.max_pairs = int(max_pairs)

    def solve(self, bucket_xy, values, hint):
        bucket, values = _vector(bucket_xy, 2, "bucket"), _vector(values, 5, "values")
        info = dict(passed=False, reason=None, candidates=[], pair_attempts=[])
        within, reach = outer_geometry_mask(self.scene, bucket[None], values[None, None], bounds=self.bounds)
        info["analytic_hint"] = dict(old_within=bool(within[0, 0]), old_workspace=bool(reach[0, 0]), hard_veto=False)
        # Preserve explicit five-parameter bounds and the existing >150mm
        # board-to-bucket task constraint. Only the wrist-sphere heuristic is
        # diagnostic; it cannot suppress actual IK/collision checks.
        if (values < self.low - 1e-9).any() or (values > self.high + 1e-9).any():
            info["reason"] = "action_bounds_rejected"; return info
        if np.linalg.norm(bucket - values[:2]) <= .15:
            info["reason"] = "board_bucket_distance_constraint_rejected"; return info
        hint = source_geometry(hint)
        exact = np.array_equal(bucket, hint["bucket_xy_m"]) and np.array_equal(values, hint["values"])
        pool = self.ik.candidates(bucket[None], values[None, :3], [values[3]], [values[4]],
            initial10=self.initial["dropper_joint_rad"], initial14=self.initial["board_joint_rad"],
            known_q10=np.asarray(hint["q10"])[None], known_q14=np.asarray(hint["q14"])[None],
            starts=16, iterations=180, preserve_known=np.asarray([exact]))
        first, second = ({k: pool[role][k][0].cpu().numpy() for k in ("q", "valid", "residual")}
                         for role in ("dropper", "board"))
        targets = {k: v[0] for k, v in pool["targets"].items()}
        info["valid_ik_branches"] = [int(first["valid"].sum()), int(second["valid"].sum())]
        seen, bad10, bad14 = set(), set(), set()
        for i, j in ordered_pairs(first, second, known10=np.asarray(hint["q10"]), known14=np.asarray(hint["q14"])):
            if i in bad10 or j in bad14: continue
            q10, q14 = first["q"][i], second["q"][j]
            identity = tuple(np.round(np.r_[q10, q14], 5))
            if identity in seen: continue
            seen.add(identity)
            checked = self.endpoint.check(q10, q14, bucket, values, targets)
            attempt = dict(branch10=int(i), branch14=int(j), checked=checked)
            info["pair_attempts"].append(attempt)
            if checked["reason"]:
                for role in checked.get("independent_reject_arms", []):
                    (bad10 if role == 10 else bad14).add(i if role == 10 else j)
                continue
            q = np.r_[q10, q14]
            errors = [native_pose_error(self.gate, role, q, target) for role, target in (
                (10, targets["dropper_flange_world"]),
                (14, base14(self.scene) @ targets["board_flange_local"]))]
            attempt["native_pose_errors"] = errors
            if any(e["position_m"] >= 1e-4 or .25 * e["rotation_rad"] >= 1e-4 for e in errors):
                attempt["pose_rejected"] = True; continue
            info["candidates"].append(dict(bucket_xy_m=bucket.tolist(), values=values.tolist(),
                q10=q10.tolist(), q14=q14.tolist(), geometry_eligible=True,
                branch10=int(i), branch14=int(j), endpoint=checked))
            if len(info["candidates"]) >= self.max_pairs: break
        info["passed"] = bool(info["candidates"])
        info["reason"] = None if info["passed"] else ("finite_ik_unresolved" if not first["valid"].any() or not second["valid"].any() else "strict_endpoint_clearance_or_pose_rejected")
        return info


def fixed_base_blockers(gate, checked):
    """Separate invariant base/bucket geometry from a movable parking failure."""
    fixed = set()
    for i, name in enumerate(gate.names):
        body = int(gate.m.geom_bodyid[i]); movable = False
        while body:
            movable |= bool(gate.m.body_dofnum[body])
            body = int(gate.m.body_parentid[body])
        if not movable and name.startswith(("Arm10/", "Arm14/")): fixed.add(name)
    contacts = checked.get("margin_contacts", []) + checked.get("unresolved_clearance_pairs", [])
    return [r for r in contacts if any(n in fixed for n in r["pair"])
            and any(n.startswith("bucket/") for n in r["pair"])]


def generate_canonical_home(scene, assets, *, coverage_layouts=None, solver=None):
    """Freeze one geometry-only initial park over declared verification layouts.

    This is a new simulator initial-state contract, never a claimed route from
    the older colliding home. Invariant fixed-base/bucket blockers stay ineligible.
    """
    solver = solver or StrictSolver(scene, assets)
    ik, gate = solver.ik, solver.gate
    points = _coverage_points(coverage_layouts)
    joint_family = (
        (0., 0., 0., 0., 0., 0.), (0., -math.pi/2, 0., 0., 0., 0.),
        (0., math.pi/2, 0., 0., 0., 0.), (0., 0., -math.pi/2, 0., 0., 0.),
        (0., 0., math.pi/2, 0., 0., 0.), (0., -.5, .5, 0., 0., 0.),
        (0., .5, -.5, 0., 0., 0.),
    )
    attempts = []

    def check_candidate(q, kind, family, branch10=None, branch14=None, targets=None):
        gate.bucket_xy = np.asarray([-10., -10.])
        initial = gate.check(q)
        attempt = dict(kind=kind, family=family, branch10=branch10, branch14=branch14,
            q10=q[:6].tolist(), q14=q[6:].tolist(), initial=initial)
        attempts.append(attempt)
        if not initial["passed"]: return None
        if targets is not None:
            errors = [native_pose_error(gate, role, q, target) for role,target in targets]
            attempt["pose_errors"] = errors
            if any(e["position_m"] >= 1e-4 or .25*e["rotation_rad"] >= 1e-4 for e in errors): return None
        coverage, failed = [], False
        for point in points:
            gate.bucket_xy = np.asarray(point); result = gate.check(q)
            blockers = fixed_base_blockers(gate, result)
            coverage.append(dict(bucket_xy_m=point, passed=result["passed"],
                fixed_base_blocked=bool(blockers), blockers=blockers,
                minimum_clearance_m=result.get("minimum_clearance_m"), reason=result.get("reason"), geometry=result))
            if not result["passed"] and not blockers: failed = True
        attempt["coverage"] = coverage
        if failed: return None
        # Geometry above was rechecked in every declared layout. FK metadata is
        # descriptive; no target or flight outcome is inferred from it.
        gate.state(q)
        board_world = np.eye(4); board_world[:3,:3] = gate.d.xmat[gate.native.board].reshape(3,3)
        board_world[:3,3] = gate.d.xpos[gate.native.board]
        return _signed(dict(version=HOME_VERSION, passed=True, scene_sha256=digest(scene),
            q10=q[:6].tolist(), q14=q[6:].tolist(), selected_family_kind=kind,
            selected_family=family, generation_joint_family=[list(v) for v in joint_family],
            generation_cartesian_family=[list(v) for v in HOME_CENTERS], coverage=coverage,
            coverage_sha256=digest(points), attempts=attempts,
            minimum_clearance_m=min(x["minimum_clearance_m"] for x in coverage if x["passed"]),
            board_world=board_world.tolist(), dropper_flange_world=body_transform(gate,10,q).tolist(),
            board_flange_world=body_transform(gate,14,q).tolist(),
            initial_state_assumption="New canonical simulator initial park; old colliding park connectivity is not claimed",
            validation_model_sha256=gate.validation_model_sha256))

    for index,q14 in enumerate(joint_family):
        result = check_candidate(np.r_[VERIFIED_PARK_Q10,q14],"declared_joint_park",index)
        if result is not None: return result

    target10 = np.eye(4); target10[:3,:3] = np.diag([1.,-1.,-1.]); target10[:3,3] = [.25,-.10,.95]
    first = ik.dropper.solve_candidates(target10[None],known=VERIFIED_PARK_Q10,
        preserve_known=np.asarray([True]),starts=16,iterations=180)
    first = {k:first[k][0].cpu().numpy() for k in ("q","valid","residual")}
    targets14 = []
    for xyz in HOME_CENTERS:
        board = np.eye(4); board[:3,:3] = Rotation.from_euler("z",scene["board"]["mount_yaw_rad"]).as_matrix()
        board[:3,3] = xyz; targets14.append(board_flange_world(scene,board))
    second = ik.board.solve_candidates(np.asarray([np.linalg.inv(base14(scene))@t for t in targets14]),
        known=REFERENCE_Q14,starts=16,iterations=180)
    second = {k:second[k].cpu().numpy() for k in ("q","valid","residual")}
    for family,target14 in enumerate(targets14):
        local = {k:v[family] for k,v in second.items()}; checked_pairs = 0
        for i,j in ordered_pairs(first,local):
            result = check_candidate(np.r_[first["q"][i],local["q"][j]],"cartesian_park",family,int(i),int(j),[(10,target10),(14,target14)])
            if result is not None: return result
            checked_pairs += 1
            if checked_pairs >= 4: break
    return _signed(dict(version=HOME_VERSION,passed=False,scene_sha256=digest(scene),
        reason="finite_global_canonical_park_unresolved",coverage_sha256=digest(points),attempts=attempts))


class SerializedPlanner:
    def __init__(self, scene, solver, home, *, rrt_nodes=192):
        self.scene, self.solver = scene, solver
        self.gate, self.ik = solver.gate, solver.ik
        self.home = validate_home(home, scene)
        self.rrt_nodes = int(rrt_nodes)
        if not 0 <= self.rrt_nodes <= 4096: raise ValueError("Finite RRT node budget required")

    def check_home(self, bucket):
        self.gate.bucket_xy = _vector(bucket, 2, "bucket")
        result = self.gate.check(np.r_[self.home["q10"], self.home["q14"]])
        blockers = fixed_base_blockers(self.gate, result)
        return dict(**result, fixed_base_blockers=blockers, hard_reject=bool(blockers))

    def _audit_path(self, points):
        forward = [audit_segment(self.gate, a, b, max_step_rad=SAMPLE_STEP_RAD) for a, b in zip(points, points[1:])]
        reverse_points = points[::-1]
        reverse = [audit_segment(self.gate, a, b, max_step_rad=SAMPLE_STEP_RAD) for a, b in zip(reverse_points, reverse_points[1:])]
        passed = all(e["passed"] for e in forward + reverse)
        return dict(passed=passed, waypoints=[np.asarray(q).tolist() for q in points],
            edges=forward, reverse_edges=reverse, scope=SCOPE,
            minimum_clearance_m=min((e.get("minimum_clearance_m", 0.) for e in forward + reverse), default=0.),
            max_joint_sample_step_rad=SAMPLE_STEP_RAD)

    @staticmethod
    def _serial_points(start, target, order):
        points = [start.copy()]; q = start.copy()
        for role in order:
            offset = 0 if role == 10 else 6
            q = q.copy(); q[offset:offset+6] = target[offset:offset+6]; points.append(q)
        return points

    def _cartesian(self, start, target, order, height):
        q, points, attempts = start.copy(), [start.copy()], []
        for role in order:
            begin = body_transform(self.gate, role, q)
            end = body_transform(self.gate, role, target)
            middle = begin.copy(); middle[2, 3] = height if role == 14 else max(float(end[2, 3]), min(float(begin[2, 3]), .82))
            translate = middle.copy(); translate[:2, 3] = end[:2, 3]
            orient = translate.copy(); orient[:3, :3] = end[:3, :3]
            for label, pose in (("vertical_clearance", middle), ("translate_xy", translate), ("orient_at_clearance", orient), ("vertical_to_target", end)):
                q, result, frames = TransitAdmission._move(self, q, role, pose,
                    f"arm{role}_{label}", {}, initial_target=target[:6] if role == 10 else target[6:])
                attempts.append(result)
                points.extend(np.r_[f["q10"], f["q14"]] for f in frames)
                if not result["passed"]: return None, attempts
            exact = q.copy(); offset = 0 if role == 10 else 6; exact[offset:offset+6] = target[offset:offset+6]
            checked = audit_segment(self.gate, q, exact)
            attempts.append(dict(name=f"arm{role}_exact_branch", **checked))
            if not checked["passed"]: return None, attempts
            q = exact; points.append(q.copy())
        return points, attempts

    def _rrt_arm(self, start, target, role, seed):
        offset = 0 if role == 10 else 6
        lo = (self.gate.native.limits10 if role == 10 else self.gate.native.limits14)[:, 0]
        hi = (self.gate.native.limits10 if role == 10 else self.gate.native.limits14)[:, 1]
        rng = np.random.default_rng(seed)
        trees = [[dict(q=start[offset:offset+6].copy(), parent=-1)], [dict(q=target[offset:offset+6].copy(), parent=-1)]]
        attempts = []
        def full(q):
            out = start.copy(); out[offset:offset+6] = q; return out
        def extend(tree, goal, tree_id):
            idx = min(range(len(tree)), key=lambda i: float(np.linalg.norm(tree[i]["q"] - goal)))
            current = tree[idx]["q"]; delta = goal - current
            proposed = current + delta * min(1., .22 / max(float(np.abs(delta).max()), 1e-12))
            checked = audit_segment(self.gate, full(current), full(proposed))
            attempts.append(dict(tree=tree_id, parent=idx, from_q=current.tolist(), to_q=proposed.tolist(), checked=checked))
            if not checked["passed"]: return "trapped", idx
            tree.append(dict(q=proposed, parent=idx))
            return ("reached" if np.max(np.abs(proposed-goal)) < 1e-10 else "advanced"), len(tree)-1
        def path(tree, index):
            nodes = []
            while index >= 0:
                nodes.append(tree[index]["q"]); index = tree[index]["parent"]
            return nodes[::-1]
        for step in range(self.rrt_nodes):
            if sum(len(t) for t in trees) >= self.rrt_nodes + 2 or len(attempts) >= max(1, self.rrt_nodes * 4): break
            side = step % 2; other = 1-side
            sample = trees[other][0]["q"] if step % 5 == 0 else rng.uniform(lo, hi)
            status, index = extend(trees[side], sample, side)
            if status == "trapped": continue
            goal = trees[side][index]["q"]
            for _ in range(64):
                if sum(len(t) for t in trees) >= self.rrt_nodes + 2 or len(attempts) >= max(1, self.rrt_nodes * 4):
                    return None, dict(passed=False, reason="finite_rrt_budget_exhausted", seed=int(seed), attempts=attempts, nodes=[len(t) for t in trees])
                status2, index2 = extend(trees[other], goal, other)
                if status2 != "advanced": break
            if status2 == "reached":
                indices = [None, None]; indices[side], indices[other] = index, index2
                vectors = path(trees[0], indices[0]) + path(trees[1], indices[1])[-2::-1]
                return [full(q) for q in vectors], dict(passed=True, seed=int(seed), attempts=attempts, nodes=[len(t) for t in trees])
        return None, dict(passed=False, reason="finite_rrt_unresolved", seed=int(seed), attempts=attempts, nodes=[len(t) for t in trees])

    def plan(self, target, bucket, *, start=None):
        target = _vector(target, 12, "target joints")
        start = np.r_[self.home["q10"], self.home["q14"]] if start is None else _vector(start, 12, "start joints")
        self.gate.bucket_xy = _vector(bucket, 2, "bucket")
        attempts = []
        for name, q in (("start", start), ("target", target)):
            check = self.gate.check(q)
            if not check["passed"]:
                return dict(passed=False, reason=f"{name}_strict_clearance_rejected", checked=check, attempts=attempts)
        for order in ((14, 10), (10, 14)):
            proof = self._audit_path(self._serial_points(start, target, order))
            attempts.append(dict(kind="serialized_joint", order=list(order), proof=proof))
            if proof["passed"]: return dict(**proof, attempts=attempts, reason=None)
        for height in (.55, .65, .75):
            for order in ((14, 10), (10, 14)):
                points, phases = self._cartesian(start, target, order, height)
                attempt = dict(kind="separated_vertical_translate_orient", order=list(order), clearance_height_m=height, phases=phases)
                attempts.append(attempt)
                if points is not None:
                    proof = self._audit_path(points); attempt["proof"] = proof
                    if proof["passed"]: return dict(**proof, attempts=attempts, reason=None)
        for order in ((14, 10), (10, 14)):
            q, points, arms = start.copy(), [start.copy()], []
            for role in order:
                endpoint = q.copy(); off = 0 if role == 10 else 6; endpoint[off:off+6] = target[off:off+6]
                seed = int(digest(dict(start=q.tolist(), target=endpoint.tolist(), bucket=np.asarray(bucket).tolist(), role=role))[:16], 16)
                route, result = self._rrt_arm(q, endpoint, role, seed); arms.append(result)
                if route is None: break
                points.extend(route[1:]); q = endpoint
            attempt = dict(kind="bounded_serialized_rrt", order=list(order), arms=arms)
            attempts.append(attempt)
            if np.array_equal(q, target):
                proof = self._audit_path(points); attempt["proof"] = proof
                if proof["passed"]: return dict(**proof, attempts=attempts, reason=None)
        return dict(passed=False, reason="finite_setup_connection_unresolved", attempts=attempts, scope=SCOPE)


class GeometryResolver:
    """Shared pre-action source choice; every requested action is checked afresh."""
    def __init__(self, ref, assets, directory, sources, *, home=None, coverage_layouts=None,
                 bounds=None, rrt_nodes=192, solver=None, planner=None):
        self.ref, self.scene = ref, copy.deepcopy(ref["scene"])
        self.directory = Path(directory); self.directory.mkdir(parents=True, exist_ok=True)
        self.sources = [source_geometry(s) for s in sources]
        if not self.sources or len({s["id"] for s in self.sources}) != len(self.sources):
            raise ValueError("Unique nonempty shared geometric sources required")
        self.bounds = copy.deepcopy(bounds or WIDE_BOUNDS)
        self.solver = solver or StrictSolver(self.scene, assets, bounds=self.bounds)
        self.home = validate_home(home, self.scene) if home is not None else generate_canonical_home(
            self.scene, assets, coverage_layouts=coverage_layouts, solver=self.solver)
        if not self.home.get("passed"):
            raise ValueError("Finite canonical home generation failed; preserve its control evidence")
        self.planner = planner or SerializedPlanner(self.scene, self.solver, self.home, rrt_nodes=rrt_nodes)
        self._resolved, self._connected = {}, {}

    def _save(self, folder, key, record):
        path = self.directory / folder / (key + ".json")
        if path.exists(): raise ValueError("Refusing to overwrite admission evidence")
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(_plain(record), indent=2, allow_nan=False) + "\n"
        path.write_text(raw)
        import hashlib
        return dict(file=str(path.relative_to(self.directory)), sha256=hashlib.sha256(raw.encode()).hexdigest())

    def resolve(self, bucket_xy):
        bucket = _vector(bucket_xy, 2, "bucket").tolist()
        search = anchor_search_spec()
        key = digest(dict(version=VERSION, bucket_xy_m=bucket, home_sha256=self.home["sha256"],
            sources=self.sources, bounds=self.bounds, anchor_search=search))
        if key in self._resolved:
            return dict(copy.deepcopy(self._resolved[key]), cache_hit=True)
        begin = time.monotonic(); timing = dict(home_s=0., endpoint_s=0., path_s=0., wall_s=0.)
        start = time.monotonic(); checked_home = self.planner.check_home(bucket); timing["home_s"] += time.monotonic()-start
        plan = anchor_candidates(self.sources,bucket); first = plan[0]
        record = dict(version=VERSION, key=key, bucket_xy_m=bucket, home_sha256=self.home["sha256"],
            home_passed=bool(checked_home["passed"]), layout_eligible=bool(checked_home["passed"]),
            hard_reject=bool(checked_home.get("hard_reject")), home_geometry=checked_home,
            passed=False, anchor_admitted=False, source_hint=first["source"], resolved=None, proof=None,
            anchor_values=first["derivation"]["values"], anchor_derivation=first["derivation"],
            effective_source_hint=effective_source(first["source"],bucket,first["derivation"]),
            anchor_search=search, attempts=[], reason=checked_home.get("reason") if not checked_home["passed"] else None,
            timing=timing, selection_uses_flight_labels=False)
        if checked_home["passed"]:
            for candidate in plan:
                hint, derivation = candidate["source"],candidate["derivation"]
                start = time.monotonic(); endpoint = self.solver.solve(bucket,derivation["values"],hint); timing["endpoint_s"] += time.monotonic()-start
                attempt = dict(source_id=hint["id"],derivation=derivation,endpoint=endpoint,paths=[])
                record["attempts"].append(attempt)
                for row in endpoint["candidates"]:
                    start = time.monotonic(); proof = self.planner.plan(np.r_[row["q10"],row["q14"]],bucket); timing["path_s"] += time.monotonic()-start
                    attempt["paths"].append(proof)
                    if proof["passed"]:
                        record.update(passed=True,anchor_admitted=True,source_hint=hint,resolved=row,proof=proof,reason=None,
                            anchor_values=derivation["values"],anchor_derivation=derivation,
                            effective_source_hint=effective_source(hint,bucket,derivation,row))
                        break
                if record["passed"]: break
            if not record["passed"]: record["reason"] = "finite_geometry_anchor_budget_unresolved"
        timing["wall_s"] = time.monotonic()-begin
        record.update(self._save("resolutions",key,record))
        self._resolved[key] = copy.deepcopy(record)
        return dict(record,cache_hit=False)

    def connect(self, anchor, requested_values):
        verify_resolution(anchor,self.sources,self.home)
        values = _vector(requested_values, 5, "requested values").tolist()
        bucket = _vector(anchor["bucket_xy_m"], 2, "bucket").tolist()
        key = digest(dict(version=VERSION, resolution_sha256=anchor["sha256"], values=values, bucket_xy_m=bucket))
        if key in self._connected: return dict(copy.deepcopy(self._connected[key]), cache_hit=True)
        begin = time.monotonic(); timing = dict(home_s=0., endpoint_s=0., path_s=0., wall_s=0.)
        record = dict(version=VERSION, bucket_xy_m=bucket, values=values, resolution_file=anchor["file"],
            resolution_sha256=anchor["sha256"], home_sha256=self.home["sha256"],
            passed=False, geometry_eligible=False, endpoint_eligible=None, reason=None,
            q10=[0.]*6, q14=[0.]*6, proof=None, endpoint_attempts=None, path_attempts=[], timing=timing,
            action_substituted=False, anchor_values=anchor["anchor_values"],
            anchor_derivation_sha256=digest(anchor["anchor_derivation"]))
        if not anchor["home_passed"]:
            record["reason"] = "fixed_base_clearance_rejected" if anchor.get("hard_reject") else "canonical_home_unresolved_in_layout"
        else:
            hint = source_geometry(anchor["effective_source_hint"])
            if anchor.get("resolved"):
                hint.update({k: anchor["resolved"][k] for k in ("q10", "q14", "values", "bucket_xy_m")})
            start = time.monotonic(); endpoint = self.solver.solve(bucket, values, hint); timing["endpoint_s"] += time.monotonic()-start
            record["endpoint_attempts"] = endpoint; record["endpoint_eligible"] = endpoint["passed"]
            record["reason"] = endpoint["reason"]
            for row in endpoint["candidates"]:
                start = time.monotonic(); proof = self.planner.plan(np.r_[row["q10"], row["q14"]], bucket); timing["path_s"] += time.monotonic()-start
                record["path_attempts"].append(proof)
                if proof["passed"]:
                    record.update(passed=True, geometry_eligible=True, q10=row["q10"], q14=row["q14"],
                        proof=proof, endpoint=row["endpoint"], reason=None); break
            if endpoint["passed"] and not record["passed"]: record["reason"] = "finite_requested_setup_path_unresolved"
        timing["wall_s"] = time.monotonic()-begin
        record.update(self._save("connections", key, record))
        self._connected[key] = copy.deepcopy(record)
        return dict(record, cache_hit=False)


def verify_path(proof, start, target):
    """Check recorded sampled-path structure; does not rerun collision geometry."""
    if not proof.get("passed") or proof.get("scope") != SCOPE:
        raise ValueError("A passed versioned sampled setup proof is required")
    points = np.asarray(proof.get("waypoints"), float)
    if (points.ndim != 2 or points.shape[1] != 12 or len(points) < 2
            or not np.isfinite(points).all()
            or not np.array_equal(points[0], _vector(start, 12, "path start"))
            or not np.array_equal(points[-1], _vector(target, 12, "path target"))):
        raise ValueError("Setup proof exact endpoints changed")
    edges, reverse = proof.get("edges", []), proof.get("reverse_edges", [])
    if len(edges) != len(points)-1 or len(reverse) != len(edges):
        raise ValueError("Forward/reverse proof edges missing")
    pairs = list(zip(points, points[1:]))
    for recorded, (a, b) in zip(edges + reverse, pairs + [(b, a) for a, b in pairs[::-1]]):
        changed = np.abs(b-a) > 1e-12
        if changed[:6].any() and changed[6:].any():
            raise ValueError("Path edge moved both arms simultaneously")
        needed = max(1, int(np.ceil(np.max(np.abs(b-a)) / SAMPLE_STEP_RAD))) + 1
        if (recorded.get("passed") is not True
                or recorded.get("minimum_clearance_m", -1) < CLEARANCE_M
                or recorded.get("total_samples", 0) != needed
                or not 0 <= recorded.get("maximum_actual_joint_sample_step_rad", -1) <= SAMPLE_STEP_RAD+1e-12):
            raise ValueError("Sampled setup edge lacks strict clearance/resolution")
    if proof.get("minimum_clearance_m", -1) < CLEARANCE_M:
        raise ValueError("Setup proof clearance changed")
    return True


def verify_resolution(record, sources, home):
    """Validate deterministic geometry-only anchor derivation, never new physics."""
    if record.get("version") != VERSION or record.get("home_sha256") != home["sha256"]:
        raise ValueError("Resolution version/home changed")
    if record.get("anchor_search") != anchor_search_spec():
        raise ValueError("Geometry anchor search specification changed")
    plan = anchor_candidates(sources,record["bucket_xy_m"])
    attempted = record.get("attempts",[])
    if len(attempted)>len(plan): raise ValueError("Geometry anchor attempt budget exceeded")
    for actual,expected in zip(attempted,plan):
        if actual["source_id"] != expected["source"]["id"] or actual.get("derivation") != expected["derivation"]:
            raise ValueError("Source/offset attempts changed deterministic order")
    if not record.get("home_passed") and (attempted or record.get("passed") or record.get("anchor_admitted")):
        raise ValueError("Blocked home cannot have admitted source attempts")
    selected = plan[len(attempted)-1] if record.get("passed") and attempted else plan[0]
    if source_geometry(record["source_hint"]) != selected["source"]:
        raise ValueError("Original selected source geometry changed")
    if record.get("anchor_derivation") != selected["derivation"] or record.get("anchor_values") != selected["derivation"]["values"]:
        raise ValueError("Pre-policy anchor derivation/action changed")
    row = record["resolved"] if record.get("passed") else None
    if record.get("effective_source_hint") != effective_source(selected["source"],record["bucket_xy_m"],selected["derivation"],row):
        raise ValueError("Effective geometric source chain changed")
    if record.get("passed"):
        if not attempted or not record.get("anchor_admitted"):
            raise ValueError("Admitted anchor requires attempted source proof")
        if row["values"] != record["anchor_values"] or row["bucket_xy_m"] != record["bucket_xy_m"]:
            raise ValueError("Nominal/effective anchor action was replaced")
        paths = attempted[-1]["paths"]
        if not paths or row != attempted[-1]["endpoint"]["candidates"][len(paths)-1] or record["proof"] != paths[-1]:
            raise ValueError("Selected source endpoint/path association changed")
        if any(p.get("passed") for p in paths[:-1]): raise ValueError("Skipped earlier admitted branch")
        verify_path(record["proof"],np.r_[home["q10"],home["q14"]],np.r_[row["q10"],row["q14"]])
        for a in attempted[:-1]:
            if any(p.get("passed") for p in a.get("paths",[])): raise ValueError("Skipped earlier admitted source")
    else:
        if record.get("home_passed") and len(attempted)!=len(plan):
            raise ValueError("Unresolved selection did not exhaust its declared budget")
        if any(p.get("passed") for a in attempted for p in a.get("paths",[])):
            raise ValueError("Unresolved selection contains an admitted source")
    if record.get("selection_uses_flight_labels") is not False:
        raise ValueError("Geometry source selection must exclude flight labels")
    return True


def verify_connection(record, anchor, home, requested_values):
    if (record.get("version") != VERSION
            or record.get("home_sha256") != home["sha256"]
            or record.get("resolution_sha256") != anchor["sha256"]
            or record.get("resolution_file") != anchor["file"]
            or record.get("values") != _vector(requested_values, 5, "requested values").tolist()
            or record.get("bucket_xy_m") != anchor["bucket_xy_m"]
            or record.get("action_substituted") is not False
            or record.get("anchor_values") != anchor["anchor_values"]
            or record.get("anchor_derivation_sha256") != digest(anchor["anchor_derivation"])):
        raise ValueError("Exact requested action/resolution binding changed")
    if record.get("passed"):
        if not record.get("endpoint_eligible") or not record.get("geometry_eligible") or not anchor["home_passed"]:
            raise ValueError("Admitted request lacks actual endpoint/home checks")
        candidates = record["endpoint_attempts"]["candidates"]
        if not any(c["values"] == record["values"] and c["q10"] == record["q10"] and c["q14"] == record["q14"] for c in candidates):
            raise ValueError("Requested endpoint branch association changed")
        if record["proof"] not in record["path_attempts"]:
            raise ValueError("Requested path association changed")
        verify_path(record["proof"], np.r_[home["q10"], home["q14"]], np.r_[record["q10"], record["q14"]])
    return True
