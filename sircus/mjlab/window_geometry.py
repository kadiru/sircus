"""CPU-only fixed-board branches for conditional release-gap experiments.

One mathematical board branch is selected per height/tilt context before any
release outcomes exist. It is held bit-for-bit for all requests in that context.
The dropper is solved from the same immutable continuation hint and target-hashed
starts, so query order never selects the next query's seed. Strict endpoint
admission is distinct from a swept path, ball flight or physical validation.
"""

import copy
import json
from pathlib import Path

import numpy as np

from .admission import VERIFIED_PARK_Q10, board_flange_world, native_pose_error
from .corrected_geometry import REASONS, StrictEndpointGate
from .expand_geometry import outer_geometry_mask, validate_bounds
from .learning_contract import digest
from .learning_ik import TOLERANCE, shot_targets
from .stable_ik import StableShotIK

VERSION = "sircus-fixed-board-gap-window-geometry-v1"
BOUNDS = dict(
    low=[.30, -.50, .04, 8., .02],
    high=[1.04, .35, .56, 64., .65],
    version="sircus-fixed-board-gap-window-actions-v1",
)
SCOPE = "Strict CPU endpoints with a fixed board branch; no swept-path or ball-flight guarantee"


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _pool(value):
    result = {name: _array(value[name]) for name in ("q", "valid", "residual")}
    result["query_sha256"] = list(value["query_sha256"])
    return result


def _ordered(pool, hint):
    ids = np.flatnonzero(pool["valid"])
    return sorted(map(int, ids), key=lambda i: (
        not np.array_equal(pool["q"][i], hint),
        float(np.linalg.norm(pool["q"][i] - hint)),
        float(pool["residual"][i]), i,
    ))


class WindowGeometry:
    """Freeze board contexts and audit requested gap values without GPU use.

    ``anchor_hint`` needs finite ``values[5]``, ``bucket_xy_m[2]`` and both
    six-joint arrays. It supplies geometry only; source rewards/outcomes are not
    retained or inspected. ``board_checker`` is an explicit test dependency;
    production uses the strict model plus a native full-pose comparison.
    """

    def __init__(self, scene, assets, anchor_hint, bounds=None, *, ik=None,
                 gate=None, initials=None, board_checker=None, pose_checker=None):
        self.scene, self.assets = copy.deepcopy(scene), Path(assets)
        self.bounds = copy.deepcopy(BOUNDS if bounds is None else bounds)
        self.low, self.high = validate_bounds(self.bounds)
        arrays = {name: np.asarray(anchor_hint[name], float) for name in
                  ("values", "bucket_xy_m", "q10", "q14")}
        for name, shape in (("values", (5,)), ("bucket_xy_m", (2,)),
                            ("q10", (6,)), ("q14", (6,))):
            if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
                raise ValueError("Finite complete geometry anchor required: " + name)
        self.anchor = {name: value.copy() for name, value in arrays.items()}
        self.bucket_xy = arrays["bucket_xy_m"].copy()
        self.board_xy = arrays["values"][:2].copy()
        if np.linalg.norm(self.bucket_xy - self.board_xy) <= .15:
            raise ValueError("This version retains the >150mm horizontal prior")
        self.ik = ik if ik is not None else StableShotIK(self.scene, self.assets, device="cpu")
        self.gate = gate if gate is not None else StrictEndpointGate(self.scene, self.assets)
        if getattr(self.gate, "clearance_m", .005) != .005:
            raise ValueError("This geometry version requires exactly5mm strict clearance")
        self.initials = copy.deepcopy(initials) if initials is not None else json.loads((self.assets / "poses.json").read_text())
        self.park_q10 = np.asarray(VERIFIED_PARK_Q10, float).copy()
        self.board_checker = board_checker
        self.pose_checker = pose_checker
        self._contexts = {}
        self._cache = {}

    @property
    def contexts(self):
        return copy.deepcopy(list(self._contexts.values()))

    def _context_values(self, context, gap):
        return np.r_[self.board_xy, context["board_height_m"], context["tilt_deg"], gap]

    def _check_board(self, q14, board_world):
        if self.board_checker is not None:
            return self.board_checker(q14.copy(), board_world.copy())
        strict = self.gate.strict
        strict.bucket_xy = self.bucket_xy.copy()
        joints = np.r_[self.park_q10, q14]
        result = strict.check(joints)
        result = copy.deepcopy(result)
        if not result.get("passed"):
            return result
        error = native_pose_error(strict, 14, joints,
                                  board_flange_world(self.scene, board_world))
        result["board_pose_error"] = error
        if (error["position_m"] >= TOLERANCE or .25 * error["rotation_rad"] >= TOLERANCE
                or not np.isfinite([error["position_m"], error["rotation_rad"]]).all()):
            result.update(passed=False, reason="native_board_full_pose_mismatch")
        return result

    def _check_pair(self, q10, q14, values, targets):
        result = self.gate.check(q10, q14, self.bucket_xy, values, targets)
        if result["reason"]:
            return result
        if self.pose_checker is not None:
            errors = self.pose_checker(q10.copy(), q14.copy(), copy.deepcopy(targets))
        else:
            joints = np.r_[q10, q14]
            errors = {
                "10": native_pose_error(self.gate.strict, 10, joints, targets["dropper_flange_world"]),
                "14": native_pose_error(self.gate.strict, 14, joints,
                                        board_flange_world(self.scene, targets["board_world"])),
            }
        result = copy.deepcopy(result)
        result["full_pose_error"] = errors
        if any(e["position_m"] >= TOLERANCE or .25 * e["rotation_rad"] >= TOLERANCE
               or not np.isfinite([e["position_m"], e["rotation_rad"]]).all() for e in errors.values()):
            result.update(reason=5, full_pose_failure="native_full_pose_mismatch")
        return result

    def prepare_contexts(self, contexts, progress=None):
        """Add/freeze contexts without considering any requested release gaps.

        Context IDs can be reused only with the identical height/tilt. Branch
        preparation checks a common parked dropper; failure is conservatively
        unresolved for this staging choice, not proof of global impossibility.
        """
        declared = []
        for raw in contexts:
            name = raw.get("id")
            h, tilt = float(raw["board_height_m"]), float(raw["tilt_deg"])
            if not isinstance(name, str) or not name or not np.isfinite([h, tilt]).all():
                raise ValueError("Distinct IDs and finite height/tilt required")
            if name in {c["id"] for c in declared}:
                raise ValueError("Duplicate context ID")
            context = dict(id=name, board_height_m=h, tilt_deg=tilt)
            if name in self._contexts and any(self._contexts[name][k] != context[k]
                                             for k in ("board_height_m", "tilt_deg")):
                raise ValueError("A prepared context cannot change geometry")
            declared.append(context)
        pending = sorted([c for c in declared if c["id"] not in self._contexts], key=lambda c: c["id"])
        valid = []
        for context in pending:
            value = self._context_values(context, self.anchor["values"][4])
            within = bool((value[:4] >= self.low[:4]).all() and (value[:4] <= self.high[:4]).all())
            record = dict(**context, prepared=True, board_admitted=False,
                          reason="context_bounds_rejected" if not within else "board_ik_unresolved",
                          q14=None, board_branch_sha256=None, scope=SCOPE)
            self._contexts[context["id"]] = record
            if within:
                valid.append(context)
        if valid:
            values = np.asarray([self._context_values(c, self.anchor["values"][4]) for c in valid])
            xy = np.broadcast_to(self.bucket_xy, (len(valid), 2)).copy()
            targets = shot_targets(self.scene, xy, values[:, :3], values[:, 3], values[:, 4])
            exact = np.asarray([np.array_equal(value[:4], self.anchor["values"][:4]) for value in values])
            pool = _pool(self.ik.board.solve_candidates(
                targets["board_flange_local"], self.initials["board_joint_rad"],
                known=self.anchor["q14"], starts=16, iterations=180, preserve_known=exact,
            ))
            for index, context in enumerate(valid):
                record = self._contexts[context["id"]]
                current = {k: pool[k][index] for k in ("q", "valid", "residual")}
                record["board_query_sha256"] = pool["query_sha256"][index]
                record["board_world"] = targets["board_world"][index].tolist()
                record["board_branch_checks"] = []
                branches = _ordered(current, self.anchor["q14"])
                if exact[index]:
                    # A nominal control must retain its source branch. Replacing
                    # a failed control would hide an anchor compatibility fault.
                    branches = [b for b in branches if np.array_equal(current["q"][b], self.anchor["q14"])]
                    if not branches:
                        record["reason"] = "anchor_board_branch_unresolved"
                for branch in branches:
                    q14 = current["q"][branch]
                    check = self._check_board(q14, targets["board_world"][index])
                    record["board_branch_checks"].append(dict(branch=branch, **check))
                    if not check.get("passed"):
                        record["reason"] = "board_at_common_park_" + str(check.get("reason", "clearance_unresolved"))
                        continue
                    proof = dict(version=VERSION, scene_sha256=digest(self.scene),
                                 bucket_xy_m=self.bucket_xy.tolist(),
                                 board_world=record["board_world"], q14=q14.tolist(),
                                 park_q10=self.park_q10.tolist())
                    record.update(board_admitted=True, reason=None, q14=q14.tolist(),
                                  board_branch_id=branch, board_branch_sha256=digest(proof),
                                  board_geometry=check,
                                  preserved_anchor_board=bool(np.array_equal(q14, self.anchor["q14"]) and exact[index]))
                    break
                if progress:
                    progress(copy.deepcopy(record))
        return copy.deepcopy([self._contexts[c["id"]] for c in declared])

    def audit(self, context_ids, gaps, progress=None):
        """Audit exact gap requests with the context's unchanged board joints.

        No outcome-dependent fallback or candidate substitution is performed.
        Failed requests remain records with ``eligible=False``. The continuation
        hint is always the immutable anchor, not a previously processed query.
        """
        ids, gaps = list(context_ids), np.asarray(gaps, float)
        if gaps.shape != (len(ids),):
            raise ValueError("One gap per context ID required")
        if not np.isfinite(gaps).all():
            raise ValueError("Finite release gaps required")
        if any(name not in self._contexts for name in ids):
            raise ValueError("Prepare every context before auditing gaps")
        result, todo = [], []
        for name, gap in zip(ids, gaps):
            key = (name, float(gap).hex())
            if key in self._cache:
                result.append(copy.deepcopy(self._cache[key])); continue
            c = self._contexts[name]
            value = self._context_values(c, gap)
            finite = np.isfinite(value).all()
            within, reach = (False, False)
            if finite:
                a, b = outer_geometry_mask(self.scene, self.bucket_xy[None], value[None, None], bounds=self.bounds)
                within, reach = bool(a[0, 0]), bool(b[0, 0])
            reason = "action_bounds_or_invalid" if not within else "board_context_unresolved" if not c["board_admitted"] else "workspace_prior_reject" if not reach else "dropper_ik_unresolved"
            row = dict(context_id=name, eligible=False, reason=reason,
                       values=value.tolist(), bucket_xy_m=self.bucket_xy.tolist(),
                       q10=[0.] * 6, q14=copy.deepcopy(c["q14"]) if c["board_admitted"] else [0.] * 6,
                       board_branch_sha256=c["board_branch_sha256"], minimum_clearance_lower_bound_m=None,
                       board_branch_id=c.get("board_branch_id"), geometry_version=VERSION,
                       scope=SCOPE, native_release=False, path_admitted=None)
            result.append(row)
            if finite and within and reach and c["board_admitted"]:
                todo.append((len(result) - 1, key))
            else:
                self._cache[key] = copy.deepcopy(row)
        if todo:
            values = np.asarray([result[i]["values"] for i, _ in todo])
            xy = np.broadcast_to(self.bucket_xy, (len(todo), 2)).copy()
            targets = shot_targets(self.scene, xy, values[:, :3], values[:, 3], values[:, 4])
            exact = np.asarray([np.array_equal(value, self.anchor["values"]) for value in values])
            pool = _pool(self.ik.dropper.solve_candidates(
                targets["dropper_flange_world"], self.initials["dropper_joint_rad"],
                known=self.anchor["q10"], starts=16, iterations=180, preserve_known=exact,
            ))
            for index, (position, key) in enumerate(todo):
                row = result[position]; current = {k: pool[k][index] for k in ("q", "valid", "residual")}
                row["dropper_query_sha256"] = pool["query_sha256"][index]
                row["branch_checks"] = []
                branches = _ordered(current, self.anchor["q10"])
                if exact[index]:
                    branches = [b for b in branches if np.array_equal(current["q"][b], self.anchor["q10"])]
                    if not branches:
                        row["reason"] = "anchor_dropper_branch_unresolved"
                for branch in branches:
                    q10 = current["q"][branch]
                    checked = self._check_pair(q10, np.asarray(row["q14"]), values[index],
                                               {k: v[index] for k, v in targets.items()})
                    row["branch_checks"].append(dict(branch=branch, **checked))
                    if checked["reason"]:
                        row["reason"] = REASONS.get(checked["reason"], "native_gate_rejected")
                        continue
                    row.update(eligible=True, reason=None, q10=q10.tolist(), dropper_branch_id=branch,
                               minimum_clearance_lower_bound_m=checked.get("minimum_clearance_lower_bound_m"),
                               preserved_anchor_dropper=bool(exact[index] and np.array_equal(q10, self.anchor["q10"])))
                    break
                self._cache[key] = copy.deepcopy(row)
                if progress:
                    progress(copy.deepcopy(row))
        return copy.deepcopy(result)
