"""Outcome-independent strict endpoints for the finite clean-650 search.

Only mathematical IK branch pools are cached. Every requested joint pair still
receives the native FK/collider check and independent 5 mm clearance query.
This is held-endpoint admission, not grasp, swept-path or actuator validation.
"""

from __future__ import annotations

import copy

import numpy as np

from .corrected_geometry import StrictEndpointGate
from .expanded_actions import WIDE_BOUNDS
from .learning_contract import digest
from .learning_ik import shot_targets
from .replay650_geometry import footprint
from .stable_geometry import ordered_pairs
from .stable_ik import StableShotIK, target_digest


VERSION = "sircus-clean650-strict-endpoints-v1"
STARTS = 16
ITERATIONS = 180
MAX_BRANCHES = STARTS + 1
MAX_PAIRS = MAX_BRANCHES ** 2
ROLE_TARGETS = {"dropper": "dropper_flange_world", "board": "board_flange_local"}


def finite_vector(value, size, label):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError("Expected finite " + label)
    return result


class CleanGeometry:
    """CPU branch resolution with exact frozen controls and fresh pair checks."""

    def __init__(self, scene, assets, seed, *, ik=None, gate=None):
        self.scene, self.assets, self.seed = copy.deepcopy(scene), assets, copy.deepcopy(seed)
        finite_vector(self.seed["values"], 5, "seed action")
        finite_vector(self.seed["bucket_xy_m"], 2, "seed bucket XY")
        self.initial = {
            "dropper": finite_vector(self.seed["q10"], 6, "seed dropper joints"),
            "board": finite_vector(self.seed["q14"], 6, "seed board joints"),
        }
        self.ik = ik
        self.gate = gate if gate is not None else StrictEndpointGate(self.scene, assets)
        self.proofs = []
        self.pools = {role: {} for role in ROLE_TARGETS}

    def _prepare(self, request):
        group = copy.deepcopy(request)
        action = finite_vector(group["values"], 5, "requested action")
        bucket = finite_vector(group["bucket_xy_m"], 2, "requested bucket XY")
        group["values"], group["bucket_xy_m"] = action.tolist(), bucket.tolist()
        within = bool((action >= WIDE_BOUNDS["low"]).all() and (action <= WIDE_BOUNDS["high"]).all())
        distance = float(np.linalg.norm(bucket-action[:2]))
        table = footprint(self.scene, bucket)
        exact_seed = np.array_equal(action, self.seed["values"]) and np.array_equal(bucket, self.seed["bucket_xy_m"])
        control = group.get("role") in ("positive_control", "negative_control")
        frozen = control or group.get("freeze_joints") is True
        if control:
            mode = "exact_control"
        elif frozen:
            mode = "frozen_verification"
        elif exact_seed:
            mode = "exact_seed"
        else:
            mode = "solved_cpu"
        proposed = None
        if frozen:
            try:
                proposed = (finite_vector(group.get("q10"), 6, "frozen dropper joints"),
                            finite_vector(group.get("q14"), 6, "frozen board joints"))
            except ValueError:
                pass
        elif exact_seed:
            proposed = (self.initial["dropper"].copy(), self.initial["board"].copy())
        proof = dict(version=VERSION, group_id=group["id"], values=action.tolist(), bucket_xy_m=bucket.tolist(),
                     bounds_passed=within, bucket_footprint=table, distance_m=distance,
                     separated_passed=distance > .15, source_mode=mode,
                     preserved_source=mode != "solved_cpu", held_endpoints_only=True,
                     ik_starts=STARTS, ik_iterations=ITERATIONS, pair_budget=MAX_PAIRS,
                     attempts=[], independent_pairs_pruned=0)
        reason = ("action_bounds_or_invalid" if not within else "bucket_table_footprint" if not table["passed"]
                  else "board_bucket_distance" if distance <= .15 else
                  "invalid_frozen_joints" if frozen and proposed is None else None)
        targets = None
        if reason is None:
            targets = {k: v[0] for k, v in shot_targets(self.scene, bucket[None], action[None, :3],
                                                       [action[3]], [action[4]]).items()}
        return dict(group=group, action=action, bucket=bucket, targets=targets, proof=proof,
                    mode=mode, proposed=proposed, rejection=reason)

    def _fill_pools(self, prepared):
        """Solve each exact role/target once; target order never changes its starts."""
        for role, name in ROLE_TARGETS.items():
            missing = {}
            for item in prepared:
                if item["rejection"] is not None or item["mode"] != "solved_cpu":
                    continue
                target = item["targets"][name]
                key = target_digest(target, role=role)
                item.setdefault("pool_keys", {})[role] = key
                if key not in self.pools[role]:
                    missing.setdefault(key, target)
            if not missing:
                continue
            if self.ik is None:
                self.ik = StableShotIK(self.scene, self.assets, device="cpu")
            keys = sorted(missing)
            for offset in range(0, len(keys), 128):
                selected = keys[offset:offset+128]
                matrices = np.stack([missing[key] for key in selected])
                known = np.repeat(self.initial[role][None], len(selected), axis=0)
                pool = getattr(self.ik, role).solve_candidates(
                    matrices, initial=self.initial[role], known=known, starts=STARTS,
                    iterations=ITERATIONS, preserve_known=np.zeros(len(selected), dtype=bool))
                q = pool["q"].cpu().numpy()
                valid = pool["valid"].cpu().numpy()
                if q.ndim != 3 or q.shape[0] != len(selected) or q.shape[2] != 6 or not 1 <= q.shape[1] <= MAX_BRANCHES or valid.shape != q.shape[:2]:
                    raise ValueError("Unexpected bounded IK branch-pool shape")
                for index, key in enumerate(selected):
                    joints = np.asarray(q[index], dtype=float).copy()
                    accepted = np.asarray(valid[index], dtype=bool) & np.isfinite(joints).all(-1)
                    joints.setflags(write=False)
                    accepted.setflags(write=False)
                    self.pools[role][key] = dict(q=joints, valid=accepted)

    def _check(self, item):
        group, proof, reason = item["group"], item["proof"], item["rejection"]
        chosen = None
        if reason is None and item["mode"] != "solved_cpu":
            pairs = [(None, None, *item["proposed"])]
        elif reason is None:
            first, second = (self.pools[role][item["pool_keys"][role]] for role in ("dropper", "board"))
            proof["ik_query_sha256"] = dict(item["pool_keys"])
            proof["valid_ik_branches"] = dict(dropper=int(first["valid"].sum()), board=int(second["valid"].sum()))
            branch_pairs = ordered_pairs(first, second, known10=self.initial["dropper"], known14=self.initial["board"])
            pairs = [(a, b, first["q"][a], second["q"][b]) for a, b in branch_pairs]
            if not pairs:
                reason = "ik_unresolved"
        else:
            pairs = []
        rejected10, rejected14 = set(), set()
        for a, b, first, second in pairs:
            if a in rejected10 or b in rejected14:
                proof["independent_pairs_pruned"] += 1
                continue
            q10, q14 = first.copy(), second.copy()
            checked = self.gate.check(q10, q14, item["bucket"], item["action"], item["targets"])
            proof["attempts"].append(dict(branch10=a, branch14=b, q10=q10.tolist(), q14=q14.tolist(), checked=checked))
            if len(proof["attempts"]) > MAX_PAIRS:
                raise RuntimeError("Finite joint-pair budget exceeded")
            if checked["reason"] == 0:
                clearance = float(checked.get("minimum_clearance_lower_bound_m", -1.))
                if not np.isfinite(clearance) or clearance < .005-1e-12:
                    raise RuntimeError("Successful strict endpoint lacks 5 mm clearance proof")
                chosen = (q10.tolist(), q14.tolist())
                proof.update(minimum_clearance_lower_bound_m=clearance,
                             selected_branches=dict(dropper=a, board=b))
                break
            for role in checked.get("independent_reject_arms", []):
                (rejected10 if role == 10 else rejected14).add(a if role == 10 else b)
        if chosen is None and reason is None:
            reason = "strict_endpoint_rejected"
        proof.update(eligible=chosen is not None, reason=None if chosen else reason)
        group.update(eligible=proof["eligible"], geometry_row=len(self.proofs), geometry_sha256=digest(proof))
        # A rejected request remains at its original ID with a data-only proposed
        # branch; it can never be executed merely because joints are present.
        if chosen is None:
            chosen = item["proposed"] or (self.initial["dropper"], self.initial["board"])
            chosen = tuple(np.asarray(row).tolist() for row in chosen)
        group.update(q10=chosen[0], q14=chosen[1])
        self.proofs.append(proof)
        return group

    def check_many(self, requests):
        prepared = [self._prepare(request) for request in requests]
        self._fill_pools(prepared)
        return [self._check(item) for item in prepared]
