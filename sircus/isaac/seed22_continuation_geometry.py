"""Per-environment continuous geometry hook; original native guards still decide release.

Installation starts no simulator and performs no inference, geometry solve or
movement. A later explicit ``env.prepare(actions)`` filters each paid proposal
from the actual joint readback, then delegates to the original bound method.
The proposal's policy latent/log-probability never enters or changes here.
"""

import copy
import math
import time
from types import MethodType

import numpy as np

import continuous_feasibility as geometry
import continuous_geometry_layer as layer

ARMS = ("Arm10", "Arm14")
SLOTS = 35
_MARKER = "_seed22_geometry_installation"


def require(value, message):
    if not value:
        raise ValueError(message)


def _mask(value, name):
    values = np.asarray(value)
    require(values.shape == (SLOTS,) and values.dtype == np.dtype(bool),
            name + " must contain exactly 35 Boolean flags")
    return values.copy()


def _joint_state(env):
    require(set(env.arms) == set(env.continuous_links) == set(ARMS),
            "Geometry hook requires the two original arm associations")
    result = {}
    for name in ARMS:
        values = env.arms[name].data.joint_pos
        if hasattr(values, "detach"):
            values = values.detach()
        if hasattr(values, "cpu"):
            values = values.cpu()
        if hasattr(values, "numpy"):
            values = values.numpy()
        values = np.asarray(values, dtype=float)
        require(values.ndim == 2 and values.shape[0] == SLOTS and values.shape[1] >= 6
                and np.isfinite(values[:, :6]).all(), "Invalid actual six-joint readback")
        result[name] = values[:, :6].copy()
    return result


def _action(cell, action):
    require(isinstance(action, dict) and isinstance(action.get("id"), str)
            and action["id"], "Expected a named continuous shot action")
    expected = geometry.decode_action(cell, [action[k] for k in geometry.ACTION_NAMES], action["id"])
    require(all(k in action and type(action[k]) in (int, float)
                and math.isfinite(action[k]) and abs(action[k] - v) <= 1e-8
                for k, v in expected.items() if k != "id"),
            "Continuous shot fields disagree with the original action decode")
    return expected


def _record(record, scene, cell, original, links, starts, *, original_filter):
    """Fail closed before mutation; the native controller independently rechecks fits."""
    require(isinstance(record, dict), "Geometry filter returned no record")
    body = copy.deepcopy(record)
    signature = body.pop("fingerprint", None)
    require(signature == geometry.canonical_hash(body), "Geometry record fingerprint mismatch")
    require(record.get("schema") == layer.VERSION
            and record.get("release_authorized") is False
            and record.get("bucket_impossibility_proven") is False
            and type(record.get("native_shots")) is int
            and type(record.get("training_updates")) is int
            and record["native_shots"] == record["training_updates"] == 0,
            "Geometry record cannot authorize motion, release or learning")
    require(record.get("original_action") == _action(cell, original),
            "Geometry record belongs to a different original proposal")
    require(record.get("binding", {}).get("protocol") == layer._protocol(True, 24, 512, .10),
            "Geometry hook may not change the frozen repair limits")
    classification = record.get("classification")
    require(classification in ("unchanged", "adjusted", "rejected")
            and record.get("accepted") is (classification != "rejected"),
            "Geometry classification and admission disagree")
    if original_filter:
        # This performs no new solve. It binds the exact original layer/source,
        # actual starts, final action, FK, path and independently verified fits.
        layer.validate_result(record, scene, cell, original, links, starts)
    if classification == "rejected":
        require(record.get("action") is None and record.get("final_preflight") is None,
                "Rejected geometry record contains an executable action")
        return None
    chosen = record.get("action")
    require(_action(cell, chosen) == chosen and chosen["id"] == original["id"],
            "Admitted geometry action is inconsistent or has a changed identity")
    require((chosen == record["original_action"]) is (classification == "unchanged"),
            "Unchanged/adjusted classification contradicts its action")
    delta = (np.asarray([chosen[k] - original[k] for k in geometry.ACTION_NAMES])
             / (geometry.HIGH - geometry.LOW))
    require(np.isfinite(delta).all() and np.max(np.abs(delta)) <= .10 + 1e-12,
            "Admitted adjustment exceeds the frozen per-component limit")
    proof = record.get("final_preflight")
    require(isinstance(proof, dict) and proof.get("accepted") is True
            and proof.get("release_authorized") is False and proof.get("action") == chosen,
            "Admitted geometry action lacks its final preflight")
    fits = proof.get("qualification", {}).get("ik")
    require(isinstance(fits, dict) and set(fits) == set(ARMS), "Missing admitted arm fits")
    for name in ARMS:
        q = np.asarray(fits[name].get("q"), dtype=float)
        require(q.shape == (6,) and np.isfinite(q).all(), "Invalid admitted IK seed")
    return copy.deepcopy(fits)


def install(env, enabled=True, filter_fn=layer.filter_shot):
    """Attach one instance-only hook and return an ownership-checked cleanup.

    ``raw_parent`` trial mode bypasses this layer and restores the admission/seed
    state captured at installation. Custom filters are explicitly synthetic test
    injection; their use is exposed by ``geometry_filter_synthetic``. Production
    callers must use the default unchanged layer.
    """
    require(type(enabled) is bool and callable(filter_fn), "Explicit enabled flag and filter required")
    require(not hasattr(env, _MARKER), "Geometry hook is already installed")
    original = env.prepare
    require(callable(original), "Native environment has no original prepare method")
    had_instance = "prepare" in vars(env)
    instance_value = vars(env).get("prepare")
    saved_support = copy.deepcopy(getattr(env, "support_missing", [False] * SLOTS))
    support = _mask(saved_support, "Original support flags")
    saved_seeds = copy.deepcopy(getattr(env, "proposal_ik_seeds", None))
    token = object()
    original_filter = filter_fn is layer.filter_shot

    def prepare(self, actions):
        require(getattr(self, _MARKER, None) is token, "Geometry hook ownership changed")
        require(type(actions) is list and len(actions) == SLOTS
                and len(self.base_cells) == SLOTS and getattr(self, "n", SLOTS) == SLOTS,
                "Geometry hook requires exactly 35 native action/context slots")
        active = _mask(self.active_mask, "Paid active mask")
        require(len({id(a) for a in actions}) == SLOTS,
                "Native shot slots must not alias action dictionaries")
        self.proposed_actions = copy.deepcopy(actions)
        self.geometry_records = [None] * SLOTS
        self.geometry_starts = [None] * SLOTS
        self.geometry_filter_synthetic = not original_filter
        self.geometry_wall_s = self.geometry_process_cpu_s = 0.
        use_geometry = enabled and self.trial_mode != "raw_parent"
        self.geometry_enabled_this_batch = use_geometry
        if not use_geometry:
            self.support_missing = support.tolist()
            self.proposal_ik_seeds = copy.deepcopy(saved_seeds)
            self.geometry_actions = copy.deepcopy(actions)
            return original(actions)
        began, cpu = time.perf_counter(), time.process_time()
        try:
            # No physics tick occurs while CPU filtering these measured starts.
            joint_state = _joint_state(self)
            missing, seeds = support.copy(), [None] * SLOTS
            replacements = {}
            for i in np.flatnonzero(active):
                starts = {n: joint_state[n][i].tolist() for n in ARMS}
                self.geometry_starts[i] = copy.deepcopy(starts)
                proposed = copy.deepcopy(actions[i])
                _action(self.base_cells[i], proposed)
                result = filter_fn(self.cfg, self.base_cells[i], copy.deepcopy(proposed), self.continuous_links,
                                   copy.deepcopy(starts), repair_max_nfev=24, repair_max_evaluations=512,
                                   max_normalized_adjustment=.10)
                fits = _record(result, self.cfg, self.base_cells[i], proposed,
                               self.continuous_links, starts, original_filter=original_filter)
                self.geometry_records[i] = copy.deepcopy(result)
                if result["accepted"]:
                    seeds[i] = fits
                    if result["classification"] == "adjusted":
                        replacements[i] = copy.deepcopy(result["action"])
                else:
                    missing[i] = True
            current = _joint_state(self)
            require(all(np.array_equal(current[n], joint_state[n]) for n in ARMS),
                    "Actual joint state changed during geometry filtering")
            # Commit only after all active results validate. Paid declines retain
            # active=True and use the existing no_geometry_support native gate.
            for i, action in replacements.items():
                actions[i].update(action)
            self.support_missing = missing.tolist()
            self.proposal_ik_seeds = seeds
            self.geometry_actions = copy.deepcopy(actions)
        finally:
            # Geometry-only wall/process CPU. Excludes delegated native setup,
            # physical stepping, scoring, learner work and later serialization.
            self.geometry_wall_s = time.perf_counter() - began
            self.geometry_process_cpu_s = time.process_time() - cpu
        return original(actions)

    env.prepare = MethodType(prepare, env)
    setattr(env, _MARKER, token)

    def cleanup():
        require(getattr(env, _MARKER, None) is token
                and getattr(env.prepare, "__self__", None) is env
                and getattr(env.prepare, "__func__", None) is prepare,
                "Refuse to remove another owner's prepare hook")
        if had_instance:
            env.prepare = instance_value
        else:
            del env.prepare
        env.support_missing = copy.deepcopy(saved_support)
        env.proposal_ik_seeds = copy.deepcopy(saved_seeds)
        delattr(env, _MARKER)

    return cleanup
