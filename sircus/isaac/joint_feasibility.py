"""Reusable, geometry-only admission for exact cooperative shot candidates.

This layer prepares a finite set of coupled dropper/board endpoints before a
policy or rollout consumes it. It neither modifies actions nor learns from ball
outcomes. Admission means the inherited endpoint and sampled path proxies pass
at the supplied measured six-joint starts. It never authorizes motion/release:
the native supervisor must check the actual state, settling and release guards.
"""

import copy
import hashlib
import math
from pathlib import Path

import numpy as np

import continuous_feasibility as geometry
import expanded_pose_menu
import expanded_pose_native
import feasibility_contract
import prepare_assets
import robust_ik

VERSION = "sircus-joint-feasibility-v1"
ARMS = ("Arm10", "Arm14")
PATH_SAMPLES = 16
MAX_CANDIDATES = 4096
SCOPE = (
    "Finite exact-action geometry admission only; coupled arm poses, independent "
    "FK and original residual/joint limits, provisional capsules and 16 sampled "
    "joint-interpolation checks. No ball-flight outcome, exhaustive workspace "
    "proof, self-collision or mesh/path certificate. Board IK constrains center "
    "and normal, not full in-plane twist. Fresh native motion/settling/release "
    "checks remain mandatory; this artifact never authorizes release."
)


def _require(value, message):
    if not value:
        raise ValueError(message)


def _plain(value):
    """Normalize numerical result containers without accepting nonfinite JSON."""
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, dict):
        _require(all(isinstance(k, str) for k in value), "Expected string keys")
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    _require(value is None or type(value) in (str, bool, int, float), "Expected JSON data")
    _require(type(value) is not float or math.isfinite(value), "Nonfinite feasibility data")
    return value


def _signed(value):
    value = _plain(value)
    value["fingerprint"] = geometry.canonical_hash(value)
    return value


def _unsigned(value):
    _require(isinstance(value, dict), "Expected signed feasibility record")
    result = copy.deepcopy(value)
    fingerprint = result.pop("fingerprint", None)
    _require(fingerprint == geometry.canonical_hash(_plain(result)), "Feasibility fingerprint mismatch")
    return result


def _vector(value, length, label):
    result = np.asarray(value, dtype=float)
    _require(result.shape == (length,) and np.isfinite(result).all(), "Expected finite " + label)
    return result


def _scene(scene):
    result = _plain(copy.deepcopy(scene))
    # The inherited transform implementation uses the original opposing base.
    _require(math.isclose(float(result["arms"]["board_base_yaw_rad"]), math.pi,
                          rel_tol=0, abs_tol=1e-12), "Unsupported board-arm base yaw")
    for value in (result["arms"]["base_spacing_m"], result["ball"]["radius_m"],
                  result["bucket"]["assumed_interior_depth_m"],
                  result["bucket"]["assumed_bottom_thickness_m"],
                  result["bucket"]["outer_top_diameter_m"]):
        _require(type(value) in (int, float) and value > 0, "Invalid scene dimensions")
    for name in ("board", "table"):
        _require((_vector(result[name]["size_m"], 3, name + " size") > 0).all(),
                 "Invalid scene box size")
    _vector(result["table"]["center_m"], 3, "table center")
    _require(type(result["board"]["tool_offset_m"]) in (int, float)
             and result["board"]["tool_offset_m"] >= 0, "Invalid board mount offset")
    _require(type(result["board"]["mount_yaw_rad"]) in (int, float), "Invalid board mount yaw")
    return result


def _cell(cell):
    _require(isinstance(cell, dict) and isinstance(cell.get("id"), str) and cell["id"],
             "Expected identified bucket context")
    # Split IDs, rewards, measured outcomes and other metadata are not inputs.
    return dict(id=cell["id"], bucket_xy_m=_vector(cell["bucket_xy_m"], 2, "bucket XY").tolist())


def _chains(links):
    _require(isinstance(links, dict) and set(links) == set(ARMS), "Expected both arm chains")
    result = {}
    for name in ARMS:
        _require(len(links[name]) == 6, "Expected six-joint arm chain")
        result[name] = []
        for item in links[name]:
            _require(len(item) == 4, "Expected chain origin, axis and two limits")
            origin, axis, lo, hi = item
            origin = np.asarray(origin, dtype=float)
            axis = _vector(axis, 3, "joint axis")
            _require(origin.shape == (4, 4) and np.isfinite(origin).all()
                     and np.allclose(origin[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
                     and np.allclose(origin[:3, :3].T @ origin[:3, :3], np.eye(3), atol=1e-8, rtol=0)
                     and abs(np.linalg.det(origin[:3, :3]) - 1) <= 1e-8,
                     "Expected rigid joint origin")
            _require(abs(float(np.linalg.norm(axis)) - 1) <= 1e-8, "Expected unit joint axis")
            _require(math.isfinite(float(lo)) and math.isfinite(float(hi)) and lo < hi,
                     "Invalid joint limits")
            result[name].append([origin.tolist(), axis.tolist(), float(lo), float(hi)])
    return result


def _joints(values, links, label):
    _require(isinstance(values, dict) and set(values) == set(ARMS), "Expected both " + label)
    result = {}
    for name in ARMS:
        q = _vector(values[name], 6, label)
        lo, hi = np.array([x[2] for x in links[name]]), np.array([x[3] for x in links[name]])
        _require((q >= lo - 1e-6).all() and (q <= hi + 1e-6).all(), label + " outside joint limits")
        result[name] = q.tolist()
    return result


def _action(cell, candidate, identifier):
    if not isinstance(candidate, dict):
        return geometry.decode_action(cell, candidate, identifier)
    _require(isinstance(candidate.get("id"), str) and candidate["id"], "Expected action identity")
    values = [candidate[k] for k in geometry.ACTION_NAMES]
    expected = geometry.decode_action(cell, values, candidate["id"])
    for key, value in expected.items():
        if key == "id":
            continue
        actual = candidate.get(key)
        _require(type(actual) in (int, float) and math.isfinite(actual)
                 and abs(float(actual) - value) <= 1e-8,
                 "Action geometry disagrees with its physical parameters: " + key)
    # Keep every supplied physical value, but discard unrelated outcome fields.
    return _plain({key: copy.deepcopy(candidate[key]) for key in expected})


def _solver_protocol(solver, solver_protocol):
    if solver_protocol is None:
        _require(solver is None, "Custom solver requires explicit solver_protocol")
        return copy.deepcopy(geometry.IK_PROTOCOLS["full_robust_v2"])
    if isinstance(solver_protocol, str):
        _require(bool(solver_protocol.strip()), "Empty solver protocol")
        return dict(name=solver_protocol)
    value = _plain(copy.deepcopy(solver_protocol))
    _require(isinstance(value, dict) and value, "Expected explicit solver protocol")
    return value


def _source_hashes():
    from settled_transit import __file__ as controller_source
    files = [__file__, controller_source] + [m.__file__ for m in
        (geometry, expanded_pose_menu, expanded_pose_native, feasibility_contract,
         prepare_assets, robust_ik)]
    return {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}


def _inputs(scene, cell, candidate, links, starts, seeds, solver, solver_protocol, identifier):
    scene, cell = _scene(scene), _cell(cell)
    chains = _chains(links)
    starts = _joints(starts, links, "measured starts")
    seeds = _joints(starts if seeds is None else seeds, links, "IK seeds")
    action = _action(cell, candidate, identifier)
    binding = dict(version=VERSION, scene_sha256=geometry.canonical_hash(scene),
                   chain_sha256=geometry.canonical_hash(chains), context=cell,
                   action=action, measured_starts_rad=starts, ik_seeds_rad=seeds,
                   solver_protocol=_solver_protocol(solver, solver_protocol),
                   path_samples=PATH_SAMPLES, source_sha256=_source_hashes())
    return scene, cell, action, starts, seeds, _signed(binding)


def preflight(scene, cell, action, links, starts, *, seeds=None, solver=None, solver_protocol=None):
    """Admit or decline one unchanged action; no native movement or release.

    ``solver`` has the original (target, links, seed, normal_only) signature.
    Saved-fit replay must explicitly label its protocol and fail if the requested
    arm was not evaluated in the saved evidence. Numerical failure is unresolved.
    """
    scene, cell, action, starts, seeds, binding = _inputs(
        scene, cell, action, links, starts, seeds, solver, solver_protocol, "candidate-0000")
    from settled_transit import equivalent_fit
    solver = robust_ik.solve if solver is None else solver

    def fresh(target, chain, seed, normal_only):
        name = "Arm14" if normal_only else "Arm10"
        fit = _plain(copy.deepcopy(solver(target, chain, np.asarray(seed), normal_only)))
        _require(isinstance(fit, dict) and type(fit.get("solved")) is bool,
                 "Missing solver acceptance flag")
        q = _vector(fit.get("q"), 6, "solver joints")
        lo, hi = np.array([x[2] for x in chain]), np.array([x[3] for x in chain])
        _require((q >= lo).all() and (q <= hi).all(), "Solver joints outside original limits")
        _require(type(fit.get("residual")) in (int, float) and fit["residual"] >= 0
                 and fit["solved"] == (fit["residual"] < robust_ik.TOLERANCE),
                 "Solver acceptance disagrees with original residual gate")
        return equivalent_fit(target, fit, chain, starts[name], normal_only)

    qualification = geometry.qualify_action(scene, cell, action, links, seeds, solver=fresh)
    targets = expanded_pose_menu.endpoint_targets(scene, cell, action)
    path, joint_targets = None, None
    if qualification["accepted"]:
        joint_targets = {name: qualification["ik"][name]["q"] for name in ARMS}
        path = geometry.approximate_path_check(scene, links, starts, joint_targets,
                                              samples=PATH_SAMPLES, cell=cell)
    accepted = bool(qualification["accepted"] and path and path["accepted"])
    reason = qualification["reason"] if not qualification["accepted"] else (
        path["reasons"][0] if not path["accepted"] else None)
    status = "admitted" if accepted else ("unresolved" if reason.startswith("endpoint_ik_unresolved_")
                                           else "rejected_candidate")
    return _signed(dict(schema=VERSION, binding=binding, action=action,
                        targets={name: np.asarray(value).tolist() for name, value in targets.items()},
                        joint_targets_rad=joint_targets, qualification=qualification, path=path,
                        accepted=accepted, status=status, reason=reason,
                        candidate_outside_necessary_reach_bound=reason == "outer_link_length_bound",
                        bucket_impossibility_proven=False, release_authorized=False, scope=SCOPE))


def prepare_candidates(scene, cell, candidates, links, starts, *, seeds=None, solver=None,
                       solver_protocol=None):
    """Keep all finite candidates and expose only their admitted exact indices.

    This is finite support, not a certified continuous box. There is no outcome
    ranking, projection, silent retry or replacement. A later policy must define
    its own distribution over this support; old continuous PPO likelihoods must
    not be attached to changed/chosen catalogue actions.
    """
    _require(isinstance(candidates, (list, tuple, np.ndarray))
             and 0 <= len(candidates) <= MAX_CANDIDATES, "Expected bounded finite candidate sequence")
    # Validate geometry/state even for an empty finite support request.
    context, scene_value, chains = _cell(cell), _scene(scene), _chains(links)
    starts_value = _joints(starts, links, "measured starts")
    seeds_value = _joints(starts if seeds is None else seeds, links, "IK seeds")
    binding = _signed(dict(version=VERSION, scene_sha256=geometry.canonical_hash(scene_value),
                           chain_sha256=geometry.canonical_hash(chains), context=context,
                           measured_starts_rad=starts_value, ik_seeds_rad=seeds_value,
                           solver_protocol=_solver_protocol(solver, solver_protocol),
                           path_samples=PATH_SAMPLES, source_sha256=_source_hashes()))
    records = []
    for i, candidate in enumerate(candidates):
        action = _action(_cell(cell), candidate, f"candidate-{i:04d}")
        records.append(preflight(scene, cell, action, links, starts, seeds=seeds,
                                 solver=solver, solver_protocol=solver_protocol))
    indices = [i for i, record in enumerate(records) if record["accepted"]]
    return _signed(dict(schema=VERSION + "-support", binding=binding, records=records,
                        requests=len(records), admitted_indices=indices, admitted=len(indices),
                        status="supported" if indices else "unresolved",
                        reason=None if indices else "no_candidate_admitted_within_finite_budget",
                        bucket_impossibility_proven=False, release_authorized=False, scope=SCOPE))


def validate_preflight(record, scene, cell, action, links, starts, *, seeds=None,
                       solver_protocol=None):
    """Fail closed on stale inputs and independently recheck admitted fits/path.

    This is CPU-only evidence validation, never a native release authorization.
    No IK optimization is performed. Measured drift or any changed target/state,
    geometry, seed, solver protocol or source invalidates reuse.
    """
    value = _unsigned(record)
    # An explicit protocol is validated without invoking its producer solver.
    scene, cell, action, starts, seeds, binding = _inputs(
        scene, cell, action, links, starts, seeds, None, solver_protocol, "candidate-0000")
    _require(value.get("schema") == VERSION and value.get("binding") == binding
             and value.get("action") == action, "Stale feasibility binding")
    _require(value.get("release_authorized") is False
             and value.get("bucket_impossibility_proven") is False,
             "Feasibility cannot authorize release or prove bucket impossibility")
    qualification, path = value["qualification"], value["path"]
    _require(qualification.get("action") == action
             and type(qualification.get("accepted")) is bool,
             "Qualification action or flag changed")
    accepted = bool(qualification["accepted"] and path and path["accepted"])
    _require(type(value.get("accepted")) is bool and value["accepted"] == accepted,
             "Inconsistent admission flags")
    targets = expanded_pose_menu.endpoint_targets(scene, cell, action)
    _require(value["targets"] == {n: np.asarray(v).tolist() for n, v in targets.items()},
             "Changed coupled endpoint targets")
    if accepted:
        _require(value.get("status") == "admitted" and value.get("reason") is None,
                 "Inconsistent admission status")
        _require(geometry.geometry_check(scene, cell, action)["accepted"], "Geometry does not admit action")
        _require(expanded_pose_menu.outer_bounds(targets, links)["eligible"], "Outside necessary arm bound")
        fits = qualification["ik"]
        for name in ARMS:
            expanded_pose_native.verified_fit(targets[name], fits[name], links[name], name == "Arm14")
        goals = {name: fits[name]["q"] for name in ARMS}
        _require(value["joint_targets_rad"] == goals, "Changed prepared joint targets")
        _require(geometry.approximate_clearance(scene, links, fits)["accepted"], "Endpoint proxy no longer admits action")
        expected = geometry.approximate_path_check(scene, links, starts, goals,
                                                  samples=PATH_SAMPLES, cell=cell)
        _require(path == _plain(expected) and expected["accepted"], "Path proxy does not match admission")
    return accepted
