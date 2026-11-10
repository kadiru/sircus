"""Geometry-only continuous shot admission and bounded nearby repair.

The neural output remains a continuous five-vector.  This module has no bank,
policy model, reward, ball-flight evaluator, simulator, or release authority.
Valid requests are preserved exactly. A declined request may receive one local
continuous action/joint optimization, followed by all unchanged original gates.
The soft optimizer is only a proposal mechanism; it never certifies admission.
"""

import copy
import hashlib
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

import continuous_feasibility as geometry
import expanded_pose_menu
import joint_feasibility as joint
from prepare_assets import fk

VERSION = "sircus-continuous-geometry-layer-v1"
SCOPE = (
    "Geometry-only continuous output adapter; exact original shot preservation or "
    "one bounded local continuous action/joint repair followed by unchanged coupled "
    "IK, independent FK, endpoint clearance and 16-sample path proxies. No bank, "
    "capture/reward search, globally nearest repair, exhaustive feasibility, mesh/"
    "self-collision certificate, policy update, native movement or release authority. "
    "A declined repair is unresolved within this finite budget, not bucket impossibility."
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _signed(value):
    value = joint._plain(value)
    return dict(value, fingerprint=geometry.canonical_hash(value))


def _unsigned(value):
    _require(isinstance(value, dict), "Expected geometry-layer result")
    result = copy.deepcopy(value)
    signature = result.pop("fingerprint", None)
    _require(signature == geometry.canonical_hash(result), "Geometry result fingerprint mismatch")
    return result


def _sources():
    return {Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _protocol(repair, repair_max_nfev, repair_max_evaluations, max_normalized_adjustment):
    _require(type(repair) is bool, "Expected explicit repair boolean")
    _require(type(repair_max_nfev) is int and 1 <= repair_max_nfev <= 256,
             "Repair max_nfev must be 1..256")
    _require(type(repair_max_evaluations) is int and 1 <= repair_max_evaluations <= 4096,
             "Repair residual-evaluation cap must be 1..4096")
    _require(type(max_normalized_adjustment) in (int, float)
             and math.isfinite(max_normalized_adjustment)
             and 0 < max_normalized_adjustment <= .25,
             "Local normalized adjustment must be in (0, .25]")
    return dict(name="bounded-continuous-coupled-least-squares-v1", enabled=repair,
                max_nfev=repair_max_nfev, max_residual_evaluations=repair_max_evaluations,
                max_normalized_adjustment=float(max_normalized_adjustment),
                distance="L2 of five action deltas divided by original HIGH-LOW",
                local_bound="Per-component absolute normalized displacement",
                optimizer="scipy.optimize.least_squares; one deterministic start",
                pose_residual_weight=10000., action_distance_weight=1.,
                clearance_residual_weight=1000.,
                admission="Only unchanged joint_feasibility preflight plus independent validation")


def _pose_error(target, q, links, normal_only):
    current = fk(q, links)
    angular = (current[:3, 2] - target[:3, 2] if normal_only else
               Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec())
    return np.r_[current[:3, 3] - target[:3, 3], .25 * angular]


def _soft_geometry(scene, cell, action, links, qs):
    """Continuous proxy penalties only; original final gates decide admission."""
    targets = expanded_pose_menu.endpoint_targets(scene, cell, action)
    check = geometry.geometry_check(scene, cell, action)
    m = check["metrics"]
    margins = [m["board_bottom_z_m"] - m["table_top_z_m"] - geometry.CLEARANCE_M,
               m["ball_board_plane_clearance_m"] - geometry.CLEARANCE_M,
               m["release_ball_center_m"][2] - m["bucket_rim_z_m"] - scene["ball"]["radius_m"]]
    # The unchanged bucket proxy accepts vertical OR horizontal separation.
    horizontal_margin = (action["radius_m"] - np.linalg.norm(m["projected_board_half_extent_m"][:2])
                         - scene["bucket"]["outer_top_diameter_m"] / 2 - geometry.CLEARANCE_M)
    margins.append(max(m["board_bottom_z_m"] - m["bucket_rim_z_m"] - geometry.CLEARANCE_M,
                       float(horizontal_margin)))
    bounds = expanded_pose_menu.outer_bounds(targets, links)
    margins.extend(row["outer_radius_m"] + 1e-6 - row["distance_m"]
                   for row in bounds["arms"].values())
    clearance = geometry.approximate_clearance(scene, links,
                                              {name: dict(q=q) for name, q in qs.items()})
    margins.append(clearance["min_inter_arm_clearance_m"] - geometry.CLEARANCE_M)
    table = clearance["min_arm_table_clearance_m"]
    # None means all tested distal points are outside the table footprint.
    margins.append(1. if table is None else table - geometry.CLEARANCE_M)
    return np.minimum(np.asarray(margins, dtype=float), 0.)


class _ResidualBudget(Exception):
    pass


def _repair_proposal(scene, cell, original, links, starts, seeds, first, protocol):
    """One continuous local optimization; no enumerated per-target candidates."""
    values = np.asarray([original[name] for name in geometry.ACTION_NAMES], dtype=float)
    span = geometry.HIGH - geometry.LOW
    normalized = (values - geometry.LOW) / span
    radius = protocol["max_normalized_adjustment"]
    action_lo, action_hi = np.maximum(0., normalized - radius), np.minimum(1., normalized + radius)
    # Any partial original IK answer is only an initialization, not acceptance.
    initial_qs = {}
    for name in joint.ARMS:
        initial_qs[name] = np.asarray(first["qualification"]["ik"].get(name, {}).get("q", seeds[name]), dtype=float)
    lower = np.r_[action_lo, *[np.array([row[2] for row in links[name]]) for name in joint.ARMS]]
    upper = np.r_[action_hi, *[np.array([row[3] for row in links[name]]) for name in joint.ARMS]]
    initial = np.r_[normalized, *[initial_qs[name] for name in joint.ARMS]]
    initial = np.clip(initial, lower, upper)
    evaluations, best = 0, None

    def residual(vector):
        nonlocal evaluations, best
        if evaluations >= protocol["max_residual_evaluations"]:
            raise _ResidualBudget()
        evaluations += 1
        action = geometry.decode_action(cell, geometry.LOW + vector[:5] * span, original["id"])
        targets = expanded_pose_menu.endpoint_targets(scene, cell, action)
        qs = {name: vector[5 + i*6:11 + i*6] for i, name in enumerate(joint.ARMS)}
        errors = np.r_[*[_pose_error(targets[name], qs[name], links[name], name == "Arm14")
                         for name in joint.ARMS]]
        distance = vector[:5] - normalized
        penalties = _soft_geometry(scene, cell, action, links, qs)
        result = np.r_[protocol["pose_residual_weight"] * errors,
                       protocol["action_distance_weight"] * distance,
                       protocol["clearance_residual_weight"] * penalties]
        _require(np.isfinite(result).all(), "Nonfinite continuous repair residual")
        cost = float(result @ result)
        if best is None or cost < best[0]:
            best = (cost, vector.copy())
        return result

    optimizer = dict(termination="not_started", success=False, nfev=0)
    vector = None
    try:
        result = least_squares(residual, initial, bounds=(lower, upper),
                               max_nfev=protocol["max_nfev"],
                               ftol=1e-9, xtol=1e-9, gtol=1e-9, x_scale="jac")
        vector = result.x
        optimizer = dict(termination="optimizer_returned", success=bool(result.success),
                         status=int(result.status), nfev=int(result.nfev),
                         cost=float(result.cost))
    except _ResidualBudget:
        optimizer = dict(termination="residual_budget_exhausted", success=False, nfev=None)
        vector = None if best is None else best[1]
    _require(evaluations <= protocol["max_residual_evaluations"], "Repair work exceeded cap")
    if vector is None:
        return dict(candidate=None, ik_seeds_rad=None, optimizer=optimizer,
                    residual_evaluations=evaluations, reason="no_continuous_proposal")
    _require(np.isfinite(vector).all() and (vector >= lower).all() and (vector <= upper).all(),
             "Continuous optimizer returned a nonfinite or out-of-bounds proposal")
    candidate = geometry.decode_action(cell, geometry.LOW + vector[:5] * span, original["id"])
    return dict(candidate=candidate,
                ik_seeds_rad={name: vector[5+i*6:11+i*6].tolist() for i, name in enumerate(joint.ARMS)},
                optimizer=optimizer, residual_evaluations=evaluations, reason=None)


def filter_shot(scene, cell, action, links, starts, *, seeds=None, repair=True,
                solver=None, solver_protocol=None, repair_max_nfev=24,
                repair_max_evaluations=512, max_normalized_adjustment=.10):
    """Preserve a valid shot, qualify one local continuous repair, or decline.

    ``seeds`` are geometry solver initializations only, not a shot lookup. Supply
    no seeds to initialize from the measured starts, matching the native direct
    path; explicit seeds must be separately identified. The optional custom
    solver follows joint_feasibility's explicitly labelled test/replay contract.
    Rejection means no verified answer under this bounded method, not infeasibility
    of the whole bucket workspace. No motion or ball flight occurs here.
    """
    protocol = _protocol(repair, repair_max_nfev, repair_max_evaluations, max_normalized_adjustment)
    first = joint.preflight(scene, cell, action, links, starts, seeds=seeds,
                            solver=solver, solver_protocol=solver_protocol)
    original = first["action"]
    actual_seeds = copy.deepcopy(first["binding"]["ik_seeds_rad"])
    joint.validate_preflight(first, scene, cell, original, links, starts,
                             seeds=actual_seeds, solver_protocol=solver_protocol)
    classification, chosen, final = "rejected", None, None
    repair_record = dict(attempted=False, candidate=None, candidate_preflight=None,
                         ik_seeds_rad=None, residual_evaluations=0,
                         optimizer=None, reason="repair_not_needed" if first["accepted"] else "repair_disabled")
    if first["accepted"]:
        classification, chosen, final = "unchanged", copy.deepcopy(original), first
    elif repair:
        proposal = _repair_proposal(scene, cell, original, links, starts, actual_seeds, first, protocol)
        repair_record.update(proposal, attempted=True)
        candidate = proposal["candidate"]
        if candidate is not None:
            original_values = [original[name] for name in geometry.ACTION_NAMES]
            candidate_values = [candidate[name] for name in geometry.ACTION_NAMES]
            if candidate_values == original_values:
                repair_record["reason"] = "continuous_proposal_unchanged_and_declined"
            else:
                last = joint.preflight(scene, cell, candidate, links, starts,
                                       seeds=proposal["ik_seeds_rad"], solver=solver,
                                       solver_protocol=solver_protocol)
                joint.validate_preflight(last, scene, cell, candidate, links, starts,
                                         seeds=proposal["ik_seeds_rad"], solver_protocol=solver_protocol)
                repair_record["candidate_preflight"] = last
                repair_record["reason"] = last["reason"]
                if last["accepted"]:
                    classification, chosen, final = "adjusted", copy.deepcopy(candidate), last
    delta = None if chosen is None else [chosen[name] - original[name] for name in geometry.ACTION_NAMES]
    normalized_delta = None if delta is None else (np.asarray(delta) / (geometry.HIGH-geometry.LOW)).tolist()
    result = _signed(dict(schema=VERSION, classification=classification,
                           accepted=classification != "rejected", original_action=original, action=chosen,
                           original_preflight=first, final_preflight=final, repair=repair_record,
                           original_reason=first["reason"],
                           reason=None if final is not None else "no_verified_repair_within_bounded_work",
                           action_names=list(geometry.ACTION_NAMES), delta_physical=delta,
                           delta_normalized=normalized_delta,
                           normalized_adjustment_l2=None if delta is None else float(np.linalg.norm(normalized_delta)),
                           binding=dict(original_preflight_fingerprint=first["fingerprint"],
                                        protocol=protocol, source_sha256=_sources()),
                           preflight_calls=1 + int(repair_record["candidate_preflight"] is not None),
                           training_updates=0, native_shots=0, release_authorized=False,
                           bucket_impossibility_proven=False, scope=SCOPE))
    validate_result(result, scene, cell, action, links, starts, seeds=seeds,
                    solver_protocol=solver_protocol)
    return result


def validate_result(record, scene, cell, action, links, starts, *, seeds=None, solver_protocol=None):
    """Validate unchanged input/state/source bindings and original admission proof.

    No solver or proposal optimization runs here. Actual movement still requires
    the existing fresh native state, settling, contact and release guards.
    """
    value = _unsigned(record)
    _require(value.get("schema") == VERSION and value.get("action_names") == list(geometry.ACTION_NAMES),
             "Unknown geometry-layer schema or action names")
    _require(value.get("release_authorized") is False and value.get("bucket_impossibility_proven") is False
             and value.get("training_updates") == value.get("native_shots") == 0,
             "Geometry layer cannot train, execute shots, authorize release or prove impossibility")
    first, binding, repair_record = value["original_preflight"], value["binding"], value["repair"]
    _require(binding["source_sha256"] == _sources()
             and binding["original_preflight_fingerprint"] == first["fingerprint"],
             "Stale geometry-layer source or first proof")
    protocol = binding["protocol"]
    _require(protocol == _protocol(protocol["enabled"], protocol["max_nfev"],
                                   protocol["max_residual_evaluations"], protocol["max_normalized_adjustment"]),
             "Changed continuous repair protocol")
    admitted = joint.validate_preflight(first, scene, cell, action, links, starts,
                                        seeds=seeds, solver_protocol=solver_protocol)
    original = first["action"]
    _require(value["original_action"] == original and value["original_reason"] == first["reason"],
             "Changed original shot or first decline cause")
    _require(type(repair_record["attempted"]) is bool
             and type(repair_record["residual_evaluations"]) is int
             and 0 <= repair_record["residual_evaluations"] <= protocol["max_residual_evaluations"],
             "Unbounded continuous repair work")
    classification = value["classification"]
    _require(classification in ("unchanged", "adjusted", "rejected")
             and value["accepted"] is (classification != "rejected"), "Inconsistent geometry result")
    _require(value["preflight_calls"] == 1 + int(repair_record["candidate_preflight"] is not None),
             "Changed preflight count")
    if admitted:
        _require(classification == "unchanged" and not repair_record["attempted"]
                 and repair_record["candidate"] is None and repair_record["residual_evaluations"] == 0
                 and value["action"] == original and value["final_preflight"] == first,
                 "An originally admitted shot must remain exactly unchanged")
    else:
        _require(classification != "unchanged"
                 and repair_record["attempted"] is protocol["enabled"], "Invalid declined-shot repair status")
        candidate = repair_record["candidate"]
        if candidate is not None:
            numeric = np.asarray([candidate[name] for name in geometry.ACTION_NAMES])
            delta = (numeric - np.asarray([original[name] for name in geometry.ACTION_NAMES])) / (geometry.HIGH-geometry.LOW)
            _require(np.isfinite(delta).all()
                     and np.max(abs(delta)) <= protocol["max_normalized_adjustment"] + 1e-12,
                     "Repair proposal exceeds local continuous bound")
        if repair_record["candidate_preflight"] is not None:
            joint.validate_preflight(repair_record["candidate_preflight"], scene, cell, candidate,
                                     links, starts, seeds=repair_record["ik_seeds_rad"],
                                     solver_protocol=solver_protocol)
        if classification == "adjusted":
            _require(repair_record["attempted"] and candidate != original
                     and value["action"] == candidate
                     and value["final_preflight"] == repair_record["candidate_preflight"],
                     "Adjusted output differs from its verified continuous proposal")
            _require(joint.validate_preflight(value["final_preflight"], scene, cell, candidate,
                                              links, starts, seeds=repair_record["ik_seeds_rad"],
                                              solver_protocol=solver_protocol),
                     "Adjusted action is not admitted by original gates")
        else:
            _require(value["action"] is None and value["final_preflight"] is None
                     and not (repair_record["candidate_preflight"] or {}).get("accepted", False),
                     "Rejected geometry result contains an admitted output")
    chosen = value["action"]
    expected_delta = None if chosen is None else [chosen[name] - original[name] for name in geometry.ACTION_NAMES]
    expected_normalized = None if chosen is None else (np.asarray(expected_delta) / (geometry.HIGH-geometry.LOW)).tolist()
    _require(value["delta_physical"] == expected_delta and value["delta_normalized"] == expected_normalized
             and value["normalized_adjustment_l2"] == (None if chosen is None else float(np.linalg.norm(expected_normalized)))
             and value["reason"] == (None if chosen is not None else "no_verified_repair_within_bounded_work"),
             "Changed adjustment metrics or final reason")
    return classification
