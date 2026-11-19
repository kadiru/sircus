"""Independent CPU evidence checks for continuous shot geometry filtering.

No candidate catalogue, policy inference, IK optimization, simulation or learning
is performed here.  Every reached original admission gate is recomputed from
the supplied geometry and recorded finite IK answer, including rejection causes.
An admitted action is not a capture prediction or permission to release.
"""

import copy
import hashlib
import math
from collections import Counter
from pathlib import Path

import numpy as np

from continuous_feasibility import (ACTION_NAMES, HIGH, LOW, approximate_clearance,
                                    approximate_path_check, canonical_hash,
                                    decode_action, geometry_check)
from expanded_pose_menu import endpoint_targets, outer_bounds
from joint_feasibility import validate_preflight
from shot_benchmark_reach_audit import fit_details


VERSION = "sircus-continuous-geometry-audit-v1"
CLASSIFICATIONS = ("unchanged", "adjusted", "rejected")


def require(value, message):
    if not value:
        raise ValueError(message)


def _signed(value):
    require(isinstance(value, dict), "Expected signed artifact")
    unsigned = copy.deepcopy(value)
    supplied = unsigned.pop("fingerprint", None)
    require(supplied == canonical_hash(unsigned), "Artifact fingerprint differs")


def _values(action):
    values = np.asarray([action[key] for key in ACTION_NAMES], dtype=float)
    require(values.shape == (5,) and np.isfinite(values).all(), "Invalid physical action")
    return values


def audit_decision(decision, scene, cell, action, links, starts, *, seeds=None):
    """Independently rederive exact-action admission and every reached cause."""
    _signed(decision)
    binding = decision["binding"]
    supplied_seeds = binding["ik_seeds_rad"] if seeds is None else seeds
    validate_preflight(decision, scene, cell, action, links, starts,
                       seeds=supplied_seeds,
                       solver_protocol=binding["solver_protocol"])
    qualification = decision["qualification"]
    geometry = geometry_check(scene, cell, action)
    require(qualification["geometry"] == geometry, "Geometry readback differs")
    reason, goals = None, None
    if not geometry["accepted"]:
        reason = geometry["reasons"][0]
        require(not qualification["ik"] and qualification.get("outer_bounds") is None
                and qualification["approximate_clearance"] is None,
                "Later gates ran after geometry decline")
    else:
        targets = endpoint_targets(scene, cell, action)
        bounds = outer_bounds(targets, links)
        require(qualification["outer_bounds"] == bounds, "Necessary arm bound differs")
        if not bounds["eligible"]:
            reason = "outer_link_length_bound"
            require(not qualification["ik"] and qualification["approximate_clearance"] is None,
                    "IK or clearance ran after necessary arm bound decline")
        else:
            fits = qualification["ik"]
            reached = []
            for arm in ("Arm10", "Arm14"):
                require(arm in fits, "Missing reached arm fit")
                reached.append(arm)
                fit_details(targets[arm], fits[arm], links[arm], np.eye(4), arm == "Arm14")
                if not fits[arm]["solved"]:
                    reason = "endpoint_ik_unresolved_" + arm
                    break
            require(list(fits) == reached, "Original IK short-circuit order differs")
            if reason is not None:
                require(qualification["approximate_clearance"] is None,
                        "Clearance ran after unresolved IK")
            else:
                clearance = approximate_clearance(scene, links, fits)
                require(qualification["approximate_clearance"] == clearance,
                        "Endpoint clearance readback differs")
                if not clearance["accepted"]:
                    reason = clearance["reasons"][0]
                else:
                    goals = {arm: fits[arm]["q"] for arm in ("Arm10", "Arm14")}
    endpoint_accepted = reason is None
    require(type(qualification["accepted"]) is bool
            and qualification["accepted"] == endpoint_accepted
            and qualification["reason"] == reason, "Endpoint cause differs")
    if endpoint_accepted:
        path = approximate_path_check(scene, links, starts, goals, samples=16, cell=cell)
        require(decision["path"] == path, "Sampled path readback differs")
        if not path["accepted"]:
            reason = path["reasons"][0]
    else:
        require(decision["path"] is None and decision["joint_targets_rad"] is None,
                "Path or joints fabricated after endpoint decline")
    require(type(decision["accepted"]) is bool and decision["accepted"] == (reason is None)
            and decision["reason"] == reason, "Admission cause differs")
    expected_status = "admitted" if reason is None else (
        "unresolved" if reason.startswith("endpoint_ik_unresolved_") else "rejected_candidate")
    require(decision["status"] == expected_status, "Admission status differs")
    require(decision["release_authorized"] is False
            and decision["bucket_impossibility_proven"] is False,
            "Geometry cannot authorize release or prove impossibility")
    return decision["accepted"]


def audit_result(result, scene, cell, action, links, starts, *, seeds=None):
    """Verify unchanged / adjusted / rejected independently of the producer."""
    _signed(result)
    require(result.get("schema") == "sircus-continuous-geometry-layer-v1",
            "Unknown continuous geometry result")
    require(result.get("original_action") == action, "Original action was changed")
    require(result.get("release_authorized") is False
            and result.get("bucket_impossibility_proven") is False
            and type(result.get("training_updates")) is int and result["training_updates"] == 0
            and type(result.get("native_shots")) is int and result["native_shots"] == 0,
            "Result cannot train, run shots, authorize release or prove impossibility")
    require(result.get("action_names") == list(ACTION_NAMES), "Physical parameter ordering differs")
    classification = result.get("classification")
    require(classification in CLASSIFICATIONS, "Unknown geometry classification")
    first = result["original_preflight"]
    binding = result["binding"]
    require(binding["original_preflight_fingerprint"] == first["fingerprint"],
            "Original proof reference differs")
    source = Path(__file__).with_name("continuous_geometry_layer.py")
    require(binding["source_sha256"] == {source.name: hashlib.sha256(source.read_bytes()).hexdigest()},
            "Continuous producer source bytes changed")
    protocol = binding["protocol"]
    require(type(protocol.get("enabled")) is bool
            and type(protocol.get("max_nfev")) is int and 1 <= protocol["max_nfev"] <= 256
            and type(protocol.get("max_residual_evaluations")) is int
            and 1 <= protocol["max_residual_evaluations"] <= 4096
            and type(protocol.get("max_normalized_adjustment")) in (int, float)
            and math.isfinite(protocol["max_normalized_adjustment"])
            and 0 < protocol["max_normalized_adjustment"] <= .25,
            "Unbounded local continuous repair protocol")
    fixed_protocol = dict(name="bounded-continuous-coupled-least-squares-v1",
                          distance="L2 of five action deltas divided by original HIGH-LOW",
                          local_bound="Per-component absolute normalized displacement",
                          optimizer="scipy.optimize.least_squares; one deterministic start",
                          pose_residual_weight=10000., action_distance_weight=1.,
                          clearance_residual_weight=1000.,
                          admission="Only unchanged joint_feasibility preflight plus independent validation")
    require(set(protocol) == set(fixed_protocol) | {"enabled", "max_nfev", "max_residual_evaluations",
                                                   "max_normalized_adjustment"}
            and all(protocol[key] == value for key, value in fixed_protocol.items()),
            "Continuous repair protocol changed")
    first_accepted = audit_decision(first, scene, cell, action, links, starts, seeds=seeds)
    require(result.get("original_reason") == first["reason"], "Original cause was relabelled")
    repair = result["repair"]
    require(type(repair.get("attempted")) is bool
            and type(repair.get("residual_evaluations")) is int
            and 0 <= repair["residual_evaluations"] <= protocol["max_residual_evaluations"],
            "Repair exceeded recorded work cap")
    require(type(result.get("preflight_calls")) is int
            and result["preflight_calls"] == 1 + int(repair.get("candidate_preflight") is not None),
            "Geometry preflight count differs")
    candidate = repair.get("candidate")
    candidate_proof = repair.get("candidate_preflight")
    if candidate is not None:
        require(isinstance(candidate, dict) and isinstance(candidate.get("id"), str),
                "Continuous proposal is not an identified action")
        require(candidate == decode_action(cell, _values(candidate), candidate["id"]),
                "Continuous proposal derived geometry differs")
        displacement = (_values(candidate)-_values(action))/(HIGH-LOW)
        require(np.max(abs(displacement)) <= protocol["max_normalized_adjustment"] + 1e-12,
                "Repair exceeds normalized local bound")
    if candidate_proof is not None:
        require(candidate is not None, "Repair proof lacks its continuous action")
        audit_decision(candidate_proof, scene, cell, candidate, links, starts,
                       seeds=repair["ik_seeds_rad"])
    final = result.get("final_preflight")
    chosen = result.get("action")
    if classification == "unchanged":
        require(first_accepted and result.get("accepted") is True and chosen == action
                and final == first, "Unchanged result did not preserve exact admitted action")
        require(repair["attempted"] is False and candidate is None and candidate_proof is None
                and repair["residual_evaluations"] == 0, "Repair ran on an admitted action")
        expected_delta = np.zeros(5)
    elif classification == "adjusted":
        require(not first_accepted and result.get("accepted") is True
                and isinstance(chosen, dict) and isinstance(final, dict)
                and chosen != action and repair["attempted"] is True and protocol["enabled"] is True
                and chosen == candidate and final == candidate_proof,
                "Adjusted result lacks original decline or exact final action")
        require(audit_decision(final, scene, cell, chosen, links, starts),
                "Adjusted action does not pass original gates")
        expected_delta = _values(chosen) - _values(action)
        require(np.any(expected_delta != 0), "Adjustment changed identity but no settings")
    else:
        require(not first_accepted and result.get("accepted") is False
                and chosen is None and final is None,
                "Rejected result fabricated an executable action")
        require(repair["attempted"] is protocol["enabled"]
                and not (candidate_proof or {}).get("accepted", False),
                "Rejected result omitted an accepted repair or changed the repair rule")
        expected_delta = np.zeros(5)
    physical, normalized = result.get("delta_physical"), result.get("delta_normalized")
    # No selected action means no selected-action displacement.
    if classification == "rejected":
        require(physical is None and normalized is None,
                "Rejected result contains a selected-action displacement")
    else:
        require(physical == expected_delta.tolist()
                and normalized == (expected_delta / (HIGH - LOW)).tolist(),
                "Action displacement readback differs")
    require(result.get("normalized_adjustment_l2") == (
        None if classification == "rejected" else float(np.linalg.norm(expected_delta/(HIGH-LOW))))
        and result.get("reason") == (
            "no_verified_repair_within_bounded_work" if classification == "rejected" else None),
        "Final adjustment magnitude or cause differs")
    return classification


def _file_hashes(values):
    require(isinstance(values, dict) and values, "Missing protected input hashes")
    for path, expected in values.items():
        require(isinstance(path, str) and isinstance(expected, str) and len(expected) == 64,
                "Invalid protected input hash")
        require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected,
                "Protected input bytes changed: " + Path(path).name)


def audit_analysis(report, inputs, *, verify_files=True):
    """Recheck complete successful-TRAIN coverage and immutable input identities.

    Real analysis requires all 1,534 original successes and all 485 clean cases.
    Synthetic tests explicitly provide their own tiny expected population and
    receive a distinct status that cannot pass as native historical evidence.
    """
    _signed(report)
    synthetic = inputs.get("synthetic") is True
    expected = inputs.get("expected_counts") if synthetic else dict(requested=1534, clean=485)
    require(isinstance(expected, dict) and set(expected) == {"requested", "clean"},
            "Missing expected historical population")
    require(report.get("schema") == "sircus-continuous-geometry-train-retention-v1"
            and report.get("seed") == 20261122, "Analysis lineage differs")
    require(type(report.get("optimizer_updates")) is int and report["optimizer_updates"] == 0
            and type(report.get("new_native_shots")) is int and report["new_native_shots"] == 0
            and report.get("release_authorized") is False,
            "Analysis claims new learning, shots or release")
    require(report.get("synthetic") is synthetic
            and report.get("candidate_bank_read") is False
            and report.get("model_loaded") is False
            and type(report.get("held_out_shot_files_read")) is int
            and report["held_out_shot_files_read"] == 0,
            "Analysis mode, bank/model or held-out boundary differs")
    require(report.get("bindings") == inputs["bindings"], "Input provenance changed")
    if not synthetic:
        require(inputs["bindings"].get("seed_strategy") == "actual_per_shot_measured_starts"
                and inputs["bindings"].get("historical_endpoint_witnesses_used") is False
                and inputs["bindings"].get("candidate_bank_read") is False
                and inputs["bindings"].get("model_loaded") is False
                and inputs["bindings"].get("held_out_shot_files_read") == 0
                and inputs["bindings"].get("verified_paid_train_packets") == 289
                and inputs["bindings"].get("verified_paid_train_requests") == 10000,
                "Primary TRAIN provenance is incomplete or uses a witness/bank")
    before = inputs["protected_before"]
    require(report.get("protected_after") == before, "Protected hashes changed during analysis")
    if "protected_before" in report:
        require(report["protected_before"] == before, "Initial protected hashes changed")
    if verify_files:
        _file_hashes(before)
    require(synthetic or verify_files, "Real audit must verify current input bytes")
    cases = inputs["cases"]
    require(len(cases) == expected["requested"] and len({c["case_id"] for c in cases}) == len(cases)
            and sum(c["clean"] is True for c in cases) == expected["clean"],
            "Historical success population differs")
    require(len(report.get("rows", [])) == len(cases), "Analysis omitted or added cases")
    counts, clean_counts = Counter(), Counter()
    reasons = Counter()
    for row, case in zip(report["rows"], cases):
        require(case["cell"].get("split") == "train" and type(case["clean"]) is bool,
                "Non-TRAIN or malformed historical case")
        require(synthetic or case.get("seeds", case["starts"]) == case["starts"],
                "Real primary analysis must seed IK from actual starts, without a historical witness lookup")
        require(row["case_id"] == case["case_id"] and row["cell_id"] == case["cell"]["id"]
                and row["clean"] is case["clean"], "Historical case identity/order differs")
        classification = audit_result(row["result"], inputs["scene"], case["cell"],
                                      case["action"], inputs["links"], case["starts"],
                                      seeds=case.get("seeds"))
        counts[classification] += 1
        if case["clean"]:
            clean_counts[classification] += 1
        reason = row["result"]["original_preflight"]["reason"]
        if reason is not None:
            reasons[reason] += 1
    totals = dict(requested=len(cases), **{k: counts[k] for k in CLASSIFICATIONS})
    clean_totals = dict(requested=expected["clean"], **{k: clean_counts[k] for k in CLASSIFICATIONS})
    require(report.get("counts") == totals and report.get("clean_counts") == clean_totals,
            "Summary classification accounting differs")
    require(report.get("reasons") == dict(sorted(reasons.items())), "Summary causes differ")
    result = dict(schema=VERSION, status="passed_synthetic" if synthetic else "passed",
                  analysis_fingerprint=report["fingerprint"], counts=totals,
                  clean_counts=clean_totals, reasons=dict(sorted(reasons.items())),
                  geometry_only=True, capture_replayed=False, optimizer_updates=0,
                  new_native_shots=0, release_authorized=False,
                  protected_input_count=len(before),
                  scope="Recorded TRAIN actions only; independent original geometry/FK/path proxy checks; no capture or mesh-collision guarantee")
    result["fingerprint"] = canonical_hash(result)
    return result
