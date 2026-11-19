"""Independent geometry-only audit for rejected original seed-22 TRAIN inputs.

No IK search, ball-flight replay, learning, simulator or shot library is used.
The original native decline remains historical: passing these geometric gates
does not establish that a native motion/settling decline or a shot is repaired.
"""

from collections import Counter
import hashlib
import math
from pathlib import Path
import re

from continuous_feasibility import IK_PROTOCOLS, canonical_hash
from continuous_geometry_audit import audit_result


VERSION = "sircus-rejected-geometry-audit-v1"
ANALYSIS_VERSION = "sircus-rejected-geometry-analysis-v1"
CLASSIFICATIONS = ("unchanged", "adjusted", "rejected")
NATIVE_REASONS = frozenset(("endpoints_not_settled", "setup_motion_not_quiet",
                            "setup_transit_incomplete"))
ORIGINAL_REJECTION_COUNTS = {
    "arm_table_proxy_overlap_Arm14": 4,
    "board_bucket_proxy_overlap": 27,
    "board_table_proxy_overlap": 41,
    "endpoint_ik_unresolved_Arm10": 2186,
    "endpoint_ik_unresolved_Arm14": 2479,
    "endpoints_not_settled": 13,
    "outer_link_length_bound": 1259,
    "setup_motion_not_quiet": 147,
    "setup_transit_incomplete": 188,
}
FROZEN_SOURCE_SHA256 = {
    "continuous_geometry_layer.py": "c6489a44e268610470edd2a4c37bd4cfb1a78c89989883113f76996bd8fbad23",
    "continuous_geometry_audit.py": "a76de94d28e01594fb9d399ee56caf844e6a30fc20bb428f67257a1391b63ab3",
}
REJECTED_EXPORT_SHA256 = "9f48315598ff581ca43f85a82ff9feaf579da9b58035f97aa642530f8d10e1ad"
REJECTED_EXPORT_FINGERPRINT = "ded7c9eb70a85bd9fd49f944be27a67c4b5114c0741be4be6dbbb494bf3b53fd"
TRAIN_PATH = re.compile(r"trials/train-([0-9]{5})\.json\.gz")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _signed(value, label):
    require(isinstance(value, dict), "Expected signed " + label)
    unsigned = {key: item for key, item in value.items() if key != "fingerprint"}
    require(value.get("fingerprint") == canonical_hash(unsigned), label + " fingerprint differs")


def _counts(counter):
    return dict(requested=sum(counter.values()), **{key: counter[key] for key in CLASSIFICATIONS})


def _fixed_protocol(result, *, real):
    protocol = result["binding"]["protocol"]
    require(protocol.get("enabled") is True
            and type(protocol.get("max_nfev")) is int and protocol["max_nfev"] == 24
            and type(protocol.get("max_residual_evaluations")) is int
            and protocol["max_residual_evaluations"] == 512
            and type(protocol.get("max_normalized_adjustment")) is float
            and protocol["max_normalized_adjustment"] == .10,
            "Frozen repair defaults changed")
    if real:
        decisions = [result["original_preflight"]]
        if result["repair"].get("candidate_preflight") is not None:
            decisions.append(result["repair"]["candidate_preflight"])
        for decision in decisions:
            require(decision["binding"]["solver_protocol"] == IK_PROTOCOLS["full_robust_v2"],
                    "Real historical check replaced original fresh robust IK")


def _current_files(protected):
    require(isinstance(protected, dict) and protected, "Missing protected file identities")
    for name, expected in protected.items():
        require(isinstance(name, str) and isinstance(expected, str)
                and re.fullmatch(r"[0-9a-f]{64}", expected), "Invalid protected file identity")
        path = Path(name)
        require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                "Protected input bytes changed: " + path.name)


def _frozen_sources():
    source = Path(__file__).resolve().parent
    for name, expected in FROZEN_SOURCE_SHA256.items():
        require(hashlib.sha256((source/name).read_bytes()).hexdigest() == expected,
                "Previously frozen geometry source changed: " + name)


def audit_row(row, case, inputs):
    """Validate one exact durable CPU result without searching for new IK.

    The caller must bind its durable-row manifest separately. This helper is
    also used before reusing a pure CPU record; it cannot authorize a task shot.
    """
    _signed(row, "Row")
    reason = case["original_reason"]
    stage = "native_motion_or_settling" if reason in NATIVE_REASONS else "geometry_or_path"
    require(isinstance(reason, str) and reason
            and case["rejection_stage"] == stage
            and case["cell"].get("split") == "train"
            and type(case["home_path_eligible"]) is bool
            and type(case.get("original_geometry_admitted")) is bool,
            "Original rejection stage or TRAIN context was relabelled")
    require(stage != "native_motion_or_settling" or
            (case["home_path_eligible"] and case["original_geometry_admitted"]),
            "Native decline lost its original accepted geometry/path provenance")
    require(case.get("seeds", case["starts"]) == case["starts"],
            "Primary check used a historical endpoint answer")
    require(row.get("case_id") == case["case_id"]
            and row.get("cell_id") == case["cell"]["id"]
            and row.get("original_reason") == reason
            and row.get("rejection_stage") == stage
            and row.get("home_path_eligible") is case["home_path_eligible"],
            "Original rejected request identity/order differs")
    for name in ("wall_s", "cpu_s"):
        require(type(row.get(name)) in (int, float) and math.isfinite(row[name]) and row[name] >= 0,
                "Invalid row resource scope")
    result = row["result"]
    _fixed_protocol(result, real=inputs.get("synthetic") is not True)
    return audit_result(result, inputs["scene"], case["cell"], case["action"],
                        inputs["links"], case["starts"], seeds=case["starts"])


def _manifest_fingerprint(inputs):
    # Reconstruct the fixed public structure independently of the analyzer.
    return canonical_hash(dict(schema=ANALYSIS_VERSION+"-manifest", bindings=inputs["bindings"],
        synthetic=inputs.get("synthetic") is True, case_ids=[c["case_id"] for c in inputs["cases"]],
        cases_fingerprint=canonical_hash(inputs["cases"]),
        protected_before=inputs["protected_before"], protected_roots=inputs.get("protected_roots", []),
        expected_counts=inputs["expected_counts"], layer_parameters=dict(
            repair=True, repair_max_nfev=24, repair_max_evaluations=512, max_normalized_adjustment=.10)))


def audit_analysis(report, inputs, *, verify_files=True):
    """Independently check every rejected TRAIN request and final cause counts.

    Full real evidence requires all 6,344 original declines. Synthetic tests use
    explicitly tiny populations and can receive only ``passed_synthetic``.
    Unknown or failed geometry is never proof that a bucket is impossible.
    """
    _signed(report, "Analysis")
    synthetic = inputs.get("synthetic") is True
    require(report.get("schema") == ANALYSIS_VERSION and report.get("seed") == 20261122,
            "Wrong rejected-TRAIN analysis lineage")
    require(report.get("synthetic") is synthetic, "Synthetic/real analysis mode differs")
    require(type(verify_files) is bool and (synthetic or verify_files),
            "Real audit cannot skip protected inputs")
    for key in ("optimizer_updates", "new_native_shots", "held_out_shot_files_read"):
        require(type(report.get(key)) is int and report[key] == 0,
                "Analysis crossed the no-training/no-shots/no-held-out boundary")
    for key in ("candidate_bank_read", "model_loaded", "release_authorized",
                "capture_replayed", "native_execution_verified"):
        require(report.get(key) is False, "Geometry-only boundary differs: " + key)
    require(report.get("bindings") == inputs["bindings"], "Original input bindings differ")
    before = inputs["protected_before"]
    require(report.get("protected_before") == before and report.get("protected_after") == before,
            "Protected input identities changed during analysis")
    require(report.get("protected_roots", []) == inputs.get("protected_roots", []),
            "Protected original roots changed")
    if verify_files:
        _current_files(before)
    _frozen_sources()
    expected = inputs["expected_counts"]["requested"] if synthetic else 6344
    require(type(expected) is int and expected > 0, "Invalid rejected request denominator")
    cases = inputs["cases"]
    require(len(cases) == expected and len({case["case_id"] for case in cases}) == expected,
            "Rejected population contains omitted or duplicate requests")
    require(isinstance(report.get("rows"), list) and len(report["rows"]) == expected,
            "Analysis omitted or added rejected requests")
    manifest = report.get("analysis_manifest_fingerprint")
    if not synthetic or manifest is not None:
        require(manifest == _manifest_fingerprint(inputs), "Durable CPU manifest differs")
        require(all(row.get("analysis_binding") == manifest for row in report["rows"]),
                "Durable CPU row uses a different source/input manifest")
    if not synthetic:
        binding = inputs["bindings"]
        require(inputs["expected_counts"] == dict(requested=6344)
                and binding.get("rejected_export_sha256") == REJECTED_EXPORT_SHA256
                and binding.get("rejected_export_fingerprint") == REJECTED_EXPORT_FINGERPRINT
                and binding.get("seed_strategy") == "actual_per_shot_measured_starts"
                and binding.get("historical_endpoint_witnesses_used") is False
                and binding.get("candidate_bank_read") is False
                and binding.get("model_loaded") is False
                and type(binding.get("held_out_shot_files_read")) is int
                and binding["held_out_shot_files_read"] == 0
                and binding.get("verified_paid_train_packets") == 289
                and binding.get("verified_paid_train_requests") == 10000,
                "Original TRAIN-only evidence provenance is incomplete")
        require(Counter(case["original_reason"] for case in cases) == ORIGINAL_REJECTION_COUNTS,
                "Original native rejection population changed")
        manifest = inputs.get("train_trial_sha256")
        require(isinstance(manifest, dict) and len(manifest) == 289
                and all(isinstance(name, str) and TRAIN_PATH.fullmatch(name)
                        and isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha)
                        for name,sha in manifest.items()),
                "Missing original 289-packet TRAIN manifest")
        for case in cases:
            relative, slot = case["source_trial_file"], case["slot"]
            require(isinstance(relative, str) and TRAIN_PATH.fullmatch(relative)
                    and type(slot) is int and 0 <= slot < 35
                    and case["case_id"] == relative+"#"+str(slot)
                    and relative in manifest and manifest[relative] == case["source_trial_sha256"],
                    "Rejected case is not bound to an original paid TRAIN slot")

    counts, first_reasons, remaining, terminations = Counter(), Counter(), Counter(), Counter()
    by_original = {}
    unchanged_native, adjusted_native = 0, 0
    for row, case in zip(report["rows"], cases):
        classification = audit_row(row, case, inputs)
        reason = case["original_reason"]
        stage = "native_motion_or_settling" if reason in NATIVE_REASONS else "geometry_or_path"
        result = row["result"]
        counts[classification] += 1
        by_original.setdefault(reason, Counter())[classification] += 1
        first_reasons[result["original_preflight"]["reason"] or "admitted"] += 1
        repair = result["repair"]
        if repair["attempted"]:
            optimizer = repair.get("optimizer")
            require(isinstance(optimizer, dict) and isinstance(optimizer.get("termination"), str)
                    and optimizer["termination"], "Attempted repair lacks its optimizer termination")
            require(optimizer.get("nfev") is None or
                    (type(optimizer["nfev"]) is int and 0 <= optimizer["nfev"] <= 24),
                    "Recorded optimizer work exceeds frozen limit")
            termination = optimizer["termination"]
        else:
            termination = "not_attempted"
        terminations[termination] += 1
        if classification == "rejected":
            final = repair.get("candidate_preflight") or result["original_preflight"]
            require(isinstance(final["reason"], str) and final["reason"],
                    "Remaining rejection lacks a verified geometry cause")
            remaining[final["reason"]] += 1
        if stage == "native_motion_or_settling":
            unchanged_native += classification == "unchanged"
            adjusted_native += classification == "adjusted"

    aggregates = dict(counts=_counts(counts),
                      by_original_reason={reason:_counts(value) for reason,value in sorted(by_original.items())},
                      first_geometry_reasons=dict(sorted(first_reasons.items())),
                      remaining_geometry_reasons=dict(sorted(remaining.items())),
                      repair_terminations=dict(sorted(terminations.items())),
                      unchanged_native_rejections=unchanged_native,
                      adjusted_native_rejections=adjusted_native)
    require(all(report.get(key) == value for key,value in aggregates.items()),
            "Rejected geometry aggregate counts or causes differ")
    audited = dict(schema=VERSION, status="passed_synthetic" if synthetic else "passed",
                   analysis_fingerprint=report["fingerprint"], **aggregates,
                   protected_input_count=len(before), geometry_only=True,
                   frozen_geometry_source_sha256=dict(FROZEN_SOURCE_SHA256),
                   capture_replayed=False, native_execution_verified=False,
                   optimizer_updates=0, new_native_shots=0, release_authorized=False,
                   scope="Original rejected TRAIN proposals only; continuous geometric adjustment or decline. Passing geometry is not proof that native motion/settling or capture succeeds. Finite solver failure is unresolved, not target impossibility.")
    audited["fingerprint"] = canonical_hash(audited)
    return audited
