"""Fresh instrumentation controls with a declared FP32 comparison envelope.

Two uninstrumented throws and one instrumented throw use the same native
environment and exact saved endpoints. Only force/velocity observations permit
up to eight adjacent finite FP32 representations per element. Every position,
clock, contact geometry/ID and complete scalar scoring output must match exactly.
This tolerance belongs to an engineering logger-equivalence check. It does not
modify physics, contact calibration, capture scoring, or benchmark results.
"""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from .calibrated_score import FORCE_THRESHOLD_N, score_trace
from .corrected_contract import sha
from .learning_contract import digest
from .sweep_worker import atomic, load_reference


VERSION = "sircus-impact-pose-equivalence-controls-v2"
MAX_FP32_ULPS = 8
APPROXIMATE_KEYS = frozenset(("velocity_m_s", "forces_N"))
SCOPE = ("Three fresh engineering controls, outside sweep/training/evaluation budgets. "
         "Endpoint-reset dynamics and frozen capture scorer remain unchanged. "
         "Eight-FP32-ULP tolerance applies only to logger comparison of forces and "
         "velocities, never geometry, clocks, category thresholds, or score outputs.")


def exact_array(a, b):
    """Byte identity, including signed zeros and NaN/Inf representations."""
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def ordered_fp32(value):
    """Map finite FP32 numbers to monotonic integers; signed zeros coincide."""
    value = np.asarray(value)
    if value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError("Finite FP32 observations required for ULP distance")
    bits = value.view(np.uint32).astype(np.int64)
    return np.where(bits & 0x80000000, 0x80000000-(bits & 0x7fffffff), 0x80000000+bits)


def compare_observations(reference, observed, score_reference, score_observed):
    """Apply the predeclared envelope and exact downstream-decision checks."""
    fields, failures = {}, []
    for key, first in reference.items():
        if key not in observed:
            failures.append(key+"/missing")
            continue
        a, b = np.asarray(first), np.asarray(observed[key])
        if a.shape != b.shape or a.dtype != b.dtype:
            failures.append(key+"/shape_or_dtype")
            fields[key] = dict(passed=False, rule="matching shapes/dtypes")
            continue
        finite_a, finite_b = np.isfinite(a), np.isfinite(b)
        finite = finite_a & finite_b
        nonfinite_match = (np.array_equal(finite_a, finite_b)
                           and exact_array(a[~finite_a], b[~finite_b]))
        bits_equal = exact_array(a, b)
        field = dict(bit_identical=bits_equal, nonfinite_pattern_identical=nonfinite_match,
                     unequal_values=int(np.count_nonzero(a[finite] != b[finite])),
                     maximum_absolute_difference=float(np.max(np.abs(a[finite].astype(float)-b[finite].astype(float)))) if finite.any() else 0.)
        if key in APPROXIMATE_KEYS:
            if a.dtype != np.float32:
                passed = False
                field.update(rule="finite FP32 only, maximum8ULP", maximum_ulp_distance=None)
            else:
                delta = np.abs(ordered_fp32(a[finite])-ordered_fp32(b[finite]))
                ulps = int(delta.max()) if len(delta) else 0
                passed = nonfinite_match and ulps <= MAX_FP32_ULPS
                field.update(rule="maximum8FP32ULPper_element", maximum_ulp_distance=ulps,
                             permitted_ulp_distance=MAX_FP32_ULPS)
        else:
            passed = bits_equal
            field["rule"] = "byte_identical"
        field["passed"] = bool(passed)
        fields[key] = field
        if not passed: failures.append(key+"/comparison")
    # Pose_* are additive observations, not unexplained replacement arrays.
    extras = sorted(set(observed)-set(reference))
    if any(not key.startswith("pose_") for key in extras):
        failures.append("unexpected_original_observation_keys")
    force_mask_equal = False
    if "forces_N" in reference and "forces_N" in observed:
        force_mask_equal = np.array_equal(np.asarray(reference["forces_N"], float) > FORCE_THRESHOLD_N,
                                         np.asarray(observed["forces_N"], float) > FORCE_THRESHOLD_N)
    if not force_mask_equal: failures.append("force_category_threshold_masks")
    score_equal = digest(score_reference) == digest(score_observed)
    if not score_equal: failures.append("complete_scalar_score_output")
    return dict(status="passed" if not failures else "failed", failures=failures,
                original_fields=fields, additive_pose_fields=extras,
                force_category_threshold_N=FORCE_THRESHOLD_N,
                force_category_threshold_masks_exact=bool(force_mask_equal),
                complete_scalar_score_output_exact=bool(score_equal),
                score_keys=sorted(score_reference), scope=SCOPE)


def run(args):
    import torch
    from .bank_geometry import fixed_endpoints
    from .calibrated_env import CalibratedEnvironment
    from .expanded_actions import WIDE_BOUNDS
    from .gpu_slot import GPU_UUID, gpu_slot
    from .impact_pose_env import ImpactPoseEnvironment, verify_pose_arrays
    from .learning_contract import constraints
    from .validation_failure_audit import member, verify_signed

    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_UUID:
        raise RuntimeError("Select only the authorized RTX4090 UUID")
    if args.output.exists():
        raise ValueError("Controls require a new output directory; old failed evidence is preserved")
    for root in (args.study, args.reference, args.assets, args.plan.parent):
        if args.output.resolve().is_relative_to(root.resolve()):
            raise ValueError("Controls must be outside original evidence, plan, reference and assets")
    plan = json.loads(args.plan.read_text()); verify_signed(plan, "Prepared impact sweep plan")
    repository = Path(__file__).resolve().parents[2]
    for name, expected in plan["source_code_sha256"].items():
        if sha(member(repository, name)) != expected:
            raise ValueError("Prepared instrumentation/sweep source changed: "+name)
    contract_path = args.study / "contract.json"
    contract = json.loads(contract_path.read_text())
    if contract["fingerprint"] != plan["source"]["contract_fingerprint"]:
        raise ValueError("Control and benchmark contract identities differ")
    source = plan["layouts"][0]
    if "-validation-" not in source["source_row_id"] or "-test-" in source["source_row_id"]:
        raise ValueError("Only the declared validation source posture may be used")
    path = member(args.study, source["source_rows_file"])
    if sha(path) != source["source_rows_sha256"]:
        raise ValueError("Frozen source row bytes changed")
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    rows = [r for r in rows if r["id"] == source["source_row_id"]]
    if len(rows) != 1 or digest(rows[0]) != source["source_row_sha256"]:
        raise ValueError("Frozen control row identity changed")
    row = rows[0]
    bucket, values = np.asarray([row["bucket_xy_m"]]), np.asarray([row["values"]])
    q10, q14 = (np.asarray([row[key]]) for key in ("q10", "q14"))
    ref = load_reference(args.reference, args.assets)
    if digest(ref["scene"]) != contract["identity"]["scene_sha256"]:
        raise ValueError("Control scene differs from frozen scene")
    calibration = contract["identity"]["calibration"]["score_config"]
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = dict(version=VERSION, status="prepared", engineering_requests=3,
        ordered_runs=["baseline1", "baseline2", "instrumented"], maximum_fp32_ulps=MAX_FP32_ULPS,
        approximate_fields=sorted(APPROXIMATE_KEYS), exact_fields="Every other original array, nonfinite pattern, force threshold mask, and complete scalar score output",
        source_row_id=source["source_row_id"], source=plan["source"], plan_sha256=sha(args.plan),
        control_source_sha256=sha(Path(__file__)), scope=SCOPE)
    atomic(args.output / "protocol.json", protocol)
    start = time.monotonic()
    with gpu_slot(args.gpu_lock, timeout_s=600, poll_s=2):
        torch.set_num_threads(2)
        env = ImpactPoseEnvironment(ref["scene"], args.assets, calibration, n=1, bounds=WIDE_BOUNDS)
        observed, saved, timings, scores = {}, {}, {}, {}
        for label in protocol["ordered_runs"]:
            with fixed_endpoints(env, bucket, values, q10, q14, np.ones(1, dtype=bool)):
                if label == "instrumented":
                    result, metadata = env.run(bucket, values, constraints(1), record=True)
                else:
                    result, metadata = CalibratedEnvironment.run(env, bucket, values, constraints(1), record=True)
            observed[label], timings[label] = result, metadata
            saved[label] = {key: value.copy() for key, value in env.last_arrays.items()}
            np.savez_compressed(args.output / (label+".npz"), **saved[label])
            trace = {key: value[:, 0] if value.ndim >= 2 and key != "time_s" else value
                     for key, value in saved[label].items() if not key.startswith("pose_")}
            scores[label] = score_trace(ref["scene"], bucket[0], result[0]["board_normal"], calibration, trace)
        comparisons = {a+"_vs_"+b: compare_observations(saved[a], saved[b], scores[a], scores[b])
                       for a, b in (("baseline1", "baseline2"), ("baseline1", "instrumented"), ("baseline2", "instrumented"))}
        fk = verify_pose_arrays(env.sim.mj_model, saved["instrumented"], observed["instrumented"])
    report = dict(protocol, status="passed" if all(v["status"] == "passed" for v in comparisons.values()) and fk["status"] == "passed" else "failed",
        comparisons=comparisons, complete_scores=scores,
        fk={key: value for key, value in fk.items() if key != "summaries"},
        instrumented_pose_summary=observed["instrumented"][0]["pose_logging"],
        evidence_sha256={str(p.relative_to(args.output)): sha(p) for p in args.output.iterdir() if p.is_file()},
        timings=timings, wall_s=time.monotonic()-start,
        conclusion="Numerical equivalence within declared engineering bounds; does not prove hardware fidelity or isolate the cause of GPU floating-point variation")
    atomic(args.output / "report.json", report)
    print(json.dumps(dict(status=report["status"], engineering_requests=3,
                         report_sha256=sha(args.output / "report.json"), comparisons=comparisons,
                         fk=report["fk"])), flush=True)
    if report["status"] != "passed":
        raise RuntimeError("Fresh instrumentation controls failed declared comparison bounds")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "study", "reference", "assets", "gpu-lock", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
