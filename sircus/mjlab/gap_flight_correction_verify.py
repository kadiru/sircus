"""Independent CPU audit of the bounded +60 mm gap development experiment.

This checker never launches native physics. Original gap-comparison outcomes are
immutable; the three selected failures are development cases, not fresh holdouts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np

from . import calibrated_benchmark_verify as calibrated
from . import policy64_verify as policy_audit
from . import replay650_verify as replay
from . import xy650_verify as xy_audit
from .benchmark_verify import EvidenceError, contained, digest, read, sha


VERSION = "sircus-gap-flight-correction-audit-v1"
STUDY_VERSION = "sircus-gap-flight-correction-v1"
PARENT_REPORT_SHA = "110b8b7f48f9f0cc153e0dc0c9341ed78bb04d861090abef592bfc98f2f5bf1b"
PARENT_MANIFEST_SHA = "7af5b7087d3ae24b47a2949e5a5cf2cd6c9a783411387186b0926fd2ae9e091e"
PARENT_AUDIT_SHA = "5c6bb8965ef264f9744c0f72c6da5c5bb4904d724938519eb21f45952d1c925a"
CONTEXTS = tuple(f"gap-coverage-test-{i:02d}" for i in (12, 24, 48))
MAX_REQUESTS = 30
GAP_DELTA_M = .06
REPEATS = 3


def same(actual, expected, checks, label, tolerance=1e-9):
    if isinstance(expected, dict):
        checks.require(isinstance(actual, dict), label+"/dictionary")
        checks.equal(set(actual), set(expected), label+"/complete fields")
        for key in expected: same(actual[key], expected[key], checks, label+"/"+str(key), tolerance)
    elif isinstance(expected, (list, tuple)):
        checks.require(isinstance(actual, (list, tuple)), label+"/sequence")
        checks.equal(len(actual), len(expected), label+"/length")
        for i, value in enumerate(expected): same(actual[i], value, checks, label+"/"+str(i), tolerance)
    elif type(expected) is bool:
        checks.require(type(actual) is bool and actual is expected, label+"/explicit boolean")
    elif isinstance(expected, (float, np.floating)):
        checks.numeric(actual, expected, label, tolerance)
    else: checks.equal(actual, expected, label)


def native_counts(rows):
    return dict(attempts=len(rows), native_releases=sum(r.get("native_release") is True for r in rows),
        captures=sum(policy_audit.captured(r) for r in rows),
        clean=sum(policy_audit.captured(r) and r.get("route_quality") == 2 for r in rows),
        rejected=sum(r.get("native_release") is False for r in rows),
        invalid_physics=sum(bool(r.get("invalid_physics")) for r in rows))


def native_control_gate(rows):
    positives = [r for r in rows if r["role"] == "positive_control"]
    negatives = [r for r in rows if r["role"] == "negative_control"]
    passed = (len(rows) == 6 and len(positives) == len(negatives) == 3
        and all(policy_audit.captured(r) for r in positives)
        and all(r.get("native_release") is True and not policy_audit.captured(r) and not r.get("whole_ball_entry") for r in negatives)
        and all(not r.get("invalid") and not r.get("invalid_physics") and r.get("scalar_score_audit_passed") is True for r in rows))
    return dict(status="passed" if passed else "failed", counts=native_counts(rows))


def audit_parent(study, audit_path, checks):
    """Bind the completed 112-slot comparison and its corrected full proof."""
    for path, expected in ((study/"report.json", PARENT_REPORT_SHA),
                           (study/"evidence-manifest.json", PARENT_MANIFEST_SHA),
                           (audit_path, PARENT_AUDIT_SHA)):
        checks.equal(sha(path), expected, "Exact completed parent bytes/"+path.name)
    parent, report, proof = read(study/"contract.json"), read(study/"report.json"), read(audit_path)
    unsigned = dict(parent); fingerprint = unsigned.pop("fingerprint")
    checks.equal(digest(unsigned), fingerprint, "Parent complete contract fingerprint")
    checks.equal((report["status"], report["phase"]), ("completed", "completed"), "Parent completed")
    checks.equal((parent["version"], report["version"], proof["version"]),
                 ("sircus-gap-coverage-study-v1", "sircus-gap-coverage-study-v1", "sircus-gap-coverage-audit-v2"), "Exact parent study and corrected audit versions")
    checks.equal(proof["status"], "passed", "Corrected original audit passed")
    checks.require(proof["checks_count"] == len(proof["checks"]) == 1441075
                   and not proof.get("failures") and all(c.get("passed") is True for c in proof["checks"]),
                   "All original1441075 checks passed")
    checks.equal((proof["report_sha256"], proof["evidence_manifest_sha256"]),
                 (PARENT_REPORT_SHA, PARENT_MANIFEST_SHA), "Original audit binds complete evidence")
    checks.equal((proof["contract_fingerprint"], report["contract_fingerprint"]),
                 (fingerprint, fingerprint), "Original report and proof bind parent contract")
    manifest = replay.check_manifest(study, checks)
    checks.equal(report["evidence_sha256"], manifest, "Original report complete manifest")
    rows = replay.read_records(study/"records.jsonl")
    checks.equal((len(rows), sum(r["native_release"] is True for r in rows)), (112, 96), "Original requested/native denominators")
    checks.equal(Counter(r["method"] for r in rows if r["role"] == "test"),
                 Counter(consistent=50, gap_coverage=50), "Original paired50 methods")
    return parent, report, rows, manifest


def expected_plan_core(rows):
    """Literal independent intervention specification; never choose new outcomes."""
    controls = [next(r for r in rows if r["phase"] == "controls_pre" and r["role"] == role and r.get("repeat") == 0)
                for role in ("positive_control", "negative_control")]
    sources = {}
    for context in CONTEXTS:
        matches = [r for r in rows if r.get("method") == "consistent" and r.get("context_id") == context]
        if len(matches) != 1:
            raise EvidenceError("Exactly one original selected case required: "+context)
        source = matches[0]
        if source.get("native_release") is not True:
            raise EvidenceError("Selected original must have a native flight: "+context)
        sources[context] = source
    plan = []
    def append(source, context, variant, phase, repeat):
        values = copy.deepcopy(source["values"])
        changed = variant == "gap_plus60"
        if changed: values[4] += GAP_DELTA_M
        result = dict(phase=phase, role=source["role"] if phase.startswith("controls_") else "diagnostic",
            context_id=context, variant=variant, kind="coordinate_swap" if changed else "exact_replay", repeat=repeat,
            substeps=64, origin_method=source["method"], parent_record_id=source["id"], parent_record_sha256=digest(source),
            bucket_xy_m=copy.deepcopy(source["bucket_xy_m"]), values=values, command_values=copy.deepcopy(values),
            original_native_release=True, swapped_coordinates=[4] if changed else [], selection_or_learning=False,
            source_teacher_id=source["source_teacher_id"], observation=copy.deepcopy(source["observation"]))
        for key in ("source_hint", "effective_source_hint", "context_resolution_file", "context_resolution_sha256",
                    "context_home_sha256", "context_layout_eligible", "context_anchor_admitted", "context_hard_reject"):
            result[key] = copy.deepcopy(source[key])
        if not changed:
            for key in ("q10", "q14", "connection_file", "connection_sha256"):
                result[key] = copy.deepcopy(source[key])
            result.update(parent_native_cache_file=source["native_cache_file"], parent_native_cache_sha256=source["native_cache_sha256"])
        plan.append(result)
    for phase in ("controls_pre", "diagnostic", "controls_post"):
        if phase.startswith("controls_"):
            for source in controls:
                for repeat in range(REPEATS): append(source, source["context_id"], "control_exact", phase, repeat)
        else:
            for context in CONTEXTS:
                for variant in ("original", "gap_plus60"):
                    for repeat in range(REPEATS): append(sources[context], context, variant, phase, repeat)
    return plan


def audit_plan(plan, rows, checks):
    expected = expected_plan_core(rows)
    checks.equal(len(plan), MAX_REQUESTS, "Exactly30 prescribed request slots")
    checks.equal(len({r["id"] for r in plan}), MAX_REQUESTS, "Every prescribed slot has unique identity")
    for request, wanted in zip(plan, expected):
        for key, value in wanted.items():
            same(request.get(key), value, checks, request["id"]+"/literal plan/"+key, 0)
        if wanted["kind"] == "coordinate_swap":
            checks.require(not any(k in request for k in ("q10", "q14", "connection_file", "connection_sha256",
                            "parent_native_cache_file", "parent_native_cache_sha256")),
                           request["id"]+"/changed gap has no inherited endpoint proof")
    return expected


def audit_contract(spec, report, parent, original_rows, manifest, checks):
    checks.equal(spec["version"], STUDY_VERSION, "Bounded gap diagnostic version")
    unsigned = dict(spec); fingerprint = unsigned.pop("fingerprint")
    checks.equal(digest(unsigned), fingerprint, "Complete frozen diagnostic contract")
    checks.equal((report["version"], report["status"], report["phase"], report["contract_fingerprint"]),
                 (STUDY_VERSION, "completed", "completed", fingerprint), "All prescribed native requests completed")
    for key, value in dict(substeps=64, maximum_requests=MAX_REQUESTS, maximum_acquired_s=3600, contexts=["12", "24", "48"],
                           gap_delta_m=GAP_DELTA_M, repeats=REPEATS).items():
        checks.equal(spec[key], value, "Literal single correction/"+key)
    checks.require("development" in spec["scope"].lower() and "64" in spec["scope"], "Development-only fixed64 scope")
    for key in ("scene", "identity", "packages", "native_source_sha256"):
        same(spec[key], parent[key], checks, "Unchanged original physics/"+key, 0)
    checks.require(spec.get("training") is False and spec.get("noise") is False,
                   "No training or noise intervention")
    audit_plan(spec["plan"], original_rows, checks)
    wanted = dict(parent_report_sha256=PARENT_REPORT_SHA, parent_contract_sha256=manifest["contract.json"],
        parent_manifest_sha256=PARENT_MANIFEST_SHA, parent_audit_sha256=PARENT_AUDIT_SHA,
        parent_contract_fingerprint=parent["fingerprint"],
        parent_records_sha256={r["parent_record_id"]: r["parent_record_sha256"] for r in spec["plan"]})
    for key, value in wanted.items():
        same(spec["parent_bindings"][key], value, checks, "Exact parent provenance/"+key, 0)
    checks.equal(spec["parent_bindings"]["parent_rows_sha256"], manifest["records.jsonl"], "Exact original requested records")
    same(spec["parent_bindings"]["parent_evidence_sha256"], manifest, checks, "Complete parent evidence provenance", 0)


def audit_requests(study, parent_dir, rows, plan, parent, checks):
    from .calibrated_admission import verify_resolution, verify_connection
    checks.equal([r["id"] for r in rows], [r["id"] for r in plan], "All30 ordered slots retained")
    checks.equal(replay.read_records(study/"native/shots.jsonl"), rows, "Raw native request log retains every result")
    home = parent["identity"]["setup"]["home"]
    prior = read(parent_dir/"prior.json")["records"]
    for row, request in zip(rows, plan):
        label = row["id"]
        for key, value in request.items(): same(row.get(key), value, checks, label+"/unchanged request/"+key, 0)
        checks.require(type(row.get("native_release")) is bool and row.get("action_substituted") is False,
                       label+"/explicit native outcome without substitutions")
        path = contained(study/"native", row["context_resolution_file"])
        checks.equal(sha(path), row["context_resolution_sha256"], label+"/copied resolution bytes")
        checks.equal(sha(contained(parent_dir/"native", row["context_resolution_file"])), sha(path), label+"/original resolution unchanged")
        resolution = read(path); resolution.update(file=row["context_resolution_file"], sha256=sha(path), cache_hit=True)
        checks.require(verify_resolution(resolution, prior, home), label+"/source home resolution proof")
        path = contained(study/"native", row["connection_file"])
        checks.equal(sha(path), row["connection_sha256"], label+"/same-action connection bytes")
        connection = read(path)
        checks.require(verify_connection(connection, resolution, home, row["values"]), label+"/exact requested-action endpoint proof")
        if row["kind"] == "exact_replay":
            checks.equal(sha(contained(parent_dir/"native", request["connection_file"])), sha(path), label+"/original exact endpoint proof")
            checks.require(row["native_release"] and connection["passed"], label+"/original native endpoint actually replayed")
        else:
            checks.equal(row["kind"], "coordinate_swap", label+"/only prescribed changed gap")
            checks.equal(row["native_release"], bool(connection["passed"]), label+"/new gap respects strict geometry admission")
        if row["native_release"]:
            for key in ("q10", "q14"):
                checks.numeric(row[key], connection[key], label+"/admitted joints/"+key, 0)
                checks.numeric(row["native_initial_"+key], np.asarray(row[key], dtype=np.float32).astype(float), label+"/observed native joints/"+key, 0)
            checks.equal(row["contact_protocol"], dict(substeps=64, board_mode="articulated", gravity_mode="passive_only"), label+"/preserved native physics")
            checks.equal(row["scoring_version"], "sircus-calibrated-native-capture-v2", label+"/same scorer")
            checks.require(row.get("scalar_score_audit_passed") is True, label+"/online independent score passed")
            checks.equal(row["adapter_directory"], "native", label+"/native directory")
            checks.equal((row["effective_options_file"], row["effective_options_sha256"]),
                         ("effective-options.json", sha(study/"native/effective-options.json")), label+"/actual options file")
        else:
            checks.require(not row.get("route_success") and not row.get("invalid_physics") and bool(row.get("invalid")),
                           label+"/setup reject retained as failed attempt")
            checks.require(not any(k in row for k in ("native_cache_file", "native_cache_sha256", "native_initial_q10", "native_initial_q14")),
                           label+"/rejection has no fabricated native observation")
            missing = xy_audit.first_flight(parent["scene"], row["bucket_xy_m"], [0., 0., 1.],
                                           parent["identity"]["calibration"]["score_config"], {})
            same(row["first_flight"], missing, checks, label+"/rejected first flight unavailable", 0)


def audit_first_flights(study, rows, parent, checks):
    grouped = {}
    for row in rows:
        if row["native_release"]: grouped.setdefault(row["native_cache_file"], []).append(row)
    for filename, cohort in grouped.items():
        with np.load(contained(study/"native", filename), allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in ("position_m", "velocity_m_s", "forces_N", "time_s", "dt_s", "contact_distance_m")}
            for row in cohort:
                column = row["native_cache_row"]
                trace = {k: v[:, column] if v.ndim >= 2 and k != "time_s" else v for k, v in arrays.items()}
                expected = xy_audit.first_flight(parent["scene"], row["bucket_xy_m"], row["board_normal"],
                    parent["identity"]["calibration"]["score_config"], trace, valid=row.get("invalid_physics") is False)
                same(row["first_flight"], expected, checks, row["id"]+"/independent first flight")
                checks.numeric(row["trajectory_m"], arrays["position_m"][256::256, column], row["id"]+"/measured display trace", 0)
                checks.numeric(row["trajectory_dt_s"], 1/60, row["id"]+"/display cadence", 0)


def audit_report(report, rows, checks):
    same(report["counts"], native_counts(rows), checks, "All diagnostic counts reconstructed")
    gates = {key: native_control_gate([r for r in rows if r["phase"] == phase])
             for key, phase in (("pre", "controls_pre"), ("post", "controls_post"))}
    same(report["control_gates"], gates, checks, "Pre/post control outcomes reconstructed")
    checks.require(all(g["status"] == "passed" for g in gates.values()), "Both control gates passed")
    contexts = [dict(context_id=context, variants={variant: native_counts([r for r in rows if r["context_id"] == context and r["variant"] == variant])
                for variant in ("original", "gap_plus60")}) for context in CONTEXTS]
    same(report["contexts"], contexts, checks, "Every exact and changed-gap repeat accounted")
    elapsed = report["reported_worker_elapsed_s"]
    checks.require(type(elapsed) in (int, float) and 0 <= elapsed <= 3600, "Bounded worker wall time")
    return gates, contexts


def audit_sources(source, native_source, spec, checks):
    for root, key in ((source, "diagnostic_source_sha256"), (native_source, "native_source_sha256")):
        checks.require(isinstance(spec[key], dict) and bool(spec[key]), "Complete source binding/"+key)
        for name, expected in spec[key].items():
            checks.equal(sha(contained(root, name)), expected, "Actual executed source/"+key+"/"+name)
    own = "sircus/mjlab/gap_flight_correction_verify.py"
    checks.equal(spec["diagnostic_source_sha256"].get(own), sha(Path(__file__)), "This independent auditor is byte-bound")
    imported = {}
    for module in tuple(sys.modules.values()):
        name, filename = getattr(module, "__name__", ""), getattr(module, "__file__", None)
        if not filename or not name.startswith(("sircus.mjlab.", "sircus.isaac.")): continue
        path = Path(filename).resolve()
        root = source if path.is_relative_to(source) else native_source if path.is_relative_to(native_source) else None
        checks.require(root is not None, "Only explicit audited source roots imported/"+name)
        relative = str(path.relative_to(root))
        if len(Path(relative).parts) != 3 or path.suffix != ".py": continue
        expected = spec["diagnostic_source_sha256"].get(relative, spec["native_source_sha256"].get(relative))
        checks.require(expected is not None, "Every imported core dependency byte-bound/"+relative)
        checks.equal(sha(path), expected, "Actual imported dependency/"+relative)
        imported[relative] = sha(path)
    checks.require("sircus/mjlab/calibrated_score_oracle.py" in imported, "Independent scorer actually imported")
    checks.equal({k: importlib.metadata.version(k) for k in spec["packages"]}, spec["packages"], "Pinned CPU audit packages")
    return imported


def audit(study, parent_dir, parent_audit, source, native_source, assets):
    start = time.monotonic(); checks = replay.AuditChecks()
    study, parent_dir, parent_audit, source, native_source, assets = [Path(p).resolve() for p in
        (study, parent_dir, parent_audit, source, native_source, assets)]
    parent, _, original_rows, parent_manifest = audit_parent(parent_dir, parent_audit, checks)
    manifest = replay.check_manifest(study, checks)
    spec, report = read(study/"contract.json"), read(study/"report.json")
    same(report["evidence_sha256"], manifest, checks, "Completed report binds every artifact")
    audit_contract(spec, report, parent, original_rows, parent_manifest, checks)
    same(read(study/"source-bindings.json"), spec["parent_bindings"], checks, "Frozen consumed source binding file", 0)
    packet = read(study/"prediction-plan.json")
    same(packet["requests"], spec["plan"], checks, "Frozen packet equals contracted requests", 0)
    checks.equal(packet["contract_fingerprint"], spec["fingerprint"], "Frozen packet contract identity")
    checks.equal(report["plan_sha256"], sha(study/"prediction-plan.json"), "Completed report binds preoutcome plan")
    checks.require(packet["frozen_before_any_native_outcome"] is True and type(packet["frozen_unix_s"]) in (int, float), "Finite preoutcome freeze")
    if "acquired_unix_s" in report:
        checks.require(packet["frozen_unix_s"] <= report["acquired_unix_s"], "Plan frozen before GPU acquisition")
    for filename in ("calibration.json", "verification.json", "setup-home.json", "prior.json"):
        checks.equal(sha(study/filename), parent_manifest[filename], "Unchanged copied source input/"+filename)
    if "analysis_sha256" in spec:
        checks.equal(sha(study/"analysis.json"), spec["analysis_sha256"], "Preoutcome diagnostic analysis frozen")
        checks.equal(report["analysis_sha256"], spec["analysis_sha256"], "Completed report retains frozen diagnosis")
        checks.equal(spec["parent_bindings"]["analysis_sha256"], spec["analysis_sha256"], "Source binding retains frozen diagnosis")
    rows = replay.read_records(study/"records.jsonl")
    audit_requests(study, parent_dir, rows, spec["plan"], parent, checks)
    calibrated.check_rows(rows, checks)
    calibration = replay.check_calibration(study, spec["identity"]["calibration"], checks)
    from .learning_solve import required_asset_paths
    checks.equal({name: sha(contained(assets.parent, name)) for name in required_asset_paths(assets)},
                 spec["identity"]["asset_sha256"], "Actual unchanged robot collision assets")
    policy_audit.audit_native_options(study/"native", parent["scene"], calibration, assets, checks)
    traces = calibrated.rescore_rows(study/"native", rows, parent["scene"], calibration, checks,
        "Bounded gap correction64", parent["identity"]["trace_schema"], assets=assets)
    audit_first_flights(study, rows, parent, checks)
    gates, contexts = audit_report(report, rows, checks)
    dependencies = audit_sources(source, native_source, spec, checks)
    return dict(version=VERSION, status="passed", checks_count=checks.count, checks=checks.entries, failures=[],
        contract_fingerprint=spec["fingerprint"], report_sha256=sha(study/"report.json"), plan_sha256=sha(study/"prediction-plan.json"), evidence_manifest_sha256=sha(study/"evidence-manifest.json"),
        auditor_sha256=sha(Path(__file__)), producer_source_sha256=spec["diagnostic_source_sha256"], native_source_sha256=spec["native_source_sha256"],
        audit_dependencies_sha256=dependencies, source_parent=dict(report_sha256=PARENT_REPORT_SHA, manifest_sha256=PARENT_MANIFEST_SHA, audit_sha256=PARENT_AUDIT_SHA),
        requested_trials=len(rows), native_releases=len(traces), counts=native_counts(rows), control_gates=gates, contexts=contexts,
        elapsed_s=time.monotonic()-start, maximum_numeric_error=checks.numeric_error,
        limitations=["Post-hoc development on three original TEST failures; not fresh held-out policy evaluation.",
            "Gap increase changes both held-dropper clearance and impact energy; no single-mechanism causal claim.",
            "Three exact repeats test nominal reproducibility only. Fixed64, no noise or128-step validation.",
            "Held endpoints and sampled setup proofs do not validate driven arm trajectories or physical hardware.",
            "No retraining, policy activation or original scoring change."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("study", "parent", "parent-audit", "source", "native-source", "assets", "output"):
        parser.add_argument("--"+key, type=Path, required=True)
    args = parser.parse_args()
    if any(args.output.resolve().is_relative_to(p.resolve()) for p in (args.study, args.parent)):
        raise ValueError("Independent proof must remain outside immutable studies")
    try:
        result = audit(args.study, args.parent, args.parent_audit, args.source, args.native_source, args.assets)
    except Exception as error:
        args.output.write_text(json.dumps(dict(version=VERSION, status="failed", failures=[type(error).__name__+": "+str(error)]), indent=2)+"\n")
        raise
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: result[k] for k in ("status", "checks_count", "requested_trials", "native_releases")}))


if __name__ == "__main__": main()
