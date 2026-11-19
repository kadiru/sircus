"""Independent CPU verification of the five previously unselected30° shots.

This module only reads frozen native observations. It retains the calibrated
capture oracle, literal source targets and exact selected joint branches.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np

from .benchmark_verify import EvidenceError, contained, digest, read, sha
from . import clean650_verify as clean_audit
from . import replay650_verify as replay
from . import xy650_verify as xy_audit

VERSION = "sircus-tilt30-650-independent-audit-v1"
STUDY_VERSION = "sircus-tilt30-650-verification-v1"
PARENT_FP = "7db7440c12005c39808ee4dfbf831d8b81000cda048a8d4eddc896d9b1d31137"
PARENT_AUDIT_SHA = "9a605b7b8fbb1cb99db966ae92f6488aab0895a5db4bc9116b05e5a7fdc0b439"
SOURCE_IDS = tuple(f"coarse-64-xy-grid-{index:04d}-r0" for index in (43, 70, 96, 151, 177))
SOURCE_CONFIGS = (([-.01, .06], .41), ([-.01, .08], .41), ([0., .04], .39),
                  ([0., .08], .41), ([.01, .04], .39))
STEPS = (64, 128)
MAX_REQUESTS = 54
AuditChecks = replay.AuditChecks


def audit_parent(study, audit_path, bindings, checks):
    """Bind one complete already-audited parent without copying its checklist."""
    expected_names = {"contract.json", "report.json", "records.jsonl", "groups.json", "initial-groups.json", "evidence-manifest.json"}
    checks.equal(set(bindings), {"parent", "parent_audit_sha256"}, "Only explicit parent and corrected external proof bindings")
    checks.equal(set(bindings["parent"]), expected_names, "Every exact parent input declared")
    for name, expected in bindings["parent"].items():
        checks.equal(sha(contained(study, name)), expected, "Parent exact input bytes/"+name)
    actual_audit_sha = sha(audit_path)
    checks.equal(bindings["parent_audit_sha256"], actual_audit_sha, "Explicit external parent proof bytes")
    checks.equal(actual_audit_sha, PARENT_AUDIT_SHA, "Exact previously verified external audit-v2")
    spec, report, proof = read(study/"contract.json"), read(study/"report.json"), read(audit_path)
    unsigned = dict(spec);fingerprint = unsigned.pop("fingerprint", None)
    checks.equal(digest(unsigned), PARENT_FP, "Complete parent contract identity")
    checks.equal(fingerprint, PARENT_FP, "Exact translated development parent")
    checks.equal(spec["version"], "sircus-xy650-shift-study-v1", "Historical XY protocol unchanged")
    checks.equal((report["version"], report["status"], report["phase"]),
                 (spec["version"], "completed", "completed"), "Parent native development complete")
    checks.equal(proof["version"], "sircus-xy650-independent-audit-v2", "Parent correction explicitly preserved")
    checks.equal(proof["status"], "passed", "Parent independent audit passed")
    checks.require(not proof.get("failures") and proof["checks_count"] == len(proof["checks"]) == 80138
                   and all(check.get("passed") is True for check in proof["checks"]), "Full80138 parent checks strictly passed")
    checks.equal(proof["report_sha256"], sha(study/"report.json"), "Parent proof binds report bytes")
    checks.equal(proof["evidence_manifest_sha256"], sha(study/"evidence-manifest.json"), "Parent proof binds all native evidence")
    checks.equal(proof["contract_fingerprint"], PARENT_FP, "Parent proof exact contract")
    checks.equal(report["contract_fingerprint"], PARENT_FP, "Parent report exact contract")
    checks.equal(proof["producer_source_sha256"], spec["source_sha256"], "Parent proof binds exact producer sources")
    checks.equal(proof["auditor_identity"]["original_producer_auditor_sha256"], spec["source_sha256"]["sircus/mjlab/xy650_verify.py"],
                 "Parent original and revised checker identities remain separate")
    manifest = replay.check_manifest(study, checks)
    checks.equal(report["evidence_sha256"], manifest, "Parent report complete evidence binding")
    checks.equal((proof["requested_trials"], proof["native_releases"]), (388, 228), "Historical parent denominators unchanged")
    checks.equal(proof["gate"], report["gate"], "Parent qualification unchanged")
    rows = replay.read_records(study/"records.jsonl")
    checks.equal(len(rows), 388, "Complete parent ledger retained")
    return spec, report, rows, read(study/"groups.json")


def select_sources(rows, parent_report, checks):
    lookup = {row["id"]: row for row in rows}
    chosen = [lookup[ident] for ident in SOURCE_IDS]
    eligible = sorted((row for row in rows if row["phase"] == "coarse" and row["substeps"] == 64
                       and row["values"][3] == 30. and xy_audit.native_valid(row) and replay.captured(row)), key=lambda row: row["id"])
    checks.equal([row["id"] for row in eligible], list(SOURCE_IDS), "All and only the five30-degree coarse captures selected")
    previous = {xy_audit.target_key(row) for row in parent_report["verification"]}
    for source, (translation, gap) in zip(chosen, SOURCE_CONFIGS):
        label = source["id"]
        checks.require(xy_audit.native_valid(source) and replay.captured(source), label+"/actual valid parent capture")
        checks.equal(source["route_quality"], 1, label+"/historical rim label retained")
        checks.equal(source["rim_band_contact"], True, label+"/actual historical rim contact")
        checks.equal(source["phase"], "coarse", label+"/original coarse observation")
        checks.equal(source["substeps"], 64, label+"/original64-only observation")
        checks.equal(source["translation_m"], translation, label+"/literal selected translated layout")
        checks.equal(source["values"][2:], [.38, 30., gap], label+"/literal unselected30-degree recipe")
        checks.require(xy_audit.target_key(source) not in previous, label+"/not one of the earlier verified targets")
    checks.equal(len({xy_audit.target_key(row) for row in chosen}), 5, "Five distinct prescribed target geometries")
    return chosen


def audit_groups(groups, selected, parent_controls, scene_fp, checks):
    checks.equal(len(groups), 7, "Two exact controls plus five prescribed verification targets")
    for actual, source in zip(groups[:2], parent_controls):
        for key in ("id", "role", "phase", "query_kind", "values", "bucket_xy_m", "q10", "q14", "source_record_id", "source_record_sha256", "translation_m", "query_fingerprint"):
            checks.equal(actual[key], source[key], "Historical control remains exact/"+key)
    checks.equal(len(selected), 5, "All and only five previously unselected30-degree captures")
    for index, (group, source) in enumerate(zip(groups[2:], selected)):
        label = f"tilt30-{index:02d}"
        checks.equal(group["id"], label, "Literal target ordering")
        checks.equal(group["role"], "verification", label+"/nominal verification role")
        checks.equal(group["phase"], "verification", label+"/fixed phase")
        checks.equal(group["query_kind"], "nominal_verification", label+"/exact replay query")
        checks.require(group["freeze_joints"] is True, label+"/source joint branch cannot be solved again")
        for key in ("values", "bucket_xy_m", "q10", "q14", "translation_m"):
            checks.equal(group[key], source[key], label+"/exact original target and requested joints/"+key)
        checks.equal(group["source_record_id"], source["id"], label+"/exact parent record ID")
        checks.equal(group["source_record_sha256"], digest(source), label+"/complete parent record bytes")
        checks.equal(group["source_query_fingerprint"], source["query_fingerprint"], label+"/historical query provenance")
        checks.equal(group["parent_route_quality"], 1, label+"/original rim source class")
    for group in groups:
        checks.equal(group["query_fingerprint"], xy_audit.query_fingerprint(group, scene_fp), group["id"]+"/new query identity")


def qualification(rows, positions, *, controls_passed, parent_source_capture):
    """Fresh nominal evidence qualifies capture and clean tiers independently."""
    present = len(rows) == 6 and all(sum(row["substeps"] == step for row in rows) == 3 for step in STEPS)
    capture_both = present and all(replay.captured(row) for row in rows)
    clean_both = present and all(xy_audit.clean(row) for row in rows)
    physics = present and all(xy_audit.native_valid(row) for row in rows)
    oracle = present and all(row.get("scalar_score_audit_passed") is True for row in rows)
    drift = replay.paired_sensitivity(rows, positions)["nominal_position_delta_350ms_max_m"] if present else None
    common = bool(controls_passed and parent_source_capture and physics and oracle and drift is not None and np.isfinite(drift) and drift <= .005)
    clean_ok, capture_ok = bool(common and clean_both), bool(common and capture_both)
    return dict(status="pending" if not present else "qualified" if clean_ok else "capture_only" if capture_ok else "failed",
                qualified=clean_ok, capture_qualified=capture_ok, clean_qualified=clean_ok,
                clean_both=bool(clean_both), capture_both=bool(capture_both), physics_valid=bool(physics),
                oracle_agreement=bool(oracle), max_drift_m=drift, parent_source_capture=bool(parent_source_capture))


def audit_settings(settings, checks):
    expected = dict(source_count=5, source_ids=list(SOURCE_IDS), repeats_per_resolution=3, substeps=list(STEPS),
                    control_requests=24, verification_requests=30, max_requests=54, minimum_clearance_m=.005, max_drift_350ms_m=.005)
    for key, value in expected.items():
        checks.equal(settings[key], value, "Literal fixed five-recipe protocol/"+key)
    checks.require("no" in settings["scope"].lower() and "noise" in settings["scope"].lower(), "No noise qualification claimed")


def audit_report(report, rows, groups, selected, positions, pre, post, scene, checks):
    stat = clean_audit.audit_statistics
    stat(report["counts"], rows, checks, "Complete54-request accounting")
    for field, key in (("phase_counts", "phase"), ("substep_counts", "substeps")):
        checks.equal(report["counts"][field], dict(Counter(str(row[key]) for row in rows)), "All phase/resolution denominators/"+field)
    checks.equal(report["control_gates"], dict(controls_pre=pre, controls_post=post), "Before/after independent control gates")
    checks.equal(report["top_settings"], [], "No optimization, ranking or selected best-trial display")
    checks.equal(len(report["groups"]), len(groups), "All requested groups visible")
    for actual, group in zip(report["groups"], groups):
        xy_audit.audit_group_metadata(actual, group, scene, checks, group["id"])
        for step in STEPS:
            stat(actual["by_substeps"][str(step)], [row for row in rows if row["group_id"] == group["id"] and row["substeps"] == step],
                 checks, group["id"]+"/actual outcomes/"+str(step))
    keys = [(phase, group["id"], group["role"], step)
            for phase in ("controls_pre", "controls_post") for group in groups[:2] for step in STEPS]
    checks.equal([(row["phase"], row["group_id"], row["role"], row["substeps"]) for row in report["controls"]], keys,
                 "Both control groups/resolutions before and after verification")
    for actual in report["controls"]:
        cohort = [row for row in rows if all(row[key] == actual[key] for key in ("phase", "group_id", "substeps"))]
        stat(actual["stats"], cohort, checks, "Actual control cohort")
    checks.equal(len(report["verification"]), 5, "Every prescribed target reported regardless of outcome")
    clean_count, capture_count = 0, 0
    for actual, group, source in zip(report["verification"], groups[2:], selected):
        xy_audit.audit_group_metadata(actual, group, scene, checks, group["id"])
        checks.equal(actual["parent_route_quality"], 1, group["id"]+"/parent rim source label retained")
        cohort = [row for row in rows if row["group_id"] == group["id"]]
        for step in STEPS:
            stat(actual["by_substeps"][str(step)], [row for row in cohort if row["substeps"] == step], checks,
                 group["id"]+"/matched fresh repeats/"+str(step))
        replay.check_pairing(cohort, checks)
        checks.equal(actual["sensitivity"], replay.paired_sensitivity(cohort, positions), group["id"]+"/matched joint/action and position sensitivity")
        result = qualification(cohort, positions, controls_passed=pre["status"] == post["status"] == "passed", parent_source_capture=replay.captured(source))
        checks.equal(actual["qualification"], result, group["id"]+"/independent fresh capture and clean qualifications")
        clean_count += int(result["clean_qualified"])
        capture_count += int(result["capture_qualified"])
    status = "blocked_controls" if pre["status"] != "passed" or post["status"] != "passed" else "passed" if clean_count else "capture_only" if capture_count else "failed"
    gate = dict(status=status, controls_pre_passed=pre["status"] == "passed", controls_post_passed=post["status"] == "passed",
                qualified_settings=clean_count, capture_qualified_settings=capture_count, more_ppo_started=False)
    checks.equal(report["gate"], gate, "Independent final clean and capture-tier gate")
    return gate


def auditor_identity(producer_source, producer_hashes, checks):
    root = Path(__file__).resolve().parents[2]
    own = "sircus/mjlab/tilt650_verify.py"
    dependencies = {}
    for module in tuple(sys.modules.values()):
        name, filename = getattr(module, "__name__", ""), getattr(module, "__file__", None)
        if not filename or not (name.startswith("sircus.mjlab.") or name.startswith("sircus.isaac.")):
            continue
        path = Path(filename).resolve()
        checks.require(path.is_relative_to(root), "One explicit auditor source snapshot/"+name)
        relative = str(path.relative_to(root))
        if len(Path(relative).parts) != 3 or path.suffix != ".py" or relative == own:
            continue
        if relative.startswith("sircus/isaac/") and relative not in producer_hashes:
            continue
        checks.require(relative in producer_hashes, "Actual imported dependency declared/"+relative)
        actual = sha(path)
        checks.equal(actual, producer_hashes[relative], "Unchanged imported audit dependency/"+relative)
        dependencies[relative] = actual
    checks.require("sircus/mjlab/calibrated_score_oracle.py" in dependencies, "Independent calibrated oracle retained")
    checks.equal(sha(contained(producer_source, own)), producer_hashes[own], "Original native checker bytes retained")
    return dict(version=VERSION, auditor_sha256=sha(Path(__file__)),
                original_producer_auditor_sha256=producer_hashes[own], dependencies_sha256=dependencies)


def audit(study, source, assets, *, parent_study, parent_audit):
    started, checks = time.monotonic(), AuditChecks()
    study, source, assets, parent_study, parent_audit = map(Path, (study, source, assets, parent_study, parent_audit))
    manifest = replay.check_manifest(study, checks)
    spec, report = read(study/"contract.json"), read(study/"report.json")
    unsigned = dict(spec);fingerprint = unsigned.pop("fingerprint", None)
    checks.equal(digest(unsigned), fingerprint, "Complete immutable follow-up contract")
    checks.equal(spec["version"], STUDY_VERSION, "New finite30-degree verification version")
    checks.equal(report["version"], STUDY_VERSION, "Report belongs to this fixed verification")
    checks.equal((report["status"], report["phase"]), ("completed", "completed"), "Finite fresh verification completed")
    checks.equal(report["contract_fingerprint"], fingerprint, "Report exact full contract")
    checks.equal(spec["evidence_kind"], "native_calibrated_tilt30_verification", "Native calibrated held-endpoint evidence")
    checks.equal(report["evidence_sha256"], manifest, "Report binds every raw artifact")
    replay.check_source_files(source, spec["source_sha256"], checks, "Immutable actual native source")
    paths = list((source/"sircus/mjlab").glob("*.py")) + [source/"sircus/mjlab/requirements.txt"]
    paths += [source/"sircus/isaac"/name for name in ("geometry.py", "rl_contract.py", "entry_quality.py", "board_route.py")]
    checks.equal(set(spec["source_sha256"]), {str(path.relative_to(source)) for path in paths}, "All actual execution/audit modules declared")
    from .learning_solve import required_asset_paths
    actual_assets = {name: sha(contained(assets.parent, name)) for name in required_asset_paths(assets)}
    checks.equal(spec["asset_sha256"], actual_assets, "Exact actual native collision assets")
    replay.audit_packages(spec["packages"], checks)
    calibration = replay.check_calibration(study, spec["calibration"], checks)
    parent, parent_report, parent_rows, _ = audit_parent(parent_study, parent_audit, spec["input_bindings"], checks)
    for key in ("scene", "scene_fingerprint", "calibration", "packages", "asset_sha256"):
        checks.equal(spec[key], parent[key], "Unchanged audited parent/"+key)
    checks.equal(spec["parent_contract_fingerprint"], PARENT_FP, "Contract preserves parent identity")
    checks.equal(report["parent_contract_fingerprint"], PARENT_FP, "Display preserves parent identity")
    selected = select_sources(parent_rows, parent_report, checks)
    checks.equal(read(study/"source-selection.json"), selected, "Complete literal parent records frozen before fresh outcomes")
    frozen = {"initial-groups.json", "initial-geometry.json", "initial-plan.json", "source-selection.json", "calibration.json", "verification.json"}
    checks.equal(set(spec["frozen_files_sha256"]), frozen, "All six pre-outcome source/geometry/plan inputs frozen")
    for name, expected in spec["frozen_files_sha256"].items():
        checks.equal(sha(contained(study, name)), expected, "Exact immutable pre-outcome bytes/"+name)
    audit_settings(spec["settings"], checks)
    checks.equal(report["settings"], spec["settings"], "Report exact prescribed protocol")
    groups, proofs = read(study/"initial-groups.json"), read(study/"initial-geometry.json")
    controls = [group for group in read(parent_study/"initial-groups.json") if group["role"].endswith("_control")]
    checks.equal([group["role"] for group in controls], ["positive_control", "negative_control"], "Exact ordered historical control pair")
    audit_groups(groups, selected, controls, spec["scene_fingerprint"], checks)
    checks.equal(read(study/"groups.json"), groups, "No conditional or changed group after outcomes")
    checks.equal(read(study/"geometry.json"), proofs, "No conditional or changed geometry proof")
    clean_audit.audit_geometry(groups, proofs, spec["scene"], assets, checks)
    for index, proof in enumerate(proofs):
        checks.equal(proof["source_mode"], "exact_control" if index < 2 else "frozen_verification", "No branch search during exact replay")
        checks.equal(proof["preserved_source"], True, "Historical joint branch preserved")
        checks.require(len(proof["attempts"]) <= 1, "No alternate-branch rescue or reselection")
    before = xy_audit.make_plan(groups[:2], "controls_pre", steps=STEPS, repeats=3)
    verify = xy_audit.make_plan(groups[2:], "verification", steps=STEPS, repeats=3)
    after = xy_audit.make_plan(groups[:2], "controls_post", steps=STEPS, repeats=3)
    checks.equal(read(study/"initial-plan.json"), dict(controls_pre=before, verification=verify, controls_post=after), "All54 requests frozen in exact order")
    checks.require(not (study/"adaptive-stages.json").exists(), "No adaptive selection or refinement stage")
    rows = replay.read_records(study/"records.jsonl")
    checks.require(12 <= len(rows) <= MAX_REQUESTS, "At most54 fresh requests, every reject retained")
    checks.equal([row["id"] for row in rows[:12]], [row["id"] for row in before], "Original controls precede all fresh target drops")
    pre = clean_audit.control_gate(rows[:12])
    checks.equal(read(study/"initial-control-gate.json"), pre, "Independent initial control gate")
    expected = before+verify+after if pre["status"] == "passed" else before
    clean_audit.audit_native_rows(rows, expected, groups, checks)
    xy_audit.audit_fingerprints(spec, report, groups, rows, selected[0], checks)
    replay.audit_batches(study, rows, checks)
    post = clean_audit.control_gate([row for row in rows if row["phase"] == "controls_post"])
    if pre["status"] != "passed":
        post["status"] = "not_run"
    checks.equal(read(study/"final-control-gate.json"), post, "Independent final control gate")
    actual_steps = {row["substeps"] for row in rows if row["native_release"]}
    checks.require(actual_steps in (set(), set(STEPS)), "Both resolutions allocated or wholly rejected controls")
    if actual_steps:
        options = read(study/"effective-options.json")
        replay.audit_options(options, checks)
        replay.audit_compiled(study, spec["scene"], assets, calibration, options, checks)
        positions, traces = replay.audit_native(study, rows, spec["scene"], calibration, assets, checks)
    else:
        checks.require(not (study/"effective-options.json").exists(), "No fabricated native options")
        checks.require(not list((study/"native").rglob("*.npz")), "No fabricated native observations")
        positions, traces = {}, {}
    xy_audit.audit_first_flight(study, rows, spec["scene"], calibration, checks)
    gate = audit_report(report, rows, groups, selected, positions, pre, post, spec["scene"], checks)
    xy_audit.audit_traces(study, report, rows, groups, spec["scene"], checks)
    identity = auditor_identity(source, spec["source_sha256"], checks)
    return dict(version=VERSION, status="passed", contract_fingerprint=fingerprint, parent_contract_fingerprint=PARENT_FP,
                scene_fingerprint=spec["scene_fingerprint"], checks_count=checks.count, checks=checks.entries, failures=[],
                auditor_sha256=sha(Path(__file__)), auditor_identity=identity, producer_source_sha256=spec["source_sha256"],
                report_sha256=sha(study/"report.json"), evidence_manifest_sha256=sha(study/"evidence-manifest.json"),
                requested_trials=len(rows), native_releases=len(traces), gate=gate, elapsed_s=time.monotonic()-started,
                limitations=["Five exact previously observed development captures, not held-out policy evaluation or workspace coverage.",
                             "Fresh nominal capture and clean tiers are separate; parent rim labels remain unchanged.",
                             "Calibrated simulated contacts and held endpoints only; no noise, grasp/transit, actuator, perception or physical validation."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "source", "assets", "parent-study", "parent-audit", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.study.resolve()):
        raise ValueError("This independent proof must be saved outside the frozen native evidence")
    try:
        result = audit(args.study, args.source, args.assets, parent_study=args.parent_study, parent_audit=args.parent_audit)
    except (EvidenceError, ValueError, KeyError, OSError) as error:
        result = dict(version=VERSION, status="failed", failures=[str(error)])
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        raise
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: result[key] for key in ("status", "checks_count", "requested_trials", "native_releases", "gate")}))


if __name__ == "__main__":
    main()
