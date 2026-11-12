"""Independent CPU audit of the finite native64 corner capture sweep.

The literal proposal/selection schedule is reconstructed here without importing
the runner. Saved dense contacts are rescored under both historical v2 and v3;
no physics steps, IK search, GPU lease, or mutation of original evidence occurs.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from . import fixed35_imitation_verify as geometry
from . import calibrated_benchmark_verify as admission
from . import coverage_training_verify as setup_namespace
from . import policy64_verify as models
from . import replay650_verify as replay
from . import table_training_verify as native
from .benchmark_verify import EvidenceError, contained, digest, read, sha

VERSION = "corner-sweep-audit-v1"
CONTEXT = "fixed35-dev-grid-00-00"
META = ("source_hint", "effective_source_hint", "source_teacher_id",
        "context_resolution_file", "context_resolution_sha256", "context_home_sha256",
        "context_layout_eligible", "context_anchor_admitted", "context_hard_reject", "observation")
PARENT_FILES = ("report.json", "rows.jsonl", "evidence-manifest.json", "contract.json")
LOW, HIGH = (.30, -.50, .12, 8., .05), (1.04, .35, .60, 60., .65)
AuditChecks = replay.AuditChecks


def fingerprint(value, checks, label):
    unsigned = dict(value)
    actual = unsigned.pop("fingerprint")
    checks.equal(digest(unsigned), actual, label + "/fingerprint")
    return actual


def request(source, identity, phase, method, values=None, **extra):
    values = copy.deepcopy(source["values"] if values is None else values)
    return dict({k: copy.deepcopy(source[k]) for k in META}, id=identity, phase=phase,
        role="development", method=method, context_id=source["context_id"],
        bucket_xy_m=copy.deepcopy(source["bucket_xy_m"]), values=values,
        command_values=copy.deepcopy(values), parent_row_id=source["id"],
        parent_row_sha256=digest(source), substeps=64, original_role=source.get("role"),
        development_reclassified=True, selection_or_learning=False,
        perturbation_mixed_units=[0.] * 5, **extra)


def independent_plan(seed, controls):
    """Thirty signed axis offsets, then sixteen ordered tilt/gap combinations."""
    search = []
    offsets = []
    for axis, name in enumerate(("board_x", "board_y", "height", "tilt", "gap")):
        for size in ((.5, 1.5, 3.) if axis == 3 else (.005, .015, .030)):
            for sign in (-1, 1):
                delta = [0.] * 5
                delta[axis] = sign * size
                offsets.append((name, delta))
    for tilt in (-3., -1., 1., 3.):
        for gap in (-.025, -.010, .010, .025):
            offsets.append(("tilt_gap", [0., 0., 0., tilt, gap]))
    for index, (family, delta) in enumerate(offsets):
        values = [v + d for v, d in zip(seed["values"], delta)]
        if not all(lo <= v <= hi for lo, v, hi in zip(LOW, values, HIGH)):
            raise EvidenceError("A literal corner proposal exceeds original action bounds")
        row = request(seed, f"corner-search-{index:02d}", "search", "corner_sweep",
                      values=values, family=family)
        row["perturbation_mixed_units"] = delta
        search.append(row)
    exact = [request(seed, f"corner-{phase}-{i}", phase, "seed_replay", exact_parent_action=True)
             for phase in ("seed_pre", "seed_post") for i in range(3)]
    control_rows = [dict({k: copy.deepcopy(s[k]) for k in META},
        id=f"corner-{phase}-{s['role']}-{i}", phase=phase, method="control", role=s["role"],
        context_id=s["context_id"], bucket_xy_m=s["bucket_xy_m"], values=s["values"],
        command_values=s["values"], source_record_id=s["id"], source_record_sha256=digest(s),
        variant="control_exact", repeat=i)
        for phase in ("controls_pre", "controls_post") for s in controls for i in range(3)]
    return dict(seed=exact, search=search, controls=control_rows, repeat_count=3, repeat_limit=6)


def independent_repeats(seed, search_rows):
    captures = [r for r in search_rows if native.captured(r)]
    selected, families = [], set()
    for row in captures:
        if row["family"] not in families:
            families.add(row["family"])
            selected.append(row)
    selected = (selected + [r for r in captures if r not in selected])[:6]
    repeats = [request(seed, f"{r['id']}-repeat-{i}", "repeat", "corner_repeat", values=r["values"],
        selected_from_id=r["id"], selected_from_sha256=digest(r), selected_q10=r["q10"],
        selected_q14=r["q14"], selected_connection_sha256=r["connection_sha256"], repeat=i)
        for r in selected for i in range(3)]
    return dict(rows=repeats, selected_from_ids=[r["id"] for r in selected])


def authenticate_parent(study, parent, audit_path, spec, checks):
    wanted = {name: sha(parent / name) for name in PARENT_FILES}
    wanted["audit"] = sha(audit_path)
    checks.equal(spec["parent_sha256"], wanted, "Exact unchanged parent inputs and audit")
    report, proof, contract, manifest = map(read, (parent / "report.json", audit_path,
        parent / "contract.json", parent / "evidence-manifest.json"))
    checks.equal(report["status"], "completed", "Completed parent experiment")
    checks.equal((proof["status"], proof["full_checks_verified"], proof["native_checks_verified"]),
        ("passed", True, True), "Full passing independent parent audit")
    checks.require(not proof.get("failures"), "No parent audit failures")
    checks.require(type(proof.get("checks_count")) is int and proof["checks_count"] > 0,
                   "Positive actual independent parent checks")
    checks.equal(len(proof["checks"]), proof["checks_count"], "Complete parent audit check ledger")
    checks.require(all(c.get("passed") is True for c in proof["checks"]), "Every original parent check passed")
    checks.equal(proof["report_sha256"], wanted["report.json"], "Parent audit binds report")
    checks.equal(proof["evidence_manifest_sha256"], wanted["evidence-manifest.json"], "Parent audit binds evidence")
    # The six-hour parent embeds the complete manifest; this small successor
    # instead records its file hash. Accept either explicit, exact binding.
    if "evidence_manifest_sha256" in report:
        checks.equal(report["evidence_manifest_sha256"], wanted["evidence-manifest.json"], "Parent report binds evidence hash")
    else:
        checks.equal(report["evidence_sha256"], manifest, "Parent report binds complete evidence manifest")
    parent_fp = fingerprint(contract, checks, "Parent")
    checks.equal(proof["contract_fingerprint"], parent_fp, "Parent audit binds contract")
    checks.equal(report["contract_fingerprint"], parent_fp, "Parent report binds contract")
    checks.equal(spec["identity"], contract["identity"], "Unchanged parent experiment identity")
    checks.equal(spec["scene"], contract["scene"], "Unchanged parent scene")

    def bound(name):
        path = contained(parent, name)
        checks.equal(sha(path), manifest[name], "Parent consumed evidence/" + name)
        return path

    for name in ("rows.jsonl", "contract.json", "setup-home.json", "prior.json", "old-prior.json",
                 "control-replays.json", "native-yaw/effective-options.json", "native-yaw/native-compiled-geometry.json"):
        bound(name)
    for name in ("calibration.json", "verification.json"):
        checks.equal(sha(study / name), sha(bound(name)), "Unchanged copied " + name)
    parents = replay.read_records(bound("rows.jsonl"))
    found = [r for r in parents if r["context_id"] == CONTEXT and r["phase"] == "preserved" and native.captured(r)]
    checks.equal(len(found), 1, "Exactly one preserved corner capture")
    seed = found[0]
    checks.equal((spec["parent_seed_id"], spec["parent_seed_sha256"]),
                 (seed["id"], digest(seed)), "Exact authenticated corner seed")
    checks.equal((seed["bucket_xy_m"], spec["bucket_xy_m"]), ([.1, -.65], [.1, -.65]), "Fixed corner bucket")
    controls = read(bound("control-replays.json"))
    checks.equal([r["role"] for r in controls], ["direct_control", "table_control", "negative_control"],
                 "Three original formal control sources")
    for source, namespace in [(seed, "native-yaw")] + [(s, "control-parent/native") for s in controls]:
        for prefix in ("context_resolution", "connection"):
            checks.equal(sha(bound(namespace + "/" + source[prefix + "_file"])),
                         source[prefix + "_sha256"], "Parent exact proof/" + source["id"] + "/" + prefix)
    return seed, controls, bound


def audit_packets(study, rows, plan, repeat_plan, checks):
    groups = [("controls_pre", [r for r in plan["controls"] if r["phase"] == "controls_pre"]),
              ("seed_pre", plan["seed"][:3])]
    groups += [("search", plan["search"][start:start + 8]) for start in range(0, 46, 8)]
    if repeat_plan["rows"]:
        groups.append(("repeat", repeat_plan["rows"]))
    groups += [("seed_post", plan["seed"][3:]),
               ("controls_post", [r for r in plan["controls"] if r["phase"] == "controls_post"])]
    checks.equal(sorted(p.name for p in (study / "requests").glob("*.json")),
                 [f"batch-{i:03d}.json" for i in range(len(groups))], "Only prescribed request packets")
    expected_ids = [r["id"] for _, group in groups for r in group]
    checks.equal([r["id"] for r in rows], expected_ids, "Complete chronological requests including setup declines")
    checks.equal(len(set(expected_ids)), len(expected_ids), "Unique paid request identities")
    cursor = 0
    for index, (phase, expected) in enumerate(groups):
        name = f"requests/batch-{index:03d}.json"
        checks.equal(read(study / name), dict(phase=phase, rows=expected), name + "/independently reconstructed packet")
        for row, request_row in zip(rows[cursor:cursor + len(expected)], expected):
            for key, value in request_row.items():
                checks.equal(row[key], value, row["id"] + "/request field/" + key)
            checks.equal((row["request_packet"], row["request_packet_sha256"]),
                         (name, sha(study / name)), row["id"] + "/frozen packet bytes")
            checks.equal(row["adapter_directory"], "native-controls" if phase.startswith("controls_") else "native-yaw",
                         row["id"] + "/explicit native namespace")
        cursor += len(expected)


def exact_branch(row, source, checks):
    for key in (*META, "values", "bucket_xy_m", "q10", "q14", "connection_file", "connection_sha256", "native_release"):
        checks.equal(row.get(key), source.get(key), row["id"] + "/exact recorded branch/" + key)
    if row.get("native_release"):
        for key in ("q10", "q14"):
            checks.equal(row["native_initial_" + key], np.asarray(source[key], np.float32).astype(float).tolist(),
                         row["id"] + "/observed exact native reset/" + key)


def check_rows(rows, checks):
    """This experiment records native reward, not a learner-specific field."""
    for row in rows:
        label = row["id"]
        values = np.asarray(row["values"], float)
        checks.require(values.shape == (5,) and np.isfinite(values).all(), label + "/finite exact action")
        checks.equal(row["values"], row["command_values"], label + "/requested action unchanged")
        checks.equal(row["action_substituted"], False, label + "/no action substitution")
        if row.get("native_release"):
            checks.numeric(row["reward"], native.oracle.reward(row), label + "/independent recorded native reward", 1e-8)
            for key, expected in (("scoring_version", native.oracle.VERSION),
                                  ("route_version", native.oracle.ROUTE_VERSION),
                                  ("reward_version", native.oracle.REWARD_VERSION)):
                checks.equal(row[key], expected, label + "/explicit v3/" + key)
            checks.equal(row["contact_protocol"], dict(substeps=64, board_mode="articulated", gravity_mode="passive_only"),
                         label + "/unchanged native64 contact protocol")
            checks.equal(row["scalar_score_audit_passed"], True, label + "/online oracle agreement")
            checks.equal(row["score_adapter_version"], "sircus-table-assisted-held-endpoint-v3", label + "/explicit scorer adapter")
        else:
            checks.require(bool(row.get("invalid")) and not native.captured(row), label + "/setup decline counted")
        if native.captured(row):
            checks.require(row.get("board_bounce") and row.get("whole_ball_entry")
                and row.get("max_retention_s", 0) >= 1 - 1e-9 and not row.get("forbidden_contact_steps"), label + "/complete capture route")
            checks.equal(int(row.get("direct_capture")) + int(row.get("table_assisted_capture")), 1, label + "/capture route partition")


def audit(study, parent_study, parent_audit, assets):
    study, parent, audit_path, assets = map(lambda p: Path(p).resolve(), (study, parent_study, parent_audit, assets))
    started = time.monotonic()
    checks = AuditChecks()
    spec, report = read(study / "contract.json"), read(study / "report.json")
    checks.equal(spec["version"], "corner-sweep-v1", "Explicit corner study")
    fingerprint(spec, checks, "Corner")
    checks.equal((report["status"], report["phase"]), ("completed", "completed"), "Completed finite sweep")
    checks.require(type(spec["wall_time_limit_s"]) is int and 60 <= spec["wall_time_limit_s"] <= 1800, "Finite worker cap")
    checks.require(math.isfinite(report["elapsed_s"]) and 0 < report["elapsed_s"] <= spec["wall_time_limit_s"], "Reported elapsed within worker cap")
    checks.equal(spec["substeps"], 64, "Canonical64 only")
    # The parent's full v2 schema also documents array shapes and interval
    # alignment. Preserve it through exact parent-identity equality below while
    # checking every literal cadence/horizon/category field in the short helper.
    for key, expected in replay.native_trace_spec(64).items():
        checks.equal(spec["identity"]["trace_schema"].get(key), expected, "Unchanged dense64 clock/" + key)
    checks.equal(digest(spec["scene"]), spec["identity"]["scene_sha256"], "Authenticated scene geometry")
    for name, expected in spec["identity"]["asset_sha256"].items():
        checks.equal(sha(contained(assets.parent, name)), expected, "Exact native asset/" + name)
    replay.check_manifest(study, checks)
    checks.equal(report["evidence_manifest_sha256"], sha(study / "evidence-manifest.json"), "Report binds complete frozen evidence")
    seed, controls, bound = authenticate_parent(study, parent, audit_path, spec, checks)
    calibration = replay.check_calibration(study, spec["identity"]["calibration"], checks)
    plan = independent_plan(seed, controls)
    checks.equal(read(study / "plan.json"), plan, "Entire literal frozen corner plan")
    checks.equal(sha(study / "plan.json"), spec["plan_sha256"], "Contract binds plan before outcomes")
    checks.equal(len(plan["search"]), 46, "Exactly46 distinct requested offsets")
    checks.equal(len({tuple(r["values"]) for r in plan["search"]}), 46, "No duplicate numeric search settings")
    rows = replay.read_records(study / "rows.jsonl")
    search = [r for r in rows if r["phase"] == "search"]
    repeats = independent_repeats(seed, search)
    checks.equal(read(study / "repeat-plan.json"), repeats, "First successful family representatives then plan-order fill")
    audit_packets(study, rows, plan, repeats, checks)

    # Deterministic offsets are validated in the literal plan, not relabeled as
    # sampled noise. No learner-specific reward field is invented for this run.
    check_rows(rows, checks)
    by_id = {r["id"]: r for r in rows}
    source_controls = {r["id"]: r for r in controls}
    for row in rows:
        ns = study / row["adapter_directory"]
        for prefix in ("context_resolution", "connection"):
            checks.equal(sha(contained(ns, row[prefix + "_file"])), row[prefix + "_sha256"],
                         row["id"] + "/immutable proof/" + prefix)
        connection = read(contained(ns, row["connection_file"]))
        checks.equal((connection["values"], connection["bucket_xy_m"]),
                     (row["values"], row["bucket_xy_m"]), row["id"] + "/proof solves exact requested shot")
        checks.equal(bool(connection["passed"]), bool(row["native_release"]), row["id"] + "/release follows admission")
        if row["phase"].startswith("seed_"):
            exact_branch(row, seed, checks)
            checks.require(native.captured(row), row["id"] + "/seed capture preserved")
        elif row["phase"].startswith("controls_"):
            exact_branch(row, source_controls[row["source_record_id"]], checks)
        elif row["phase"] == "repeat":
            exact_branch(row, by_id[row["selected_from_id"]], checks)
        if row.get("native_release"):
            for key in ("q10", "q14"):
                checks.equal(row[key], connection[key], row["id"] + "/proof joint branch/" + key)
                checks.equal(row["native_initial_" + key], np.asarray(row[key], np.float32).astype(float).tolist(),
                             row["id"] + "/observed float32 joint branch/" + key)
    native_count = 0
    setup = {}
    home = read(bound("setup-home.json"))
    for namespace, parent_namespace in (("native-yaw", "native-yaw"), ("native-controls", "control-parent/native")):
        members = [r for r in rows if r["adapter_directory"] == namespace]
        path = study / namespace
        prior = read(bound("prior.json" if namespace == "native-yaw" else "old-prior.json"))
        setup[namespace] = admission.check_setup_evidence(setup_namespace._NativeNamespace(path), members, prior, home, checks)
        models.audit_native_options(path, spec["scene"], calibration, assets, checks)
        for name in ("effective-options.json", "native-compiled-geometry.json"):
            checks.equal(read(path / name), read(bound(parent_namespace + "/" + name)), namespace + "/unchanged parent physics/" + name)
        for row in members:
            if row.get("native_release"):
                checks.equal((row["effective_options_file"], row["effective_options_sha256"]),
                             ("effective-options.json", sha(path / "effective-options.json")), row["id"] + "/allocated native64 options")
        native_count += len(native.rescore_rows(path, members, spec["scene"], calibration,
            spec["identity"]["trace_schema"], assets, checks))
    classified = [dict(r, phase="test") if r["adapter_directory"] == "native-yaw" else r for r in rows]
    geometry_result = geometry.audit_geometry(study, classified, spec["scene"], assets, checks)
    checks.equal(native_count, sum(bool(r.get("native_release")) for r in rows), "Every fresh native flight independently rescored")
    gates = {}
    for phase in ("controls_pre", "controls_post"):
        gates[phase] = native.control_gate([r for r in rows if r["phase"] == phase])
        checks.equal(gates[phase]["status"], "passed", phase + "/all direct/table/negative controls pass")
        checks.equal(report[phase], gates[phase], phase + "/reported control gate")
    for phase in ("seed_pre", "seed_post", "search", "repeat"):
        checks.equal(report[phase], native.statistics([r for r in rows if r["phase"] == phase]), phase + "/independent result counts")
    groups = [dict(selected_from_id=identity, values=by_id[identity]["values"],
        captures=sum(native.captured(r) for r in rows if r.get("selected_from_id") == identity), requests=3)
        for identity in repeats["selected_from_ids"]]
    confirmed = [g for g in groups if g["captures"] == 3]
    checks.equal(report["repeat_results"], groups, "All selected repeat outcomes retained")
    checks.equal(report["confirmed"], confirmed, "Only3/3 fresh captures count as confirmed")
    # Bind the actual imported audit/core source separately from immutable
    # execution, which may predate this newly added verifier module.
    root = Path(__file__).resolve().parents[2]
    dependencies = {}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename or not getattr(module, "__name__", "").startswith(("sircus.mjlab.", "sircus.isaac.")):
            continue
        path = Path(filename).resolve()
        checks.require(path.is_relative_to(root), "Auditor imports a single explicit source root")
        name = str(path.relative_to(root))
        if path == Path(__file__).resolve():
            continue
        checks.require(name in spec["source_sha256"], "Imported dependency bound by producer/" + name)
        checks.equal(sha(path), spec["source_sha256"][name], "Unchanged imported audit/core dependency/" + name)
        dependencies[name] = sha(path)
    checks.equal({name: sha(parent / name) for name in PARENT_FILES} | {"audit": sha(audit_path)},
                 spec["parent_sha256"], "Parent bytes unchanged after audit")
    replay.check_manifest(study, checks)
    return dict(version=VERSION, status="passed", full_checks_verified=True, native_checks_verified=True,
        checks_count=checks.count, checks=checks.entries, failures=[], contract_fingerprint=spec["fingerprint"],
        report_sha256=sha(study / "report.json"), rows_sha256=sha(study / "rows.jsonl"),
        evidence_manifest_sha256=sha(study / "evidence-manifest.json"), parent_sha256=spec["parent_sha256"],
        auditor_sha256=sha(Path(__file__)), dependency_sha256=dependencies,
        native_releases=native_count, counts=native.statistics(rows), search=native.statistics(search),
        discovered_settings=sum(native.captured(r) for r in search), selected_settings=len(groups),
        confirmed_settings=len(confirmed), confirmed=confirmed, repeat_results=groups, controls=gates,
        setup=setup, geometry=geometry_result, elapsed_s=time.monotonic() - started,
        scope="Fixed corner recipe development; dense native64/v3, held endpoints, no policy training or held-out evaluation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "parent-study", "parent-audit", "assets", "audit"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.audit.resolve()
    if output.exists() or output.is_relative_to(args.study.resolve()) or output.is_relative_to(args.parent_study.resolve()):
        parser.error("Write a new independent audit outside both immutable evidence directories")
    try:
        result = audit(args.study, args.parent_study, args.parent_audit, args.assets)
    except Exception as exc:
        result = dict(version=VERSION, status="failed", full_checks_verified=False, native_checks_verified=False,
                      failures=[type(exc).__name__ + ": " + str(exc)], auditor_sha256=sha(Path(__file__)))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k: result.get(k) for k in ("status", "checks_count", "discovered_settings", "confirmed_settings", "failures")}))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
