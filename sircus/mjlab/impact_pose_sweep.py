"""Small validation-only gap/tilt experiment with native impact-pose evidence.

This creates a new study; the completed benchmark and diagnostic are read-only
inputs. It is controlled development search, not PPO training or fresh TEST.
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np

from .corrected_contract import PHYSICS, validate_packages
from .expanded_actions import WIDE_BOUNDS
from .learning_contract import digest
from .sweep_worker import atomic, load_reference
from .validation_failure_audit import (bound_file, diagnose_native, member, read,
    read_validation_rows, require, sha, signed, verify_signed, verify_source)

VERSION = "sircus-impact-pose-sweep-v1"
AUDIT_VERSION = "sircus-impact-pose-sweep-independent-audit-v1"
REPLAY_VERSION = "sircus-impact-pose-replay-v1"
GAPS = (0., .06, .12, .18)
TILTS = (-4., 0., 4.)
REPEATS = 3
SCOPE = ("Validation-selected short-miss development search. Fixed bucket and board XYZ; "
         "only board tilt and vertical ball-bottom gap above board CENTRE plane vary. "
         "Prescribed joint endpoints and spawned free-flight ball, not grasp/release or "
         "compliant arm control. No final TEST or benchmark replacement.")


def source_cases(diagnostic, study, contract, manifest):
    """Use source-bound selected short examples, one posture/method/layout."""
    candidates = [r for r in diagnostic["representatives"]
                  if r.get("primary_selected") is True and r.get("recorded_replay_available") is True
                  and r.get("category") == "no_whole_ball_entry"
                  and r.get("miss_region", {}).get("region") == "short"]
    groups = {}
    for item in candidates:
        key = (item["context_id"], item["method"])
        rank = lambda r: (r["seed"], r["source_row_id"])
        if key not in groups or rank(item) < rank(groups[key]):
            groups[key] = item
    # Predeclared deterministic cap: first two observed-short layouts, both
    # methods where available. A missing layout/method is never invented.
    layout_ids = sorted({key[0] for key in groups})[:2]
    selected = [groups[key] for key in sorted(groups) if key[0] in layout_ids]
    require(selected, "No source-bound selected short validation examples available")
    cases = []
    for item in selected:
        prefix = f"runs/{item['method']}-seed{item['seed']}"
        rows_path = bound_file(study, manifest, prefix + "/rows.jsonl")
        selection = read(bound_file(study, manifest, prefix + "/selected.json"))
        rows = read_validation_rows(rows_path, item["method"], item["seed"], contract)
        matches = [r for r in rows if r["id"] == item["source_row_id"]]
        require(len(matches) == 1, "Short source row missing or duplicated")
        row = matches[0]
        require(row["samples"] == selection["selected_samples"] and selection["frozen_before_test"] is True,
                "Short source is not the selected validation checkpoint")
        require(row["native_release"] and row["bucket_xy_m"] == item["bucket_xy_m"], "Source row/diagnostic mismatch")
        name = row["native_cache_file"]
        require(isinstance(name, str) and "/" not in name and "\\" not in name and name.endswith(".npz"), "Unsafe source native name")
        native_path = bound_file(study, manifest, prefix + "/native/" + name)
        require(sha(native_path) == row["native_cache_sha256"], "Short source native bytes changed")
        with np.load(native_path, allow_pickle=False) as packet:
            ids = packet["record_ids"].tolist()
            require(all("-validation-" in v and "-test-" not in v and "-train-" not in v for v in ids), "Only validation source packets allowed")
            require(ids[row["native_cache_row"]] == row["id"], "Short source packet column changed")
        cases.append(dict(id=item["context_id"] + "-" + item["method"], context_id=item["context_id"],
            source_method=item["method"], source_row_id=row["id"], source_seed=item["seed"],
            source_samples=row["samples"], bucket_xy_m=copy.deepcopy(row["bucket_xy_m"]),
            baseline_values=copy.deepcopy(row["values"]), source_row_sha256=digest(row),
            source_rows_file=prefix + "/rows.jsonl", source_rows_sha256=sha(rows_path),
            source_native_file=prefix + "/native/" + name, source_native_sha256=sha(native_path),
            source_category=item["category"], source_miss_region=item["miss_region"],
            source_pose={key: copy.deepcopy(row[key]) for key in
                ("q10", "q14", "native_initial_q10", "native_initial_q14", "board_center_m", "board_normal",
                 "board_quaternion_wxyz", "native_initial_ball_center_m", "release_ball_center_m")},
            source_route_success=bool(row["route_success"])))
    return cases


def grid_requests(cases):
    requests = []
    for case in cases:
        for gi, gap in enumerate(GAPS):
            for ti, tilt in enumerate(TILTS):
                values = np.asarray(case["baseline_values"], float).copy()
                values[3] += tilt
                values[4] += gap
                requests.append(dict(id=f"impact-{case['id']}-g{gi}-t{ti}", case_id=case["id"],
                    context_id=case["context_id"], phase="sweep", baseline=gap == 0 and tilt == 0,
                    bucket_xy_m=copy.deepcopy(case["bucket_xy_m"]), values=values.tolist(),
                    gap_offset_m=gap, tilt_offset_deg=tilt, source_row_id=case["source_row_id"]))
    return requests


def selection_key(row):
    """Capture, quality, retention, observed proximity, then stable trial ID."""
    proximity = float(row.get("closest_opening_distance_m", 2.))
    require(np.isfinite(proximity), "Non-finite selection proximity")
    return (-int(bool(row.get("native_release") and row.get("route_success") and not row.get("invalid"))),
            -int(row.get("route_quality", 0)), -float(row.get("max_retention_s", 0.)),
            proximity, row["id"])


def choose(cases, rows):
    result = {}
    for case in cases:
        pool = [r for r in rows if r["case_id"] == case["id"]]
        require(len(pool) == len(GAPS) * len(TILTS), "Incomplete grid cannot select a repeat posture")
        native = [r for r in pool if r.get("native_release") and not r.get("invalid")]
        # Preserve a finite unresolved result. Repeat the requested baseline if
        # all requests rejected, still charging all three requested repeats.
        selected = min(native, key=selection_key) if native else next(r for r in pool if r["baseline"])
        result[case["id"]] = selected
    return result


def confirmation_requests(cases, selected):
    return [dict(id=f"impact-{case['id']}-repeat-{repeat}", case_id=case["id"],
                 context_id=case["context_id"], phase="confirmation", baseline=False,
                 bucket_xy_m=copy.deepcopy(case["bucket_xy_m"]), values=copy.deepcopy(selected[case["id"]]["values"]),
                 selected_trial_id=selected[case["id"]]["id"], repeat=repeat, source_row_id=case["source_row_id"])
            for case in cases for repeat in range(REPEATS)]


def count(rows):
    return dict(attempts=len(rows), native_releases=sum(bool(r.get("native_release")) for r in rows),
                captures=sum(bool(r.get("native_release") and r.get("route_success") and not r.get("invalid")) for r in rows))


def inspect_inputs(args):
    repository = Path(args.repository).resolve()
    study = Path(args.benchmark_dir).resolve()
    contract, benchmark, manifest, audit, scene, calibration, source = verify_source(
        study, Path(args.benchmark_audit), repository, args.contract_fingerprint, args.benchmark_audit_sha256)
    path = Path(args.diagnostic_dir) / "report.json"
    require(sha(path) == args.diagnostic_sha256, "Pinned validation diagnostic bytes changed")
    diagnostic = read(path); verify_signed(diagnostic, "Validation diagnostic")
    require(diagnostic.get("status") == "passed" and diagnostic.get("synthetic_fixture") is False,
            "Passed native validation diagnostic required")
    require(diagnostic["source"] == source, "Diagnostic and frozen benchmark sources differ")
    require(diagnostic["diagnostic_source_sha256"] == sha(repository / "sircus/mjlab/validation_failure_audit.py"),
            "Validation diagnostic producer source changed")
    prior_path = bound_file(study, manifest, "prior.json")
    prior = read(prior_path)
    require(digest(prior) == contract["prior_sha256"], "Frozen shared prior changed")
    cases = source_cases(diagnostic, study, contract, manifest)
    source = dict(source, benchmark_report_sha256=source["report_sha256"],
                  benchmark_manifest_sha256=source["study_manifest_sha256"],
                  benchmark_audit_sha256=source["audit_sha256"],
                  validation_report_sha256=args.diagnostic_sha256, prior_sha256=sha(prior_path))
    return contract, scene, calibration, source, prior, cases


def prepare(args):
    output = Path(args.output).resolve()
    for root in (args.benchmark_dir, args.diagnostic_dir, args.reference, args.assets):
        require(not output.is_relative_to(Path(root).resolve()), "Output must be outside frozen evidence and assets")
    require(not output.exists(), "New output directory required; existing experiment is never overwritten")
    contract, scene, calibration, source, prior, cases = inspect_inputs(args)
    source_code = {name: sha(Path(args.repository) / name) for name in
        ("sircus/mjlab/impact_pose_sweep.py", "sircus/mjlab/impact_pose_env.py")}
    protocol = dict(gap_offsets_m=list(GAPS), tilt_offsets_deg=list(TILTS), confirmation_repeats=REPEATS,
        maximum_distinct_bucket_positions=2, source_postures=len(cases), distinct_bucket_positions=len({r["context_id"] for r in cases}),
        sweep_requests=len(cases) * len(GAPS) * len(TILTS), confirmation_requests=len(cases) * REPEATS,
        selection_rule="native non-invalid only; capture, route quality, retention, observed opening proximity, stable ID; if none admitted repeat baseline as rejection",
        confirmation_rule="Three newly executed nominal repeats of selected posture; not robust, perturbed or independent layout evaluation",
        baseline_rule="Exactly one gap=0/tilt=0 request per source posture; fresh execution compared descriptively with original short miss",
        ball_radius_m=scene["ball"]["radius_m"], requested_gap_definition="ball bottom above board CENTRE plane, not gap to tilted top surface",
        requested_release_z_formula="board_center_z + gap + ball_radius", action_bounds=copy.deepcopy(WIDE_BOUNDS),
        invalid_request_policy="No clipping/projection/substitution; geometry rejections remain in requested denominators",
        scoring=copy.deepcopy(PHYSICS), scope=SCOPE)
    plan = signed(dict(version=VERSION, role="validation_diagnostic", synthetic_fixture=False,
        source=source, source_code_sha256=source_code, protocol=protocol, layouts=cases, requests=grid_requests(cases)))
    output.mkdir(parents=True)
    atomic(output / "plan.json", plan)
    atomic(output / "progress.json", dict(version=VERSION, status="prepared", phase="awaiting_4090", plan_sha256=sha(output / "plan.json"),
        planned_requests=protocol["sweep_requests"] + protocol["confirmation_requests"], scope=SCOPE))
    return plan


def request_metadata(adapter, prior, case):
    resolution, timing = adapter.resolve_context(dict(id=case["context_id"], bucket_xy_m=case["bucket_xy_m"]))
    anchor = next(r for r in prior["records"] if r["id"] == resolution["source_hint"]["id"])
    require(anchor["values"] == resolution["source_hint"]["values"]
            and anchor["bucket_xy_m"] == resolution["source_hint"]["bucket_xy_m"], "Resolver source changed")
    metadata = dict(source_teacher_id=anchor["id"], source_hint=anchor,
        effective_source_hint=resolution["effective_source_hint"], context_resolution_file=resolution["file"],
        context_resolution_sha256=resolution["sha256"], context_home_sha256=resolution["home_sha256"],
        context_layout_eligible=resolution["layout_eligible"], context_anchor_admitted=resolution["anchor_admitted"],
        context_hard_reject=resolution["hard_reject"])
    return metadata, timing


def run(args):
    from .calibrated_adapter import CalibratedAdapter
    from .gpu_slot import GPU_UUID, gpu_slot
    from .impact_pose_env import ImpactPoseEnvironment
    import torch

    require(os.environ.get("CUDA_VISIBLE_DEVICES") == GPU_UUID, "Only the authorized 4090 UUID may run the sweep")
    versions = {name: importlib.metadata.version(name) for name in
        ("mjlab", "mujoco", "mujoco-warp", "torch", "warp-lang", "numpy", "ml-dtypes")}
    validate_packages(versions)
    output = Path(args.output).resolve()
    plan = read(output / "plan.json") if args.resume_plan else prepare(args)
    verify_signed(plan, "Prepared sweep plan")
    require(not (output / "native").exists() and not (output / "rows.jsonl").exists(), "Executed studies cannot be resumed/replaced")
    contract, scene, calibration, source, prior, cases = inspect_inputs(args)
    require(plan["source"] == source and plan["layouts"] == cases and plan["requests"] == grid_requests(cases), "Prepared inputs or request grid changed")
    for name, expected in plan["source_code_sha256"].items():
        require(sha(Path(args.repository) / name) == expected, "Prepared source code changed")
    started = time.monotonic()
    def publish(phase, **kw):
        atomic(output / "progress.json", dict(version=VERSION, status="running", phase=phase,
            plan_sha256=sha(output / "plan.json"), planned_requests=len(plan["requests"]) + len(cases)*REPEATS,
            elapsed_wall_s=time.monotonic()-started, scope=SCOPE, **kw))
        print(json.dumps(dict(phase=phase, **kw)), flush=True)
    with gpu_slot(args.gpu_lock, timeout_s=args.queue_timeout, poll_s=2, notify=lambda v: publish("waiting_for_4090", queue=v)):
        torch.set_num_threads(2)
        torch.manual_seed(20261104); np.random.seed(20261104)
        ref = load_reference(args.reference, args.assets)
        require(digest(ref["scene"]) == contract["identity"]["scene_sha256"], "Native scene differs from frozen scene")
        adapter = CalibratedAdapter(ref, args.assets, output / "native", prior_records=prior["records"],
            calibration=calibration, home=contract["identity"]["setup"]["home"], batch=32,
            environment_factory=lambda: ImpactPoseEnvironment(scene, args.assets, calibration, n=32, bounds=WIDE_BOUNDS))
        metadata, resolutions = {}, []
        publish("resolving_geometry")
        for case in cases:
            metadata[case["id"]], timing = request_metadata(adapter, prior, case)
            resolutions.append(dict(case_id=case["id"], timing=timing, **metadata[case["id"]]))
        atomic(output / "resolutions.json", resolutions)
        all_rows, timings = [], []
        def execute(requests, phase):
            normalized = [dict(copy.deepcopy(r), **copy.deepcopy(metadata[r["case_id"]])) for r in requests]
            rows, timing = adapter.evaluate(normalized, "validation_diagnostic_" + phase)
            for row, request in zip(rows, requests):
                row.update(copy.deepcopy(request))
                row["role"] = "validation_diagnostic"
            all_rows.extend(rows); timings.append(dict(phase=phase, **timing))
            with (output / "rows.jsonl").open("a") as stream:
                for row in rows: stream.write(json.dumps(row, allow_nan=False) + "\n")
            return rows
        publish("height_tilt_sweep", completed_requests=0)
        sweep = execute(plan["requests"], "sweep")
        selected = choose(cases, sweep)
        atomic(output / "selection.json", {key: dict(id=r["id"], values=r["values"]) for key, r in selected.items()})
        publish("fresh_nominal_repeats", completed_requests=len(sweep))
        execute(confirmation_requests(cases, selected), "confirmation")
        atomic(output / "timings.json", dict(resolutions=resolutions, native_batches=timings,
            total_wall_s=time.monotonic()-started, dependency_versions=versions))
    # Build compact replay/report using only these new observations. Native
    # contact scoring itself already used both independent implementations.
    publish("building_report", completed_requests=len(all_rows))
    report = build_report(plan, all_rows, output, scene, calibration)
    report["resources"] = dict(total_wall_s=time.monotonic()-started, dependency_versions=versions)
    atomic(output / "report.json", signed(report))
    files = {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()
             and p.name not in ("manifest.json", "audit.json", "progress.json")}
    atomic(output / "manifest.json", files)
    audit = verify(args)
    atomic(output / "audit.json", audit)
    atomic(output / "progress.json", dict(version=VERSION, status="completed", phase="completed", audit_status=audit["status"],
        completed_requests=len(all_rows), planned_requests=len(all_rows), scope=SCOPE, report_sha256=sha(output / "report.json")))
    return audit


def trajectory_and_pose(arrays, column, base_spacing_m):
    """Downsample actual ball states; pose history uses actual logged FK samples."""
    t = arrays["time_s"]
    step = max(1, round((1/60) / float(t[1]-t[0])))
    ix = np.unique(np.r_[np.arange(0, len(t), step), len(t)-1]).astype(int)
    trajectory = [dict(time=float(t[i]), ball=arrays["position_m"][i, column].tolist()) for i in ix]
    # Logger arrays use a padded, per-world native index; invalid padding never
    # becomes an interpolated or fabricated observation.
    history = []
    if "pose_sample_valid" in arrays:
        valid = arrays["pose_sample_valid"][:, column]
        for k in np.flatnonzero(valid):
            history.append(dict(time=float(arrays["pose_sample_time_s"][k, column]),
                board_center=arrays["pose_board_center_m"][k, column].tolist(),
                board_quaternion_wxyz=arrays["pose_board_quaternion_wxyz"][k, column].tolist(),
                board_normal=arrays["pose_board_normal"][k, column].tolist(),
                link_positions10=[[0., 0., 0.], *arrays["pose_link_positions10_m"][k, column].tolist()],
                link_positions14=[[float(base_spacing_m), 0., 0.], *arrays["pose_link_positions14_m"][k, column].tolist()]))
    return trajectory, history


def trial_view(row, scene, replay=None):
    release = [*row["values"][:2], row["values"][2] + row["values"][4] + scene["ball"]["radius_m"]]
    result = {key: copy.deepcopy(row.get(key)) for key in
        ("id", "case_id", "phase", "baseline", "values", "bucket_xy_m", "native_release", "invalid", "gap_offset_m", "tilt_offset_deg")}
    result.update(success=bool(row.get("route_success") and not row.get("invalid")),
        route_quality=row.get("route_quality", 0), release_ball_center_m=release,
        category=row.get("diagnosis", {}).get("category", "setup_rejected"),
        miss_region=row.get("diagnosis", {}).get("miss_region"), pose_summary=row.get("pose_logging"),
        max_retention_s=row.get("max_retention_s", 0.), closest_opening_distance_m=row.get("closest_opening_distance_m"))
    if replay: result.update(replay)
    return result


def build_report(plan, rows, output, scene, calibration):
    by_file = {}
    for row in rows:
        if row.get("native_release"): by_file.setdefault(row["native_cache_file"], []).append(row)
    replay_refs = {}
    for filename, members in by_file.items():
        path = output / "native" / filename
        with np.load(path, allow_pickle=False) as raw:
            arrays = {key: raw[key] for key in raw.files if key != "record_ids"}
            ids = raw["record_ids"].tolist()
        for row in members:
            col = row["native_cache_row"]
            require(ids[col] == row["id"], "New packet column changed")
            trace = {key: value[:, col] if value.ndim >= 2 and key not in ("time_s",) else value
                     for key, value in arrays.items() if key in ("position_m", "velocity_m_s", "forces_N", "time_s", "contact_distance_m", "native_time_s")}
            row["diagnosis"] = diagnose_native(row, scene, calibration, trace)
            trajectory, history = trajectory_and_pose(arrays, col, scene["arms"]["base_spacing_m"])
            replay_id = "impact-" + digest(row["id"])[:32]
            configuration = dict(board_center=row["board_center_m"], board_normal=row["board_normal"],
                board_quaternion_wxyz=row["board_quaternion_wxyz"], release_center=arrays["position_m"][0, col].tolist(),
                q10=row["native_initial_q10"], q14=row["native_initial_q14"],
                link_positions10=row["link_positions10"], link_positions14=row["link_positions14"])
            payload = dict(version=REPLAY_VERSION, id=replay_id, role="validation_diagnostic", source=plan["source"],
                source_row_id=row["source_row_id"], result=dict(request_id=row["id"], method="impact_sweep_recorded_replay",
                    bucket_xy_m=row["bucket_xy_m"], values=row["values"], success=bool(row["route_success"] and not row.get("invalid")),
                    configuration=configuration, trajectory=trajectory, pose_history=history,
                    metrics=dict(pose_logging=row.get("pose_logging"), category=row["diagnosis"]["category"],
                        observed_miss_region=row["diagnosis"]["miss_region"], route_quality=row["route_quality"], max_retention_s=row["max_retention_s"])),
                scope=SCOPE + " Native arm/board FK history is reconstructed from recorded integrated joints; no invented arm travel.")
            replay_path = output / "replays" / (replay_id + ".json")
            atomic(replay_path, payload)
            replay_refs[row["id"]] = dict(replay_id=replay_id, replay_sha256=sha(replay_path))
    # Keep diagnostic fields in the separately saved row evidence; adapter's
    # original native shot rows remain exact execution observations.
    atomic(output / "diagnosed-rows.json", rows)
    sweep = [r for r in rows if r["phase"] == "sweep"]
    selected = choose(plan["layouts"], sweep)
    layouts = []
    for case in plan["layouts"]:
        pool = [r for r in rows if r["case_id"] == case["id"]]
        repeats = [r for r in pool if r["phase"] == "confirmation"]
        best = selected[case["id"]]
        layouts.append(dict(copy.deepcopy(case), trials=[trial_view(r, scene, replay_refs.get(r["id"])) for r in pool],
            selected=dict(id=best["id"], values=best["values"], success=bool(best.get("route_success") and not best.get("invalid"))),
            confirmation=count(repeats)))
    return dict(version=VERSION, status="completed", synthetic_fixture=False, role="validation_diagnostic",
        source=copy.deepcopy(plan["source"]), source_code_sha256=plan["source_code_sha256"], protocol=plan["protocol"],
        plan_sha256=sha(output / "plan.json"), layouts=layouts,
        summary=dict(requests=len(rows), **{k: v for k, v in count(rows).items() if k != "attempts"},
                     sweep=count(sweep), confirmation=count([r for r in rows if r["phase"] == "confirmation"])),
        limitations=[SCOPE, f"Only {len({c['context_id'] for c in plan['layouts']})} observed-short validation bucket positions and {len(plan['layouts'])} source postures are diagnosed; no all-table coverage.",
            "Three repeated nominal executions establish local repeatability only; no perturbation robustness or generalization claim.",
            "Frozen benchmark policies, original results, calibration/scoring and final TEST remain unchanged."])


def verify(args):
    """Separate CPU-only pass over exact requests, raw observations and scoring."""
    output = Path(args.output).resolve()
    plan, report, manifest = (read(output / n) for n in ("plan.json", "report.json", "manifest.json"))
    verify_signed(plan, "Sweep plan"); verify_signed(report, "Sweep report")
    checks = 0
    def check(condition, label):
        nonlocal checks
        require(condition, label); checks += 1
    contract, scene, calibration, source, prior, cases = inspect_inputs(args)
    check(plan["source"] == report["source"] == source, "Frozen sources changed")
    check(plan["layouts"] == cases and plan["requests"] == grid_requests(cases), "Exact predeclared sweep changed")
    check(report["version"] == VERSION and report["status"] == "completed" and report["synthetic_fixture"] is False,
          "Completed native diagnostic report required")
    check(report["protocol"] == plan["protocol"] and report["source_code_sha256"] == plan["source_code_sha256"]
          and report["plan_sha256"] == sha(output / "plan.json"), "Report protocol/source/plan changed")
    check(len(report["layouts"]) == len(cases), "Report layout/posture count changed")
    check(all(name in manifest for name in ("plan.json", "report.json", "rows.jsonl", "diagnosed-rows.json", "selection.json")),
          "Incomplete derived evidence manifest")
    for name, expected in manifest.items(): check(sha(member(output, name)) == expected, "Derived evidence changed: " + name)
    for name, expected in plan["source_code_sha256"].items(): check(sha(member(args.repository, name)) == expected, "Executed source changed")
    rows = [json.loads(s) for s in (output / "rows.jsonl").read_text().splitlines()]
    selected = choose(cases, [r for r in rows if r.get("phase") == "sweep"])
    expected = plan["requests"] + confirmation_requests(cases, selected)
    check(len(rows) == len(expected) and len({r["id"] for r in rows}) == len(rows), "Requested budget/identity changed")
    by_file = {}
    for row, requested in zip(rows, expected):
        for key, value in requested.items(): check(row.get(key) == value, "Requested action or phase changed: " + key)
        check(row.get("role") == "validation_diagnostic", "Training/TEST role forbidden")
        check(row["values"][:3] == next(c for c in cases if c["id"] == row["case_id"])["baseline_values"][:3], "Board position changed")
        if not row.get("native_release"):
            check(bool(row.get("invalid")) and not row.get("route_success"), "Rejected setup cannot capture")
        else:
            check(row.get("scalar_score_audit_passed") is True, "Missing native oracle agreement")
            by_file.setdefault(row["native_cache_file"], []).append(row)
    for filename, group in by_file.items():
        path = member(output, "native/" + filename)
        check(all(r["native_cache_sha256"] == sha(path) for r in group), "Native packet hash differs")
        with np.load(path, allow_pickle=False) as raw:
            ids = raw["record_ids"].tolist()
            check(set(ids) == {r["id"] for r in group} and len(ids) == len(group), "Unknown native trial column")
            arrays = {k: raw[k] for k in raw.files if k != "record_ids"}
        dt = float(arrays["dt_s"]); t = arrays["time_s"]
        check(abs(dt - PHYSICS["native_dt_s"]) < 1e-15 and np.array_equal(t, np.arange(len(t))*dt), "Native clock changed")
        check(len(t) == round(PHYSICS["horizon_s"]/dt) + 1, "Native horizon changed")
        from .contact_model import make_model
        from .impact_pose_env import verify_pose_arrays
        if "pose_model" not in locals():
            pose_spec, _ = make_model(scene, args.assets, PHYSICS["board_mode"])
            pose_model = pose_spec.compile()
        ordered = [next(r for r in group if r["id"] == name) for name in ids]
        pose_audit = verify_pose_arrays(pose_model, arrays, ordered)
        check(pose_audit["status"] == "passed", "Independent CPU pose reconstruction failed")
        checks += pose_audit["checked_fk_samples"]
        for row in group:
            col = row["native_cache_row"]
            check(ids[col] == row["id"], "Native action/column association changed")
            trace = {k: v[:, col] if v.ndim >= 2 and k != "time_s" else v for k, v in arrays.items()
                     if k in ("position_m", "velocity_m_s", "forces_N", "time_s", "contact_distance_m", "native_time_s")}
            row["diagnosis"] = diagnose_native(row, scene, calibration, trace); checks += 12
            check(np.allclose(arrays["position_m"][0, col], [*row["values"][:2], row["values"][2]+row["values"][4]+scene["ball"]["radius_m"]], atol=1e-6, rtol=0), "Actual ball release differs")
            check("pose_logging" in row and bool(row["pose_logging"]), "Native pose logging missing")
            # Finite per-step joints and explicit sample indices are necessary
            # evidence, not trust in the old cached zero-displacement scalar.
            for key in ("pose_joint_qpos_rad", "pose_joint_qvel_rad_s"):
                check(key in arrays and arrays[key].shape[:2] == (len(t), len(ids)) and arrays[key].shape[2] == 12
                      and np.isfinite(arrays[key][:, col]).all(), "Missing or invalid dense joint history: " + key)
            check("pose_sample_valid" in arrays and np.count_nonzero(arrays["pose_sample_valid"][:, col]) >= 960, "Pose sampling incomplete")
            check(row["pose_logging"] == pose_audit["summaries"][col], "Compact native pose summary changed")
            trial = next(t for c in report["layouts"] for t in c["trials"] if t["id"] == row["id"])
            replay = read(member(output, "replays/" + trial["replay_id"] + ".json"))["result"]
            expected_configuration = dict(board_center=row["board_center_m"], board_normal=row["board_normal"],
                board_quaternion_wxyz=row["board_quaternion_wxyz"], release_center=arrays["position_m"][0, col].tolist(),
                q10=row["native_initial_q10"], q14=row["native_initial_q14"],
                link_positions10=row["link_positions10"], link_positions14=row["link_positions14"])
            check(replay["configuration"] == expected_configuration and replay["values"] == row["values"]
                  and replay["bucket_xy_m"] == row["bucket_xy_m"]
                  and replay["success"] is bool(row["route_success"] and not row.get("invalid")), "Replay configuration/outcome changed")
            for point in replay["trajectory"]:
                index = int(round(point["time"] / dt))
                check(0 <= index < len(t) and point["time"] == float(t[index])
                      and point["ball"] == arrays["position_m"][index, col].tolist(), "Replay ball sample/time changed")
            expected_history = trajectory_and_pose(arrays, col, scene["arms"]["base_spacing_m"])[1]
            check(replay["pose_history"] == expected_history, "Replay pose samples/time changed")
    check(report["summary"]["requests"] == len(rows)
          and report["summary"]["captures"] == count(rows)["captures"]
          and report["summary"]["native_releases"] == count(rows)["native_releases"], "Report totals changed")
    check(report["summary"]["sweep"] == count([r for r in rows if r["phase"] == "sweep"])
          and report["summary"]["confirmation"] == count([r for r in rows if r["phase"] == "confirmation"]), "Report phase totals changed")
    for case, observed in zip(cases, report["layouts"]):
        pool = [r for r in rows if r["case_id"] == case["id"]]
        check(case["id"] == observed["id"] and len(observed["trials"]) == len(pool), "Case report changed")
        for trial, row in zip(observed["trials"], pool):
            expected_view = trial_view(row, scene)
            for key, value in expected_view.items():
                check(trial.get(key) == value, "Compact trial metadata changed: " + key)
            if row["native_release"]:
                replay = member(output, "replays/" + trial["replay_id"] + ".json")
                check(sha(replay) == trial["replay_sha256"], "Recorded replay hash changed")
                payload = read(replay)
                check(payload["role"] == "validation_diagnostic" and payload["source"] == source
                      and payload["result"]["request_id"] == row["id"] and payload["source_row_id"] == row["source_row_id"], "Recorded replay identity changed")
        check(observed["selected"]["id"] == selected[case["id"]]["id"], "Selection changed")
        check(observed["confirmation"] == count([r for r in pool if r["phase"] == "confirmation"]), "Confirmation changed")
    # Re-inspection also verifies all original source hashes at the end.
    check(inspect_inputs(args)[3] == source, "Frozen evidence changed during CPU verification")
    return signed(dict(version=AUDIT_VERSION, status="passed", checks=checks, failures=[], synthetic_fixture=False, source=source,
        report_sha256=sha(output / "report.json"), plan_sha256=sha(output / "plan.json"), manifest_sha256=sha(output / "manifest.json"),
        benchmark_report_sha256=source["benchmark_report_sha256"], validation_report_sha256=source["validation_report_sha256"],
        requests=len(rows), native_releases=count(rows)["native_releases"], captures=count(rows)["captures"], scope=SCOPE))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("benchmark-dir", "benchmark-audit", "diagnostic-dir", "reference", "assets", "gpu-lock", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[2])
    for name in ("contract-fingerprint", "benchmark-audit-sha256", "diagnostic-sha256"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--queue-timeout", type=float, default=600.)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume-plan", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        result = verify(args); atomic(args.output / "audit.json", result)
    elif args.prepare_only:
        result = prepare(args)
    else:
        result = run(args)
    requests = result.get("requests", [])
    print(json.dumps(dict(status=result.get("status", "prepared"), version=result.get("version"),
        checks=result.get("checks"), requests=len(requests) if isinstance(requests, list) else requests,
        report_sha256=result.get("report_sha256"), plan_sha256=sha(args.output / "plan.json")), allow_nan=False), flush=True)


if __name__ == "__main__": main()
