"""Frozen current planner versus expanded search with full bucket-approach ranking.

No fitting, rollout adaptation, or runtime promotion occurs in this comparison.
Each lane gets one drop at each of 50 fresh placements under native64/v4 scoring.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import random
import shutil
import time

from .benchmark_verify import contained, digest, read, sha
from .sweep_worker import atomic

VERSION = "sircus-approach-comparison-v1"
METHODS = ("current", "expanded")
POSITIONS = 50
SEED = 2026112717
PROCESSING_SEED = 2026112718
PLANNING_S = 30.
GPU_CAP_S = 3600
WORKERS = 2
COPIED = ("shot-library-rows.json", "route-models.json", "route-history.json",
          "training-prior.json", "calibration.json", "verification.json")


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def publish(a, report):
    report["updated_at_unix"] = time.time()
    atomic(a.output / "report.json", report)
    print(json.dumps({k: report[k] for k in ("status", "phase", "progress", "error", "gpu_wait") if k in report}), flush=True)


def cpu_only():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Freeze/preparation requires CUDA_VISIBLE_DEVICES='' explicitly")


def current_planner_binding(a):
    """Bind the exact archived inputs used by the running hybrid service."""
    study, audit_path = a.current_planner_study, a.current_planner_audit
    report, audit = read(study/"report.json"), read(audit_path)
    manifest = read(study/"evidence-manifest.json")
    if (report.get("status") != "completed" or audit.get("status") != "passed"
            or audit.get("report_sha256") != sha(study/"report.json")
            or audit.get("evidence_manifest_sha256") != sha(study/"evidence-manifest.json")
            or report.get("evidence_manifest_sha256") != sha(study/"evidence-manifest.json")):
        raise ValueError("Current planner needs its unchanged passed input-study audit")
    archived = read(study/"contract.json")
    neural = read(a.neural_study/"contract.json")
    if (manifest.get("contract.json") != sha(study/"contract.json")
            or archived.get("fingerprint") != digest({k:v for k,v in archived.items() if k != "fingerprint"})):
        raise ValueError("Current planner contract is not bound to its audited manifest")
    for key in ("scene", "identity", "flight_ranking"):
        if archived.get(key) != neural.get(key) or key not in neural:
            raise ValueError("Current planner and neural baseline configuration differ: "+key)
    from .stalled_reference_plan import REFERENCE_PATH_BUDGET_S
    from .fast_planning_plan import EARLY_SCORE_M
    if (archived.get("planning_budget_s"), archived.get("reference_path_s"), archived.get("early_score_m"),
            archived.get("physics_substeps")) != (PLANNING_S, REFERENCE_PATH_BUDGET_S, EARLY_SCORE_M, 64):
        raise ValueError("Current planner deadline, reference fallback, early threshold or native substeps differ")
    inputs = {}
    for name in (*COPIED, "neural-checkpoint.json"):
        checksum = sha(study/name)
        if manifest.get(name) != checksum or sha(a.neural_study/name) != checksum:
            raise ValueError("Current planner input differs from selected baseline: "+name)
        inputs[name] = checksum
    return dict(study=str(study.resolve()), audit=str(audit_path.resolve()),
        report_sha256=sha(study/"report.json"), audit_sha256=sha(audit_path),
        manifest_sha256=sha(study/"evidence-manifest.json"), inputs=inputs,
        planner="StalledReferencePlanner/warm_early", decision_budget_s=PLANNING_S,
        reference_path_budget_s=REFERENCE_PATH_BUDGET_S, early_score_m=EARLY_SCORE_M,
        contract_sha256=sha(study/"contract.json"),
        configuration={key:archived[key] for key in ("scene", "identity", "flight_ranking", "physics_substeps")})


def source_inputs(a):
    from .planning_budget_study import audited_neural
    from .clearance_expansion_study import native_inputs

    neural_binding = audited_neural(a)
    current_planner_binding(a)
    spec, parent_binding, source, reference = native_inputs(a)
    neural_contract = read(a.neural_study / "contract.json")
    if spec["identity"] != neural_contract["identity"] or spec["scene"] != neural_contract["scene"]:
        raise ValueError("Historical neural and control scene identities or geometry differ")
    manifest = read(a.neural_study / "evidence-manifest.json")
    for name in ("training-dataset.jsonl", "validation-dataset.jsonl", "collection-plans.json", *COPIED):
        if sha(a.neural_study / name) != manifest[name]:
            raise ValueError("Historical dataset changed: " + name)
    return spec, parent_binding, reference, neural_contract, neural_binding


def fresh_layouts(a, reference, neural_contract):
    from .intergrid_proposer_plan import make_splits, sample_splits

    original_layouts = read(a.neural_study / "layouts.json")
    previous = [row for split in original_layouts.values() for row in split["layouts"]]
    grid = read(a.parent_study / "grid.json")
    exclusions, inventory, geometry = set(), {}, None
    for root in a.history_root:
        _, provenance = make_splits(root, grid, scene=reference["scene"], assets=a.assets,
                                    previous_layouts=previous, seed=SEED, include_train=False)
        exclusions.update(map(tuple, provenance["excluded_coordinates_m"]))
        inventory[str(root.resolve())] = provenance["historical_files_sha256"]
        if geometry is not None and geometry != provenance["geometry"]:
            raise ValueError("Historical inventory changed placement geometry")
        geometry = provenance["geometry"]
    exclusions.update(tuple(xy) for xy in read(a.library/"all-positions.json"))
    for split in ("train", "validation"):
        exclusions.update(tuple(r["request"]["bucket_xy_m"]) for r in lines(a.library/(split+"-dataset.jsonl")))
    exclusions.update(tuple(r["bucket_xy_m"]) for r in read(a.library/"candidate-sources.json"))
    previous_board_first = read(a.expanded_study/"layouts.json")
    exclusions.update(tuple(r["bucket_xy_m"]) for r in previous_board_first["layouts"])
    sampled = sample_splits(sorted(exclusions), geometry, grid, seed=SEED, include_train=False)["test"]
    layouts = [dict(row, id=f"approach-comparison-test-{i:02d}", context_id=f"approach-comparison-test-{i:02d}", split="test")
               for i, row in enumerate(sampled)]
    if len(layouts) != POSITIONS:
        raise ValueError("Expected exactly 50 fresh positions")
    return dict(layouts=layouts, provenance=dict(excluded_coordinates_m=sorted(exclusions),
                geometry=geometry, history_files=inventory, previous_comparison=str(a.expanded_study.resolve()),
                previous_layouts_sha256=sha(a.expanded_study/"layouts.json"),
                layouts_sha256=digest(layouts), minimum_separation_m=.005))


def library_inputs(a):
    from .expanded_shot_library import verify_library
    return verify_library(a.library)


def expanded_parent_binding(a):
    """Copy only inputs authenticated by the completed comparison's full audit."""
    study, audit_path = a.expanded_study, a.expanded_audit
    report, audit = read(study/"report.json"), read(audit_path)
    manifest = read(study/"evidence-manifest.json")
    if (report.get("status") != "completed" or audit.get("status") != "passed"
            or audit.get("report_sha256") != sha(study/"report.json")
            or audit.get("evidence_manifest_sha256") != sha(study/"evidence-manifest.json")
            or report.get("evidence_manifest_sha256") != sha(study/"evidence-manifest.json")):
        raise ValueError("Expanded parent needs its unchanged completed independent audit")
    names = (*COPIED, "current-checkpoint.json", "expanded-checkpoint.json", "fit.json",
             "train-dataset.jsonl", "validation-dataset.jsonl", "candidate-sources.json",
             "library-summary.json", "library-manifest.json", "all-positions.json", "layouts.json")
    hashes = {}
    for name in names:
        checksum = sha(study/name)
        if manifest.get(name) != checksum:
            raise ValueError("Expanded parent frozen input changed: "+name)
        hashes[name] = checksum
    for name in COPIED:
        if hashes[name] != sha(a.neural_study/name):
            raise ValueError("Expanded parent and current baseline source differ: "+name)
    if hashes["current-checkpoint.json"] != sha(a.neural_study/"neural-checkpoint.json"):
        raise ValueError("Expanded parent does not contain the current frozen model")
    for name in ("train-dataset.jsonl", "validation-dataset.jsonl", "candidate-sources.json",
                 "library-summary.json", "library-manifest.json", "all-positions.json"):
        if hashes[name] != sha(a.library/name):
            raise ValueError("Expanded library differs from audited parent: "+name)
    return dict(study=str(study.resolve()), audit=str(audit_path.resolve()),
        report_sha256=sha(study/"report.json"), audit_sha256=sha(audit_path),
        manifest_sha256=sha(study/"evidence-manifest.json"), inputs=hashes)


def schedule(layouts):
    """Seeded placement order and exactly 25 first turns for each method."""
    order = [row["id"] for row in layouts]
    rng = random.Random(PROCESSING_SEED)
    rng.shuffle(order)
    first = [METHODS[i % 2] for i in range(len(order))]
    rng.shuffle(first)
    return order, {identity: [method, next(m for m in METHODS if m != method)]
                   for identity, method in zip(order, first, strict=True)}


def freeze(a):
    from .board_first_score import VERSION as score_version
    from .corrected_grid import source_manifest
    from .approach_comparison_plan import VERSION as planner_version, description
    cpu_only()
    spec, parent, reference, source_contract, binding = source_inputs(a)
    library_inputs(a)
    expanded = expanded_parent_binding(a)
    if a.output.exists():
        raise ValueError("Preserve the existing study; use a fresh output")
    layout_archive = fresh_layouts(a, reference, source_contract)
    a.output.mkdir(parents=True, exist_ok=False)
    report = dict(version=VERSION, status="running", phase="freezing", audit_status="pending", controls={},
        progress=dict(prepared=0, completed_releases=0, planned_releases=None),
        gpu_lease_s=0., gpu_wait_s=0., training=dict(performed=False,
        explanation="Both checkpoints and their training libraries are copied without changes from the audited parent."))
    publish(a, report)
    for name in expanded["inputs"]:
        if name != "layouts.json":
            shutil.copyfile(a.expanded_study/name, a.output/name)
    shutil.copyfile(a.parent_study/"setup-home.json", a.output/"setup-home.json")
    atomic(a.output/"layouts.json", layout_archive)
    source = Path(__file__).resolve().parents[2]
    frozen_inputs = {p.name:sha(p) for p in a.output.iterdir() if p.is_file() and p.name != "report.json"}
    order, method_order = schedule(layout_archive["layouts"])
    contract = dict(version=VERSION, methods=list(METHODS), parent=parent, neural_parent=binding,
        expanded_parent=expanded, training_performed=False,
        identity=spec["identity"], scene=spec["scene"], flight_ranking=source_contract["flight_ranking"],
        current_planner_binding=current_planner_binding(a), approach_planner_version=planner_version,
        approach_planner=description(),
        scoring_version=score_version, frozen_inputs=frozen_inputs, source_sha256=source_manifest(source),
        library_path=str(a.library.resolve()), library_summary_sha256=sha(a.library/"library-summary.json"),
        library_manifest_sha256=sha(a.library/"library-manifest.json"),
        test_positions=POSITIONS, attempts_per_method_per_position=1, physics_substeps=64,
        planning_budget_s=PLANNING_S, reference_path_s=5., early_score_m=.005,
        gpu_cap_s=GPU_CAP_S, workers=WORKERS, seed=SEED,
        processing_seed=PROCESSING_SEED, processing_order=order, method_order=method_order,
        max_native_releases=118, matched_controls=18,
        scope="Frozen current live hybrid versus frozen expanded selector with a broader menu and full bucket approach forecast; privileged geometry, held endpoints, one shot per fresh placement; no training or live promotion")
    contract["fingerprint"] = digest(contract)
    atomic(a.output/"contract.json", contract)
    report.update(phase="models_frozen")
    publish(a, report)


def frozen(a):
    contract = read(a.output / "contract.json")
    if contract.get("fingerprint") != digest({k: v for k, v in contract.items() if k != "fingerprint"}):
        raise ValueError("Study contract fingerprint changed")
    for name, expected in contract["frozen_inputs"].items():
        if sha(a.output / name) != expected:
            raise ValueError("Frozen input changed: " + name)
    source = Path(__file__).resolve().parents[2]
    for name, expected in contract["source_sha256"].items():
        if sha(source/name) != expected:raise ValueError("Computational source changed: "+name)
    return contract


def planner_inputs(a):
    from .intergrid_library_plan import libraries
    from .expanded_selector_plan import VERSION as menu_version
    from .board_first_selector_plan import VERSION as ranking_version
    contract = frozen(a)
    _, _, reference, _, _ = source_inputs(a)
    raw = read(a.output/"shot-library-rows.json")
    current = libraries(raw["original"], raw["additions"])
    libs = dict(current=current, expanded=dict(expanded_menu_version=menu_version,
        baseline_libraries=current, expanded_sources=read(a.output/"candidate-sources.json")))
    model, history = (read(a.output/name) for name in ("route-models.json", "route-history.json"))
    checkpoints = {m:read(a.output/(m+"-checkpoint.json")) for m in METHODS}
    kwargs = dict(prior=read(a.output/"training-prior.json"), calibration=read(a.output/"calibration.json"), home=read(a.output/"setup-home.json"))
    common = dict(scene=reference["scene"], calibration=kwargs["calibration"]["score_config"], assets=str(a.assets), rule=contract["flight_ranking"]["rule"])
    configs = dict(current=common, expanded=dict(common, contact_ranking_version=ranking_version))
    from .approach_comparison_plan import configure
    libs["expanded"], configs["expanded"] = configure(libs["expanded"], configs["expanded"], checkpoints["expanded"])
    return reference, libs, model, history, checkpoints, kwargs, configs


def start_preparation(output, contract):
    """An incomplete preparation is preserved, never silently overwritten."""
    output = Path(output)
    partial = [output/name for name in ("preparation-started.json", "choices.json", "plans.json", "preparation", "native-started.json")]
    partial.extend(output.glob("native-*"))
    if any(path.exists() for path in partial):
        raise ValueError("Prior or partial preparation exists; preserve it and use a fresh study output")
    receipt = dict(contract_sha256=sha(output/"contract.json"),
                   layouts_sha256=sha(output/"layouts.json"),
                   frozen_inputs=contract["frozen_inputs"], timestamp_unix=time.time())
    with (output/"preparation-started.json").open("x") as stream:
        json.dump(receipt,stream,sort_keys=True)
    return receipt


def prepare(a):
    from contextlib import ExitStack
    from .stalled_reference_plan import StalledReferencePlanner
    from .approach_comparison_plan import ApproachComparisonPlanner
    from .planning_budget_plan import _wrap_choice
    cpu_only()
    contract = frozen(a)
    reference, libs, model, history, checkpoints, kwargs, configs = planner_inputs(a)
    start_preparation(a.output,contract)
    layouts = read(a.output/"layouts.json")["layouts"]
    by_id = {r["id"]:r for r in layouts}
    results, report = {}, read(a.output/"report.json")
    with ExitStack() as stack:
        planners = {m:stack.enter_context((StalledReferencePlanner if m == "current" else ApproachComparisonPlanner)("warm_early", libs[m], model, history, configs[m], checkpoints[m],
            reference=reference, assets=a.assets, directory=a.output/("native-"+m), engine_kwargs=kwargs, budget_s=PLANNING_S)) for m in METHODS}
        pool = stack.enter_context(ThreadPoolExecutor(max_workers=WORKERS))
        for identity in contract["processing_order"]:
            jobs = {pool.submit(planners[m].plan, by_id[identity]):m for m in contract["method_order"][identity]}
            for future in as_completed(jobs):
                method, site = jobs[future], future.result()
                if site["status"] not in {"completed", "deadline"}:
                    raise RuntimeError("Planner worker failed: "+str(site))
                results[identity,method] = site
                atomic(a.output/"preparation"/method/(identity+".json"), site)
                report["progress"]["prepared"] = len(results)
                report["phase"] = "preparing"
                publish(a, report)
    plans = [dict(variant=m, site=results[l["id"],m]) for l in layouts for m in METHODS]
    atomic(a.output/"plans.json", plans)
    slots = [_wrap_choice(p["site"]["choice"],p["variant"]) for p in plans]
    proofs = {}
    for slot in slots:
        request = slot["request"]
        if request is None:continue
        for prefix in ("context_resolution", "prepared_connection"):
            name = "native-"+slot["method"]+"/"+request[prefix+"_file"]
            expected = request[prefix+"_sha256"]
            if sha(contained(a.output,name)) != expected:raise ValueError("Incomplete prepared proof")
            proofs[name] = expected
    atomic(a.output/"admission-proof-bindings.json", proofs)
    bindings = dict(contract["frozen_inputs"], **{"contract.json":sha(a.output/"contract.json"), "plans.json":sha(a.output/"plans.json"), "preparation-started.json":sha(a.output/"preparation-started.json")})
    bindings.update(proofs)
    # Freeze all accepted and late preparation evidence, not only the winning proofs.
    bindings.update({str(p.relative_to(a.output)):sha(p) for p in a.output.rglob("*")
                     if p.is_file() and p.name not in {"report.json", "choices.json"}})
    bindings["admission-proof-bindings.json"] = sha(a.output/"admission-proof-bindings.json")
    atomic(a.output/"choices.json",dict(slots=slots,bindings=bindings))
    report["phase"] = "prepared"
    report["progress"]["planned_releases"] = sum(s["request"] is not None for s in slots)
    publish(a,report)


def attach_native_scores(scene, calibration, measured, arrays):
    import numpy as np
    from . import board_first_score, board_first_score_oracle
    from .table_assisted_env import _same

    for row in measured:
        col = row["native_cache_row"]
        if "record_ids" in arrays and str(arrays["record_ids"][col]) != row["id"]:
            raise ValueError("Native score column and record ID differ")
        trace = {key: value[:, col] if getattr(value, "ndim", 0) >= 2 and key != "time_s" else value
                 for key, value in arrays.items()}
        if not np.array_equal(trace["position_m"][0], row["native_initial_ball_center_m"]):
            raise ValueError("Native score column and release identity differ")
        args = (scene, row["bucket_xy_m"], row["board_normal"], calibration, trace)
        revised = board_first_score.score_trace(*args)
        _same(revised, board_first_score_oracle.score_trace(*args))
        for key in ("success", "board_bounce", "whole_ball_entry", "max_retention_s", "physics_steps", "invalid_physics"):
            _same(row[key], revised[key], "unchanged_kernel/" + key)
        row["board_first_score"] = revised
        row["board_first_scalar_audit_passed"] = True


def board_first_controls(rows):
    from .table_training_study import control_gate
    gate = control_gate(rows)
    expected = {"direct_control": True, "table_control": True, "negative_control": False}
    new_ok = all(row.get("board_first_scalar_audit_passed") is True
                 and not row["board_first_score"]["invalid_physics"]
                 and row["board_first_score"]["route_success"] is expected.get(row.get("role")) for row in rows)
    gate["board_first_captures"] = sum(row["board_first_score"]["route_success"] for row in rows)
    gate["board_first_passed"] = new_ok
    if not new_ok:
        gate["status"] = "failed"
    return gate


def run_native(a):
    from .cell1050_heading_native import HeadingEngine
    from .direct_path_first import DirectPathEngine, hydrate_direct_resolution
    from .stalled_reference_study import execute_mixed
    from .cell1050_rebound_study import control_requests, annotate
    from .intergrid_route_labels import annotate_routes
    from .coverage_training_study import append
    from .gpu_slot import gpu_slot, foreign_processes
    from .intergrid_system_study import gpu_deadline, require_storage
    from .parallel_capture_grid_study import control_mirror, NATIVE_BATCH
    from .parallel_capture_grid_workers import load_resolution
    from .table_correction_native import MatchedEngine

    contract = frozen(a)
    from .approach_comparison_audit import require_preflight
    require_preflight(a)
    _, _, reference, _, _ = source_inputs(a)
    report = read(a.output / "report.json")
    if report["phase"] not in {"prepared", "waiting_for_4090"}:
        raise ValueError("Complete preparation before native evaluation")
    choices = read(a.output / "choices.json")
    bindings = dict(choices["bindings"], **{"choices.json": sha(a.output / "choices.json")})
    planned = planned_requests(contract, choices["slots"])
    start_receipt = dict(bindings=bindings, requests_sha256=digest(planned), preflight_sha256=sha(a.preflight_output))
    if (a.output/"native-started.json").exists():
        raise ValueError("Preserve prior native execution; no implicit partial retry")
    kwargs = dict(prior=read(a.output / "training-prior.json"), calibration=read(a.output / "calibration.json"),
                  home=read(a.output / "setup-home.json"))
    options = read(a.parent_study / "native-yaw/effective-options.json")
    geometry = read(a.parent_study / "native-yaw/native-compiled-geometry.json")
    packet_number = 0

    def execute(engine, batch, phase, directory, control=False):
        nonlocal packet_number
        if foreign_processes():
            raise RuntimeError("A foreign workload appeared on the leased4090")
        require_storage(a.output)
        if any(sha(a.output / name) != expected for name, expected in bindings.items()):
            raise ValueError("Frozen choices changed during native execution")
        if not control:
            for xy in {tuple(row["bucket_xy_m"]) for row in batch}:
                anchor = hydrate_direct_resolution(engine, list(xy)) if isinstance(engine, DirectPathEngine) else load_resolution(engine, list(xy))
                wanted = {r["context_resolution_sha256"] for r in batch if tuple(r["bucket_xy_m"]) == xy}
                if anchor is None or wanted != {anchor["sha256"]}:
                    raise ValueError("Exact prepared resolution unavailable; native replanning is forbidden")
        name = f"requests/batch-{packet_number:05d}.json"
        packet_number += 1
        atomic(a.output / name, dict(rows=batch, phase=phase, control=control, adapter_directory=directory))
        measured, timing = engine.evaluate(batch, phase)
        if [row["id"] for row in measured] != [row["id"] for row in batch] or any(row["native_release"] is not True for row in measured):
            raise ValueError("Frozen admitted command did not release exactly once")
        adapter = engine.engine.adapter if control else engine.adapter
        if adapter.env.effective_options != options or adapter.env.compiled_geometry != geometry:
            raise ValueError("Native physics differs from matched controls")
        if not control:
            annotate(measured, a.output / directory)
            annotate_routes(measured, a.output / directory)
        attach_native_scores(reference["scene"], kwargs["calibration"]["score_config"], measured, adapter.env.last_arrays)
        for row in measured:
            row.update(request_packet=name, request_packet_sha256=sha(a.output / name), adapter_directory=directory)
        append(a.output / "rows.jsonl", measured)
        append(a.output / "timing.jsonl", [dict(packet=name, phase=phase, timing=timing)])
        return measured

    prior_wait_s = float(report.get("gpu_wait_s", 0.))
    def waiting(value):
        report.pop("queue_expired", None)
        report.pop("error", None)
        report.update(phase="waiting_for_4090", gpu_wait=dict(status=value["status"], wait_s=value["wait_s"]),
                      gpu_wait_s=prior_wait_s + value["wait_s"])
        publish(a, report)

    wait_start = time.monotonic()
    with gpu_slot(a.gpu_lock, timeout_s=a.gpu_wait_s, poll_s=30, notify=waiting):
        report["gpu_wait_s"] = prior_wait_s + time.monotonic() - wait_start
        report.pop("queue_expired", None)
        report.pop("error", None)
        if foreign_processes():raise RuntimeError("4090 has foreign work despite the lease")
        frozen(a)
        if any(sha(a.output/name) != value for name,value in bindings.items()):
            raise ValueError("Frozen inputs changed while waiting for the GPU")
        with (a.output/"native-started.json").open("x") as stream:
            json.dump(start_receipt,stream)
        begin = time.monotonic()
        with gpu_deadline(GPU_CAP_S):
            engines = {method:dict(reference=HeadingEngine(reference,a.assets,a.output/("native-"+method),**kwargs),
                                   direct_first=DirectPathEngine(reference,a.assets,a.output/("native-"+method),**kwargs)) for method in METHODS}
            sources = read(a.parent_study / "control-replays.json")
            mirror = control_mirror(a.parent_study, a.output, sources, False)
            controls = MatchedEngine(reference, a.assets, a.output / "native-controls", parent=mirror,
                         sources=sources, prior=read(a.parent_study / "old-prior.json"), calibration=kwargs["calibration"], home=kwargs["home"])
            adapters = [e.adapter for pair in engines.values() for e in pair.values()] + [controls.engine.adapter]
            factory, allocated = adapters[0].environment_factory, []
            def shared():
                if not allocated:
                    allocated.append(factory())
                return allocated[0]
            for adapter in adapters:
                adapter.environment_factory = shared
            for phase in ("controls_pre", "test", "controls_post"):
                if phase != "test":
                    gate = board_first_controls(execute(controls, control_requests(sources, phase), phase, "native-controls", True))
                    report["controls"][phase] = gate
                    if gate["status"] != "passed":
                        raise RuntimeError("Matched controls failed: " + phase)
                else:
                    by_slot = {(s["context_id"],s["method"]):s for s in choices["slots"]}
                    for identity in contract["processing_order"]:
                        for method in contract["method_order"][identity]:
                            request = by_slot[identity,method]["request"]
                            if request is not None:
                                execute_mixed(engines[method], [request], "test_" + method, "native-" + method, execute)
                                report["progress"]["completed_releases"] += 1
                                report.update(phase="test_" + method, gpu_lease_s=time.monotonic() - begin)
                                publish(a, report)
                report["phase"] = phase
                publish(a, report)
        report["gpu_lease_s"] = time.monotonic() - begin
    report.update(native_completed=True, phase="native_completed")
    report.pop("gpu_wait", None)
    publish(a, report)


def planned_requests(contract, slots):
    lookup = {(s["context_id"], s["method"]):s["request"] for s in slots}
    return {method:[lookup[identity,method] for identity in contract["processing_order"]
                    if lookup[identity,method] is not None] for method in METHODS}


def percentile90(values):
    """Nearest-rank p90; declared rather than interpolated after seeing timings."""
    import math
    return sorted(values)[max(0, math.ceil(.9*len(values))-1)]


def summarize(layouts, slots, rows, plans):
    from statistics import median
    observed = [row for row in rows if row["adapter_directory"] != "native-controls"]
    lookup = {row["id"]: row for row in observed}
    expected = {slot["id"] for slot in slots if slot["request"] is not None}
    if len(lookup) != len(observed) or len(lookup) != len(expected) or set(lookup) != expected:
        raise ValueError("Exactly one observed drop per admitted slot required")
    methods, per_position = {}, []
    success = {}
    for method in METHODS:
        selected = [slot for slot in slots if slot["method"] == method]
        if len(selected) != len(layouts) or {s["context_id"] for s in selected} != {l["id"] for l in layouts}:
            raise ValueError("Every method needs every frozen position")
        native = [lookup[s["id"]] for s in selected if s["request"] is not None]
        for row in native:
            if row.get("board_first_scalar_audit_passed") is not True:
                raise ValueError("Native score must pass its independent oracle")
        hits = sum(row["board_first_score"]["route_success"] for row in native)
        invalid = sum(row["board_first_score"]["invalid_physics"] for row in native)
        times = [plan["site"]["decision_latency_s"] for plan in plans if plan["variant"] == method]
        requests = [plan["site"]["request_wall_s"] for plan in plans if plan["variant"] == method]
        methods[method] = dict(positions=len(layouts), captures=hits, releases=len(native),
              setup_rejections=len(layouts)-len(native), invalid_physics=invalid,
              flight_misses=len(native)-hits-invalid, capture_rate=hits/len(layouts),
              mean_planning_s=sum(times)/len(times), median_planning_s=median(times), max_planning_s=max(times), p90_planning_s=percentile90(times), mean_request_s=sum(requests)/len(requests), median_request_s=median(requests), max_request_s=max(requests), p90_request_s=percentile90(requests))
        for slot in selected:
            success[slot["context_id"], method] = bool(slot["request"] is not None and lookup[slot["id"]]["board_first_score"]["route_success"])
    paired = dict(both=0, current_only=0, expanded_only=0, neither=0)
    for layout in layouts:
        first, second = [success[layout["id"], method] for method in METHODS]
        paired["both" if first and second else "current_only" if first else "expanded_only" if second else "neither"] += 1
        per_position.append(dict(id=layout["id"], bucket_xy_m=layout["bucket_xy_m"], captures={method: success[layout["id"], method] for method in METHODS}))
    return dict(methods=methods, paired=paired, positions=per_position)


def finalize(a):
    frozen(a)
    report = read(a.output / "report.json")
    if report.get("native_completed") is not True:
        raise ValueError("Native evaluation and controls must finish")
    report["comparison"] = summarize(read(a.output / "layouts.json")["layouts"], read(a.output / "choices.json")["slots"],
                                      lines(a.output / "rows.jsonl"), read(a.output / "plans.json"))
    atomic(a.output / "evidence-manifest.json", {str(p.relative_to(a.output)): sha(p) for p in sorted(a.output.rglob("*"))
           if p.is_file() and p.name not in {"report.json", "evidence-manifest.json"}})
    report.update(status="completed", phase="completed", audit_status="pending",
                  evidence_manifest_sha256=sha(a.output / "evidence-manifest.json"),
                  next="Independent read-only audit before publishing or promoting any selector.")
    publish(a, report)


def record_failure(a, stage, existed, exc):
    """A queue expiration may be explicitly retried; a partial native run may not."""
    report_path = a.output/"report.json"
    if not report_path.exists() or (stage == "freeze" and existed):
        return
    report = read(report_path)
    if report.get("version") != VERSION or report.get("status") in {"completed", "failed"}:
        return
    if stage == "run" and isinstance(exc, TimeoutError) and not (a.output/"native-started.json").exists():
        report.update(status="running", phase="waiting_for_4090", queue_expired=True,
                      error="Queue wait expired without starting native simulation. Explicit run retry is permitted.")
        report.setdefault("queue_expirations", []).append(dict(timestamp_unix=time.time(), cumulative_wait_s=report.get("gpu_wait_s",0.)))
    else:
        report.update(status="failed", phase=stage+"_failed", error=type(exc).__name__+": "+str(exc))
    publish(a, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("neural-study", "neural-audit", "current-planner-study", "current-planner-audit", "library", "expanded-study", "expanded-audit", "parent-study", "parent-audit",
                 "reference", "assets", "output", "gpu-lock"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--history-root", action="append", type=Path, required=True)
    parser.add_argument("--gpu-wait-s", type=int, default=43200)
    parser.add_argument("--stage", choices=("freeze", "prepare", "preflight", "run", "finalize", "audit"), required=True)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--original-source-root", type=Path)
    args = parser.parse_args()
    if args.gpu_wait_s <= 0:
        parser.error("Positive GPU wait limit required")
    if args.stage in {"audit", "preflight"}:
        from .approach_comparison_audit import audit, preflight
        if args.stage == "preflight":
            args.audit_output = args.preflight_output
        if args.audit_output is None or args.audit_output.exists() or args.audit_output.resolve().is_relative_to(args.output.resolve()):
            parser.error("Use a fresh audit output outside immutable evidence")
        result = (preflight if args.stage == "preflight" else audit)(args)
        atomic(args.audit_output, result)
        print(json.dumps({k:v for k,v in result.items() if k != "comparison"}), flush=True)
    else:
        existed = args.output.exists()
        try:
            dict(freeze=freeze, prepare=prepare, run=run_native, finalize=finalize)[args.stage](args)
        except Exception as exc:
            record_failure(args, args.stage, existed, exc)
            raise


if __name__ == "__main__":
    main()
