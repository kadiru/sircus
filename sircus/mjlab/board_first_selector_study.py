"""Re-label the existing TRAIN/VAL only, freeze two selectors, test 50 new sites.

Native physics and setup checks remain the original audited native64 engine.
The board-first score is attached alongside its historical v3 observations.
Both selectors receive the same contact-permitting ballistic ranking rule.
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

VERSION = "sircus-board-first-selector-comparison-v1"
METHODS = ("original", "retrained")
POSITIONS = 50
SEED = 2026112401
PLANNING_S = 30.
GPU_CAP_S = 3600.
WORKERS = 4
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
        raise ValueError("Fit/preparation requires CUDA_VISIBLE_DEVICES='' explicitly")


def source_inputs(a):
    from .planning_budget_study import audited_neural
    from .clearance_expansion_study import native_inputs

    neural_binding = audited_neural(a)
    spec, parent_binding, source, reference = native_inputs(a)
    neural_contract = read(a.neural_study / "contract.json")
    if spec["identity"] != neural_contract["identity"]:
        raise ValueError("Historical neural and control scene identities differ")
    manifest = read(a.neural_study / "evidence-manifest.json")
    for name in ("training-dataset.jsonl", "validation-dataset.jsonl", "collection-plans.json", *COPIED):
        if sha(a.neural_study / name) != manifest[name]:
            raise ValueError("Historical dataset changed: " + name)
    return spec, parent_binding, reference, neural_contract, neural_binding


def ledger_by_id(a):
    report, audit = read(a.rescore_study / "summary.json"), read(a.rescore_audit)
    manifest = read(a.rescore_study / "manifest.json")
    if (report.get("status") != "verified" or audit.get("status") != "passed"
            or audit.get("summary_sha256") != sha(a.rescore_study / "summary.json")
            or audit.get("manifest_sha256") != sha(a.rescore_study / "manifest.json")
            or report.get("manifest_sha256") != sha(a.rescore_study / "manifest.json")):
        raise ValueError("A passed audit bound to the new rescoring report is required")
    for name, expected in manifest["artifacts"].items():
        if sha(contained(a.rescore_study, name)) != expected:
            raise ValueError("Rescoring artifact changed: " + name)
    expected_study = a.neural_study.resolve().relative_to(a.rescore_root.resolve()).as_posix()
    indexed = {}
    for item in lines(a.rescore_study / "records.jsonl"):
        if item.get("status") != "scored":
            continue
        if item["source_study"] != expected_study:
            continue
        identity = item["original"]["id"]
        if identity in indexed and indexed[identity] != item:
            raise ValueError("Conflicting rescored native identity: " + identity)
        indexed[identity] = item
    if not indexed:
        raise ValueError("No rescored neural-source rows found")
    return indexed


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
    sampled = sample_splits(sorted(exclusions), geometry, grid, seed=SEED, include_train=False)["test"]
    layouts = [dict(row, id=f"board-first-test-{i:02d}", context_id=f"board-first-test-{i:02d}", split="test")
               for i, row in enumerate(sampled)]
    if len(layouts) != POSITIONS:
        raise ValueError("Expected exactly 50 fresh positions")
    return dict(layouts=layouts, provenance=dict(excluded_coordinates_m=sorted(exclusions),
                geometry=geometry, history_files=inventory, layouts_sha256=digest(layouts), minimum_separation_m=.005))


def fit(a):
    from .board_first_selector_model import attach_scores, fit as train, label_changes
    from .board_first_selector_plan import VERSION as ranking_version
    from .board_first_score import VERSION as score_version
    from .corrected_grid import source_manifest
    from .coverage_training_study import append

    cpu_only()
    spec, parent, reference, source_contract, binding = source_inputs(a)
    revised = ledger_by_id(a)
    original_checkpoint = read(a.neural_study / "neural-checkpoint.json")
    data = {split: attach_scores(lines(a.neural_study / name), revised)
            for split, name in (("train", "training-dataset.jsonl"), ("validation", "validation-dataset.jsonl"))}
    if a.output.exists():
        raise ValueError("Preserve the existing study; use a fresh output")
    # Inventory history before creating our own mutable progress report, which
    # would otherwise become part of the immutable historical exclusion set.
    layout_archive = fresh_layouts(a, reference, source_contract)
    a.output.mkdir(parents=True, exist_ok=False)
    report = dict(version=VERSION, status="running", phase="fitting", audit_status="pending", controls={},
                  progress=dict(prepared=0, completed_releases=0, planned_releases=None), gpu_lease_s=0.)
    publish(a, report)
    for name in COPIED:
        shutil.copyfile(a.neural_study / name, a.output / name)
    shutil.copyfile(a.parent_study / "setup-home.json", a.output / "setup-home.json")
    shutil.copyfile(a.neural_study / "neural-checkpoint.json", a.output / "original-checkpoint.json")
    atomic(a.output / "layouts.json", layout_archive)
    for split, records in data.items():
        append(a.output / (split + "-dataset.jsonl"), records)
    fitted = train(data["train"], data["validation"], seed=original_checkpoint["seed"],
                   epochs=tuple(original_checkpoint["checkpoint_epochs"]))
    atomic(a.output / "retrained-checkpoint.json", fitted["checkpoint"])
    atomic(a.output / "fit.json", dict(validation=fitted["validation"],
           label_changes={split: label_changes(records) for split, records in data.items()}))
    source = Path(__file__).resolve().parents[2]
    frozen = {p.name: sha(p) for p in a.output.iterdir() if p.is_file() and p.name != "report.json"}
    contract = dict(version=VERSION, methods=list(METHODS), parent=parent, neural_parent=binding,
                    identity=spec["identity"], scene=spec["scene"], flight_ranking=source_contract["flight_ranking"],
                    scoring_version=score_version, contact_ranking_version=ranking_version,
                    rescore_summary_sha256=sha(a.rescore_study / "summary.json"), rescore_audit_sha256=sha(a.rescore_audit),
                    rescore_manifest_sha256=sha(a.rescore_study / "manifest.json"),
                    rescore_records_sha256=sha(a.rescore_study / "records.jsonl"),
                    frozen_inputs=frozen, source_sha256=source_manifest(source),
                    test_positions=POSITIONS, attempts_per_method_per_position=1, physics_substeps=64,
                    planning_budget_s=PLANNING_S, gpu_cap_s=GPU_CAP_S, workers=WORKERS, seed=SEED,
                    scope="Same frozen proposal library, features and architecture; only corrected TRAIN/VAL labels; same new contact-permitting ranker; privileged geometry and held endpoints; no live promotion")
    contract["fingerprint"] = digest(contract)
    atomic(a.output / "contract.json", contract)
    report.update(phase="models_frozen", training=dict(selected_epoch=fitted["checkpoint"]["epoch"],
                  training_rows=len(data["train"]), validation_rows=len(data["validation"]),
                  label_changes=read(a.output / "fit.json")["label_changes"]))
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
        if sha(source / name) != expected:
            raise ValueError("Computational source changed: " + name)
    return contract


def prepare(a):
    from .board_first_selector_plan import run_job
    from .intergrid_library_plan import libraries

    cpu_only()
    contract = frozen(a)
    _, _, reference, _, _ = source_inputs(a)
    if (a.output / "choices.json").exists():
        raise ValueError("Preserve frozen test choices")
    raw = read(a.output / "shot-library-rows.json")
    libs = libraries(raw["original"], raw["additions"])
    source_model, history = [read(a.output / name) for name in ("route-models.json", "route-history.json")]
    checkpoints = {method: read(a.output / (method + "-checkpoint.json")) for method in METHODS}
    kwargs = dict(prior=read(a.output / "training-prior.json"), calibration=read(a.output / "calibration.json"),
                  home=read(a.output / "setup-home.json"))
    config = dict(scene=reference["scene"], calibration=kwargs["calibration"]["score_config"],
                  assets=str(a.assets), rule=contract["flight_ranking"]["rule"])
    layouts = read(a.output / "layouts.json")["layouts"]
    report = read(a.output / "report.json")
    tasks = [(layout, method) for layout in layouts for method in METHODS]
    random.Random(SEED).shuffle(tasks)
    results = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        jobs = {pool.submit(run_job, layout, method, libs, source_model, history, config, checkpoints[method],
                    reference=reference, assets=a.assets, directory=a.output / ("native-" + method),
                    engine_kwargs=kwargs, budget_s=PLANNING_S): (layout, method) for layout, method in tasks}
        for future in as_completed(jobs):
            layout, method = jobs[future]
            result = future.result()
            results[layout["id"], method] = result
            atomic(a.output / "preparation" / method / (layout["id"] + ".json"), result)
            report["progress"]["prepared"] = len(results)
            report["phase"] = "preparing"
            publish(a, report)
    plans = [results[layout["id"], method] for layout in layouts for method in METHODS]
    atomic(a.output / "plans.json", plans)
    slots = [plan["choice"] for plan in plans]
    proofs = {}
    for slot in slots:
        request = slot["request"]
        if request is None:
            continue
        directory = "native-" + slot["method"]
        for prefix in ("context_resolution", "prepared_connection"):
            name = directory + "/" + request[prefix + "_file"]
            expected = request[prefix + "_sha256"]
            if sha(contained(a.output, name)) != expected:
                raise ValueError("Complete prepared proof unavailable: " + name)
            proofs[name] = expected
    atomic(a.output / "admission-proof-bindings.json", proofs)
    bindings = dict(contract["frozen_inputs"], **{"plans.json": sha(a.output / "plans.json")})
    bindings.update(proofs)
    bindings["admission-proof-bindings.json"] = sha(a.output / "admission-proof-bindings.json")
    atomic(a.output / "choices.json", dict(slots=slots, bindings=bindings))
    report["phase"] = "prepared"
    report["progress"]["planned_releases"] = sum(slot["request"] is not None for slot in slots)
    publish(a, report)


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
    from .cell1050_rebound_study import control_requests, annotate
    from .intergrid_route_labels import annotate_routes
    from .coverage_training_study import append
    from .gpu_slot import gpu_slot, foreign_processes
    from .intergrid_system_study import gpu_deadline, require_storage
    from .parallel_capture_grid_study import control_mirror, NATIVE_BATCH
    from .parallel_capture_grid_workers import load_resolution
    from .table_correction_native import MatchedEngine

    frozen(a)
    _, _, reference, _, _ = source_inputs(a)
    report = read(a.output / "report.json")
    if report["phase"] != "prepared":
        raise ValueError("Complete preparation before native evaluation")
    choices = read(a.output / "choices.json")
    bindings = dict(choices["bindings"], **{"choices.json": sha(a.output / "choices.json")})
    planned = {method: [slot["request"] for slot in choices["slots"] if slot["method"] == method and slot["request"] is not None]
               for method in METHODS}
    with (a.output / "native-started.json").open("x") as stream:
        json.dump(dict(bindings=bindings, requests_sha256=digest(planned)), stream)
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
                anchor = load_resolution(engine, list(xy))
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
        append(a.output / "timing.jsonl", [dict(packet=name, timing=timing)])
        return measured

    def waiting(value):
        report.update(phase="waiting_for_4090", gpu_wait=value)
        publish(a, report)

    with gpu_slot(a.gpu_lock, timeout_s=a.gpu_wait_s, poll_s=30, notify=waiting):
        begin = time.monotonic()
        with gpu_deadline(GPU_CAP_S):
            engines = {method: HeadingEngine(reference, a.assets, a.output / ("native-" + method), **kwargs) for method in METHODS}
            sources = read(a.parent_study / "control-replays.json")
            mirror = control_mirror(a.parent_study, a.output, sources, False)
            controls = MatchedEngine(reference, a.assets, a.output / "native-controls", parent=mirror,
                         sources=sources, prior=read(a.parent_study / "old-prior.json"), calibration=kwargs["calibration"], home=kwargs["home"])
            adapters = [e.adapter for e in engines.values()] + [controls.engine.adapter]
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
                    for offset in range(0, max(map(len, planned.values())), NATIVE_BATCH):
                        for method in METHODS:
                            batch = planned[method][offset:offset + NATIVE_BATCH]
                            if batch:
                                execute(engines[method], batch, "test_" + method, "native-" + method)
                                report["progress"]["completed_releases"] += len(batch)
                                report.update(phase="test_" + method, gpu_lease_s=time.monotonic() - begin)
                                publish(a, report)
                report["phase"] = phase
                publish(a, report)
        report["gpu_lease_s"] = time.monotonic() - begin
    report.update(native_completed=True, phase="native_completed")
    report.pop("gpu_wait", None)
    publish(a, report)


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
        times = [plan["decision_latency_s"] for plan in plans if plan["method"] == method]
        methods[method] = dict(positions=len(layouts), captures=hits, releases=len(native),
              setup_rejections=len(layouts)-len(native), invalid_physics=invalid,
              flight_misses=len(native)-hits-invalid, capture_rate=hits/len(layouts),
              mean_planning_s=sum(times)/len(times), median_planning_s=median(times), max_planning_s=max(times))
        for slot in selected:
            success[slot["context_id"], method] = bool(slot["request"] is not None and lookup[slot["id"]]["board_first_score"]["route_success"])
    paired = dict(both=0, original_only=0, retrained_only=0, neither=0)
    for layout in layouts:
        first, second = [success[layout["id"], method] for method in METHODS]
        paired["both" if first and second else "original_only" if first else "retrained_only" if second else "neither"] += 1
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("neural-study", "neural-audit", "rescore-study", "rescore-root", "rescore-audit", "parent-study", "parent-audit",
                 "reference", "assets", "output", "gpu-lock"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--history-root", action="append", type=Path, required=True)
    parser.add_argument("--gpu-wait-s", type=int, default=43200)
    parser.add_argument("--stage", choices=("fit", "prepare", "run", "finalize", "audit"), required=True)
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()
    if args.gpu_wait_s <= 0:
        parser.error("Positive GPU wait limit required")
    if args.stage == "audit":
        from .board_first_selector_audit import audit
        if args.audit_output is None or args.audit_output.exists() or args.audit_output.resolve().is_relative_to(args.output.resolve()):
            parser.error("Use a fresh audit output outside immutable evidence")
        result = audit(args)
        atomic(args.audit_output, result)
        print(json.dumps({k:v for k,v in result.items() if k != "comparison"}), flush=True)
    else:
        dict(fit=fit, prepare=prepare, run=run_native, finalize=finalize)[args.stage](args)


if __name__ == "__main__":
    main()
