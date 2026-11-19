"""Bounded geometry-to-shot learning, followed by untouched one-drop testing."""
from __future__ import annotations

import argparse
from concurrent.futures import as_completed
import json
from pathlib import Path
import random
import shutil
import time

from . import clearance_expansion_study as parent
from .benchmark_verify import digest, read, sha
from .cell1050_rebound_study import annotate, control_requests
from .corrected_grid import source_manifest
from .coverage_training_study import append
from .gpu_slot import foreign_processes, gpu_slot
from .intergrid_library_plan import libraries
from .intergrid_proposer_plan import make_splits, sample_splits
from .intergrid_route_labels import LABEL_VERSION, annotate_routes
from .intergrid_system_study import gpu_deadline, require_storage
from .neural_shot_plan import METHODS, prepare_collection, prepare_neural
from .parallel_capture_grid_study import NATIVE_BATCH, control_mirror
from .parallel_capture_grid_workers import load_resolution, open_workers
from .sweep_worker import atomic
from .table_training_study import control_gate

VERSION = "sircus-neural-shot-study-v1"
SPLIT_COUNTS = dict(train=160, validation=40, test=200)
SEED = 2026112901
WORKERS = 4
GPU_CAP_S = 7 * 3600
STAGE_CAPS = dict(collection=4 * 3600, test=3 * 3600)
NATIVE_DIRS = dict(train="native-collection", validation="native-collection",
                   test_baseline="native-test-baseline", test_neural="native-test-neural")


def split_layouts(excluded, geometry, grid, *, seed=SEED):
    """400 spatially disjoint placements; preserve the sampler's geometry gates."""
    points = [tuple(xy) for xy in excluded]
    rows = []
    for batch in range(8):
        sampled = sample_splits(points, geometry, grid, seed=seed + batch, include_train=False)["test"]
        rows.extend(sampled)
        points.extend(tuple(row["bucket_xy_m"]) for row in sampled)
    # Each 50-location batch spans all grid rectangles. A frozen shuffle avoids
    # making the final split exclusively the last rectangles or boundary cases.
    random.Random(seed).shuffle(rows)
    result, offset = {}, 0
    for split, count in SPLIT_COUNTS.items():
        selected = [dict(row, id=f"neural-shot-{split}-{i:03d}",
                         context_id=f"neural-shot-{split}-{i:03d}", split=split)
                    for i, row in enumerate(rows[offset:offset + count])]
        result[split] = dict(layouts=selected, layouts_sha256=digest(selected))
        offset += count
    return result


def publish(args, report, phase=None):
    if phase is not None:
        report["phase"] = phase
    report["updated_at_unix"] = time.time()
    atomic(args.output / "report.json", report)
    if args.display_report:
        display = read(args.display_report) if args.display_report.exists() else {}
        display["neural_shot_study"] = report
        atomic(args.display_report, display)
    print(json.dumps({key: report[key] for key in ("status", "phase", "progress", "gpu_wait", "error")
                      if key in report}), flush=True)


def initialize(args):
    spec, parent_binding, source, ref = parent.native_inputs(args)
    model, original, additions, previous, binding, history = parent.inputs(args)
    # Include the last comparison as exclusions, without using its outcomes to
    # train this model or alter the frozen planner rules.
    prior_audit, prior_report = read(args.expansion_audit), read(args.expansion_study / "report.json")
    if (prior_audit["status"] != "passed" or prior_report["status"] != "completed"
            or prior_audit["report_sha256"] != sha(args.expansion_study / "report.json")):
        raise ValueError("Latest clearance comparison must have passed its audit")
    previous += read(args.expansion_study / "layouts.json")["layouts"]
    grid = read(args.parent_study / "grid.json")
    _, provenance = make_splits(args.history_root, grid, scene=ref["scene"], assets=args.assets,
                                previous_layouts=previous, seed=SEED, include_train=False)
    layouts = split_layouts(provenance["excluded_coordinates_m"], provenance["geometry"], grid)
    args.output.mkdir(parents=True, exist_ok=False)
    require_storage(args.output)
    atomic(args.output / "layouts.json", layouts)
    atomic(args.output / "layout-provenance.json", provenance)
    atomic(args.output / "shot-library-rows.json", dict(original=original, additions=additions))
    atomic(args.output / "route-history.json", history)
    for name, source_path in (
        ("training-prior.json", args.model_study / "training-prior.json"),
        ("route-models.json", args.refit_study / "route-models.json"),
        ("calibration.json", args.parent_study / "calibration.json"),
        ("verification.json", args.parent_study / "verification.json"),
    ):
        shutil.copyfile(source_path, args.output / name)
    frozen = {p.name: sha(p) for p in args.output.glob("*.json")}
    contract = dict(version=VERSION, parent=parent_binding, identity=spec["identity"], scene=spec["scene"],
                    **binding, expansion_audit_sha256=sha(args.expansion_audit), split_counts=SPLIT_COUNTS,
                    seed=SEED, methods=list(METHODS), physics_substeps=64, label_version=LABEL_VERSION,
                    attempts_per_method_per_position=1, gpu_cap_s=GPU_CAP_S, stage_caps_s=STAGE_CAPS,
                    workers=WORKERS, frozen_inputs=frozen, source_sha256=source_manifest(source),
                    test_scope="Unseen continuous bucket XY, privileged geometry, endpoint-held native64",
                    model_scope="Pre-IK supervised ranking, exact setup and physics selection retained")
    contract["fingerprint"] = digest(contract)
    atomic(args.output / "contract.json", contract)
    report = dict(version=VERSION, status="running", phase="initialized", audit_status="pending",
                  split_positions=SPLIT_COUNTS, controls={}, gpu_lease_s=0., gpu_cap_s=GPU_CAP_S,
                  progress={split: dict(prepared_positions=0, completed_releases=0, planned_releases=None)
                            for split in SPLIT_COUNTS},
                  next="Prepare fresh shots; GPU work waits cooperatively for the 4090.")
    publish(args, report)


def frozen_check(args):
    contract = read(args.output / "contract.json")
    for name, expected in contract["frozen_inputs"].items():
        if sha(args.output / name) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    source = Path(__file__).resolve().parents[2]
    # Bind all computational Python sources, not just the runner entrypoint.
    for name, expected in contract["source_sha256"].items():
        if sha(source / name) != expected:
            raise ValueError(f"Producer source changed: {name}")
    return contract


def preparation_context(args):
    contract = frozen_check(args)
    _, _, _, ref = parent.native_inputs(args)
    kwargs = dict(prior=read(args.output / "training-prior.json"),
                  calibration=read(args.output / "calibration.json"),
                  home=read(args.parent_study / "setup-home.json"))
    config = dict(scene=ref["scene"], calibration=kwargs["calibration"]["score_config"],
                  assets=str(args.assets), rule=contract["flight_ranking"]["rule"])
    return ref, kwargs, config


def prepare(args, stage):
    from .clearance_expansion_plan import prepare_layout

    output = args.output
    ref, kwargs, config = preparation_context(args)
    layouts = read(output / "layouts.json")
    library_rows = read(output / "shot-library-rows.json")
    libs = libraries(library_rows["original"], library_rows["additions"])
    model, history = (read(output / name) for name in ("route-models.json", "route-history.json"))
    report = read(output / "report.json")
    if stage == "collection":
        rows = layouts["train"]["layouts"] + layouts["validation"]["layouts"]
        directory, function = "native-collection", prepare_collection
        extra = (libs, model, config)
    elif stage == "baseline":
        rows = layouts["test"]["layouts"]
        directory, function = "native-test-baseline", prepare_layout
        extra = (libs, model, history, config)
    else:
        rows = layouts["test"]["layouts"]
        directory, function = "native-test-neural", prepare_neural
        extra = (libs, model, history, config, read(output / "neural-checkpoint.json"))
    archive = output / ("collection-plans.json" if stage == "collection" else f"plans-test-{stage}.json")
    if archive.exists():
        raise ValueError("Preparation is already complete; preserve it")
    sites = {}
    with open_workers(WORKERS, ref, args.assets, output / directory, kwargs) as pool:
        futures = {pool.submit(function, row, *extra): row for row in rows}
        for future in as_completed(futures):
            row = futures[future]
            site = future.result()
            sites[row["id"]] = site
            folder = row["split"] if stage == "collection" else "test-" + stage
            atomic(output / "preparation" / folder / (row["id"] + ".json"), site)
            split = row["split"]
            count = sum(item["layout"]["split"] == split for item in sites.values())
            report["progress"][split]["prepared_positions"] = count
            if stage != "collection":
                report["progress"][split][stage + "_prepared_positions"] = count
            require_storage(output)
            publish(args, report, "preparing_" + stage)
    ordered = [sites[row["id"]] for row in rows]
    atomic(archive, ordered)
    if stage == "collection":
        for split in ("train", "validation"):
            report["progress"][split]["planned_releases"] = sum(
                len(site["requests"]) for site in ordered if site["layout"]["split"] == split)
    publish(args, report, stage + "_prepared")


def bind_test(args):
    output = args.output
    if (output / "choices-test.json").exists():
        raise ValueError("Test commands already frozen")
    plans = [read(output / f"plans-test-{name}.json") for name in ("baseline", "neural")]
    slots = []
    for first, second in zip(*plans, strict=True):
        if first["layout"] != second["layout"]:
            raise ValueError("Test planners must use identical bucket placements")
        slots.extend(first["choices"] + second["choices"])
    bindings = {name: sha(output / name) for name in
                ("plans-test-baseline.json", "plans-test-neural.json", "neural-checkpoint.json", "fit.json",
                 "training-dataset.jsonl", "validation-dataset.jsonl")}
    atomic(output / "choices-test.json", dict(slots=slots, bindings=bindings))
    report = read(output / "report.json")
    report["progress"]["test"]["planned_releases"] = sum(slot["request"] is not None for slot in slots)
    publish(args, report, "test_frozen")


def stage_requests(output, stage):
    if stage == "collection":
        plans = read(output / "collection-plans.json")
        return {split: [request for site in plans if site["layout"]["split"] == split
                        for request in site["requests"]] for split in ("train", "validation")}
    slots = read(output / "choices-test.json")["slots"]
    return {key: [slot["request"] for slot in slots if slot["request"] is not None
                   and (slot["method"] == "neural_selector") == (key == "test_neural")]
            for key in ("test_baseline", "test_neural")}


def native(args, stage):
    from .cell1050_heading_native import HeadingEngine
    from .table_correction_native import MatchedEngine

    output = args.output
    ref, kwargs, _ = preparation_context(args)
    contract = read(output / "contract.json")
    report = read(output / "report.json")
    prior_gpu = report.get("gpu_lease_s", 0.)
    requests = stage_requests(output, stage)
    names = ["collection-plans.json"] if stage == "collection" else [
        "choices-test.json", "plans-test-baseline.json", "plans-test-neural.json", "neural-checkpoint.json",
        "fit.json", "training-dataset.jsonl", "validation-dataset.jsonl"]
    bindings = dict(contract["frozen_inputs"], **{name: sha(output / name) for name in names})
    receipt = output / f"native-started-{stage}.json"
    if receipt.exists():
        raise ValueError("Native stage was already started; do not silently repeat drops")
    atomic(receipt, dict(stage=stage, bindings=bindings, requests_sha256=digest(requests),
                         timestamp_unix=time.time()))
    expected_options = read(args.parent_study / "native-yaw/effective-options.json")
    expected_geometry = read(args.parent_study / "native-yaw/native-compiled-geometry.json")
    packet_number = len(list((output / "requests").glob("*.json")))

    def execute(engine, batch, phase, directory, control=False):
        nonlocal packet_number
        if foreign_processes():
            raise RuntimeError("Foreign process appeared on the leased 4090")
        require_storage(output)
        if any(sha(output / name) != value for name, value in bindings.items()):
            raise ValueError("A frozen study input changed during native execution")
        if not control:
            for xy in {tuple(row["bucket_xy_m"]) for row in batch}:
                load_resolution(engine, list(xy))
        packet = f"requests/batch-{packet_number:05d}.json"
        packet_number += 1
        atomic(output / packet, dict(phase=phase, stage=stage, control=control,
                                     adapter_directory=directory, rows=batch))
        measured, timing = engine.evaluate(batch, phase)
        if [row["id"] for row in measured] != [row["id"] for row in batch] or any(
                row["native_release"] is not True for row in measured):
            raise RuntimeError("Admitted shot did not execute exactly once")
        adapter = engine.engine.adapter if control else engine.adapter
        if adapter.env.effective_options != expected_options or adapter.env.compiled_geometry != expected_geometry:
            raise RuntimeError("Native settings changed from audited controls")
        if not control:
            annotate(measured, output / directory)
            annotate_routes(measured, output / directory)
            for row, request in zip(measured, batch, strict=True):
                row["intended_route"] = request["intended_route"]
        for row in measured:
            row.update(request_packet=packet, request_packet_sha256=sha(output / packet),
                       adapter_directory=directory, study_stage=stage)
        append(output / "rows.jsonl", measured)
        append(output / "timing.jsonl", [dict(packet=packet, phase=phase, timing=timing)])
        return measured

    def waiting(status):
        report["gpu_wait"] = status
        publish(args, report, "waiting_for_4090_" + stage)

    started = None
    try:
        with gpu_slot(args.gpu_lock, timeout_s=args.gpu_wait_s, poll_s=30, notify=waiting):
            started = time.monotonic()
            with gpu_deadline(STAGE_CAPS[stage]):
                report.pop("gpu_wait", None)
                engines = {key: HeadingEngine(ref, args.assets, output / NATIVE_DIRS[key], **kwargs)
                           for key in requests}
                # Train/validation share a proof directory but use one engine.
                if stage == "collection":
                    engines["validation"] = engines["train"]
                control_sources = read(args.parent_study / "control-replays.json")
                mirror = control_mirror(args.parent_study, output, control_sources,
                                        (output / "control-parent").exists())
                control_directory = "native-controls-" + stage
                controls = MatchedEngine(ref, args.assets, output / control_directory, parent=mirror,
                                        sources=control_sources, prior=read(args.parent_study / "old-prior.json"),
                                        calibration=kwargs["calibration"], home=kwargs["home"])
                adapters = list({id(engine.adapter): engine.adapter for engine in engines.values()}.values())
                adapters.append(controls.engine.adapter)
                factory, allocated = adapters[0].environment_factory, []
                def shared_factory():
                    if not allocated:
                        allocated.append(factory())
                    return allocated[0]
                for adapter in adapters:
                    adapter.environment_factory = shared_factory
                for timing in ("pre", "post"):
                    phase = f"{stage}_controls_{timing}"
                    gate = control_gate(execute(controls, control_requests(control_sources, phase), phase,
                                                control_directory, True))
                    report["controls"][phase] = gate
                    if gate["status"] != "passed":
                        raise RuntimeError(f"Matched controls failed: {phase}")
                    publish(args, report, phase)
                    if timing == "post":
                        break
                    for split, planned in requests.items():
                        counter = split if split in SPLIT_COUNTS else "test"
                        for offset in range(0, len(planned), NATIVE_BATCH):
                            measured = execute(engines[split], planned[offset:offset + NATIVE_BATCH], split,
                                               NATIVE_DIRS[split])
                            report["progress"][counter]["completed_releases"] += len(measured)
                            report["gpu_lease_s"] = prior_gpu + time.monotonic() - started
                            publish(args, report, split)
        report["gpu_lease_s"] = prior_gpu + time.monotonic() - started
        report.setdefault("stages", {})[stage] = "completed"
        publish(args, report, stage + "_native_completed")
    finally:
        if started is not None:
            report["gpu_lease_s"] = prior_gpu + time.monotonic() - started
            atomic(output / "report.json", report)


def learning_rows(sites, observations):
    """Bind learning labels only to the exact selected collection commands."""
    wanted = [request for site in sites for request in site["requests"]]
    if len({row["id"] for row in wanted}) != len(wanted) or set(observations) != {row["id"] for row in wanted}:
        raise ValueError("Every and only selected collection drop must have one observation")
    for request in wanted:
        row = observations[request["id"]]
        if any(row[key] != request[key] for key in ("id", "proposal_id", "bucket_xy_m", "values")):
            raise ValueError("Native label differs from the exact proposed shot")
    datasets = dict(train=[], validation=[])
    for site in sites:
        admitted = {row["proposal_id"]: row for row in site["admitted"]}
        if len(admitted) != len(site["admitted"]):
            raise ValueError("Collection must admit at most one branch per proposal")
        split = site["layout"]["split"]
        if split not in datasets:
            raise ValueError("TEST cannot enter the learning data")
        for proposal in site["proposals"]:
            request = admitted.get(proposal["id"])
            datasets[split].append(dict(request=proposal, setup_ok=request is not None,
                                        observation=observations.get(request["id"]) if request else None,
                                        split=split))
    return datasets


def dataset(args):
    from .neural_shot_model import fit
    from .parallel_capture_grid_audit import observed_rows

    output = args.output
    if (output / "neural-checkpoint.json").exists():
        raise ValueError("Preserve the frozen checkpoint")
    if read(output / "report.json").get("stages", {}).get("collection") != "completed":
        raise ValueError("All collection drops and controls must complete before fitting")
    observations = {row["id"]: row for row in observed_rows(output / "rows.jsonl")
                    if row.get("adapter_directory") == "native-collection"}
    datasets = learning_rows(read(output / "collection-plans.json"), observations)
    for split, name in (("train", "training-dataset.jsonl"), ("validation", "validation-dataset.jsonl")):
        if (output / name).exists():
            raise ValueError("Preserve the recorded training datasets")
        append(output / name, datasets[split])
    result = fit(datasets["train"], datasets["validation"])
    atomic(output / "neural-checkpoint.json", result["checkpoint"])
    atomic(output / "fit.json", {key: value for key, value in result.items() if key != "checkpoint"})
    report = read(output / "report.json")
    report["training"] = dict(checkpoint_sha256=sha(output / "neural-checkpoint.json"),
                               selected_epoch=result["checkpoint"]["epoch"],
                               training_rows=len(datasets["train"]), validation_rows=len(datasets["validation"]))
    publish(args, report, "model_frozen")


def finalize(args):
    from .neural_shot_audit import summarize
    from .parallel_capture_grid_audit import observed_rows

    output = args.output
    report = read(output / "report.json")
    if report.get("stages", {}).get("test") != "completed":
        raise ValueError("Complete all test drops and controls before summarizing")
    rows = {row["id"]: row for row in observed_rows(output / "rows.jsonl")
            if row.get("adapter_directory") in {"native-test-baseline", "native-test-neural"}}
    report["comparison"] = summarize(read(output / "layouts.json")["test"]["layouts"],
                                      read(output / "choices-test.json")["slots"], rows,
                                      read(output / "plans-test-baseline.json"),
                                      read(output / "plans-test-neural.json"))
    atomic(output / "evidence-manifest.json", {str(path.relative_to(output)): sha(path)
           for path in sorted(output.rglob("*")) if path.is_file() and path.name not in {"report.json", "evidence-manifest.json"}})
    report.update(status="completed", audit_status="pending", evidence_manifest_sha256=sha(output / "evidence-manifest.json"),
                  next="Independent evidence audit before publishing scores or promoting a planner.")
    publish(args, report, "completed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("history-root", "parent-study", "parent-audit", "reference", "assets", "gpu-lock", "output",
                 "rule", "system-study", "system-audit", "interactive-history", "expansion-study", "expansion-audit"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in (*parent.SOURCE_NAMES, *parent.baseline.EXTRA_SOURCES, "prior-comparison"):
        for kind in ("study", "audit"):
            parser.add_argument(f"--{name}-{kind}", type=Path, required=True)
    parser.add_argument("--display-report", type=Path)
    parser.add_argument("--gpu-wait-s", type=int, default=12 * 3600)
    parser.add_argument("--stage", required=True, choices=("initialize", "prepare-collection", "run-collection",
                        "fit", "prepare-baseline", "prepare-neural", "bind-test", "run-test", "finalize", "audit"))
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()
    if args.gpu_wait_s <= 0:
        parser.error("Use a positive GPU wait time")
    try:
        if args.stage == "initialize":
            initialize(args)
        elif args.stage.startswith("prepare-"):
            prepare(args, args.stage.removeprefix("prepare-"))
        elif args.stage.startswith("run-"):
            native(args, args.stage.removeprefix("run-"))
        elif args.stage == "fit":
            frozen_check(args)
            dataset(args)
        elif args.stage == "bind-test":
            frozen_check(args)
            bind_test(args)
        elif args.stage == "finalize":
            finalize(args)
        else:
            from .neural_shot_audit import audit
            if args.audit_output is None or args.audit_output.exists() or args.audit_output.is_relative_to(args.output):
                parser.error("Use a fresh audit output outside study evidence")
            result = audit(args)
            atomic(args.audit_output, result)
            if result["status"] != "passed":
                raise RuntimeError("Independent audit did not pass")
            # Do not change the audited report. Publish the receipt in derived UI only.
            if args.display_report:
                display = read(args.display_report)
                display["neural_shot_study"] = dict(read(args.output / "report.json"), audit_status="passed",
                    next="Review capture and planning time together before any live planner promotion.")
                atomic(args.display_report, display)
            print(json.dumps(result), flush=True)
    except Exception as exc:
        if (args.output / "report.json").exists() and args.stage != "audit":
            report = read(args.output / "report.json")
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            publish(args, report, "failed_" + args.stage)
        raise


if __name__ == "__main__":
    main()
