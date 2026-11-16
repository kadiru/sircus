"""Train a shot ranker and benchmark a wider setup search on fresh bucket placements."""
from __future__ import annotations

import argparse
from concurrent.futures import as_completed
from contextlib import contextmanager
import importlib.metadata
import json
from pathlib import Path
import shutil
import signal
import time

from . import intergrid_flight_ranking_study as baseline
from .benchmark_verify import digest, read, sha
from .cell1050_rebound_study import annotate, control_requests, parent_inputs
from .corrected_grid import source_manifest
from .coverage_training_study import append, share_native_environment
from .gpu_slot import foreign_processes, gpu_slot
from .intergrid_candidate_replay_plan import command_key
from .intergrid_library_plan import libraries
from .intergrid_matched_route_plan import make_layouts as make_matched_layouts
from .intergrid_route_labels import LABEL_VERSION, annotate_routes
from .intergrid_route_refit_inputs import _audited_source
from .intergrid_route_selector_study import SOURCE_NAMES, WORKERS
from .intergrid_system_learn import fit_ranker, learned_order
from .intergrid_system_plan import BASE_PER_ROUTE, FALLBACK_PER_ROUTE, METHODS, ROUTES
from .intergrid_system_plan import base_proposals, prepare_layout, select_requests
from .parallel_capture_grid_study import NATIVE_BATCH, STORAGE_RESERVE_BYTES, control_mirror
from .parallel_capture_grid_workers import load_resolution, open_workers
from .policy64_contract import load_runtime_reference
from .sweep_worker import atomic
from .table_training_study import control_gate

VERSION = "sircus-intergrid-system-planner-v1"
TRAIN_SEED = 2026112501
TEST_SEED = 2026112601
POSITIONS = 50
TRAIN_MAX_PER_SITE = 8
GPU_CAP_S = 7 * 3600
LEARNED = "learned_ranker"


def inputs(args):
    checkpoint, original, additions, previous, binding, history = baseline._inputs(args)
    source, receipt = args.prior_comparison_study, args.prior_comparison_audit
    report, _, audit, archive, contract = _audited_source(
        source, receipt, baseline.VERSION, ("layouts.json", "choices.json"),
    )
    if (audit["version"] != baseline.AUDIT_VERSION or audit["comparison"] != report["comparison"]
            or any(contract[key] != value for key, value in binding.items())):
        raise ValueError("The previous comparison does not bind the current frozen baseline")
    layouts = read(source / "layouts.json")
    if (len(layouts["layouts"]) != POSITIONS
            or digest(layouts["layouts"]) != layouts["provenance"]["layouts_sha256"]):
        raise ValueError("Previous comparison exclusions are incomplete")
    return checkpoint, original, additions, previous + layouts["layouts"], dict(
        binding, prior_comparison=archive,
    ), history


def layout_ids(layouts, split):
    return [dict(row, id=f"intergrid-system-{split}-{i:02d}", context_id=f"intergrid-system-{split}-{i:02d}",
                 split=split) for i, row in enumerate(layouts)]


def make_layouts(history_root, grid, scene, assets, previous, seed, split):
    layouts, provenance = make_matched_layouts(history_root, grid, scene=scene, assets=assets,
                                                previous_layouts=previous, seed=seed)
    layouts = layout_ids(layouts, split)
    if len(layouts) != POSITIONS or len({tuple(row["bucket_xy_m"]) for row in layouts}) != POSITIONS:
        raise ValueError("Expected 50 distinct positions per split")
    return dict(layouts=layouts, provenance=dict(provenance, layouts_sha256=digest(layouts)))


def admitted_pool(plan):
    return plan["base"]["admitted"] or plan["fallback"]["admitted"]


def training_requests(site, ranker):
    ordered = ranker.order(admitted_pool(site["plan"]))
    chosen = {}
    for request in ordered:
        chosen.setdefault(command_key(request), request)
    return dict(layout=site["layout"], requests=list(chosen.values())[:TRAIN_MAX_PER_SITE])


def training_predictions(cases, sites):
    by_site = {site["layout"]["id"]: {row["id"]: row for row in site["predictions"]} for site in sites}
    return [[by_site[case["layout"]["id"]][row["id"]] for row in case["requests"]] for case in cases]


def test_requests(site, history, ranker, checkpoint):
    layout, plan = site["layout"], site["plan"]
    slots = select_requests(layout, plan, history, ranker)
    if checkpoint is None:
        return slots
    ordered = learned_order(admitted_pool(plan), site["predictions"], checkpoint)
    request = ordered[0] if ordered else None
    identifier = f"{layout['id']}-{LEARNED}"
    slots.append(dict(slots[1], id=identifier, method=LEARNED,
                      ranked_candidate_ids=[row["id"] for row in ordered],
                      prediction=None if request is None else next(
                          pred for pred in site["predictions"] if pred["id"] == request["id"]),
                      intended_route=None if request is None else request["intended_route"],
                      request=None if request is None else dict(request, id=identifier, method=LEARNED,
                                                                 stratum=layout["stratum"])))
    return slots


def require_storage(output):
    if shutil.disk_usage(output).free < STORAGE_RESERVE_BYTES:
        raise RuntimeError("Insufficient space for native trace evidence")


@contextmanager
def gpu_deadline(seconds):
    """Bound this study's own lease; the signal is delivered only to this runner."""
    def expire(*_):
        raise TimeoutError("Study reached its 4090 time limit")

    old = signal.signal(signal.SIGALRM, expire)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def run(args):
    from .cell1050_heading_native import HeadingEngine
    from .intergrid_system_audit import summarize
    from .table_correction_native import MatchedEngine

    output, parent = args.output, args.parent_study
    spec, parent_binding, _ = parent_inputs(parent, args.parent_audit)
    source = Path(__file__).resolve().parents[2]
    ref = load_runtime_reference(args.reference, args.assets, source)
    if (digest(ref["scene"]) != spec["identity"]["scene_sha256"]
            or ref["asset_hashes"] != spec["identity"]["asset_sha256"]
            or any(importlib.metadata.version(name) != pinned for name, pinned in spec["packages"].items())):
        raise ValueError("Scene, assets or dependencies differ from the audited controls")
    model, original, additions, previous, binding, history = inputs(args)
    ranker = baseline._ranker(args, spec["scene"], binding)
    grid = read(parent / "grid.json")
    train = make_layouts(args.history_root, grid, ref["scene"], args.assets, previous, TRAIN_SEED, "train")
    test = make_layouts(args.history_root, grid, ref["scene"], args.assets,
                        previous + train["layouts"], TEST_SEED, "test")
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", controls={}, prepared=dict(train=0, test=0), train_releases=0, test_releases=0)
    packet_number = 0

    def publish(phase):
        report["phase"] = phase
        atomic(output / "report.json", report)
        print(json.dumps({k: v for k, v in report.items() if k not in {"comparison", "validation"}}), flush=True)

    def execute(engine, requests, phase):
        nonlocal packet_number
        if foreign_processes():
            raise RuntimeError("Another process is using the leased 4090")
        require_storage(output)
        control = phase.startswith("controls_")
        directory = "native-controls" if control else "native-heading"
        if not control:
            for xy in {tuple(request["bucket_xy_m"]) for request in requests}:
                load_resolution(engine, list(xy))
        packet = f"requests/batch-{packet_number:05d}.json"
        packet_number += 1
        atomic(output / packet, dict(phase=phase, rows=requests))
        measured, timing = engine.evaluate(requests, phase)
        if ([row["id"] for row in measured] != [row["id"] for row in requests]
                or any(row["native_release"] is not True for row in measured)):
            raise RuntimeError("An exact admitted request did not release")
        adapter = engine.engine.adapter if control else engine.adapter
        if adapter.env.effective_options != expected_options or adapter.env.compiled_geometry != expected_geometry:
            raise RuntimeError("Native physics differs from pinned controls")
        if not control:
            annotate(measured, output / directory)
            annotate_routes(measured, output / directory)
            for row, request in zip(measured, requests, strict=True):
                row["intended_route"] = request["intended_route"]
        packet_hash = sha(output / packet)
        for row in measured:
            row.update(request_packet=packet, request_packet_sha256=packet_hash, adapter_directory=directory)
        append(output / "rows.jsonl", measured)
        append(output / "timing.jsonl", [dict(packet=packet, phase=phase, timing=timing)])
        return measured

    def releases(engine, requests, phase, frozen):
        observations = {}
        for start in range(0, len(requests), NATIVE_BATCH):
            if any(sha(output / name) != expected for name, expected in frozen.items()):
                raise ValueError("Frozen inputs changed during native execution")
            batch = execute(engine, requests[start:start + NATIVE_BATCH], phase)
            observations.update((row["id"], row) for row in batch)
            report[phase + "_releases"] = len(observations)
            publish(phase)
        return observations

    def run_controls(phase):
        gate = control_gate(execute(controls, control_requests(control_sources, phase), phase))
        report["controls"][phase] = gate
        if gate["status"] != "passed":
            raise RuntimeError(f"{phase} failed")
        publish(phase)

    def waiting(status):
        report["gpu_wait"] = status
        publish("waiting_for_4090")

    try:
        require_storage(output)
        atomic(output / "layouts.json", dict(train=train, test=test))
        shutil.copyfile(args.model_study / "training-prior.json", output / "training-prior.json")
        shutil.copyfile(args.refit_study / "route-models.json", output / "route-models.json")
        atomic(output / "route-history.json", history)
        frozen = {name: sha(output / name) for name in
                  ("layouts.json", "training-prior.json", "route-models.json", "route-history.json")}
        if (frozen["route-models.json"] != binding["refit_study"]["checkpoint_sha256"]
                or frozen["training-prior.json"] != binding["model_study"]["geometry_prior"]):
            raise ValueError("The copied baseline models or arm settings differ from the audited files")
        contract = dict(version=VERSION, parent=parent_binding, identity=spec["identity"], scene=spec["scene"],
                        **binding, test_positions=POSITIONS, train_positions=POSITIONS,
                        methods=list(METHODS), routes=list(ROUTES), attempts_per_method_per_position=1,
                        candidate_budget_per_route=BASE_PER_ROUTE, fallback_budget_per_route=FALLBACK_PER_ROUTE,
                        train_max_per_site=TRAIN_MAX_PER_SITE, physics_substeps=64, label_version=LABEL_VERSION,
                        split_seeds=[TRAIN_SEED, TEST_SEED], source_sha256=source_manifest(source),
                        gpu_cap_s=args.gpu_cap_s, frozen_inputs=frozen.copy())
        contract["fingerprint"] = digest(contract)
        atomic(output / "contract.json", contract)
        prior = read(output / "training-prior.json")
        setup, calibration = read(parent / "setup-home.json"), read(parent / "calibration.json")
        for name in ("calibration.json", "verification.json"):
            shutil.copyfile(parent / name, output / name)
        kwargs = dict(prior=prior, calibration=calibration, home=setup)
        shot_libraries = libraries(original, additions)
        plans = {}
        archives = dict(train=train, test=test)
        publish("preparing")
        with open_workers(WORKERS, ref, args.assets, output / "native-heading", kwargs) as workers:
            for split, archive in archives.items():
                pending = {workers.submit(prepare_layout, layout, base_proposals(layout, shot_libraries, model),
                                          shot_libraries): layout for layout in archive["layouts"]}
                prepared = {}
                for future in as_completed(pending):
                    layout = pending[future]
                    plan = future.result()
                    predictions = [dict(id=req["id"], **ranker.predict(req)) for req in admitted_pool(plan)]
                    prepared[layout["id"]] = dict(layout=layout, plan=plan, predictions=predictions)
                    report["prepared"][split] = len(prepared)
                    if len(prepared) % 5 == 0:
                        require_storage(output)
                        publish("preparing")
                plans[split] = [prepared[row["id"]] for row in archive["layouts"]]
        atomic(output / "plans.json", plans)
        training = [training_requests(site, ranker) for site in plans["train"]]
        atomic(output / "train_cases.json", training)
        frozen.update({name: sha(output / name) for name in ("plans.json", "train_cases.json")})
        train_reqs = [row for case in training for row in case["requests"]]
        if not train_reqs:
            raise RuntimeError("No development shots passed the pose checks")
        report["train_admitted"] = len(train_reqs)
        publish("train_frozen")

        with gpu_slot(args.gpu_lock, timeout_s=args.gpu_wait_s, poll_s=15, notify=waiting):
            with gpu_deadline(args.gpu_cap_s):
                started = time.monotonic()
                report.pop("gpu_wait", None)
                heading = HeadingEngine(ref, args.assets, output / "native-heading", **kwargs)
                control_sources = read(parent / "control-replays.json")
                mirror = control_mirror(parent, output, control_sources, False)
                controls = MatchedEngine(ref, args.assets, output / "native-controls", parent=mirror,
                                         sources=control_sources, prior=read(parent / "old-prior.json"),
                                         calibration=calibration, home=setup)
                share_native_environment(heading, controls)
                expected_options = read(parent / "native-yaw/effective-options.json")
                expected_geometry = read(parent / "native-yaw/native-compiled-geometry.json")
                run_controls("controls_pre")
                observations = releases(heading, train_reqs, "train", frozen)
                learning = fit_ranker(training, training_predictions(training, plans["train"]), observations)
                atomic(output / "learning.json", learning)
                checkpoint = learning["checkpoint"]
                slots = [slot for site in plans["test"] for slot in test_requests(site, history, ranker, checkpoint)]
                frozen["learning.json"] = sha(output / "learning.json")
                atomic(output / "choices.json", dict(learning_sha256=frozen["learning.json"], slots=slots))
                frozen["choices.json"] = sha(output / "choices.json")
                test_reqs = [slot["request"] for slot in slots if slot["request"] is not None]
                report.update(validation=learning["validation"], learned_advance=checkpoint is not None,
                              choices_sha256=frozen["choices.json"], test_slots=len(slots),
                              test_admitted=len(test_reqs))
                publish("choices_frozen")
                observations = releases(heading, test_reqs, "test", frozen)
                methods = METHODS if checkpoint is None else (*METHODS, LEARNED)
                report["comparison"] = summarize(test["layouts"], slots, observations, plans["test"], methods)
                run_controls("controls_post")
                report["gpu_lease_s"] = time.monotonic() - started

        atomic(output / "evidence-manifest.json", {
            str(path.relative_to(output)): sha(path) for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "report.json"
        })
        report.update(status="completed", evidence_manifest_sha256=sha(output / "evidence-manifest.json"))
        publish("completed")
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        publish("failed")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("history-root", "parent-study", "parent-audit", "reference", "assets", "gpu-lock", "output", "rule"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in (*SOURCE_NAMES, *baseline.EXTRA_SOURCES, "prior-comparison"):
        for kind in ("study", "audit"):
            parser.add_argument(f"--{name}-{kind}", type=Path, required=True)
    parser.add_argument("--gpu-wait-s", type=int, default=GPU_CAP_S)
    parser.add_argument("--gpu-cap-s", type=int, default=GPU_CAP_S)
    parser.add_argument("--audit", type=Path, help="Rescore completed native evidence on CPU")
    args = parser.parse_args()
    if args.audit is None:
        if args.output.exists() or args.gpu_wait_s <= 0 or not 0 < args.gpu_cap_s <= GPU_CAP_S:
            parser.error("Require fresh output, positive wait and at most seven hours of GPU time")
        run(args)
        return
    if args.audit.exists() or args.audit.resolve().is_relative_to(args.output.resolve()):
        parser.error("Write a fresh audit outside the frozen evidence directory")
    from .intergrid_system_audit import audit

    try:
        result = audit(args)
    except Exception as exc:
        result = dict(status="failed", error=f"{type(exc).__name__}: {exc}")
    atomic(args.audit, result)
    print(json.dumps(result), flush=True)
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
