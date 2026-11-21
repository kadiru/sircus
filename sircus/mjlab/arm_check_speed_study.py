"""Compare unchanged shot selection with opt-in faster CPU arm checking.

All selections are frozen before native64 drops. The existing live server,
historical computational files, model, thresholds and physics remain untouched.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import copy
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
from .fast_planning_study import COPIED_INPUTS, PLANNING_BUDGET_S
from .gpu_slot import foreign_processes, gpu_slot
from .interactive_fast_check import coordinates, fast_inventory
from .interactive_physics_check import sample_positions, _digest
from .intergrid_library_plan import libraries
from .intergrid_route_labels import LABEL_VERSION, annotate_routes
from .intergrid_system_study import gpu_deadline, require_storage
from .parallel_capture_grid_study import NATIVE_BATCH, control_mirror
from .parallel_capture_grid_workers import load_resolution
from .planning_budget_plan import _wrap_choice
from .sweep_worker import atomic
from .table_training_study import control_gate

VERSION = "sircus-arm-check-speed-study-v1"
METHODS = ("baseline", "optimized")
POSITION_COUNT = 20
SEED = 2026112107
PROCESSING_SEED = 2026112108
GPU_CAP_S = 3600
FEATURES = dict(baseline=["original_geometric_gate", "complete_path_audit"],
                optimized=["sircus-vectorized-arm-broad-phase-v1", "sircus-arm-path-rejection-short-circuit-v1"])


def initialize_baseline(ref, assets, directory, kwargs):
    from . import parallel_capture_grid_workers as workers
    workers._initialize_worker(ref, assets, directory, kwargs)
    atomic(Path(directory)/"arm-check-features.json", dict(variant="baseline", features=FEATURES["baseline"]))


def initialize_optimized(ref, assets, directory, kwargs):
    from . import parallel_capture_grid_workers as workers
    from .arm_check_speed import install_vectorized_gate
    from .arm_path_speed import install_path_short_circuit
    workers._initialize_worker(ref, assets, directory, kwargs)
    resolver = workers._engine.adapter.resolver.base
    install_vectorized_gate(resolver.solver.gate)
    install_path_short_circuit(resolver.planner)
    atomic(Path(directory)/"arm-check-features.json", dict(variant="optimized", features=FEATURES["optimized"]))


def make_layouts(provenance, seed=SEED):
    occupied = list(map(tuple, provenance["excluded_coordinates_m"]))
    layouts = []
    for index in range(2):
        rows = sample_positions(provenance["geometry"], provenance["grid"], occupied,
                                provenance["bounds"], seed + index)
        for row in rows:
            identity = f"arm-check-speed-test-{len(layouts):02d}"
            layouts.append(dict(row, id=identity, context_id=identity, split="test",
                                stratum=row["region"]))
        occupied += [tuple(row["bucket_xy_m"]) for row in rows]
    return layouts


def publish(args, report, phase=None):
    if phase:
        report["phase"] = phase
    report["updated_at_unix"] = time.time()
    atomic(args.output / "report.json", report)
    if args.display_report:
        display = read(args.display_report) if args.display_report.exists() else {}
        display["arm_check_speed_study"] = report
        atomic(args.display_report, display)
    print(json.dumps({k: report[k] for k in ("status", "phase", "progress", "gpu_wait", "error") if k in report}), flush=True)


def initialize(args):
    spec, parent_binding, source, ref = parent.native_inputs(args)
    known, fast_binding, fast_contract = fast_inventory(args.fast_study, args.fast_audit)
    receipt = read(args.acceptance_audit)
    previous = read(args.acceptance_study / "positions.json")
    prior_report = read(args.acceptance_study / "report.json")
    if (receipt["status"] != "passed" or prior_report["status"] != "completed"
            or receipt["results_sha256"] != _digest(prior_report["results"])
            or receipt["positions_sha256"] != sha(args.acceptance_study / "positions.json")):
        raise ValueError("Need the unchanged passed live acceptance comparison")
    if fast_contract["identity"] != spec["identity"] or fast_contract["scene"] != spec["scene"]:
        raise ValueError("Audited native scene or evaluator changed")
    baseline_sources = {name: expected for name, expected in fast_contract["source_sha256"].items()
                        if not Path(name).name.startswith("._")
                        and name != "sircus/mjlab/interactive_physics_server.py"}
    for name, expected in baseline_sources.items():
        if sha(source/name) != expected:
            raise ValueError("Archived planner computational source changed: " + name)
    known.update(coordinates(previous))
    history = {str(args.acceptance_study/"positions.json"):sha(args.acceptance_study/"positions.json"),
               str(args.fast_study/"layouts.json"):sha(args.fast_study/"layouts.json")}
    if not args.evidence_root.is_dir():
        raise ValueError("Missing interactive evidence inventory")
    for name in ("request.json", "preview.json", "plan.json", "positions.json"):
        for path in sorted(args.evidence_root.rglob(name)):
            known.update(coordinates(read(path)))
            history[str(path)] = sha(path)
    provenance = dict(excluded_coordinates_m=sorted(known),
                      excluded_coordinates_sha256=digest(sorted(known)),
                      geometry=previous["geometry"], grid=previous["grid"],
                      bounds=previous["bounds"], history_files=history)
    layouts = make_layouts(provenance)
    if len(layouts) != POSITION_COUNT:
        raise ValueError("Expected 20 spatially fresh placements")
    args.output.mkdir(parents=True, exist_ok=False)
    require_storage(args.output)
    atomic(args.output / "layouts.json", dict(layouts=layouts, provenance=provenance))
    for name in COPIED_INPUTS:
        shutil.copyfile(args.fast_study / name, args.output / name)
    shutil.copyfile(args.parent_study / "setup-home.json", args.output / "setup-home.json")
    equivalence = read(args.equivalence_report)
    if (equivalence.get("exact_full_record_equivalence") is not True
            or equivalence.get("mismatch_count") != 0 or equivalence.get("segment_mismatch_count") != 0
            or equivalence.get("cpu_only") is not True or equivalence.get("samples",0) < 64):
        raise ValueError("Need passed native-scene CPU equivalence before preparation")
    shutil.copyfile(args.equivalence_report, args.output / "cpu-equivalence.json")
    frozen = {p.name: sha(p) for p in args.output.glob("*.json")}
    order = [row["id"] for row in layouts]
    random.Random(PROCESSING_SEED).shuffle(order)
    contract = dict(version=VERSION, parent=parent_binding, identity=spec["identity"], scene=spec["scene"],
        flight_ranking=fast_contract["flight_ranking"], fast_parent=fast_binding,
        baseline_computational_sha256=baseline_sources,
        acceptance_parent=dict(report_sha256=sha(args.acceptance_study/"report.json"),
                               positions_sha256=sha(args.acceptance_study/"positions.json"),
                               audit_sha256=sha(args.acceptance_audit)),
        methods=list(METHODS), features=FEATURES, early_score_m=.005, test_positions=POSITION_COUNT, seed=SEED,
        processing_seed=PROCESSING_SEED, processing_order=order, planning_budget_s=PLANNING_BUDGET_S,
        physics_substeps=64, label_version=LABEL_VERSION, attempts_per_method_per_position=1,
        gpu_cap_s=GPU_CAP_S, frozen_inputs=frozen, source_sha256=source_manifest(source),
        same_selector="warm_early; identical menus, learned weights, candidate order and early-return rule",
        optimization="Batched AABB broad phase; stop rejected paths at first failed segment",
        unchanged_checks="5 mm clearance, 0.5 degree samples, exact narrow-phase queries, full accepted forward/reverse paths",
        scheduling="Two independent CPU lanes, barrier after each placement; all choices frozen before drops",
        scope="Privileged simulator geometry; endpoint-held native64; not driven arm trajectories or hardware")
    contract["fingerprint"] = digest(contract)
    atomic(args.output/"contract.json", contract)
    report = dict(version=VERSION, status="running", phase="initialized", audit_status="pending",
        budgets=dict(positions=POSITION_COUNT, planning_s=PLANNING_BUDGET_S),
        controls={}, gpu_lease_s=0., gpu_cap_s=GPU_CAP_S,
        progress=dict(prepared=0, planned=2*POSITION_COUNT, completed_releases=0, planned_releases=None),
        next="Check both frozen planners at fresh placements before any native drops.")
    publish(args, report)


def frozen_check(args):
    contract = read(args.output/"contract.json")
    if digest({k:v for k,v in contract.items() if k != "fingerprint"}) != contract["fingerprint"]:
        raise ValueError("Study contract changed")
    for name, expected in contract["frozen_inputs"].items():
        if sha(args.output/name) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    source = Path(__file__).resolve().parents[2]
    for name, expected in contract["source_sha256"].items():
        if sha(source/name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    return contract


def planner_inputs(args):
    contract = frozen_check(args)
    _, _, _, ref = parent.native_inputs(args)
    raw = read(args.output/"shot-library-rows.json")
    libs = libraries(raw["original"], raw["additions"])
    model, history, neural = (read(args.output/name) for name in
                             ("route-models.json", "route-history.json", "neural-checkpoint.json"))
    kwargs = dict(prior=read(args.output/"training-prior.json"),
                  calibration=read(args.output/"calibration.json"),
                  home=read(args.output/"setup-home.json"))
    config = dict(scene=ref["scene"], calibration=kwargs["calibration"]["score_config"],
                  assets=str(args.assets), rule=contract["flight_ranking"]["rule"])
    return ref, libs, model, history, neural, kwargs, config


def plan_batch(args, layouts, root):
    from .fast_planning_plan import ReusablePlanner
    ref, libs, model, history, neural, kwargs, config = planner_inputs(args)
    report = read(args.output/"report.json")
    sites = {}
    with ExitStack() as stack:
        planners = {}
        for variant in METHODS:
            planner = stack.enter_context(ReusablePlanner("warm_early", libs, model, history, config, neural,
                reference=ref, assets=args.assets, directory=root/("native-"+variant),
                engine_kwargs=kwargs, budget_s=PLANNING_BUDGET_S))
            planner.worker.initializer = initialize_optimized if variant == "optimized" else initialize_baseline
            planners[variant] = planner
        pool = stack.enter_context(ThreadPoolExecutor(max_workers=2))
        for layout in layouts:
            jobs = {pool.submit(planners[v].plan, layout):v for v in METHODS}
            for future in as_completed(jobs):
                variant, site = jobs[future], future.result()
                if site["status"] not in {"completed", "deadline"}:
                    raise RuntimeError("Planner worker failed: " + str(site))
                sites[layout["id"],variant] = site
                atomic(root/"preparation"/variant/(layout["id"]+".json"), site)
                if root == args.output:
                    report["progress"]["prepared"] = len(sites)
                    publish(args, report, "preparing")
                require_storage(root)
    return sites


def prepare(args):
    contract = frozen_check(args)
    if (args.output/"plans.json").exists():
        raise ValueError("Preserve already frozen preparation")
    layouts = read(args.output/"layouts.json")["layouts"]
    by_id = {row["id"]:row for row in layouts}
    sites = plan_batch(args, [by_id[key] for key in contract["processing_order"]], args.output)
    wrappers = [dict(variant=v, site=sites[row["id"],v]) for row in layouts for v in METHODS]
    atomic(args.output/"plans.json", wrappers)
    slots = [_wrap_choice(item["site"]["choice"], item["variant"]) for item in wrappers]
    bindings = {name:sha(args.output/name) for name in (*contract["frozen_inputs"], "contract.json", "plans.json")}
    atomic(args.output/"choices.json", dict(slots=slots, bindings=bindings))
    report = read(args.output/"report.json")
    report["progress"]["planned_releases"] = sum(s["request"] is not None for s in slots)
    publish(args, report, "prepared")


def diagnose(args):
    root = args.output/"diagnostic"
    root.mkdir(exist_ok=False)
    # This previously inspected placement is development evidence, never part of fresh testing.
    layout = dict(id="diagnostic-old-case02", context_id="diagnostic-old-case02", split="test",
                  stratum="development", bucket_xy_m=[0.9712010092480674,-0.2853881822068356])
    sites = plan_batch(args, [layout], root)
    atomic(root/"plans.json", [dict(variant=v,site=sites[layout["id"],v]) for v in METHODS])
    print(json.dumps({v:dict(planning_s=sites[layout["id"],v]["request_wall_s"],
                            checked_shot=sites[layout["id"],v]["choice"]["request"] is not None)
                      for v in METHODS}), flush=True)


def run_native(args):
    from .cell1050_heading_native import HeadingEngine
    from .table_correction_native import MatchedEngine

    contract = frozen_check(args)
    _, _, _, ref = parent.native_inputs(args)
    output = args.output
    report = read(output / "report.json")
    if report["phase"] != "prepared" or report["status"] != "running":
        raise ValueError("Need complete frozen preparation before any drop")
    choices = read(output / "choices.json")
    bindings = dict(choices["bindings"], **{"choices.json": sha(output / "choices.json")})
    if any(sha(output / name) != value for name, value in bindings.items()):
        raise ValueError("Pre-release choices or their inputs changed")
    planned = {method: [slot["request"] for slot in choices["slots"]
                        if slot["method"] == method and slot["request"] is not None] for method in METHODS}
    receipt = output / "native-started.json"
    with receipt.open("x") as stream:
        json.dump(dict(bindings=bindings, requests_sha256=digest(planned), timestamp_unix=time.time()), stream)
    kwargs = dict(prior=read(output / "training-prior.json"), calibration=read(output / "calibration.json"),
                  home=read(args.output / "setup-home.json"))
    expected_options = read(args.parent_study / "native-yaw/effective-options.json")
    expected_geometry = read(args.parent_study / "native-yaw/native-compiled-geometry.json")
    packet_number = 0

    def execute(engine, batch, phase, directory, control=False):
        nonlocal packet_number
        if foreign_processes():
            raise RuntimeError("A foreign process appeared on the leased 4090")
        require_storage(output)
        if any(sha(output / name) != value for name, value in bindings.items()):
            raise ValueError("Frozen selections changed during native execution")
        if not control:
            for xy in {tuple(row["bucket_xy_m"]) for row in batch}:
                load_resolution(engine, list(xy))
        packet = f"requests/batch-{packet_number:05d}.json"
        packet_number += 1
        atomic(output / packet, dict(phase=phase, control=control, adapter_directory=directory, rows=batch))
        measured, timing = engine.evaluate(batch, phase)
        if [row["id"] for row in measured] != [row["id"] for row in batch] or any(
                row["native_release"] is not True for row in measured):
            raise RuntimeError("A frozen admitted command did not release exactly once")
        adapter = engine.engine.adapter if control else engine.adapter
        if adapter.env.effective_options != expected_options or adapter.env.compiled_geometry != expected_geometry:
            raise RuntimeError("Native options or geometry differ from audited controls")
        if not control:
            annotate(measured, output / directory)
            annotate_routes(measured, output / directory)
            for row, request in zip(measured, batch, strict=True):
                row["intended_route"] = request["intended_route"]
        for row in measured:
            row.update(request_packet=packet, request_packet_sha256=sha(output / packet), adapter_directory=directory)
        append(output / "rows.jsonl", measured)
        append(output / "timing.jsonl", [dict(packet=packet, phase=phase, timing=timing)])
        return measured

    def waiting(status):
        report["gpu_wait"] = status
        publish(args, report, "waiting_for_4090")

    with gpu_slot(args.gpu_lock, timeout_s=args.gpu_wait_s, poll_s=30, notify=waiting):
        started = time.monotonic()
        with gpu_deadline(GPU_CAP_S):
            report.pop("gpu_wait", None)
            engines = {method: HeadingEngine(ref, args.assets, output / ("native-" + method), **kwargs) for method in METHODS}
            sources = read(args.parent_study / "control-replays.json")
            mirror = control_mirror(args.parent_study, output, sources, False)
            controls = MatchedEngine(ref, args.assets, output / "native-controls", parent=mirror, sources=sources,
                                    prior=read(args.parent_study / "old-prior.json"),
                                    calibration=kwargs["calibration"], home=kwargs["home"])
            adapters = [engine.adapter for engine in engines.values()] + [controls.engine.adapter]
            factory, allocated = adapters[0].environment_factory, []
            def shared_factory():
                if not allocated:
                    allocated.append(factory())
                return allocated[0]
            for adapter in adapters:
                adapter.environment_factory = shared_factory
            for phase in ("controls_pre", "test", "controls_post"):
                if phase != "test":
                    gate = control_gate(execute(controls, control_requests(sources, phase), phase, "native-controls", True))
                    report["controls"][phase] = gate
                    if gate["status"] != "passed":
                        raise RuntimeError(f"Matched control gate failed: {phase}")
                    publish(args, report, phase)
                    continue
                # Alternate method batches without feeding any outcome back into selection.
                for offset in range(0, max(map(len, planned.values())), NATIVE_BATCH):
                    for method in METHODS:
                        batch = planned[method][offset:offset + NATIVE_BATCH]
                        if not batch:
                            continue
                        rows = execute(engines[method], batch, "test_" + method, "native-" + method)
                        report["progress"]["completed_releases"] += len(rows)
                        report["gpu_lease_s"] = time.monotonic() - started
                        publish(args, report, "test_" + method)
        report["gpu_lease_s"] = time.monotonic() - started
    report["native_completed"] = True
    publish(args, report, "native_completed")


def summarize(layouts, slots, observations, wrappers):
    from .intergrid_route_selector_plan import _account, STANDARD_CAPTURES
    from .fast_planning_audit import latency
    _, methods, attempts = _account(layouts, slots, observations, METHODS)
    for variant, counts in methods.items():
        admitted = sum(s["method"] == variant and s["request"] is not None for s in slots)
        counts.update(positions=len(layouts), admitted=admitted,
                      no_candidate=len(layouts)-admitted,
                      flight_misses=counts["outcomes"].get("flight_miss", 0))
    paired = dict(both=0, gained=0, lost=0, neither=0)
    positions = []
    for layout in layouts:
        outcomes = {v:attempts[layout["id"],v][1] for v in METHODS}
        left,right = (outcomes[v] in STANDARD_CAPTURES for v in METHODS)
        paired["both" if left and right else "lost" if left else "gained" if right else "neither"] += 1
        positions.append(dict(id=layout["id"], bucket_xy_m=layout["bucket_xy_m"], outcomes=outcomes))
    timing = {v:latency([w["site"]["request_wall_s"] for w in wrappers if w["variant"]==v]) for v in METHODS}
    decision = {v:latency([w["site"]["decision_latency_s"] for w in wrappers if w["variant"]==v]) for v in METHODS}
    return dict(methods=methods, latency=timing, decision_latency=decision, paired=paired, positions=positions)


def finalize(args):
    from .parallel_capture_grid_audit import observed_rows
    frozen_check(args)
    report = read(args.output/"report.json")
    if report.get("native_completed") is not True:
        raise ValueError("Finish both native lanes and controls first")
    rows = {r["id"]:r for r in observed_rows(args.output/"rows.jsonl") if r["adapter_directory"] != "native-controls"}
    report["comparison"] = summarize(read(args.output/"layouts.json")["layouts"],
        read(args.output/"choices.json")["slots"], rows, read(args.output/"plans.json"))
    atomic(args.output/"evidence-manifest.json", {str(p.relative_to(args.output)):sha(p)
        for p in sorted(args.output.rglob("*")) if p.is_file()
        and p.name not in {"report.json","evidence-manifest.json"}})
    report.update(status="completed", audit_status="pending",
                  evidence_manifest_sha256=sha(args.output/"evidence-manifest.json"),
                  next="Independently verify selection receipts and native capture evidence.")
    publish(args, report, "completed")


def argument_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("parent-study","parent-audit","reference","assets","fast-study","fast-audit",
                 "acceptance-study","acceptance-audit","evidence-root","equivalence-report","output","gpu-lock"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--display-report",type=Path)
    parser.add_argument("--audit-output",type=Path)
    parser.add_argument("--gpu-wait-s",type=int,default=12*3600)
    parser.add_argument("--stage",required=True,choices=("initialize","diagnose","prepare","run","finalize","audit"))
    return parser


def main():
    args=argument_parser().parse_args()
    try:
        if args.stage == "initialize": initialize(args)
        elif args.stage == "diagnose": diagnose(args)
        elif args.stage == "prepare": prepare(args)
        elif args.stage == "run": run_native(args)
        elif args.stage == "finalize": finalize(args)
        else:
            from .arm_check_speed_audit import audit
            if args.audit_output is None or args.audit_output.exists() or args.audit_output.is_relative_to(args.output):
                raise ValueError("Use a new external audit receipt")
            result=audit(args)
            atomic(args.audit_output,result)
            if result["status"] != "passed": raise RuntimeError("Independent audit failed")
            if args.display_report:
                display=read(args.display_report)
                display["arm_check_speed_study"]=dict(read(args.output/"report.json"),audit_status="passed",
                    next="Compare total planning time and first-drop captures; live planner not changed.")
                atomic(args.display_report,display)
            print(json.dumps({k:v for k,v in result.items() if k!="comparison"}),flush=True)
    except Exception as exc:
        if (args.output/"report.json").exists() and args.stage != "audit":
            report=read(args.output/"report.json")
            report.update(status="failed",error=f"{type(exc).__name__}: {exc}")
            publish(args,report,"failed_"+args.stage)
        raise


if __name__ == "__main__": main()

