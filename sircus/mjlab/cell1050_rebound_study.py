"""Run the fixed (1050, 25) mm heading and robot-rebound simulation study."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import importlib.metadata
import json
from pathlib import Path
import shutil
import signal
import time

import numpy as np

from .benchmark_verify import digest, read, sha
from .corrected_grid import source_manifest
from .corrected_imitation_study import mirror_control_proofs
from .coverage_training_study import append, share_native_environment
from .gpu_slot import foreign_processes, gpu_slot
from .policy64_contract import load_runtime_reference
from .sweep_worker import atomic
from .table_correction_native import META, MatchedEngine
from .table_training_study import control_gate

CONTEXT = "fixed35-dev-grid-03-05"
VERSION = "cell1050-rebound-study-v1"
ROUTES = ("standard_capture", "robot_assisted_capture")
INPUTS = ("rows.jsonl", "contract.json", "calibration.json", "verification.json",
          "setup-home.json", "prior.json", "old-prior.json", "control-replays.json",
          "native-yaw/effective-options.json", "native-yaw/native-compiled-geometry.json")
SEARCH_PACKET_RESERVE_S = 120


class SearchPreparationExpired(Exception):
    """Finish the study when the preparation time for the final packet expires."""


@contextmanager
def preparation_time_limit(seconds):
    if seconds is None:
        yield
        return

    def expire(_signal, _frame):
        raise SearchPreparationExpired

    if seconds <= 0:
        raise SearchPreparationExpired
    previous = signal.signal(signal.SIGALRM, expire)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def parent_inputs(parent, audit_path):
    manifest = read(parent / "evidence-manifest.json")
    report = read(parent / "report.json")
    audit = read(audit_path)
    binding = {
        "report": sha(parent / "report.json"), "audit": sha(audit_path),
        "manifest": sha(parent / "evidence-manifest.json"),
    }
    spec = read(parent / "contract.json")
    if (report["status"] != "completed" or report["evidence_sha256"] != manifest
            or audit["status"] != "passed" or audit["full_checks_verified"] is not True
            or audit["native_checks_verified"] is not True
            or audit["report_sha256"] != binding["report"]
            or audit["evidence_manifest_sha256"] != binding["manifest"]
            or audit["contract_fingerprint"] != spec["fingerprint"]
            or digest({k: v for k, v in spec.items() if k != "fingerprint"}) != spec["fingerprint"]):
        raise ValueError("The completed six-hour parent and its audit must bind the same evidence")
    for name in INPUTS:
        if sha(parent / name) != manifest[name]:
            raise ValueError(f"Parent input changed: {name}")
    with (parent / "rows.jsonl").open() as stream:
        anchor = next(row for row in map(json.loads, stream)
                      if row["context_id"] == CONTEXT and row.get("native_release"))
    return spec, binding, anchor


def control_requests(sources, phase):
    return [
        dict({key: source[key] for key in META}, id=f"cell1050-{phase}-{source['role']}-{i}",
             phase=phase, method="control", role=source["role"], context_id=source["context_id"],
             bucket_xy_m=source["bucket_xy_m"], values=source["values"], command_values=source["values"],
             source_record_id=source["id"], source_record_sha256=digest(source),
             variant="control_exact", repeat=i)
        for source in sources for i in range(3)
    ]


def annotate(rows, directory):
    from .cell1050_rebound_score import score_recorded_outcome

    packets = defaultdict(list)
    for row in rows:
        if not row.get("native_release"):
            raise RuntimeError("An admitted shot did not release")
        packets[row["native_cache_file"]].append(row)
    for name, members in packets.items():
        path = directory / name
        if sha(path) != members[0]["native_cache_sha256"]:
            raise RuntimeError("Dense trace bytes changed before scoring")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        for row in members:
            world = row["native_cache_row"]
            if arrays["record_ids"][world] != row["id"]:
                raise RuntimeError("Dense trace identity differs from the measured shot")
            row["experimental_route"] = score_recorded_outcome(row, arrays, world=world)


def run(args):
    from .cell1050_heading_native import HeadingEngine
    from .cell1050_plan import candidate_stream, command_key, load_history, load_seed_evidence

    started = time.monotonic()
    parent, output = args.parent_study.resolve(), args.output.resolve()
    spec, binding, anchor = parent_inputs(parent, args.parent_audit)
    root = Path(__file__).resolve().parents[2]
    ref = load_runtime_reference(args.reference, args.assets, root)
    if (digest(ref["scene"]) != spec["identity"]["scene_sha256"]
            or ref["asset_hashes"] != spec["identity"]["asset_sha256"]
            or any(importlib.metadata.version(name) != version for name, version in spec["packages"].items())):
        raise ValueError("Native reference or pinned dependencies differ from the audited parent")
    history = load_history(parent.parent)
    seeds = load_seed_evidence(parent, history)
    history_keys = {command_key(row) for row in history}
    continuation = None
    if args.previous_study is not None:
        from .cell1050_followup_plan import followup_candidates, geometry_key, load_followup_seeds

        near_misses, continuation = load_followup_seeds(args.previous_study, args.previous_audit)
        previous_manifest = read(args.previous_study / "evidence-manifest.json")
        previous_keys = args.previous_study / "historical-commands.json"
        if sha(previous_keys) != previous_manifest["historical-commands.json"]:
            raise ValueError("The prior historical command inventory changed")
        history_keys.update(read(previous_keys))
        used_geometry = {geometry_key(row) for row in history}
        with (args.previous_study / "rows.jsonl").open() as stream:
            for row in map(json.loads, stream):
                if row["phase"] in ("search", "repeat") and row["native_release"]:
                    history_keys.add(command_key(row))
                    used_geometry.add(geometry_key(row))
        for path in sorted((args.previous_study / "preparations").glob("*.json")):
            if sha(path) != previous_manifest[str(path.relative_to(args.previous_study))]:
                raise ValueError(f"Prior preparation changed: {path.name}")
            used_geometry.update(geometry_key(row) for row in read(path)["proposals"])
        generator = iter(followup_candidates(
            near_misses, seeds, seed=args.seed, exclude_geometry=used_geometry,
        ))
        seeds = dict(near_misses=near_misses, earlier_fallback=seeds)
    else:
        generator = iter(candidate_stream(seeds, seed=args.seed))
    output.mkdir(parents=True, exist_ok=False)
    for name in ("calibration.json", "verification.json"):
        shutil.copyfile(parent / name, output / name)
    atomic(output / "seeds.json", seeds)
    atomic(output / "historical-commands.json", sorted(history_keys))
    contract = dict(version=VERSION, target_releases=args.limit, candidate_limit=args.max_preparations,
                    generator_seed=args.seed, history_records=len(history), history_commands=len(history_keys),
                    seed_sha256=sha(output / "seeds.json"),
                    historical_commands_sha256=sha(output / "historical-commands.json"),
                    parent=binding, identity=spec["identity"], scene=spec["scene"],
                    source_sha256=source_manifest(root), preparation_batch=args.prepare_batch)
    if continuation is not None:
        contract.update(continuation=continuation, search_budget_s=args.search_seconds)
    contract["fingerprint"] = digest(contract)
    atomic(output / "contract.json", contract)
    report = dict(status="running", phase="initializing", target_releases=args.limit, search_releases=0,
                  attempts=0, admitted=0, duplicate_commands=0, preparation_declines=0,
                  route_counts={}, contact_counts={}, groups={}, confirmations=[], controls={})
    group_counts = defaultdict(Counter)
    outcomes = Counter()
    contact_counts = Counter()
    picks = defaultdict(list)
    pending = []
    packets = 0
    search_started = deadline = None
    release_reserve = SEARCH_PACKET_RESERVE_S

    def remaining():
        return deadline - time.monotonic() if deadline is not None else float("inf")

    def publish(phase):
        report.update(phase=phase, elapsed_s=round(time.monotonic() - started, 3),
                      route_counts=dict(outcomes), contact_counts=dict(contact_counts),
                      groups={group: dict(count) for group, count in group_counts.items()})
        if continuation is not None:
            report["search_budget_s"] = args.search_seconds
            if search_started is not None and "stop_reason" not in report:
                report["search_elapsed_s"] = round(time.monotonic() - search_started, 3)
        atomic(output / "report.json", report)
        print(json.dumps(report, allow_nan=False), flush=True)

    calibration, home = read(parent / "calibration.json"), read(parent / "setup-home.json")
    engine = HeadingEngine(ref, args.assets, output / "native-heading", prior=read(parent / "prior.json"),
                           calibration=calibration, home=home, anchor_source=anchor,
                           anchor_directory=parent / "native-yaw")
    source_controls = read(parent / "control-replays.json")
    mirror = mirror_control_proofs(parent, output, source_controls)
    controls = MatchedEngine(ref, args.assets, output / "native-controls", parent=mirror, sources=source_controls,
                             prior=read(parent / "old-prior.json"), calibration=calibration, home=home)
    share_native_environment(engine, controls)
    expected_options = read(parent / "native-yaw/effective-options.json")
    expected_geometry = read(parent / "native-yaw/native-compiled-geometry.json")

    def execute(requests, phase):
        nonlocal packets
        if foreign_processes():
            raise RuntimeError("Another process is using the leased 4090")
        if shutil.disk_usage(output).free < 8 * 1024**3:
            raise RuntimeError("Less than 8 GiB remains for immutable simulation evidence")
        controls_phase = phase.startswith("controls_")
        directory = "native-controls" if controls_phase else "native-heading"
        packet = f"requests/batch-{packets:05d}.json"
        packets += 1
        atomic(output / packet, dict(phase=phase, rows=requests))
        measured, timing = (controls if controls_phase else engine).evaluate(requests, phase)
        if [row["id"] for row in measured] != [request["id"] for request in requests]:
            raise RuntimeError("Native trial IDs changed")
        used = controls.engine.adapter if controls_phase else engine.adapter
        if used.env.effective_options != expected_options or used.env.compiled_geometry != expected_geometry:
            raise RuntimeError("Native physics options or collision geometry changed")
        if not controls_phase:
            annotate(measured, output / directory)
        packet_hash = sha(output / packet)
        for row in measured:
            row.update(request_packet=packet, request_packet_sha256=packet_hash, adapter_directory=directory)
        append(output / "rows.jsonl", measured)
        append(output / "timing.jsonl", [dict(packet=packet, phase=phase, timing=timing)])
        return measured

    def run_controls(phase):
        gate = control_gate(execute(control_requests(source_controls, phase), phase))
        report["controls"][phase] = gate
        if gate["status"] != "passed":
            raise RuntimeError(f"{phase} failed")
        publish(phase)

    def flush():
        nonlocal release_reserve
        if remaining() < release_reserve:
            return False
        packet_started = time.monotonic()
        measured = execute(pending, "search")
        release_reserve = max(release_reserve, 1.5 * (time.monotonic() - packet_started))
        report["search_releases"] += len(measured)
        for request, row in zip(pending, measured):
            experimental = row["experimental_route"]
            route = experimental["route_class"]
            outcomes[route] += 1
            group = group_counts[request["group"]]
            group["releases"] += 1
            group[route] += 1
            events = dict(board_bounces=bool(row["board_bounce"]),
                          robot_contacts=experimental["robot_contact_steps"] > 0,
                          whole_ball_entries=experimental["whole_ball_entry"],
                          invalid_physics=bool(row["invalid_physics"]))
            for key, occurred in events.items():
                contact_counts[key] += int(occurred)
                group[key] += int(occurred)
            if route in ROUTES and len(picks[route]) < 3:
                picks[route].append(request.copy())
        pending.clear()
        publish("search")
        return True

    try:
        publish("initializing")
        run_controls("controls_pre")
        if continuation is not None:
            search_started = time.monotonic()
            deadline = search_started + args.search_seconds
        stop_reason = None
        while report["search_releases"] + len(pending) < args.limit and report["attempts"] < args.max_preparations:
            if remaining() <= release_reserve:
                stop_reason = "time_limit"
                break
            proposals = []
            for _ in range(min(args.prepare_batch, args.max_preparations - report["attempts"])):
                item = next(generator, None)
                if item is None:
                    break
                index = report["attempts"]
                report["attempts"] += 1
                prefix = "cell1050-followup" if continuation is not None else "cell1050"
                proposals.append(dict(item, id=f"{prefix}-{index:05d}", phase="search",
                                      role="development", context_id=CONTEXT, method="heading_search"))
                group_counts[item["group"]]["proposals"] += 1
            if not proposals:
                break
            preparation = f"preparations/batch-{proposals[0]['id']}.json"
            atomic(output / preparation, dict(proposals=proposals))
            try:
                limit = remaining() - release_reserve if deadline is not None else None
                with preparation_time_limit(limit):
                    admitted, records = engine.prepare(proposals)
            except SearchPreparationExpired:
                report["interrupted_proposals"] = len(proposals)
                stop_reason = "time_limit"
                break
            report["admitted"] += len(admitted)
            admitted_ids = {row["id"] for row in admitted}
            report["preparation_declines"] += sum(
                not any(identity == item["id"] or identity.startswith(item["id"] + "-")
                        for identity in admitted_ids) for item in proposals)
            atomic(output / preparation, dict(proposals=proposals, records=records, admitted=admitted))
            for row in admitted:
                key = command_key(row)
                if key in history_keys:
                    report["duplicate_commands"] += 1
                    continue
                history_keys.add(key)
                pending.append(row)
                if len(pending) == 32 and not flush():
                    stop_reason = "time_limit"
                    break
                if report["search_releases"] + len(pending) == args.limit:
                    break
            publish("preparing")
            if stop_reason is not None:
                break
        if pending and not flush():
            stop_reason = "time_limit"
        if continuation is not None:
            if report["search_releases"] == args.limit:
                stop_reason = "release_limit"
            report.update(stop_reason=stop_reason or "proposals_exhausted",
                          search_elapsed_s=round(time.monotonic() - search_started, 3),
                          prepared_not_released=len(pending))
            publish("search_finished")
        repeat_plan = []
        for route in ROUTES:
            for shot in picks[route]:
                for index in range(3):
                    repeat_plan.append(dict(shot, id=f"{shot['id']}-repeat-{index}",
                                            phase="repeat", selected_from_id=shot["id"], expected_route=route))
        atomic(output / "repeat-plan.json", repeat_plan)
        if repeat_plan:
            measured = execute(repeat_plan, "repeat")
            counts = Counter((r["selected_from_id"], r["experimental_route"]["route_class"]) for r in measured)
            report["confirmations"] = [
                dict(id=shot["id"], route=route, captures=counts[shot["id"], route], repeats=3,
                     values=shot["values"], board_heading_offset_deg=shot["board_heading_offset_deg"],
                     board_twist_deg=shot.get("board_twist_deg", 0.0),
                     q10=shot["selected_q10"], q14=shot["selected_q14"])
                for route in ROUTES for shot in picks[route]
            ]
            publish("repeat")
        run_controls("controls_post")
        if (sha(parent / "report.json") != binding["report"] or sha(args.parent_audit) != binding["audit"]
                or sha(parent / "evidence-manifest.json") != binding["manifest"]):
            raise RuntimeError("Parent evidence changed during the study")
        atomic(output / "evidence-manifest.json", {
            str(path.relative_to(output)): sha(path) for path in sorted(output.rglob("*"))
            if path.is_file() and path.name not in ("report.json", "evidence-manifest.json")
        })
        complete = report["search_releases"] == args.limit or stop_reason == "time_limit"
        report.update(status="completed" if complete else "incomplete",
                      evidence_manifest_sha256=sha(output / "evidence-manifest.json"))
        publish(report["status"])
        if not complete:
            raise RuntimeError("The finite proposal stream did not yield the requested fresh releases")
    except Exception as exc:
        if report["status"] != "incomplete":
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            publish("failed")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-study", "parent-audit", "reference", "assets", "output", "gpu-lock"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--max-preparations", type=int, default=16000)
    parser.add_argument("--prepare-batch", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--previous-study", type=Path)
    parser.add_argument("--previous-audit", type=Path)
    parser.add_argument("--search-seconds", type=int)
    args = parser.parse_args()
    followup = args.previous_study is not None
    if followup != (args.previous_audit is not None) or followup != (args.search_seconds is not None):
        parser.error("Use the previous study, its audit and the search time budget together")
    if followup and not SEARCH_PACKET_RESERVE_S < args.search_seconds <= 7 * 3600:
        parser.error("Use a follow-up search budget over two minutes and at most seven hours")
    max_releases = 10000 if followup else 2000
    if args.output.exists() or not 1 <= args.limit <= max_releases:
        parser.error(f"Use a fresh output and a release count from 1 to {max_releases}")
    max_preparations = 40000 if followup else 32000
    if not args.limit <= args.max_preparations <= max_preparations or not 1 <= args.prepare_batch <= 64:
        parser.error("Use a finite proposal budget and a preparation batch from 1 to 64")
    with gpu_slot(args.gpu_lock, timeout_s=900, notify=lambda value: print(json.dumps(value), flush=True)):
        run(args)


if __name__ == "__main__":
    main()
