"""Verify completed mixed-layout native64/v3 searches using saved evidence on CPU."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from . import replay650_verify as replay
from . import table_assisted_score_oracle as oracle
from .benchmark_verify import Checks, contained, digest, read, sha
from .calibrated_benchmark_verify import check_native_clock
from .cell1050_followup_plan import geometry_key
from .cell1050_rebound_audit import verify_heading
from .cell1050_rebound_score import score_recorded_outcome
from .cell1050_rebound_study import parent_inputs
from .policy64_verify import audit_native_options
from .sweep_worker import atomic
from .table_training_study import control_gate
from .table_training_verify import check_native_score, rescore_rows


VERSION = "parallel-capture-grid-v1"
PHASE_ORDER = {"controls_pre": 0, "search": 1, "controls_post": 2}


def verify_frozen_inputs(study, parent, parent_audit, checks):
    spec, report = read(study / "contract.json"), read(study / "report.json")
    parent_spec, binding, _ = parent_inputs(parent, parent_audit)
    checks.equal((spec["version"], report["status"], report["phase"]),
                 (VERSION, "completed", "completed"), "Completed parallel grid study")
    checks.equal(spec["fingerprint"], digest({key: value for key, value in spec.items() if key != "fingerprint"}),
                 "Frozen grid contract")
    checks.equal((spec["identity"], spec["scene"], spec["parent"]),
                 (parent_spec["identity"], parent_spec["scene"], binding), "Exact parent physics and audit")
    checks.equal(spec["native_batch"], 32, "Native packet capacity")
    checks.require(type(spec["target_per_cell"]) is int and 1 <= spec["target_per_cell"] <= 1000,
                   "Positive per-cell release quota up to 1000")
    checks.equal(report["target_per_cell"], spec["target_per_cell"], "Report keeps the frozen quota")
    manifest = replay.check_manifest(study, checks)
    checks.equal(sha(study / "evidence-manifest.json"), report["evidence_manifest_sha256"], "Frozen final evidence")
    for name, key in (("plan.json", "plan_sha256"), ("historical-geometries.json", "history_sha256")):
        checks.equal(manifest[name], spec[key], "Frozen grid input " + name)
    cells = spec["cells"]
    parent_grid_path = parent / "grid.json"
    checks.equal(sha(parent_grid_path), read(parent / "evidence-manifest.json")["grid.json"],
                 "Frozen parent bucket grid")
    parent_cells = {row["id"]: row["bucket_xy_m"] for row in read(parent_grid_path)}
    checks.require(isinstance(cells, dict) and len(cells) == 13, "Thirteen distinct bucket locations required")
    checks.equal(set(cells), set(report["cells"]), "Report covers precisely the frozen cells")
    plan = read(study / "plan.json")
    checks.equal(set(plan), set(cells), "Plan covers precisely the frozen cells")
    for context_id, bucket_xy in cells.items():
        checks.equal(bucket_xy, parent_cells.get(context_id), context_id + "/canonical bucket location")
        setup, summary = plan[context_id], report["cells"][context_id]
        checks.equal((setup["context_id"], setup["bucket_xy_m"], summary["bucket_xy_m"]),
                     (context_id, bucket_xy, bucket_xy), context_id + "/plan and report location")
        known = {tuple(setting) for setting in setup["prior_settings"]}
        checks.require(0 < len(known) < 25, context_id + "/fewer than 25 prior captured settings")
        checks.equal((len(setup["prior_settings"]), setup["prior_count"], summary["prior_count"]),
                     (len(known), len(known), len(known)), context_id + "/unique prior capture count")
    checks.equal(len({tuple(xy) for xy in cells.values()}), len(cells), "No duplicate bucket locations")
    history = read(study / "historical-geometries.json")
    checks.equal(set(history), set(cells), "Historical exclusions cover precisely the frozen cells")
    for context_id, entries in history.items():
        checks.require(all(isinstance(entry, list) and len(entry) == 7 and all(
            type(x) in (float, int) and math.isfinite(x) for x in entry) for entry in entries),
            context_id + "/finite seven-value historical keys")
        keys = {tuple(entry) for entry in entries}
        checks.equal(len(keys), len(entries), context_id + "/unique historical keys")
        history[context_id] = keys
    return spec, report, manifest, history


def preparation_inventory(study, cells, checks):
    attempts, declines = Counter(), Counter()
    admitted = {}
    proposal_ids = set()
    for path in sorted((study / "preparations").glob("*.json")):
        packet = read(path)
        checks.equal(len(packet["proposals"]), len(packet["records"]), path.name + "/one record per proposal")
        for proposal, record in zip(packet["proposals"], packet["records"]):
            identifier, context_id = proposal["id"], proposal["context_id"]
            checks.require(identifier not in proposal_ids, identifier + "/unique prepared proposal")
            proposal_ids.add(identifier)
            checks.equal(proposal["bucket_xy_m"], cells[context_id], identifier + "/frozen bucket")
            checks.equal((record["proposal_id"], record["proposal_sha256"]),
                         (identifier, digest(proposal)), identifier + "/preparation identity")
            ids = record["admitted_request_ids"]
            checks.equal(len(ids), record["admitted"], identifier + "/admitted branch count")
            for request_id in ids:
                checks.require(request_id not in admitted, request_id + "/unique prepared branch")
                admitted[request_id] = proposal
            attempts[context_id] += 1
            declines[context_id] += int(not ids)
    return attempts, declines, admitted


def observed_rows(path):
    with path.open() as stream:
        for line in stream:
            if not line.endswith("\n"):
                raise ValueError("Incomplete observation ledger row")
            yield json.loads(line)


def verify_release(row, request, path, packet_hash, phase, checks):
    name = request["id"]
    checks.equal((row["id"], row["phase"], request["phase"]), (name, phase, phase),
                 name + "/chronological packet and phase")
    checks.equal((row["request_packet"], row["request_packet_sha256"]),
                 (path, packet_hash), name + "/frozen request packet")
    checks.equal((row["context_id"], row["values"], row["bucket_xy_m"], row["native_release"]),
                 (request["context_id"], request["values"], request["bucket_xy_m"], True),
                 name + "/observed exact release")


def rescore_packet(directory, rows, scene, calibration, manifest, used_archives, checks):
    name = rows[0]["native_cache_file"]
    checks.require(name not in used_archives, name + "/one archive per search packet")
    used_archives.add(name)
    path = contained(directory, name)
    checksum = sha(path)
    checks.equal(checksum, manifest["native-heading/" + name], name + "/frozen archive")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    checks.equal(list(arrays["record_ids"]), [row["id"] for row in rows], name + "/exact archive row order")
    ticks = np.arange(61441) / 15360
    checks.numeric(arrays["time_s"], ticks, name + "/canonical native ticks", 1e-12)
    clock = check_native_clock(arrays["native_time_s"], ticks, checks, name)
    for index, row in enumerate(rows):
        label = row["id"]
        checks.equal((row["native_cache_file"], row["native_cache_row"], row["native_cache_sha256"]),
                     (name, index, checksum), label + "/dense trace identity")
        checks.numeric(row["native_clock_drift_s"], clock[index], label + "/native clock drift", 1e-12)
        trace = {key: value[:, index] if value.ndim >= 2 else value for key, value in arrays.items()}
        checks.numeric(trace["position_m"][0], row["native_initial_ball_center_m"], label + "/observed release", 1e-9)
        release = np.asarray(row["values"][:3])
        release[2] += row["values"][4] + scene["ball"]["radius_m"]
        checks.numeric(trace["position_m"][0], release, label + "/requested release", 2e-6)
        checks.numeric(row["native_initial_bucket_center_m"], [*row["bucket_xy_m"], 0],
                       label + "/observed bucket reset", 2e-6)
        score = oracle.score_trace(scene, row["bucket_xy_m"], row["board_normal"], calibration, trace)
        check_native_score(row, score, checks)
        result = score_recorded_outcome(row, arrays, world=index)
        checks.equal(row["experimental_route"], result, label + "/recorded route reproduced")
        yield result["route_class"]


def verify_counts(spec, report, routes, attempts, declines, search_packets, mixed_packets, max_cells, checks):
    result = {}
    for context_id in spec["cells"]:
        releases = sum(routes[context_id].values())
        checks.equal(releases, spec["target_per_cell"], context_id + "/exact released quota")
        summary = dict(search_releases=releases, route_counts=dict(routes[context_id]),
                       attempts=attempts[context_id], preparation_declines=declines[context_id])
        measured = report["cells"][context_id]
        checks.equal({key: measured[key] for key in summary}, summary, context_id + "/recorded results")
        result[context_id] = summary
    total = sum(row["search_releases"] for row in result.values())
    checks.equal(report["search_releases"], total, "All per-cell search releases accounted for")
    checks.require(mixed_packets > 0, "At least one native packet contains several bucket layouts")
    checks.equal((report["search_packets"], report["mixed_packets"], report["max_cells_in_packet"]),
                 (search_packets, mixed_packets, max_cells), "Reported mixed-layout native execution")
    return total, result


def verify_controls(study, parent, spec, report, controls, calibration, assets, checks):
    sources = read(parent / "control-replays.json")
    rescore_rows(study / "native-controls", controls, spec["scene"], calibration,
                 spec["identity"]["trace_schema"], assets, checks)
    for phase in ("controls_pre", "controls_post"):
        rows = [row for row in controls if row["phase"] == phase]
        expected = [(r["id"], digest(r), r["bucket_xy_m"], r["values"]) for r in sources for _ in range(3)]
        actual = [(r["source_record_id"], r["source_record_sha256"], r["bucket_xy_m"], r["values"]) for r in rows]
        checks.equal(actual, expected, phase + "/exact original control settings")
        gate = control_gate(rows)
        checks.equal(gate["status"], "passed", phase + "/expected outcomes")
        checks.equal(gate, report["controls"][phase], phase + "/recorded control gate")


def audit(study, parent, parent_audit, assets):
    checks = Checks()
    spec, report, manifest, history = verify_frozen_inputs(study, parent, parent_audit, checks)
    attempts, declines, admitted = preparation_inventory(study, spec["cells"], checks)
    calibration = replay.check_calibration(study, spec["identity"]["calibration"], checks)
    for namespace in ("native-heading", "native-controls"):
        audit_native_options(study / namespace, spec["scene"], calibration, assets, checks)
    home, prior = read(parent / "setup-home.json"), read(parent / "prior.json")
    ledger = observed_rows(study / "rows.jsonl")
    controls, resolutions = [], {}
    routes = {context_id: Counter() for context_id in spec["cells"]}
    ids, released_proposals, used_archives = set(), set(), set()
    search_packets = mixed_packets = max_cells = 0
    previous_phase = 0
    for path in sorted((study / "requests").glob("*.json")):
        packet = read(path)
        phase, requests = packet["phase"], packet["rows"]
        label = str(path.relative_to(study))
        packet_hash = manifest[label]
        checks.require(previous_phase <= PHASE_ORDER[phase], label + "/ordered controls and search")
        previous_phase = PHASE_ORDER[phase]
        checks.require(0 < len(requests) <= spec["native_batch"], label + "/bounded native packet")
        rows = []
        for request in requests:
            row = next(ledger, None)
            checks.require(row is not None, label + "/every request has an observation")
            verify_release(row, request, label, packet_hash, phase, checks)
            name = row["id"]
            checks.require(name not in ids, name + "/unique observed trial ID")
            ids.add(name)
            if phase != "search":
                checks.equal(row["adapter_directory"], "native-controls", name + "/control archive namespace")
                controls.append(row)
                continue
            context_id = row["context_id"]
            checks.equal(row["bucket_xy_m"], spec["cells"][context_id], name + "/frozen bucket location")
            checks.equal(row["adapter_directory"], "native-heading", name + "/search archive namespace")
            proposal = admitted[name]
            checks.equal((row["proposal_id"], row["context_id"], geometry_key(row)),
                         (proposal["id"], proposal["context_id"], geometry_key(proposal)),
                         name + "/exact prepared geometry")
            checks.require(proposal["id"] not in released_proposals, name + "/one released branch per proposal")
            released_proposals.add(proposal["id"])
            key = geometry_key(row)
            checks.require(key not in history[context_id], name + "/new shot geometry at this cell")
            history[context_id].add(key)
            verify_heading(row, request, study / "native-heading", spec["scene"], home, prior, resolutions, checks)
            rows.append(row)
        if phase == "search":
            contexts = len({row["context_id"] for row in rows})
            search_packets += 1
            mixed_packets += int(contexts > 1)
            max_cells = max(max_cells, contexts)
            classes = rescore_packet(study / "native-heading", rows, spec["scene"], calibration, manifest,
                                     used_archives, checks)
            for row, route in zip(rows, classes, strict=True):
                routes[row["context_id"]][route] += 1
    checks.require(next(ledger, None) is None, "Every observation belongs to a frozen request")
    checks.equal(used_archives, {str(path.relative_to(study / "native-heading"))
                 for path in (study / "native-heading").rglob("*.npz")}, "Complete native search archive inventory")
    verify_controls(study, parent, spec, report, controls, calibration, assets, checks)
    total, cells = verify_counts(spec, report, routes, attempts, declines,
                                 search_packets, mixed_packets, max_cells, checks)
    return dict(status="passed", checks_count=checks.count, search_releases=total,
                total_releases=total + len(controls), cells=cells, search_packets=search_packets,
                mixed_packets=mixed_packets, max_cells_in_packet=max_cells,
                report_sha256=sha(study / "report.json"),
                evidence_manifest_sha256=sha(study / "evidence-manifest.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "parent-study", "parent-audit", "assets", "audit"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if args.audit.exists() or args.audit.resolve().is_relative_to(args.study.resolve()):
        parser.error("Write a new audit outside the frozen evidence directory")
    try:
        result = audit(args.study, args.parent_study, args.parent_audit, args.assets)
    except Exception as exc:
        result = dict(status="failed", error=f"{type(exc).__name__}: {exc}")
    atomic(args.audit, result)
    print(json.dumps(result), flush=True)
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
