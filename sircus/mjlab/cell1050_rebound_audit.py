"""Audit the heading study and rescore its saved dense observations on CPU."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from . import replay650_verify as replay
from . import table_assisted_score_oracle as oracle
from .benchmark_verify import contained, digest, read, sha
from .calibrated_admission import verify_connection, verify_resolution
from .calibrated_benchmark_verify import check_native_clock
from .cell1050_plan import command_key
from .cell1050_rebound_score import score_recorded_outcome
from .cell1050_rebound_study import ROUTES, VERSION, parent_inputs
from .policy64_verify import audit_native_options
from .sweep_worker import atomic
from .table_training_verify import check_native_score, rescore_rows
from .table_training_study import control_gate


def verify_heading(row, request, directory, scene, home, prior, resolutions, checks):
    name = row["id"]
    resolution_name = row["context_resolution_file"]
    if resolution_name not in resolutions:
        path = contained(directory, resolution_name)
        resolution = dict(read(path), file=resolution_name, sha256=sha(path))
        verify_resolution(resolution, prior["records"], home)
        resolutions[resolution_name] = resolution
    resolution = resolutions[resolution_name]
    checks.equal(row["context_resolution_sha256"], resolution["sha256"], name + "/frozen resolution")
    file = request["prepared_connection_file"]
    path = contained(directory, file)
    proof = dict(read(path), file=file, sha256=sha(path))
    checks.equal((row["connection_file"], row["connection_sha256"]),
                 (file, proof["sha256"]), name + "/frozen connection")
    checks.equal(proof["sha256"], request["prepared_connection_sha256"], name + "/request proof bytes")
    verify_connection(proof, resolution, home, request["values"])
    checks.require(proof["passed"], name + "/passed path proof")
    for key in ("board_heading_offset_deg", "board_twist_deg", "heading_key"):
        checks.equal((row[key], proof[key]), (request[key], request[key]), name + "/" + key)
    for key in ("q10", "q14"):
        checks.equal((row[key], proof[key]), (request["selected_" + key], request["selected_" + key]),
                     name + "/requested joint branch/" + key)
        checks.equal(row["native_initial_" + key], np.asarray(row[key], np.float32).astype(float).tolist(),
                     name + "/observed exact reset joints/" + key)
    values = np.asarray(request["values"])
    delta = np.asarray(request["bucket_xy_m"]) - values[:2]
    azimuth = np.arctan2(delta[1], delta[0]) + np.pi / 2 + np.deg2rad(request["board_heading_offset_deg"])
    spin = scene["board"]["mount_yaw_rad"] + np.deg2rad(request["board_twist_deg"])
    expected = Rotation.from_euler("z", azimuth) * Rotation.from_euler("x", values[3], degrees=True) * Rotation.from_euler("z", spin)
    actual = Rotation.from_quat(np.roll(row["board_quaternion_wxyz"], -1))
    checks.require((actual * expected.inv()).magnitude() < 4e-4, name + "/measured full board orientation")
    checks.numeric(row["board_normal"], actual.as_matrix()[:, 2], name + "/measured board normal", 2e-6)
    checks.require(np.linalg.norm(np.asarray(row["board_center_m"]) - values[:3]) < .002,
                   name + "/measured board position")


def audit(study, parent, parent_audit, assets):
    checks = replay.AuditChecks()
    spec, report = read(study / "contract.json"), read(study / "report.json")
    parent_spec, binding, _ = parent_inputs(parent, parent_audit)
    checks.equal((spec["version"], report["status"]), (VERSION, "completed"), "Completed finite heading study")
    checks.equal(spec["fingerprint"], digest({k: v for k, v in spec.items() if k != "fingerprint"}), "Frozen experiment contract")
    checks.equal((spec["identity"], spec["scene"], spec["parent"]),
                 (parent_spec["identity"], parent_spec["scene"], binding), "Exact parent physics and audit")
    replay.check_manifest(study, checks)
    checks.equal(sha(study / "evidence-manifest.json"), report["evidence_manifest_sha256"], "Frozen final evidence")
    for name, key in (("seeds.json", "seed_sha256"), ("historical-commands.json", "historical_commands_sha256")):
        checks.equal(sha(study / name), spec[key], "Frozen initial input " + name)
    calibration = replay.check_calibration(study, spec["identity"]["calibration"], checks)
    rows = replay.read_records(study / "rows.jsonl")
    search = [row for row in rows if row["phase"] == "search"]
    repeats = [row for row in rows if row["phase"] == "repeat"]
    checks.equal(len(search), spec["target_releases"], "Exact number of fresh native search releases")
    checks.equal(len({row["id"] for row in rows}), len(rows), "Unique observed trial IDs")
    history = set(read(study / "historical-commands.json"))
    checks.equal(len(history), spec["history_commands"], "Unique frozen historical command inventory")
    packets = {}
    for path in sorted((study / "requests").glob("*.json")):
        packet = read(path)
        for request in packet["rows"]:
            checks.require(request["id"] not in packets, "Each request frozen once")
            packets[request["id"]] = (request, path, packet["phase"])
    checks.equal(list(packets), [row["id"] for row in rows], "Complete chronological request ledger")
    by_id = {row["id"]: row for row in search}
    directory = study / "native-heading"
    home, prior = read(parent / "setup-home.json"), read(parent / "prior.json")
    resolutions = {}
    for row in rows:
        request, path, phase = packets[row["id"]]
        checks.equal(row["phase"], phase, row["id"] + "/request phase")
        checks.equal((row["request_packet"], row["request_packet_sha256"]),
                     (str(path.relative_to(study)), sha(path)), row["id"] + "/request bytes")
        checks.equal((row["values"], row["bucket_xy_m"], row["native_release"]),
                     (request["values"], request["bucket_xy_m"], True), row["id"] + "/observed exact release")
        if phase == "search":
            key = command_key(row)
            checks.require(key not in history, row["id"] + "/fresh physical command")
            history.add(key)
        if phase == "repeat":
            original = by_id[request["selected_from_id"]]
            for key in ("values", "bucket_xy_m", "q10", "q14", "heading_key", "connection_sha256"):
                checks.equal(row[key], original[key], row["id"] + "/exact replay/" + key)
        if phase in ("search", "repeat"):
            checks.numeric(row["bucket_xy_m"], [1.05, .025], row["id"] + "/fixed bucket", 1e-12)
            verify_heading(row, request, directory, spec["scene"], home, prior, resolutions, checks)

    for namespace in ("native-heading", "native-controls"):
        audit_native_options(study / namespace, spec["scene"], calibration, assets, checks)
    control_rows = [row for row in rows if row["phase"].startswith("controls_")]
    rescore_rows(study / "native-controls", control_rows, spec["scene"], calibration,
                 spec["identity"]["trace_schema"], assets, checks)
    for phase in ("controls_pre", "controls_post"):
        gate = control_gate([row for row in control_rows if row["phase"] == phase])
        checks.equal(gate["status"], "passed", phase + "/expected outcome")
        checks.equal(gate, report["controls"][phase], phase + "/recorded outcome")

    groups = defaultdict(list)
    for row in search + repeats:
        groups[row["native_cache_file"]].append(row)
    checks.equal(set(groups), {str(p.relative_to(directory)) for p in directory.rglob("*.npz")},
                 "All heading trace archives assigned")
    ticks = np.arange(61441) / 15360
    for name, members in groups.items():
        path = contained(directory, name)
        checksum = sha(path)
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        checks.equal({r["native_cache_row"] for r in members}, set(range(len(arrays["record_ids"]))),
                     name + "/all native columns assigned")
        checks.equal(len(members), len(arrays["record_ids"]), name + "/no duplicated native columns")
        checks.numeric(arrays["time_s"], ticks, name + "/exact canonical native ticks", 1e-12)
        clock = check_native_clock(arrays["native_time_s"], ticks, checks, name)
        for row in members:
            index = row["native_cache_row"]
            checks.equal((str(arrays["record_ids"][index]), checksum),
                         (row["id"], row["native_cache_sha256"]), row["id"] + "/dense archive identity")
            checks.numeric(row["native_clock_drift_s"], clock[index], row["id"] + "/recorded clock drift", 1e-12)
            trace = {key: value[:, index] if value.ndim >= 2 else value for key, value in arrays.items()}
            checks.numeric(trace["position_m"][0], row["native_initial_ball_center_m"], row["id"] + "/observed release", 1e-9)
            release = np.asarray(row["values"][:3])
            release[2] += row["values"][4] + spec["scene"]["ball"]["radius_m"]
            checks.numeric(trace["position_m"][0], release, row["id"] + "/requested release", 2e-6)
            checks.numeric(row["native_initial_bucket_center_m"], [*row["bucket_xy_m"], 0],
                           row["id"] + "/observed bucket reset", 2e-6)
            score = oracle.score_trace(spec["scene"], row["bucket_xy_m"], row["board_normal"], calibration, trace)
            check_native_score(row, score, checks)
            checks.equal(score_recorded_outcome(row, arrays, world=index), row["experimental_route"],
                         row["id"] + "/experimental outcome reproduced")
    counts = Counter(row["experimental_route"]["route_class"] for row in search)
    checks.equal(dict(counts), report["route_counts"], "Independent route totals")
    confirmed = []
    for item in report["confirmations"]:
        source = by_id[item["id"]]
        observed = [row for row in repeats if row["selected_from_id"] == source["id"]]
        route = source["experimental_route"]["route_class"]
        checks.require(route in ROUTES, source["id"] + "/accepted repeat source")
        checks.equal((item["route"], len(observed), item["repeats"]), (route, 3, 3),
                     source["id"] + "/exactly three selected repeats")
        captures = sum(row["experimental_route"]["route_class"] == route for row in observed)
        checks.equal(captures, item["captures"], source["id"] + "/same-route repeat count")
        if captures == 3:
            confirmed.append(item)
    return dict(status="passed", checks_count=checks.count, search_releases=len(search),
                total_releases=len(rows), route_counts=dict(counts), confirmed=confirmed,
                report_sha256=sha(study / "report.json"), evidence_manifest_sha256=sha(study / "evidence-manifest.json"))


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
