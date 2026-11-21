"""Frozen, resumable one-drop acceptance of the live fast and original planners.

Run on the service host. Only the explicit ``run`` command sends plan/drop
requests; ``freeze`` inventories coordinates and ``audit`` only reads evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import fcntl
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from statistics import median
import time
from urllib.parse import quote

from .interactive_physics_check import (
    CAPTURES, Client, _bounds, _checksum, _digest, _finite, _require,
    historical_layouts, sample_positions, verify_preview as historical_preview, verify_result,
)

VERSION = "sircus-interactive-fast-acceptance-v1"
METHODS = ("fast", "classic")
SEED = 2026120101
COUNT = 10
FINAL = {"completed", "no_setup", "failed", "expired", "unreleased_uncertain"}
SCOPE = ("Ten fresh bucket placements; one explicit drop per planner and position at 64 substeps. "
         "Simulator geometry and held arm endpoints; no training or physical validation.")


def read(path):
    return json.loads(Path(path).read_text())


def checksum(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    """Durably record intent before any potentially non-idempotent HTTP request."""
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def coordinates(value):
    """Extract declared bucket coordinates, never arbitrary pose/vector pairs."""
    result = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "bucket_xy_m":
                _require(isinstance(child, list) and len(child) == 2 and all(map(_finite, child)),
                         "Historical bucket coordinates must be finite")
                result.add(tuple(child))
            elif key == "excluded_coordinates_m":
                for xy in child:
                    _require(len(xy) == 2 and all(map(_finite, xy)), "Invalid excluded coordinates")
                    result.add(tuple(xy))
            else:
                result.update(coordinates(child))
    elif isinstance(value, list):
        for child in value:
            result.update(coordinates(child))
    return result


def fast_inventory(study, audit_path):
    report, audit, contract, manifest, layouts = (read(path) for path in (
        study / "report.json", audit_path, study / "contract.json", study / "evidence-manifest.json",
        study / "layouts.json"))
    _require(report["status"] == "completed" and audit["status"] == "passed"
             and audit["version"] == "sircus-fast-planning-independent-audit-v1"
             and contract["version"] == "sircus-fast-planning-study-v1"
             and contract["physics_substeps"] == 64, "Need a passed native64 fast-planning comparison")
    _require(audit["report_sha256"] == checksum(study / "report.json")
             and audit["evidence_manifest_sha256"] == checksum(study / "evidence-manifest.json")
             and audit["comparison"] == report["comparison"], "Fast comparison differs from its audit")
    _require(contract["fingerprint"] == _digest({k: v for k, v in contract.items() if k != "fingerprint"}),
             "Fast comparison contract changed")
    for name in ("contract.json", "layouts.json", "calibration.json", "neural-checkpoint.json", "choices.json"):
        _require(checksum(study / name) == manifest[name], f"Audited fast input changed: {name}")
    provenance = layouts["provenance"]
    _require(len(layouts["layouts"]) == 50 and _digest(layouts["layouts"]) == provenance["layouts_sha256"]
             and _digest(provenance["excluded_coordinates_m"]) == provenance["excluded_coordinates_sha256"],
             "Fast study's historical coordinate inventory changed")
    return coordinates(layouts), dict(
        fast_study_report_sha256=checksum(study / "report.json"),
        fast_study_audit_sha256=checksum(audit_path),
        fast_choices_sha256=checksum(study / "choices.json"),
        fast_evidence_manifest_sha256=checksum(study / "evidence-manifest.json"),
        neural_checkpoint_sha256=checksum(study / "neural-checkpoint.json")), contract


def ordered_rows(placements):
    return [dict(placement, planner=method, state="pending") for index, placement in enumerate(placements)
            for method in (METHODS if index % 2 == 0 else METHODS[::-1])]


def latency(values):
    values = sorted(values)
    _require(all(_finite(v) and v >= 0 for v in values), "Invalid measured latency")
    return (dict(samples=len(values), mean_s=math.fsum(values) / len(values), median_s=median(values),
                 p90_s=values[math.ceil(.9 * len(values)) - 1], max_s=max(values)) if values else dict(samples=0))


def summarize(rows, count=COUNT):
    comparison = {}
    for method in METHODS:
        own = [row for row in rows if row["planner"] == method]
        counts = Counter(row.get("outcome", row["state"]) for row in own)
        comparison[method] = dict(
            positions=count, completed=sum(row["state"] in FINAL for row in own),
            captures=sum(row.get("outcome") in CAPTURES for row in own),
            releases=sum(row["state"] == "completed" for row in own),
            no_setup=counts["no_setup"], failed=sum(counts[k] for k in FINAL - {"completed", "no_setup"}),
            outcomes=dict(counts), planning=latency([row["planning_s"] for row in own if "planning_s" in row]),
            backend_request=latency([row["preview"]["timing"]["planning_wall_s"] for row in own
                                    if "planning_wall_s" in row.get("preview", {}).get("timing", {})]),
            drop=latency([row["drop_s"] for row in own if "drop_s" in row]),
            worker_reused=sum(row.get("preview", {}).get("timing", {}).get("worker_reused") is True for row in own))
    paired = dict(both=0, gained=0, lost=0, neither=0)
    cases = {row["case"] for row in rows}
    for case in cases:
        pair = {row["planner"]: row for row in rows if row["case"] == case}
        if set(pair) != set(METHODS) or any(row["state"] not in FINAL for row in pair.values()):
            continue
        fast, classic = (pair[method].get("outcome") in CAPTURES for method in METHODS)
        paired["both" if fast and classic else "gained" if fast else "lost" if classic else "neither"] += 1
    return comparison, paired


def publish(args, report):
    report["updated_at_unix"] = time.time()
    report["progress"] = dict(completed=sum(row["state"] in FINAL for row in report["results"]), planned=COUNT * 2)
    report["comparison"], report["paired"] = summarize(report["results"])
    save(args.output / "report.json", report)
    if args.display_report:
        display = read(args.display_report) if args.display_report.exists() else {}
        display["interactive_fast_acceptance"] = {key: value for key, value in report.items() if key != "results"}
        display["interactive_fast_acceptance"]["positions"] = [
            {key: value for key, value in row.items() if key in
             {"case", "region", "bucket_xy_m", "planner", "state", "outcome", "planning_s", "drop_s", "message"}}
            for row in report["results"]]
        save(args.display_report, display)


def freeze(args):
    geometry, grid, known, system_source = historical_layouts(args.system_study, args.system_audit)
    fast_known, fast_source, contract = fast_inventory(args.fast_study, args.fast_audit)
    _require(contract["identity"]["scene_sha256"] == geometry["scene_sha256"], "Scenes differ")
    known = set(map(tuple, known)) | fast_known
    history = {}
    for root in dict.fromkeys([*args.history_evidence, args.evidence_root]):
        _require(root.is_dir(), f"Interactive evidence directory does not exist: {root}")
        for pattern in ("**/request.json", "**/preview.json", "**/plan.json", "**/positions.json"):
            for path in sorted(root.glob(pattern)):
                raw = path.read_bytes()
                known.update(coordinates(json.loads(raw)))
                history[str(path.resolve())] = sha256(raw).hexdigest()
    capability = Client(args.base_url).request()
    _require({row["id"] for row in capability.get("planners", [])} == set(METHODS), "Both live planners must be available")
    known = sorted(known)
    bounds = _bounds(capability)
    placements = sample_positions(geometry, grid, known, bounds, args.seed)
    _require(len(placements) == COUNT, "Freeze exactly ten placements")
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = dict(version=VERSION, seed=args.seed, placements=placements, positions_sha256=_digest(placements),
                  minimum_separation_m=.005, excluded_coordinates_m=known,
                  excluded_coordinates_sha256=_digest(known), history_files=history,
                  system_source=system_source, fast_source=fast_source, geometry=geometry, grid=grid,
                  bounds=bounds, scene=contract["scene"], calibration=read(args.fast_study / "calibration.json"),
                  base_url=args.base_url, evidence_root=str(args.evidence_root.resolve()),
                  physics_substeps=64, attempts_per_method_per_position=1, scope=SCOPE)
    save(args.output / "positions.json", frozen)
    report = dict(version=VERSION, status="initialized", audit_status="pending", scope=SCOPE,
                  positions_sha256=checksum(args.output / "positions.json"),
                  results=ordered_rows(placements), guards=dict(duplicate=False, stale=False),
                  next="Run the frozen 20 requests through the live loopback API.")
    publish(args, report)


def verify_preview(preview, xy, planner, frozen):
    # The old checker deliberately retains historical base/fallback vocabulary.
    _require(preview.get("candidate_pool") in {"base", "fallback", "neural", "branches", "neural_checked"}, "Unknown candidate pool")
    historical_preview(dict(preview, candidate_pool="base"), xy)
    _require(preview.get("planner") == planner, "Planner changed after request")
    expected = dict(scene_sha256=frozen["geometry"]["scene_sha256"],
                    study_audit_sha256=frozen["system_source"]["audit_sha256"], simulation_only=True,
                    substeps=64, required_retention_s=1.)
    if planner == "fast":
        expected.update(frozen["fast_source"])
    _require(all(preview["provenance"].get(key) == value for key, value in expected.items()),
             "Preview differs from the frozen audited planner")
    timing = preview["timing"]
    _require(_finite(timing["planning_wall_s"]) and timing["planning_wall_s"] >= 0, "Missing backend planning time")
    if planner == "fast":
        _require(type(timing.get("worker_reused")) is bool, "Fast planner must expose worker reuse")


def verify_decline(preview, xy, planner, frozen):
    _require(preview["bucket_xy_m"] == xy and preview["status"] in {"blocked", "unresolved"}
             and preview.get("configuration") is None and preview.get("planner") == planner,
             "A declined setup must preserve coordinates/method and expose no arm command")
    expected = dict(scene_sha256=frozen["geometry"]["scene_sha256"], simulation_only=True, substeps=64)
    if planner == "fast":
        expected.update(frozen["fast_source"])
    _require(all(preview["provenance"].get(key) == value for key, value in expected.items()),
             "Declined setup lost its frozen planner provenance")
    elapsed = preview["timing"]["planning_wall_s"]
    _require(_finite(elapsed) and elapsed >= 0, "Missing declined setup planning time")


def poll(client, row, active, final, timeout_s, interval):
    deadline = time.monotonic() + timeout_s
    while True:
        status = client.request("/" + quote(row["plan_id"], safe=""))
        _require((status["id"], status["bucket_xy_m"], status["planner"]) ==
                 (row["plan_id"], row["bucket_xy_m"], row["planner"]), "Job identity changed")
        if status["state"] in final:
            return status
        _require(status["state"] in active, f"Unexpected job state {status['state']}")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Job remains active; resume will only poll its existing ID")
        time.sleep(min(interval, remaining))


def check_row(client, row, frozen, persist, *, timeout_s=900, interval=.5, stale_id=None, before_drop=None):
    """At most one POST /drop, even if interrupted immediately before/after it."""
    if row["state"] in FINAL:
        return
    if row["state"] == "plan_intent":
        raise RuntimeError("Plan submission outcome is unknown; do not automatically create another job")
    if row["state"] == "pending":
        row.update(state="plan_intent", plan_started_unix=time.time())
        persist()
        created = client.request("/plan", body={"bucket_xy_m": row["bucket_xy_m"], "planner": row["planner"]}, expected=202)
        _require(isinstance(created.get("id"), str) and created["id"], "Missing new job ID")
        row.update(state="planning", plan_id=created["id"])
        persist()
    if row["state"] == "planning":
        status = poll(client, row, {"queued", "planning"}, {"ready", "no_setup", "failed", "expired"}, timeout_s, interval)
        row.update(state=status["state"], planning_s=time.time() - row["plan_started_unix"],
                   message=status.get("message"))
        if "preview" in status:
            row["preview"] = status["preview"]
        if row["state"] != "ready":
            if row["state"] == "no_setup":
                verify_decline(row["preview"], row["bucket_xy_m"], row["planner"], frozen)
            persist()
            return
        verify_preview(row["preview"], row["bucket_xy_m"], row["planner"], frozen)
        persist()
    identity = quote(row["plan_id"], safe="")
    if row["state"] == "ready":
        verify_preview(row["preview"], row["bucket_xy_m"], row["planner"], frozen)
        if before_drop is not None:
            before_drop(row, persist)
        if stale_id is not None:
            client.request("/" + quote(stale_id, safe="") + "/drop", body={}, expected=404)
            after = client.request("/" + identity)
            _require(after["state"] == "ready" and after["preview"] == row["preview"], "Stale drop changed current preview")
            row["stale_rejected"] = True
        row.update(state="drop_intent", drop_sent=True, drop_started_unix=time.time())
        persist()
        client.request("/" + identity + "/drop", body={}, expected=202)
        row["state"] = "releasing"
        persist()
    terminal = poll(client, row, {"queued", "running"}, {"completed", "failed", "expired", "ready"}, timeout_s, interval)
    if terminal["state"] == "ready":
        # Intent was durable, but server has no accepted drop. Never retry it.
        row.update(state="unreleased_uncertain", message="Interrupted drop intent was not accepted; no automatic retry.")
        persist()
        return
    row.update(state=terminal["state"], drop_s=time.time() - row["drop_started_unix"], message=terminal.get("message"))
    if row["state"] == "completed":
        result = terminal["result"]
        verify_result(result, row["preview"], row["bucket_xy_m"])
        row.update(result=result, outcome=result["outcome"])
        client.request("/" + identity + "/drop", body={}, expected=409)
        after = client.request("/" + identity)
        _require(after["state"] == "completed" and after["result"] == result, "Duplicate drop changed measured result")
        row["duplicate_rejected"] = True
    persist()


def load_frozen(args):
    report = read(args.output / "report.json")
    _require(report["positions_sha256"] == checksum(args.output / "positions.json"), "Frozen positions changed")
    frozen = read(args.output / "positions.json")
    _require(frozen["version"] == VERSION and frozen["base_url"] == args.base_url
             and frozen["evidence_root"] == str(args.evidence_root.resolve()), "Acceptance target changed")
    _require(_digest(frozen["placements"]) == frozen["positions_sha256"]
             and _digest(frozen["excluded_coordinates_m"]) == frozen["excluded_coordinates_sha256"], "Frozen coordinate inventory changed")
    _require(sample_positions(frozen["geometry"], frozen["grid"], frozen["excluded_coordinates_m"], frozen["bounds"], frozen["seed"])
             == frozen["placements"], "Frozen sampling cannot be reproduced")
    expected = ordered_rows(frozen["placements"])
    _require(len(report["results"]) == len(expected) and all(
        all(row.get(key) == original[key] for key in ("case", "region", "bucket_xy_m", "planner"))
        for row, original in zip(report["results"], expected, strict=True)), "Attempt order or coordinates changed")
    return frozen, report


def audit_native(row, root, frozen):
    """CPU-only independent scalar rescoring plus immutable request/trace binding."""
    import numpy as np
    from .benchmark_verify import Checks, contained
    from .cell1050_rebound_score import score_recorded_outcome
    from .interactive_physics import _trace
    from .intergrid_matched_route_study import outcome
    from .intergrid_route_labels import annotate_routes
    from .table_assisted_score_oracle import score_trace
    from .table_training_verify import check_native_score

    result, preview = row["result"], row["preview"]
    request_hash = preview["provenance"]["request_sha256"]
    matches = [path for path in root.glob("**/request.json") if checksum(path) == request_hash]
    _require(len(matches) == 1, "Release request must bind exactly one local immutable evidence directory")
    directory = matches[0].parent
    request = read(matches[0])
    measurement = read(directory / "measurement.json")
    native = measurement["row"]
    _require(measurement["request_sha256"] == request_hash and
             request["bucket_xy_m"] == native["bucket_xy_m"] == row["bucket_xy_m"] and
             request["id"] == native["id"] == result["provenance"]["native_release_id"], "Release identity changed")
    _require(request["selected_q10"] == preview["configuration"]["q10"] and
             request["selected_q14"] == preview["configuration"]["q14"] and
             native["contact_protocol"]["substeps"] == 64 and native["native_release"] is True,
             "Joint command or physics resolution changed")
    native_dir = directory / "native-heading"
    trace_path = contained(native_dir, native["native_cache_file"])
    _require(checksum(trace_path) == native["native_cache_sha256"] == result["provenance"]["native_cache_sha256"],
             "Native trace checksum changed")
    with np.load(trace_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    column = native["native_cache_row"]
    _require(str(arrays["record_ids"][column]) == native["id"] and
             np.allclose(np.diff(arrays["time_s"]), 1 / (240 * 64), atol=1e-12, rtol=0), "Native64 trace clock changed")
    trace = {key: value[:, column] if value.ndim >= 2 and key != "time_s" else value
             for key, value in arrays.items() if key != "record_ids"}
    score = score_trace(frozen["scene"], native["bucket_xy_m"], native["board_normal"], frozen["calibration"]["score_config"], trace)
    checks = Checks()
    check_native_score(native, score, checks)
    reconstructed = copy.deepcopy(native)
    reconstructed["experimental_route"] = score_recorded_outcome(native, arrays, world=column)
    annotate_routes([reconstructed], native_dir)
    _require(reconstructed["experimental_route"] == native["experimental_route"] and
             outcome(reconstructed) == result["outcome"] and _trace(native_dir, native) == result["trajectory"],
             "Public score or displayed flight differs from the independently rescored trace")
    return dict(request_sha256=request_hash, native_cache_sha256=checksum(trace_path),
                measurement_sha256=checksum(directory / "measurement.json"),
                native_release_id=native["id"], independently_rescored=True)


def audit(args):
    frozen, report = load_frozen(args)
    _require(all(row["state"] in FINAL for row in report["results"]), "All twenty attempts must terminate before final audit")
    ids, traces = set(), []
    for row in report["results"]:
        if "plan_id" in row:
            _require(row["plan_id"] not in ids, "Job IDs must be unique")
            ids.add(row["plan_id"])
        if row["state"] == "completed":
            verify_preview(row["preview"], row["bucket_xy_m"], row["planner"], frozen)
            verify_result(row["result"], row["preview"], row["bucket_xy_m"])
            traces.append(audit_native(row, args.evidence_root, frozen))
        elif row["state"] == "no_setup":
            verify_decline(row["preview"], row["bucket_xy_m"], row["planner"], frozen)
    _require(len({r["native_release_id"] for r in traces}) == len(traces), "A native release was reused")
    _require(any(row.get("duplicate_rejected") for row in report["results"])
             and any(row.get("stale_rejected") for row in report["results"]), "Duplicate and stale guards need observed checks")
    save(args.output / "audit.json", dict(version=VERSION, status="passed", positions_sha256=report["positions_sha256"],
                                          results_sha256=_digest(report["results"]), native=traces))
    report.update(status="completed", audit_status="passed", audit_sha256=checksum(args.output / "audit.json"),
                  next="Inspect planning time and first-drop captures; this ten-position batch is not all-table reliability.")
    publish(args, report)
    return report


def run(args):
    frozen, report = load_frozen(args)
    if report["status"] == "completed":
        return audit(args)
    if "error" in report:
        report.setdefault("interruptions", []).append(dict(
            error=report.pop("error"), observed_at_unix=report.get("updated_at_unix")))
    report.update(status="running", audit_status="pending", next="Plan and drop each frozen placement once per method.")
    client = Client(args.base_url)
    def persist():
        report["guards"] = dict(duplicate=any(row.get("duplicate_rejected") for row in report["results"]),
                                stale=any(row.get("stale_rejected") for row in report["results"]))
        publish(args, report)
    def before_drop(row, persist):
        if args.gpu_wait_s <= 0:
            return
        from .gpu_slot import foreign_processes
        started = time.monotonic()
        previous_wait = row.get("gpu_preflight_wait_s", 0.)
        while foreign_processes():
            row.update(message="Checked shot ready; waiting for the 4090 before requesting its drop.",
                       gpu_preflight_wait_s=previous_wait + time.monotonic() - started)
            persist()
            if time.monotonic() - started >= args.gpu_wait_s:
                raise TimeoutError("4090 remains occupied; saved ready plan can resume without replanning")
            time.sleep(max(.001, min(15, args.gpu_wait_s - (time.monotonic() - started))))
        row["gpu_preflight_wait_s"] = previous_wait + time.monotonic() - started
    try:
        for index, row in enumerate(report["results"]):
            previous = next((old["plan_id"] for old in reversed(report["results"][:index]) if "plan_id" in old), None)
            check_row(client, row, frozen, persist, timeout_s=args.timeout_s, interval=args.poll_s,
                      stale_id=previous, before_drop=before_drop)
            print(json.dumps({key: row[key] for key in ("case", "planner", "state", "outcome") if key in row}), flush=True)
        return audit(args)
    except Exception as exc:
        report.update(status="failed", audit_status="pending", error=f"{type(exc).__name__}: {exc}",
                      next="Resolve the saved interruption, then resume existing job IDs; never repeat uncertain drops.")
        persist()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "audit"))
    parser.add_argument("--base-url", default="http://127.0.0.1:18781")
    for name in ("system-study", "system-audit", "fast-study", "fast-audit", "evidence-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--history-evidence", type=Path, action="append", default=[])
    parser.add_argument("--display-report", type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--timeout-s", type=float, default=900.)
    parser.add_argument("--poll-s", type=float, default=.5)
    parser.add_argument("--gpu-wait-s", type=float, default=0.,
                        help="Optional read-only 4090 occupancy wait before drop; native worker still acquires its lease")
    args = parser.parse_args()
    if args.seed < 0 or not all(_finite(v) and v > 0 for v in (args.timeout_s, args.poll_s)) or not _finite(args.gpu_wait_s) or args.gpu_wait_s < 0:
        parser.error("Use nonnegative seed and positive finite polling limits")
    if args.command == "freeze":
        freeze(args)
    else:
        with (args.output / ".runner.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = run(args) if args.command == "run" else audit(args)
            print(json.dumps(result["comparison"], allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
