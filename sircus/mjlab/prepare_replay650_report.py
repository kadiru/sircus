"""Add read-only calibrated650 replay evidence to a separate display report.

No physics, scorer, inference, service or original evidence is changed here.
"""
import argparse
import copy
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import time

KEY = "calibrated650_replay"
VERSION = "sircus-calibrated-650-replay-v1"
AUDIT_VERSION = "sircus-calibrated-650-independent-audit-v2"
TERMINAL = {"completed", "failed", "cancelled", "blocked_geometry", "blocked_storage", "blocked_resources"}
GEOMETRY = ("id", "role", "group_id", "gap_m", "board_height_m", "tilt_deg", "release_center_height_m",
    "ball_radius_m", "distance_m", "board_xy_m", "bucket_xy_m", "scope", "source_record_id", "source_record_sha256", "source_contract_fingerprint")
STATS = ("attempts", "native_releases", "captures", "invalid", "clean", "whole_entries", "max_retention_s")


def select(value, keys):
    return {k:copy.deepcopy(value[k]) for k in keys if isinstance(value, dict) and k in value}


def optional_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def hashes(value):
    return {k:v for k,v in value.items() if isinstance(k, str) and not Path(k).is_absolute() and ".." not in Path(k).parts and
        isinstance(v, str) and re.fullmatch(r"[a-f0-9]{64}", v)} if isinstance(value, dict) else {}


def full_audit_verified(raw, audit):
    """Check full audit structure; prepare also binds its exact report/manifest bytes."""
    checks, count = audit.get("checks"), audit.get("checks_count")
    fingerprint = raw.get("contract_fingerprint")
    return bool(raw.get("version") == VERSION and raw.get("status") == "completed" and raw.get("phase") == "completed" and
        isinstance(fingerprint, str) and fingerprint and audit.get("version") == AUDIT_VERSION and audit.get("status") == "passed" and
        audit.get("contract_fingerprint") == fingerprint and not audit.get("failures") and
        isinstance(checks, list) and checks and isinstance(count, int) and not isinstance(count, bool) and count > 0 and
        count == len(checks) and all(isinstance(c, dict) and c.get("passed") is True for c in checks))


def compact(raw):
    if not isinstance(raw, dict):
        return None
    value = select(raw, ("version", "status", "phase", "contract_fingerprint", "scope", "error", "reason", "max_requests", "elapsed_s"))
    value["queue"] = select(raw.get("queue", {}), ("reason", "wait_s"))
    settings = raw.get("settings", {})
    value["settings"] = select(settings, ("version", "substeps", "gaps_m", "scope", "repeats_per_group", "max_requests", "window_scope"))
    value["settings"]["candidate"] = select(settings.get("candidate", {}), GEOMETRY)
    value["settings"]["controls"] = {}
    controls = settings.get("controls")
    if isinstance(controls, (dict, list)):
        entries = controls.items() if isinstance(controls, dict) else ((str(i), row) for i, row in enumerate(controls))
        normalized = {}
        for name, control in entries:
            entry = select(control, GEOMETRY)
            if isinstance(control, dict) and isinstance(control.get("provenance"), dict):
                entry["provenance"] = select(control["provenance"], ("kind", "source_record_id", "source_contract_fingerprint", "source_sha256", "scope"))
            normalized[str(name)] = entry
        value["settings"]["controls"] = list(normalized.values()) if isinstance(controls, list) else normalized
    value["counts"] = select(raw.get("counts", {}), STATS + ("requests", "phase_counts", "role_counts", "substep_counts"))
    value["calibration"] = select(raw.get("calibration", {}), ("score_config_sha256", "calibration_sha256", "verification_sha256"))
    value["gate"] = select(raw.get("gate", {}), ("status", "positive_control", "negative_control", "candidate", "physics_valid", "oracle_agreement", "max_drift_m", "window_opened", "reason"))
    value["groups"] = []
    for row in raw.get("groups", []):
        if not isinstance(row, dict):
            continue
        group = select(row, GEOMETRY + ("eligible",))
        group["by_substeps"] = {str(k):select(v, STATS) for k,v in row.get("by_substeps", {}).items() if str(k) in ("64", "128")}
        group["sensitivity"] = select(row.get("sensitivity", {}), ("paired_attempts", "capture_agreement", "nominal_position_delta_350ms_max_m"))
        value["groups"].append(group)
    value["source_sha256"] = hashes(raw.get("source_sha256"))
    value["evidence_sha256"] = hashes(raw.get("evidence_sha256"))
    value["trace_examples"] = []
    for row in raw.get("trace_examples", [])[:24]:
        if not isinstance(row, dict):
            continue
        points = row.get("position_m")
        if not isinstance(points, list) or not 2 <= len(points) <= 5000 or not all(
            isinstance(p, list) and len(p) == 3 and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in p) for p in points):
            continue
        trace = select(row, GEOMETRY + ("record_id", "substeps", "route_success", "whole_ball_entry", "max_retention_s", "invalid",
            "external_contact_steps", "oracle_agreement", "scalar_score_audit_passed", "invalid_physics", "sample_dt_s", "selection_scope", "trace_file", "cache_file", "cache_row", "source_sha256", "cache_sha256"))
        for name in ("trace_file", "cache_file"):
            if name in trace and (not isinstance(trace[name], str) or Path(trace[name]).is_absolute() or ".." in Path(trace[name]).parts):
                trace.pop(name)
        for name in ("source_sha256", "cache_sha256"):
            if name in trace and (not isinstance(trace[name], str) or not re.fullmatch(r"[a-f0-9]{64}", trace[name])):
                trace.pop(name)
        trace["position_m"] = copy.deepcopy(points)
        trace["selection_scope"] = row.get("selection_scope", row.get("scope", "Actual saved native replay; illustrative record, not fresh-noise evidence"))
        value["trace_examples"].append(trace)
    return value


def prepare(base_display, study, output):
    base_display, study, output = map(Path, (base_display, study, output))
    if output.resolve() == base_display.resolve() or output.resolve().is_relative_to(study.resolve()):
        raise ValueError("Write a separate derived report outside original evidence")
    base = optional_json(base_display)
    if not isinstance(base, dict):
        raise ValueError("A valid existing base display report is required")
    result = copy.deepcopy(base)
    try:
        report_bytes = (study / "report.json").read_bytes()
        raw = json.loads(report_bytes)
    except (OSError, ValueError):
        report_bytes, raw = None, None
    value = compact(raw)
    result[KEY] = value
    if value is not None:
        try:
            audit_bytes = (study / "audit-replay650.json").read_bytes()
            audit = json.loads(audit_bytes)
        except (OSError, ValueError):
            audit_bytes, audit = None, {}
        audit = audit if isinstance(audit, dict) else {}
        value["audit"] = select(audit, ("version", "status", "contract_fingerprint", "checks_count", "reason", "limitations", "report_sha256", "evidence_manifest_sha256"))
        try:
            manifest_bytes = (study / "evidence-manifest.json").read_bytes()
        except OSError:
            manifest_bytes = None
        files = {"report_sha256":report_bytes, "evidence_manifest_sha256":manifest_bytes}
        bound = all(data is not None and isinstance(audit.get(name), str) and re.fullmatch(r"[a-f0-9]{64}", audit[name]) and
            hashlib.sha256(data).hexdigest() == audit[name] for name, data in files.items())
        value["audit"]["bound_files_verified"] = bound
        value["audit"]["full_checks_verified"] = full_audit_verified(raw, audit) and bound
        for name, data in files.items():
            if data is not None:
                value["audit"]["observed_"+name] = hashlib.sha256(data).hexdigest()
        if audit_bytes is not None:
            value["audit"]["source_sha256"] = hashlib.sha256(audit_bytes).hexdigest()
        if audit.get("status") == "passed" and not value["audit"]["full_checks_verified"]:
            value["audit"]["verification_reason"] = "Audited report or evidence-manifest bytes are missing or mismatched" if not bound else "Completed exact-version evidence and a nonempty matching strictly passed full audit are required"
        value["display_updated_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(payload)
    temporary.replace(output)
    return result


def watch(base_display, study, output, seconds=0):
    if not math.isfinite(seconds) or not 0 <= seconds <= 30000:
        raise ValueError("Bounded watch must be 0–30,000 seconds")
    started = time.monotonic()
    while True:
        result = prepare(base_display, study, output)
        value = result[KEY]
        if value and value.get("status") in TERMINAL:
            return result
        remaining = seconds - (time.monotonic() - started)
        if remaining <= 0:
            return result
        time.sleep(min(5, remaining))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-display", "study", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--watch-seconds", type=float, default=0)
    args = parser.parse_args()
    watch(args.base_display, args.study, args.output, args.watch_seconds)
