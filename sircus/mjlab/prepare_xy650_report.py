"""Add read-only XY650 translated sweep evidence to a separate display report.

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

KEY = "xy650_study"
VERSION = "sircus-xy650-shift-study-v1"
AUDIT_VERSION = "sircus-xy650-independent-audit-v2"
TERMINAL = {"completed", "failed", "cancelled", "blocked_geometry", "blocked_storage", "blocked_resources"}
GEOMETRY = ("id", "role", "group_id", "phase", "stage", "substeps", "values", "gap_m", "board_height_m", "tilt_deg", "release_center_height_m",
    "ball_radius_m", "distance_m", "board_xy_m", "bucket_xy_m", "source_id", "source_record_id", "source_record_sha256", "geometry_sha256", "translation_m", "query_fingerprint", "source_query_fingerprint", "execution_fingerprint", "query_kind")
STATS = ("attempts", "native_releases", "captures", "invalid", "clean", "rim_assisted", "assisted_or_marginal", "whole_entries", "max_retention_s", "geometry_rejections")
QUALIFICATION = ("passed", "qualified", "status", "reason", "controls_passed", "initial_controls", "final_controls", "clean64", "clean128", "clean_both", "physics_valid", "oracle_agreement", "max_drift_m")


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


def first_flight(raw):
    if not isinstance(raw, dict):
        return None
    value = select(raw, ("version", "reason", "cutoff_reason", "crossing_plane"))
    if type(raw.get("available")) is bool:
        value["available"] = raw["available"]
    for name in ("start_index", "end_index", "start_time_s", "end_time_s", "mouth_clearance_m", "closest_opening_distance_m", "sample_count"):
        v = raw.get(name)
        if name in raw and (v is None or type(v) in (int, float) and math.isfinite(v)):
            value[name] = v
    for name in ("contact_channels", "cutoff_contact_channels"):
        channels = raw.get(name)
        if isinstance(channels, list):
            value[name] = [v for v in channels if v in ("board", "bucket", "table", "floor", "robot", "unattributed")]
    crossing = raw.get("first_crossing")
    if "first_crossing" in raw and crossing is None:
        value["first_crossing"] = None
    elif isinstance(crossing, dict):
        observed = {}
        for name in ("time_s", "interval_index", "fraction"):
            v = crossing.get(name)
            if type(v) in (int, float) and math.isfinite(v):
                observed[name] = v
        xy = crossing.get("xy_m")
        if isinstance(xy, list) and len(xy) == 2 and all(type(v) in (int, float) and math.isfinite(v) for v in xy):
            observed["xy_m"] = copy.deepcopy(xy)
        value["first_crossing"] = observed
    return value


def group(raw):
    result = select(raw, GEOMETRY + ("eligible", "route_success", "route_quality", "whole_ball_entry", "minimum_mouth_clearance_mm", "max_retention_s", "invalid", "invalid_physics", "rim_band_contact", "external_contact_steps", "qualified"))
    result["by_substeps"] = {str(k): select(v, STATS) for k, v in raw.get("by_substeps", {}).items() if str(k) in ("64", "128")}
    result["statistics"] = select(raw.get("statistics", raw.get("stats", {})), STATS)
    result["sensitivity"] = select(raw.get("sensitivity", {}), ("paired_attempts", "capture_agreement", "nominal_position_delta_350ms_max_m"))
    result["qualification"] = select(raw.get("qualification", {}), QUALIFICATION)
    diagnostic = first_flight(raw.get("first_flight"))
    if diagnostic is not None:
        result["first_flight"] = diagnostic
    return result


def compact(raw):
    if not isinstance(raw, dict):
        return None
    value = select(raw, ("version", "status", "phase", "contract_fingerprint", "scope", "error", "reason", "elapsed_s"))
    value["queue"] = select(raw.get("queue", {}), ("reason", "wait_s"))
    settings = raw.get("settings", {})
    value["settings"] = select(settings, ("version", "profile", "profile_label", "heights_m", "tilts_deg", "gaps_m", "substeps", "max_requests", "coarse_requests", "translated_grid_requests", "translated_reference_requests", "unshifted_reference_requests", "first_flight_version", "refinement_max_requests", "verification_limit", "verification_sources", "refinement_seed_limit", "refinement_seeds", "repeats_per_group", "repeats_per_resolution", "refinement_offsets", "selection_rule", "selection", "verification_scope", "minimum_clearance_m", "max_drift_350ms_m", "scope", "no_noise", "no_ppo"))
    value["settings"]["translations"] = select(settings.get("translations", {}), ("dx_m", "dy_m", "rule"))
    value["settings"]["grid"] = select(settings.get("grid", {}), ("heights_m", "tilts_deg", "gaps_m", "requests"))
    reference = settings.get("grid", {}).get("reference")
    if isinstance(reference, dict):
        value["settings"]["grid"]["reference"] = select(reference, GEOMETRY)
    elif isinstance(reference, list) and len(reference) == 3 and all(type(v) in (int, float) and math.isfinite(v) for v in reference):
        value["settings"]["grid"]["reference"] = copy.deepcopy(reference)
    value["settings"]["target"] = select(settings.get("target", {}), GEOMETRY)
    value["counts"] = select(raw.get("counts", {}), STATS + ("requests", "phase_counts", "role_counts", "substep_counts"))
    phases = raw.get("by_phase", raw.get("counts", {}).get("by_phase", {}))
    value["by_phase"] = {str(k):select(v, STATS) for k,v in phases.items()} if isinstance(phases, dict) else {}
    value["calibration"] = select(raw.get("calibration", {}), ("score_config_sha256", "calibration_sha256", "verification_sha256"))
    value["control_gates"] = {str(k):select(v, QUALIFICATION + ("positive_control", "negative_control")) for k,v in raw.get("control_gates", {}).items()} if isinstance(raw.get("control_gates", {}), dict) else {}
    value["gate"] = select(raw.get("gate", {}), QUALIFICATION + ("qualified_ids", "qualification_count", "qualified_settings", "controls_pre_passed", "controls_post_passed", "more_ppo_started", "search_opened"))
    for name in ("groups", "controls", "top_settings", "verification"):
        value[name] = [group(row) for row in raw.get(name, []) if isinstance(row, dict)]
    value["qualified_ids"] = [v for v in raw.get("qualified_ids", []) if isinstance(v, str)]
    value["source_sha256"] = hashes(raw.get("source_sha256"))
    value["evidence_sha256"] = hashes(raw.get("evidence_sha256"))
    value["trace_examples"] = []
    for row in raw.get("trace_examples", [])[:32]:
        if not isinstance(row, dict):
            continue
        points = row.get("position_m")
        if not isinstance(points, list) or not 2 <= len(points) <= 5000 or not all(
            isinstance(p, list) and len(p) == 3 and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in p) for p in points):
            continue
        trace = select(row, GEOMETRY + ("record_id", "substeps", "route_success", "route_quality", "whole_ball_entry", "max_retention_s", "invalid",
            "external_contact_steps", "scalar_score_audit_passed", "invalid_physics", "rim_band_contact", "minimum_mouth_clearance_mm", "sample_dt_s", "selection_scope", "cache_file", "cache_row", "cache_sha256"))
        if "cache_file" in trace and (not isinstance(trace["cache_file"], str) or Path(trace["cache_file"]).is_absolute() or ".." in Path(trace["cache_file"]).parts):
            trace.pop("cache_file")
        if "cache_sha256" in trace and (not isinstance(trace["cache_sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", trace["cache_sha256"])):
            trace.pop("cache_sha256")
        diagnostic = first_flight(row.get("first_flight"))
        if diagnostic is not None:
            trace["first_flight"] = diagnostic
        trace["position_m"] = copy.deepcopy(points)
        value["trace_examples"].append(trace)
    return value


def merge_display(base, value):
    if not isinstance(base, dict):
        raise ValueError("Existing display must be a dictionary")
    result = copy.deepcopy(base)
    result[KEY] = copy.deepcopy(value)
    return result


def prepare(base_display, study, output, audit_path=None):
    base_display, study, output = map(Path, (base_display, study, output))
    audit_path = Path(audit_path) if audit_path is not None else study / "audit-xy650.json"
    if output.resolve() in (base_display.resolve(), audit_path.resolve()) or output.resolve().is_relative_to(study.resolve()):
        raise ValueError("Write a separate derived report outside original evidence")
    base = optional_json(base_display)
    if not isinstance(base, dict):
        raise ValueError("A valid existing base display report is required")
    try:
        report_bytes = (study / "report.json").read_bytes()
        raw = json.loads(report_bytes)
    except (OSError, ValueError):
        report_bytes, raw = None, None
    value = compact(raw)
    if value is not None:
        try:
            audit_bytes = audit_path.read_bytes()
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
    result = merge_display(base, value)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(payload)
    temporary.replace(output)
    return result


def watch(base_display, study, output, seconds=0, audit_path=None):
    if not math.isfinite(seconds) or not 0 <= seconds <= 30000:
        raise ValueError("Bounded watch must be 0–30,000 seconds")
    started = time.monotonic()
    while True:
        result = prepare(base_display, study, output, audit_path=audit_path)
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
    parser.add_argument("--audit", type=Path, help="Explicit independent audit file; defaults to STUDY/audit-xy650.json")
    args = parser.parse_args()
    watch(args.base_display, args.study, args.output, args.watch_seconds, audit_path=args.audit)
