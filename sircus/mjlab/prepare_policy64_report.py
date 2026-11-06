"""Add read-only fixed64 learned-policy evidence to a separate display report.

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

KEY = "policy64_study"
VERSION = "sircus-policy64-bc-ppo-v1"
AUDIT_VERSION = "sircus-policy64-independent-audit-v1"
TERMINAL = {"completed", "failed", "cancelled", "blocked_geometry", "blocked_storage", "blocked_resources"}
GEOMETRY = ("id", "record_id", "group_id", "context_id", "method", "phase", "substeps", "values", "gap_m", "board_height_m", "tilt_deg", "release_center_height_m", "ball_radius_m", "distance_m", "board_bucket_distance_m", "board_xy_m", "bucket_xy_m", "board_center_m", "board_normal", "source_record_id", "source_record_sha256", "query_fingerprint", "execution_fingerprint", "artifact_sha256")
STATS = ("attempts", "native_releases", "captures", "invalid", "clean", "rim_assisted", "whole_entries", "max_retention_s", "geometry_rejections", "mean_reward")


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


def safe_path(value):
    return value if isinstance(value, str) and value and not Path(value).is_absolute() and ".." not in Path(value).parts and ":" not in value and "\\" not in value else None


def artifact(raw):
    result = select(raw, ("samples", "sha256", "weight_sha256"))
    for key in ("sha256", "weight_sha256"):
        if key in result and (not isinstance(result[key], str) or not re.fullmatch(r"[a-f0-9]{64}", result[key])):
            result.pop(key)
    path = safe_path(raw.get("path")) if isinstance(raw, dict) else None
    if path is not None:
        result["path"] = path
    return result


def compact(raw):
    if not isinstance(raw, dict):
        return None
    value = select(raw, ("version", "status", "phase", "contract_fingerprint", "scene_fingerprint", "scope", "error", "reason", "train_requests", "updates", "elapsed_s"))
    value["protocol"] = select(raw.get("protocol", {}), ("version", "train_requests", "updates", "validation_layouts", "test_layouts", "maximum_fresh_requests", "milestones", "substeps", "noise"))
    value["bootstrap_summary"] = select(raw.get("bootstrap_summary", {}), ("requests", "native_releases", "captures", "clean", "layouts", "successful_layouts", "teacher_count", "labels_scope"))
    value["counts"] = select(raw.get("counts", {}), STATS + ("phase_counts", "split_counts", "method_counts"))
    value["curves"] = [dict(samples=row.get("samples"),validation=select(row.get("validation", {}), STATS),artifact=artifact(row.get("artifact", {}))) for row in raw.get("curves", []) if isinstance(row, dict)]
    value["selected"] = artifact(raw.get("selected", {}))
    value["artifacts"] = {key:artifact(row) for key,row in raw.get("artifacts", {}).items() if key in ("initial", "bc", "selected") and isinstance(row, dict)}
    value["test_comparison"] = [dict(method=row.get("method"),counts=select(row.get("counts", {}), STATS),samples=row.get("samples"),artifact_sha256=row.get("artifact_sha256")) for row in raw.get("test_comparison", []) if isinstance(row, dict) and row.get("method") in ("prior", "nearest", "bc", "ppo")]
    value["test_positions"] = [select(row, ("context_id", "bucket_xy_m", "method", "values", "board_bucket_distance_m", "capture", "clean", "native_release", "invalid", "invalid_physics")) for row in raw.get("test_positions", []) if isinstance(row, dict) and row.get("method") in ("prior", "nearest", "bc", "ppo")]
    value["control_gates"] = {key:select(row, ("status", "positive_control", "negative_control", "physics_valid", "oracle_agreement")) for key,row in raw.get("control_gates", {}).items() if key in ("pre", "post") and isinstance(row, dict)}
    value["gate"] = select(raw.get("gate", {}), ("status", "reason", "complete", "substeps", "scope", "test_complete", "controls_passed", "no_test_flight_search"))
    value["source_sha256"] = hashes(raw.get("source_sha256"))
    value["evidence_sha256"] = hashes(raw.get("evidence_sha256"))
    value["trace_examples"] = []
    for row in raw.get("trace_examples", [])[:32]:
        if not isinstance(row, dict):
            continue
        points = row.get("position_m", row.get("trajectory_m"))
        if not isinstance(points, list) or not 2 <= len(points) <= 5000 or not all(isinstance(p, list) and len(p) == 3 and all(type(v) in (float, int) and math.isfinite(v) for v in p) for p in points):
            continue
        trace = select(row, GEOMETRY + ("route_success", "route_quality", "capture", "clean", "whole_ball_entry", "max_retention_s", "invalid", "invalid_physics", "rim_band_contact", "scalar_score_audit_passed", "sample_dt_s", "selection_scope", "native_cache_sha256", "native_cache_row"))
        cache = safe_path(row.get("native_cache_file"))
        if cache is not None:
            trace["native_cache_file"] = cache
        if "native_cache_sha256" in trace and (not isinstance(trace["native_cache_sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", trace["native_cache_sha256"])):
            trace.pop("native_cache_sha256")
        if "sample_dt_s" not in trace and type(row.get("trajectory_dt_s")) in (float, int) and math.isfinite(row["trajectory_dt_s"]):
            trace["sample_dt_s"] = row["trajectory_dt_s"]
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
    audit_path = Path(audit_path) if audit_path is not None else study.with_name(study.name+"-audit.json")
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
    parser.add_argument("--audit", type=Path, help="Explicit independent audit file; defaults to the external STUDY_NAME-audit.json")
    args = parser.parse_args()
    watch(args.base_display, args.study, args.output, args.watch_seconds, audit_path=args.audit)
