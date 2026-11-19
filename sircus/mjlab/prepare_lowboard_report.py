"""Separate read-only lower-board development evidence for the existing viewer.

This derives a new display document. It does not mutate study evidence, choose
or activate a policy, start dynamics, or manage a running service.
"""
import argparse
import copy
import datetime
import hashlib
import json
import math
from pathlib import Path
import time

from .prepare_distance_report import compact as distance_compact
from .prepare_expanded_report import optional_json, select
from .prepare_neighborhood_report import TERMINAL

KEY = "lowboard_evaluation"
VERSION = "sircus-low-board-far-development-v1"
SETTINGS = ("board_heights_m", "tilts_deg", "height_grid_m", "tilt_grid_deg", "headroom_m", "headrooms_m",
    "ball_radius_m", "release_ball_center_ceiling_m", "release_center_ceiling_m", "ceiling_source", "ceiling_provenance",
    "gap_formula", "search", "gates", "parent_contract_fingerprint", "controls", "repeatability")
EXTRA_CONTEXT = ("stage", "headroom_m", "release_ball_center_ceiling_m", "release_center_ceiling_m", "proposed_anchor_gap_m")


def full_audit_verified(raw, audit):
    """Require the full strictly passed audit, never a truthy summary flag."""
    checks, count = audit.get("checks"), audit.get("checks_count")
    fingerprint = raw.get("contract_fingerprint")
    return bool(raw.get("version") == VERSION and raw.get("status") == "completed" and raw.get("phase") == "completed" and
        isinstance(fingerprint, str) and fingerprint and audit.get("status") == "passed" and
        audit.get("contract_fingerprint") == fingerprint and not audit.get("failures") and
        isinstance(checks, list) and checks and isinstance(count, int) and not isinstance(count, bool) and count > 0 and
        count == len(checks) and all(isinstance(c, dict) and c.get("passed") is True for c in checks))


def compact(raw):
    value = distance_compact(raw)
    if value is None:
        return None
    value["settings"].update(select(raw.get("settings", {}), SETTINGS))
    value["settings"]["grid"] = select(raw.get("settings", {}).get("grid", {}), ("board_heights_m", "tilts_deg", "release_headroom_m"))
    value["settings"]["reach_limit"] = select(raw.get("settings", {}).get("reach_limit", {}), ("ball_center_height_m", "board_height_plus_gap_m", "formula", "scope"))
    value["settings"]["control_target"] = select(raw.get("settings", {}).get("control_target", {}), ("id", "distance_id", "ray_id", "distance_m", "board_xy_m", "bucket_xy_m", "scope"))
    value["active_policy"] = "Read-only lower-board development search; parent studies and active solver routes remain unchanged"
    for key in ("contexts", "selected_recipes"):
        originals = [r for r in raw.get(key, []) if isinstance(r, dict)]
        for result, original in zip(value[key], originals):
            result.update(select(original, EXTRA_CONTEXT))
    traces = {r.get("record_id"): r for r in raw.get("trace_examples", []) if isinstance(r, dict)}
    for trace in value["trace_examples"]:
        original = traces.get(trace.get("record_id"), {})
        trace.update(select(original, ("source_context_id", "whole_ball_entry", "max_retention_s", "external_contact_steps", "closest_opening_distance_m", "board_bounce")))
        trace["selection_scope"] = original.get("selection_scope", original.get("scope", "Recorded native flight; illustrative sample, not aggregate performance"))
    return value


def prepare(base_display, study, output):
    base_display, study, output = map(Path, (base_display, study, output))
    if output.resolve() == base_display.resolve() or output.resolve().is_relative_to(study.resolve()):
        raise ValueError("Write a separate derived report outside original evidence")
    base = optional_json(base_display)
    if not isinstance(base, dict):
        raise ValueError("A valid existing base display report is required")
    result = copy.deepcopy(base)
    raw = optional_json(study / "report.json")
    value = compact(raw)
    result[KEY] = value
    if value is not None:
        audit_path = study / "audit-lowboard.json"
        audit = optional_json(audit_path)
        audit = audit if isinstance(audit, dict) else {}
        value["audit"] = select(audit, ("status", "contract_fingerprint", "checks_count", "reason", "limitations"))
        value["audit"]["full_checks_verified"] = full_audit_verified(raw, audit)
        if audit_path.is_file():
            value["audit"]["source_sha256"] = hashlib.sha256(audit_path.read_bytes()).hexdigest()
        if audit.get("status") == "passed" and not value["audit"]["full_checks_verified"]:
            value["audit"]["verification_reason"] = "Completed exact-version evidence and a nonempty matching strictly passed full audit are required"
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
