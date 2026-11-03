"""Read-only height/tilt development evidence for the existing mjlab viewer.

This only derives a display document. It never starts inference or simulation,
selects a policy, changes source evidence, or manages a running service.
"""
import argparse
import copy
import datetime
import json
import math
from pathlib import Path
import time

from .prepare_expanded_report import optional_json, select
from .prepare_window_report import statistics

KEY = "neighborhood_evaluation"
TERMINAL = {"completed", "failed", "cancelled", "blocked_geometry", "blocked_storage", "blocked_resources"}
CONTEXT_FIELDS = ("id", "source_family", "source_anchor_id", "board_height_m", "tilt_deg", "board_admitted",
    "geometry_requests", "eligible", "search_attempts", "search_native_releases", "search_captures", "clean", "reason", "best_gap_m")
SETTINGS_FIELDS = ("bucket_xy_m", "board_xy_m", "substeps", "gap_noise_m", "height_noise_m", "tilt_noise_deg",
    "seed", "height_offsets_m", "tilt_offsets_deg", "gap_offsets_m", "scope", "budgets")
COUNT_FIELDS = ("contexts", "declared_contexts", "board_resolved", "board_contexts_admitted", "geometry_requests",
    "geometry_eligible", "requests", "native_attempts", "native_releases", "captures", "invalid", "geometry_rejections",
    "search_requests", "search_captures", "refinement_requests", "repeatability_requests", "confirmation_requests",
    "selected_recipes", "paired_gate_passed", "qualified_recipes", "control_requests", "control_captures", "clean",
    "phase_counts", "substep_counts")


def compact(raw):
    if not isinstance(raw, dict):
        return None
    result = select(raw, ("version", "status", "phase", "contract_fingerprint", "scope", "reason", "error", "elapsed_s", "max_requests"))
    result["settings"] = select(raw.get("settings", {}), SETTINGS_FIELDS)
    result["counts"] = select(raw.get("counts", {}), COUNT_FIELDS)
    result["queue"] = select(raw.get("queue", {}), ("wait_s", "reason"))
    result["active_policy"] = "Read-only development sweep; existing inference and historical studies remain unchanged"
    result["contexts"] = [select(row, CONTEXT_FIELDS) for row in raw.get("contexts", []) if isinstance(row, dict)]
    result["anchors"] = [select(row, ("id", "source_family", "board_height_m", "tilt_deg", "gap_m", "values", "bucket_xy_m"))
        for row in raw.get("anchors", []) if isinstance(row, dict)]
    result["selected_recipes"] = []
    for row in raw.get("selected_recipes", []):
        value = select(row, ("id", "context_id", "source_family", "board_height_m", "tilt_deg", "gap_m",
            "release_center_height_m", "distance_m", "selection_scope", "selection", "source_record_id", "sampled_run_width_m"))
        for key in ("by_substeps", "nominal_by_substeps", "fresh_by_substeps"):
            value[key] = {str(k):statistics(v) for k,v in row.get(key, {}).items() if str(k) in ("32", "64", "128")}
        value["sensitivity"] = select(row.get("sensitivity", {}), ("capture_agreement_64_128",
            "nominal_position_delta_350ms_max_m", "paired_engineering_gate", "scope", "paired_requests",
            "position_delta_350ms_mean_m", "capture_disagreements_64_128", "paired_attempts"))
        result["selected_recipes"].append(value)
    result["trace_examples"] = []
    for row in raw.get("trace_examples", [])[:12]:
        points = row.get("position_m", [])
        if not isinstance(points, list) or not 2 <= len(points) <= 5000:
            continue
        if not all(isinstance(p, list) and len(p) == 3 and all(isinstance(v,(int,float)) and math.isfinite(v) for v in p) for p in points):
            continue
        result["trace_examples"].append(dict(**select(row, ("context_id", "gap_m", "record_id", "substeps", "route_success",
            "method", "phase", "sample_dt_s", "selection_scope")), position_m=points,
            scope="Actual saved native positions; illustrative selection, not an analytic flight model"))
        result["trace_examples"][-1]["selection_scope"] = row.get("selection_scope",row.get("scope",""))
    result["audit"] = select(raw.get("audit", {}), ("status", "contract_fingerprint", "checks_count", "reason", "limitations"))
    return result


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
        audit = optional_json(study / "audit-neighborhood.json") or raw.get("audit", {})
        value["audit"] = select(audit, ("status", "contract_fingerprint", "checks_count", "reason", "limitations"))
        if audit.get("status") == "passed" and audit.get("contract_fingerprint") != raw.get("contract_fingerprint"):
            value["audit"].update(status="failed", reason="Audit fingerprint does not match this development study")
        value["display_updated_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(payload); temporary.replace(output)
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
