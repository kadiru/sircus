"""Derive read-only distance-sweep evidence for the existing mjlab viewer.

Write a separate display document, retaining every historical panel. This module
does not launch physics, train, select a policy or manage a running service.
"""
import argparse
import copy
import datetime
import json
import math
from pathlib import Path
import time

from .prepare_expanded_report import optional_json, select
from .prepare_neighborhood_report import compact as neighborhood_compact, TERMINAL

KEY = "distance_evaluation"
IDENTITY = ("distance_id", "ray_id", "distance_m", "bucket_xy_m", "board_xy_m")
DISTANCE_FIELDS = ("id", "ray_id", "distance_m", "bucket_xy_m", "bucket_on_table", "table_margin_m",
    "footprint_ok", "footprint_admitted", "requested_contexts", "board_resolved", "geometry_requests", "eligible",
    "search_attempts", "search_native_releases", "search_captures", "clean", "reason", "selected_recipe_ids")
COUNT_FIELDS = ("requests", "native_releases", "captures", "clean", "geometry_rejections", "contexts",
    "board_resolved", "eligible", "search_attempts", "search_native_releases", "search_captures",
    "selected_recipes", "qualified_recipes", "phase_counts", "substep_counts")


def compact(raw):
    value = neighborhood_compact(raw)
    if value is None:
        return None
    value["active_policy"] = "Read-only distance development search; existing solver, policies and historical studies remain unchanged"
    value["settings"].update(select(raw.get("settings", {}), ("version", "rays", "distances", "table_bounds_xy_m",
        "table_bucket_margin_m", "bucket_outer_radius_m", "bucket_outer_top_diameter_m", "bucket_reported_height_m",
        "bucket_height_status", "height_offsets_m", "tilt_offsets_deg", "gap_offsets_m", "repeatability", "parent_contract_fingerprint")))
    value["settings"]["rays"] = [select(row, ("id", "label", "unit_xy", "heading_deg", "distances_m", "scope"))
        for row in raw.get("settings", {}).get("rays", []) if isinstance(row, dict)]
    value["distances"] = []
    for row in raw.get("targets", raw.get("distances", [])):
        if not isinstance(row, dict):
            continue
        entry = select(row, DISTANCE_FIELDS)
        entry["footprint"] = select(row.get("footprint", {}), ("version", "passed", "minimum_edge_clearance_m",
            "required_edge_margin_m", "numerical_tolerance_m", "outer_radius_m", "table_min_xy_m", "table_max_xy_m"))
        entry["counts"] = select(row.get("counts", {}), COUNT_FIELDS)
        value["distances"].append(entry)
    for key in ("contexts", "selected_recipes"):
        originals = [row for row in raw.get(key, []) if isinstance(row, dict)]
        for target, row in zip(value[key], originals):
            target.update(select(row, IDENTITY))
    for row, original in zip(value["selected_recipes"], (r for r in raw.get("selected_recipes", []) if isinstance(r, dict))):
        row["sensitivity"].update(select(original.get("sensitivity", {}), ("all_resolution_repeat_gate",)))
    value["anchors"] = [select(row, ("id", "values", "board_height_m", "tilt_deg", "gap_m", "source_recipe_id",
        "source_contract_fingerprint", "source_family")) for row in raw.get("anchors", []) if isinstance(row, dict)]
    # A trace must carry the bucket/board association used for its own release.
    # Never reconstruct it with the previous fixed bucket's XY or distance.
    value["trace_examples"] = []
    for row in raw.get("trace_examples", [])[:24]:
        if not isinstance(row, dict):
            continue
        points = row.get("position_m", [])
        if not isinstance(points, list) or not 2 <= len(points) <= 5000 or not all(
            isinstance(p, list) and len(p) == 3 and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in p)
            for p in points):
            continue
        value["trace_examples"].append(dict(**select(row, IDENTITY + ("context_id", "recipe_id", "gap_m", "record_id",
            "substeps", "route_success", "method", "phase", "sample_dt_s", "selection_scope")), position_m=points,
            scope="Actual saved native positions; illustrative distance-specific record, not an ideal-flight model"))
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
        audit = optional_json(study / "audit-distance.json") or raw.get("audit", {})
        value["audit"] = select(audit, ("status", "contract_fingerprint", "checks_count", "reason", "limitations"))
        fingerprint = raw.get("contract_fingerprint")
        if audit.get("status") == "passed" and (not isinstance(fingerprint, str) or not fingerprint or audit.get("contract_fingerprint") != fingerprint):
            value["audit"].update(status="failed", reason="Audit fingerprint is missing or does not match this distance study")
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
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--watch-seconds", type=float, default=0)
    args = parser.parse_args()
    watch(args.base_display, args.study, args.output, args.watch_seconds)
