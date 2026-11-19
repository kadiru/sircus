"""Derive read-only distance-sweep evidence for the existing mjlab viewer.

Write a separate display document, retaining every historical panel. This module
does not launch physics, train, select a policy or manage a running service.
"""
import argparse
import copy
import datetime
import hashlib
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
VERSION = "sircus-bucket-distance-development-v1"


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
    # The frozen budget allows15 recipes ×2 final native timestep traces.
    # Retain the last (often farthest-distance) recipe as well as the early ones.
    for row in raw.get("trace_examples", [])[:30]:
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


def far_miss_examples(study, raw, audit):
    """Select at most one hash-verified saved miss per supported far target.

    This is an optional post-hoc illustration, never a performance evaluation,
    re-score, physics execution or mutation of the independently audited study.
    """
    fingerprint = raw.get("contract_fingerprint")
    checks = audit.get("checks", [])
    checks_count = audit.get("checks_count")
    if (raw.get("version") != VERSION or raw.get("status") != "completed" or raw.get("phase") != "completed" or
            not isinstance(fingerprint, str) or not fingerprint or audit.get("status") != "passed" or
            audit.get("contract_fingerprint") != fingerprint or audit.get("failures") or
            not isinstance(checks, list) or not checks or not isinstance(checks_count, int) or isinstance(checks_count, bool) or
            checks_count <= 0 or checks_count != len(checks) or any(not isinstance(c, dict) or c.get("passed") is not True for c in checks)):
        return [], ["Far-miss illustrations await a completed study and matching passed audit."]
    study = Path(study).resolve()
    manifest = optional_json(study / "evidence-manifest.json")
    if not isinstance(manifest, dict) or not manifest or manifest != raw.get("evidence_sha256"):
        return [], ["Far-miss illustrations unavailable: evidence manifest does not match the audited report."]

    def verified(name):
        if not isinstance(name, str) or not name:
            raise ValueError("Missing evidence filename")
        path = (study / name).resolve()
        expected = manifest.get(name)
        if (not path.is_relative_to(study) or not path.is_file() or not isinstance(expected, str) or
                hashlib.sha256(path.read_bytes()).hexdigest() != expected):
            raise ValueError("Missing or mismatched evidence hash: " + name)
        return path

    def finite(values, size):
        return isinstance(values, list) and len(values) == size and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values)

    try:
        records = [json.loads(line) for line in verified("records.jsonl").read_text().splitlines() if line.strip()]
        ids = [r.get("id") for r in records]
        if not all(isinstance(r, str) and r for r in ids) or len(set(ids)) != len(ids):
            raise ValueError("Saved record IDs are missing or duplicated")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        return [], ["Far-miss illustrations unavailable: " + str(error)]

    # Imported only for the opt-in completed-evidence path; no simulator/GPU import.
    import numpy as np
    examples, warnings = [], []
    targets = [t for t in raw.get("targets", raw.get("distances", []))
        if isinstance(t, dict) and t.get("bucket_on_table") is True and
        isinstance(t.get("distance_m"), (int, float)) and math.isfinite(t["distance_m"]) and t["distance_m"] >= .65]
    targets.sort(key=lambda t: (str(t.get("ray_id", "")), t["distance_m"], str(t.get("id", ""))))
    for target in targets[:4]:
        candidates = [r for r in records if r.get("distance_id") == target.get("id") and
            r.get("native_release") is True and r.get("route_success") is False and not r.get("invalid") and
            isinstance(r.get("whole_ball_entry"), bool) and
            isinstance(r.get("max_retention_s"), (int, float)) and math.isfinite(r["max_retention_s"]) and r["max_retention_s"] >= 0 and
            isinstance(r.get("closest_opening_distance_m"), (int, float)) and math.isfinite(r["closest_opening_distance_m"]) and r["closest_opening_distance_m"] >= 0]
        if not candidates:
            warnings.append("No eligible saved native miss for " + str(target.get("id")))
            continue
        row = min(candidates, key=lambda r: (not r["whole_ball_entry"], -r["max_retention_s"], r["closest_opening_distance_m"], r["id"]))
        try:
            if (not finite(target.get("bucket_xy_m"), 2) or not finite(target.get("board_xy_m"), 2) or
                    not finite(row.get("bucket_xy_m"), 2) or not finite(row.get("values"), 5) or
                    row["bucket_xy_m"] != target["bucket_xy_m"] or row["values"][:2] != target["board_xy_m"] or
                    row.get("ray_id") != target.get("ray_id") or row.get("distance_m") != target["distance_m"] or
                    not isinstance(row.get("context_id"), str) or not row["context_id"] or row.get("substeps") not in (32, 64, 128)):
                raise ValueError("Native record/target geometry association mismatch")
            index, filename = row.get("native_cache_row"), row.get("native_cache_file")
            with np.load(verified(filename), allow_pickle=False) as data:
                points, cache_ids = data["position_m"], data["record_ids"]
                if cache_ids.ndim != 1:
                    raise ValueError("Native cache record ID shape mismatch")
                cache_ids = list(map(str, cache_ids))
                if (len(cache_ids) != len(set(cache_ids)) or points.shape != (960, len(cache_ids), 3) or not np.isfinite(points).all() or
                        not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(cache_ids) or cache_ids[index] != row["id"]):
                    raise ValueError("Native cache shape, values or exact record column mismatch")
                trajectory = points[::3, index].tolist()
            external = row.get("external_contact_steps")
            details = f"Whole-ball entry {row['whole_ball_entry']}; retention {row['max_retention_s']:.3f} s; external-contact steps {external if isinstance(external, int) else 'unknown'}."
            examples.append(dict(**select(row, ("context_id", "distance_id", "ray_id", "distance_m", "bucket_xy_m", "substeps", "gap_m", "phase", "role",
                "whole_ball_entry", "max_retention_s", "closest_opening_distance_m", "external_contact_steps")),
                record_id=row["id"], board_xy_m=row["values"][:2], method="posthoc_far_miss", route_success=False,
                position_m=trajectory, sample_dt_s=3/240,
                selection_scope="Post-hoc illustrative miss: one per supported far target, ranked by whole entry, retention, closest distance, then stable ID. " + details + " Not aggregate performance or a new evaluation.",
                evidence_provenance=dict(contract_fingerprint=fingerprint, records_file="records.jsonl", records_sha256=manifest["records.jsonl"],
                    cache_file=filename, cache_sha256=manifest[filename], cache_row=index, allow_pickle=False)))
        except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
            warnings.append("Far-miss illustration omitted for " + str(target.get("id")) + ": " + str(error))
    return examples, warnings


def prepare(base_display, study, output, *, include_far_misses=False):
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
        if include_far_misses:
            examples, warnings = far_miss_examples(study, raw, audit)
            value["trace_examples"].extend(examples)
            value["far_miss_illustrations"] = dict(scope="Opt-in post-hoc native miss illustrations, separate from reported scores and gates",
                count=len(examples), record_ids=[r["record_id"] for r in examples], warnings=warnings)
        value["display_updated_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(payload)
    temporary.replace(output)
    return result


def watch(base_display, study, output, seconds=0, *, include_far_misses=False):
    if not math.isfinite(seconds) or not 0 <= seconds <= 30000:
        raise ValueError("Bounded watch must be 0–30,000 seconds")
    started = time.monotonic()
    while True:
        result = prepare(base_display, study, output, include_far_misses=include_far_misses)
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
    parser.add_argument("--include-far-misses", action="store_true", help="Append hash-verified post-hoc native misses only after completion and a matching passed audit")
    args = parser.parse_args()
    watch(args.base_display, args.study, args.output, args.watch_seconds, include_far_misses=args.include_far_misses)
