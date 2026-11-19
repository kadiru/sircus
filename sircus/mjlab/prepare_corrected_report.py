"""Read-only corrected-physics grid progress; never controls a simulator/policy."""

import argparse
import copy
from collections import defaultdict
import datetime
import json
from pathlib import Path
import time

from .prepare_expanded_report import optional_json, select
from .prepare_contact_report import verification

TERMINAL = {"completed", "failed", "blocked_geometry", "blocked_storage", "blocked_resources", "cancelled"}
FIELDS = ("version", "status", "phase", "no_training", "contract_fingerprint", "counts",
    "elapsed_s", "scope", "waiting_for_4090", "queue", "reason", "budgets", "common_home")


def compact(raw, search_counts=None):
    if not isinstance(raw, dict):
        return None
    result = select(raw, FIELDS)
    result["counts"] = dict(result.get("counts", {}))
    if "native_releases" in raw:
        result["counts"]["native_releases"] = raw["native_releases"]
    result["counts"]["qualified_settings"] = len(raw.get("qualified_sources", []))
    result["counts"]["positions_with_multiple_qualified"] = sum(
        sum(group.get("qualified") is True for group in cell.get("confirmations", [])) >= 2
        for cell in raw.get("cells", []))
    if raw.get("error") and not result.get("reason"):
        result["reason"] = str(raw["error"])[:400]
    result["active_policy"] = "Previous policy remains active; this is a new feasibility study"
    result["cells"] = []
    for cell in raw.get("cells", []):
        row = select(cell, ("id", "bucket_xy_m", "status", "reason", "geometry", "search", "qualification"))
        if search_counts is not None:
            row["search"] = search_counts.get(cell.get("id"), dict(attempts=0,captures=0,invalid=0))
            if row["search"]["attempts"] and row.get("status") == "ready_search":
                row["status"] = "native_capture" if row["search"]["captures"] else "no_capture_in_budget"
        row["home_geometry"] = select(cell.get("home_geometry", {}), ("passed", "reason", "category", "minimum_clearance_m"))
        connections = [attempt for attempt in raw.get("transit_attempts", []) if attempt.get("cell_id") == cell.get("id")]
        row["motion"] = dict(connected=any(attempt.get("passed") is True for attempt in connections),
            attempts=len(connections), scope="Fixed common-start geometric connection; dynamic tracking separate")
        row["confirmations"] = []
        for group in cell.get("confirmations", []):
            row["confirmations"].append(select(group, ("source_id", "nominal", "fresh", "motion", "qualified", "reason")))
        result["cells"].append(row)
    return result


def shot_snapshot(study, raw):
    """Read only rows covered by the last atomic report, not an in-flight append.

    The native runner fills per-cell summaries during its confirmation loop.
    Derive the already-completed search batches for the other cells from their
    saved rows so that the progress map does not call them untested.
    """
    path = Path(study) / "shots.jsonl"
    if not raw or not path.is_file() or not isinstance(raw.get("records"),int):
        return None
    counts = defaultdict(lambda: dict(attempts=0,captures=0,invalid=0))
    stages = {name:dict(requests=0,releases=0,captures=0) for name in ("search","nominal","confirmation")}
    with path.open() as stream:
        for index,line in enumerate(stream):
            if index >= raw["records"]:
                break
            row = json.loads(line)
            phase = row.get("phase")
            selected_search = phase == "search" and row.get("geometry_eligible") and row.get("status") != "not_selected_budget"
            if phase in stages and (phase != "search" or selected_search):
                stage = stages[phase]
                stage["requests"] += 1
                stage["releases"] += bool(row.get("native_release"))
                stage["captures"] += bool(row.get("native_release") and row.get("route_success"))
            if not selected_search:
                continue
            cell = counts[row["cell_id"]]
            cell["attempts"] += 1
            cell["captures"] += bool(row.get("native_release") and row.get("route_success"))
            cell["invalid"] += bool(row.get("invalid"))
    return dict(search_counts=dict(counts),stages=stages)


def prepare(base_display, study, output):
    base_display, study, output = map(Path, (base_display, study, output))
    if output.resolve() == base_display.resolve() or output.resolve().is_relative_to(study.resolve()):
        raise ValueError("Write a separate derived report outside source evidence")
    base = optional_json(base_display)
    if base is None:
        raise ValueError("A valid existing base report is required")
    result = copy.deepcopy(base)
    raw = optional_json(study / "report.json")
    snapshot = shot_snapshot(study,raw)
    value = compact(raw,snapshot["search_counts"] if snapshot is not None else None)
    result["corrected_grid_evaluation"] = value
    if value is not None:
        value["audits"] = {name:verification(raw,optional_json(study / filename)) for name,filename in (
            ("native","audit-corrected.json"),("supplemental","audit-corrected-supplement.json"))}
        statuses = [audit["status"] for audit in value["audits"].values()]
        value["verification"] = dict(status="failed" if "failed" in statuses else "passed" if all(s == "passed" for s in statuses) else "pending")
        value["stage_statistics"] = snapshot["stages"] if snapshot is not None else None
        value["display_updated_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    return result


def watch(base_display, study, output, seconds=0):
    if not 0 <= seconds <= 30000:
        raise ValueError("Bounded watch must be 0–30,000 seconds")
    started = time.monotonic()
    while True:
        result = prepare(base_display, study, output)
        value = result["corrected_grid_evaluation"]
        if value and value.get("status") in TERMINAL:
            if value["status"] != "completed" or value.get("verification", {}).get("status") in ("passed", "failed"):
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
