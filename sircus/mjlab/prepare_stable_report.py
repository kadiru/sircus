"""Append compact IK/search progress to a derived display; no control or training."""

import argparse
import copy
import datetime
import json
from pathlib import Path
import time

from .prepare_expanded_report import optional_json, select, STATS

FIELDS = (
    "version", "status", "phase", "contract_fingerprint", "elapsed_s",
    "timestamp_utc", "development_layouts", "selected_target_ids", "no_training",
    "native_search_attempts", "confirmation_attempts", "final_confirmation_attempts",
    "geometry_proposals", "waiting_for_4090", "gpu_queue", "scope", "solver_audit", "control_attempts", "control_statistics",
)


def compact(report):
    if report is None:
        return None
    value = select(report, FIELDS)
    value["active_policy"] = "Previous policy remains active; no policy training or deployment"
    value["cells"] = []
    for cell in report.get("cells", []):
        row = select(cell, ("id", "bucket_xy_m", "original_test_index", "role",
            "observed_assisted_entries", "unresolved_reason", "first_confirmation", "final_confirmation"))
        row["search"] = select(cell.get("search", {}), STATS + ("assisted",))
        promotion = cell.get("promotion")
        row["promotion"] = select(promotion, (
            "source_trial_id", "source_target_id", "source_phase", "confirmation", "first_confirmation",
            "final_confirmation", "qualification", "relative_board_xy_m", "height_m",
            "tilt_deg", "gap_m", "source_record_sha256",
        )) if isinstance(promotion, dict) else None
        value["cells"].append(row)
    return value


def confirmation_summaries(records):
    """Summarize all recorded attempts, including failed final-tested sources.

    A final-tested source owns the displayed screening counts even when another
    source screened better. This display never chooses or promotes a new source.
    """
    grouped = {}
    for index, row in enumerate(records):
        phase = row.get("phase")
        if phase not in ("confirmation", "final_confirmation"):
            continue
        target, source = row.get("target_id"), row.get("confirmation_source_id")
        if not isinstance(target, str) or not isinstance(source, str):
            continue
        key = (target, source, phase)
        value = grouped.setdefault(key, dict(source_trial_id=source, attempts=0,
            captures=0, invalid=0, first_record_index=index))
        value["attempts"] += 1
        value["captures"] += int(bool(row.get("route_success")) and not bool(row.get("invalid")))
        value["invalid"] += int(bool(row.get("invalid")))

    def summary(value, attempts, captures):
        if value is None:
            return None
        value = {k: v for k, v in value.items() if k != "first_record_index"}
        n = value["attempts"]
        status = "pending" if n < attempts else "invalid" if n > attempts else "passed" if value["captures"] >= captures else "failed"
        return dict(value, required_attempts=attempts, required_captures=captures, status=status)

    result = {}
    for target in sorted({key[0] for key in grouped}):
        finals = [value for (tid, _, phase), value in grouped.items() if tid == target and phase == "final_confirmation"]
        screens = [value for (tid, _, phase), value in grouped.items() if tid == target and phase == "confirmation"]
        if finals:
            # The protocol normally freezes one final source per target. If a
            # malformed record contains more, do not pick the best final result.
            final = min(finals, key=lambda value: value["first_record_index"])
            source = final["source_trial_id"]
            screen = grouped.get((target, source, "confirmation"))
            scope = "Final-tested source and its own screening group"
        elif screens:
            screen = max(screens, key=lambda value: (value["attempts"] == 10,
                value["captures"], -value["invalid"], value["source_trial_id"]))
            final = None
            scope = "Highest-capture completed screening group; final confirmation not recorded"
        else:
            continue
        result[target] = dict(best_screening=summary(screen, 10, 8),
            final_confirmation=summary(final, 20, 16), confirmation_summary_scope=scope,
            multiple_final_groups=len(finals) > 1)
    return result


def optional_records(path):
    """Transient/incomplete JSONL yields unknown, never a shortened denominator."""
    try:
        records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        return records if all(isinstance(row, dict) for row in records) else None
    except (OSError, ValueError):
        return None


def prepare(base_display, study, output):
    base_display, study, output = map(Path, (base_display, study, output))
    if output.resolve() == base_display.resolve() or output.resolve().is_relative_to(study.resolve()):
        raise ValueError("Write a separate derived display; never replace source evidence")
    base = optional_json(base_display)
    if base is None:
        raise ValueError("An existing valid base display report is required")
    result = copy.deepcopy(base)
    result["stable_ik_evaluation"] = compact(optional_json(study / "report.json"))
    if result["stable_ik_evaluation"] is not None:
        records = optional_records(study / "shots.jsonl")
        result["stable_ik_evaluation"]["confirmation_records_status"] = "available" if records is not None else "not_available"
        if records is not None:
            summaries = confirmation_summaries(records)
            for cell in result["stable_ik_evaluation"]["cells"]:
                cell.update(summaries.get(cell["id"], {}))
        audit = optional_json(study / "audit-stable.json")
        result["stable_ik_evaluation"]["verification"] = dict(
            status=("passed" if audit.get("passed") else "failed") if audit else "not_available",
            checks=len(audit.get("checks", [])) if audit else None,
        )
        result["stable_ik_evaluation"]["display_updated_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False))
    temporary.replace(output)
    return result


def watch(base_display, study, output, seconds=0):
    if not 0 <= seconds <= 1800:
        raise ValueError("Watch duration must be between 0 and 1800 seconds")
    start = time.monotonic()
    while True:
        value = prepare(base_display, study, output)
        evidence = value["stable_ik_evaluation"]
        if evidence and evidence.get("status") == "completed" and evidence.get("verification", {}).get("status") == "passed":
            return value
        remaining = seconds - (time.monotonic() - start)
        if remaining <= 0:
            return value
        time.sleep(min(5, remaining))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-display", "study", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--watch-seconds", type=float, default=0)
    args = parser.parse_args()
    if not 0 <= args.watch_seconds <= 1800:
        parser.error("Bounded watch duration0–1800seconds required")
    watch(args.base_display, args.study, args.output, args.watch_seconds)
