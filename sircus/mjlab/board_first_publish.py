"""Publish additive audited board-first results; never activate a planner."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import os
from pathlib import Path

from .benchmark_verify import contained, digest, read, sha

VERSION = "sircus-board-first-display-v1"
COUNTS = ("scorable_native", "old_captures", "new_captures", "unchanged_captures",
          "unchanged_misses", "invalid_native")


def normalized_counts(raw):
    result = {key: raw[key] for key in COUNTS}
    result.update(gained=raw["gained_captures"], lost=raw["lost_captures"])
    if any(type(value) is not int or value < 0 for value in result.values()):
        raise ValueError("Expected nonnegative capture accounting")
    if (result["old_captures"] != result["unchanged_captures"] + result["lost"]
        or result["new_captures"] != result["unchanged_captures"] + result["gained"]
        or result["scorable_native"] != sum(result[k] for k in
            ("unchanged_captures", "gained", "lost", "unchanged_misses", "invalid_native"))):
        raise ValueError("Capture accounting differs")
    return result


def rescore_section(study):
    study = Path(study)
    summary, audit = read(study / "summary.json"), read(study / "audit.json")
    if (summary.get("version"), summary.get("status"), audit.get("version"), audit.get("status")) != (
        "sircus-board-first-saved-rescore-v1", "verified",
        "sircus-board-first-saved-rescore-audit-v1", "passed"):
        raise ValueError("Verified saved-shot rescoring and independent audit required")
    if audit.get("summary_sha256") != sha(study / "summary.json") or audit.get("manifest_sha256") != sha(study / "manifest.json"):
        raise ValueError("Rescoring audit no longer binds the evidence")
    counts = normalized_counts(summary["totals"])
    positions = [dict(bucket_xy_m=p["bucket_xy_m"], counts=normalized_counts(p["counts"]))
                 for p in summary["buckets"]]
    if any(sum(p["counts"][key] for p in positions) != value for key, value in counts.items()):
        raise ValueError("Per-position and total accounting differ")
    return dict(status="completed", audit_status="passed", counts=counts, positions=positions,
        exclusions=summary["exclusions"], metadata=summary["metadata"],
        provenance=dict(report_sha256=sha(study / "summary.json"), audit_sha256=sha(study / "audit.json"),
                        manifest_sha256=sha(study / "manifest.json")))


def comparison_section(study, audit_path):
    """Normalize a complete independently audited native64 comparison.

    The independent receipt covers raw trajectories and physics reconstruction.
    This publisher additionally checks the small frozen inputs and reconciles
    every recorded attempt, so edited weights, choices or partial reports fail.
    """
    study, audit_path = Path(study), Path(audit_path)
    report, audit = read(study / "report.json"), read(audit_path)
    if (report.get("version"), report.get("status"), report.get("phase"),
            audit.get("version"), audit.get("status")) != (
            "sircus-board-first-selector-comparison-v1", "completed", "completed",
            "sircus-board-first-selector-independent-audit-v1", "passed"):
        raise ValueError("Completed selector comparison and independent audit required")
    if report.get("native_completed") is not True or audit.get("frozen_before_test") is not True:
        raise ValueError("Complete frozen-before-native comparison required")
    if type(audit.get("checks_count")) is not int or audit["checks_count"] <= 0:
        raise ValueError("Independent audit must record completed checks")
    for key, name in (("report_sha256", "report.json"),
                      ("evidence_manifest_sha256", "evidence-manifest.json"),
                      ("contract_sha256", "contract.json"),
                      ("choices_sha256", "choices.json"),
                      ("native_started_sha256", "native-started.json")):
        if audit.get(key) != sha(study / name):
            raise ValueError("Comparison audit no longer binds " + name)
    if (report.get("evidence_manifest_sha256") != sha(study / "evidence-manifest.json")
            or audit.get("comparison") != report.get("comparison")):
        raise ValueError("Audited comparison summary differs")
    for phase in ("controls_pre", "controls_post"):
        gate = report.get("controls", {}).get(phase, {})
        if gate.get("status") != "passed" or gate.get("board_first_passed") is not True:
            raise ValueError("Both historical and board-first control gates must pass")

    contract = read(study / "contract.json")
    unsigned = dict(contract)
    fingerprint = unsigned.pop("fingerprint", None)
    if fingerprint != digest(unsigned):
        raise ValueError("Comparison contract fingerprint changed")
    methods = ("original", "retrained")
    if (contract.get("version"), contract.get("methods"), contract.get("test_positions"),
            contract.get("physics_substeps"), contract.get("attempts_per_method_per_position"),
            contract.get("scoring_version")) != (
            "sircus-board-first-selector-comparison-v1", list(methods), 50, 64, 1,
            "sircus-board-first-native-capture-v4"):
        raise ValueError("Expected two frozen selectors, 50 positions, one attempt each, native64")
    manifest = read(study / "evidence-manifest.json")
    required = ("contract.json", "layouts.json", "plans.json", "choices.json",
                "native-started.json", "original-checkpoint.json", "retrained-checkpoint.json", "rows.jsonl")
    for name in required:
        if manifest.get(name) != sha(study / name):
            raise ValueError("Comparison evidence changed: " + name)
    choices = read(study / "choices.json")
    started = read(study / "native-started.json")
    frozen = contract.get("frozen_inputs", {})
    for name in ("original-checkpoint.json", "retrained-checkpoint.json", "layouts.json"):
        if name not in frozen:
            raise ValueError("Frozen weights and placements are required")
    for name, expected in frozen.items():
        if (sha(contained(study, name)) != expected
                or choices.get("bindings", {}).get(name) != expected):
            raise ValueError("Frozen model/input differs before native: " + name)
    expected_bindings = dict(choices["bindings"], **{"choices.json": sha(study / "choices.json")})
    if started.get("bindings") != expected_bindings:
        raise ValueError("Native start did not freeze the exact weights and choices")
    for name, expected in expected_bindings.items():
        if sha(contained(study, name)) != expected:
            raise ValueError("Native-start binding changed: " + name)
    for method in methods:
        if audit.get("checkpoint_sha256", {}).get(method) != sha(study / (method + "-checkpoint.json")):
            raise ValueError("Audit checkpoint binding differs: " + method)

    layouts = read(study / "layouts.json")["layouts"]
    lookup = {row["id"]: row for row in layouts}
    if (len(layouts) != 50 or len(lookup) != 50
            or len({tuple(row["bucket_xy_m"]) for row in layouts}) != 50
            or any(len(row["bucket_xy_m"]) != 2 or
                   not all(type(v) in (int, float) and math.isfinite(v) for v in row["bucket_xy_m"])
                   for row in layouts)):
        raise ValueError("Exactly 50 distinct finite bucket positions required")
    slots = choices["slots"]
    slot_by_id = {row["id"]: row for row in slots}
    keys = {(row["context_id"], row["method"]) for row in slots}
    expected_keys = {(layout, method) for layout in lookup for method in methods}
    if len(slots) != 100 or len(slot_by_id) != 100 or keys != expected_keys:
        raise ValueError("Exactly one choice per position and frozen selector required")
    if any(row["bucket_xy_m"] != lookup[row["context_id"]]["bucket_xy_m"] for row in slots):
        raise ValueError("Choice bucket positions differ from frozen layouts")
    planned = {method: [row["request"] for row in slots if row["method"] == method and row["request"] is not None]
               for method in methods}
    if started.get("requests_sha256") != digest(planned):
        raise ValueError("Native-start requests differ from frozen choices")
    plans = read(study / "plans.json")
    plan_by_id = {row["choice"]["id"]: row for row in plans}
    if len(plans) != 100 or set(plan_by_id) != set(slot_by_id):
        raise ValueError("Complete planning receipts required")
    times = {method: [] for method in methods}
    for identifier, plan in plan_by_id.items():
        if plan["choice"] != slot_by_id[identifier] or plan["method"] != slot_by_id[identifier]["method"]:
            raise ValueError("Planning receipts disagree with frozen choices")
        elapsed = plan["decision_latency_s"]
        if type(elapsed) not in (float, int) or not math.isfinite(elapsed) or not 0 <= elapsed <= contract["planning_budget_s"] + 1e-6:
            raise ValueError("Invalid or over-budget planning time")
        times[plan["method"]].append(elapsed)
    rows = [json.loads(line) for line in (study / "rows.jsonl").read_text().splitlines() if line.strip()]
    controls = [row for row in rows if row.get("adapter_directory") == "native-controls"]
    released = [row for row in rows if row.get("adapter_directory") != "native-controls"]
    native = {row["id"]: row for row in released}
    wanted = {row["id"] for row in slots if row["request"] is not None}
    if len(controls) != 18 or len(native) != len(released) or set(native) != wanted:
        raise ValueError("Every admitted slot needs one native drop plus 18 matched controls")
    if (report.get("progress", {}).get("prepared") != 100
            or report["progress"].get("planned_releases") != len(native)
            or report["progress"].get("completed_releases") != len(native)):
        raise ValueError("Partial native comparison cannot be published")
    states = {}
    for identifier, row in native.items():
        score = row.get("board_first_score", {})
        slot = slot_by_id[identifier]
        if (row.get("native_release") is not True or row.get("board_first_scalar_audit_passed") is not True
                or score.get("scoring_version") != contract["scoring_version"]
                or type(score.get("route_success")) is not bool or type(score.get("invalid_physics")) is not bool
                or (score["route_success"] and score["invalid_physics"])
                or row.get("adapter_directory") != "native-" + slot["method"]
                or row.get("bucket_xy_m") != slot["bucket_xy_m"]):
            raise ValueError("Invalid, unaudited or mismatched native result")
        states[slot["context_id"], slot["method"]] = score["route_success"]
    normalized = []
    for method in methods:
        selected = [row for identifier, row in native.items() if slot_by_id[identifier]["method"] == method]
        captures = sum(row["board_first_score"]["route_success"] for row in selected)
        invalid = sum(row["board_first_score"]["invalid_physics"] for row in selected)
        statistics_row = dict(positions=50, captures=captures, releases=len(selected),
            setup_rejections=50-len(selected), invalid_physics=invalid, flight_misses=len(selected)-captures-invalid,
            capture_rate=captures/50, mean_planning_s=statistics.mean(times[method]),
            median_planning_s=statistics.median(times[method]), max_planning_s=max(times[method]))
        observed = report["comparison"]["methods"][method]
        for key, value in statistics_row.items():
            actual = observed.get(key)
            if isinstance(value, float):
                same = type(actual) in (float, int) and math.isclose(actual, value, rel_tol=0, abs_tol=1e-9)
            else:
                same = type(actual) is int and actual == value
            if not same:
                raise ValueError("Comparison totals differ from native evidence: " + method + "/" + key)
        normalized.append(dict(id=method, label="Original checkpoint" if method == "original" else "Retrained checkpoint",
            checkpoint_sha256=sha(study / (method + "-checkpoint.json")), attempts=50,
            captures=captures, setup_rejections=statistics_row["setup_rejections"], released_misses=statistics_row["flight_misses"],
            invalid_physics=invalid, mean_planning_s=statistics_row["mean_planning_s"],
            median_planning_s=statistics_row["median_planning_s"], max_planning_s=statistics_row["max_planning_s"]))
    paired = dict(both=0, original_only=0, retrained_only=0, neither=0)
    for layout in layouts:
        old, new = [states.get((layout["id"], method), False) for method in methods]
        paired["both" if old and new else "original_only" if old else "retrained_only" if new else "neither"] += 1
    if report["comparison"]["paired"] != paired:
        raise ValueError("Paired comparison differs from native outcomes")
    old, new = (row["captures"] for row in normalized)
    return dict(status="completed", phase="completed", audit_status="passed", positions=50,
        physics_substeps=64, frozen_before_test=True, methods=normalized, paired=paired,
        conclusion=(f"Retrained checkpoint: {new}/50 captures; original checkpoint: {old}/50 ({new-old:+d}). "
                    "Both used the same board-first scoring and contact-permitting planner. The live planner is unchanged."),
        provenance=dict(report_sha256=sha(study / "report.json"), audit_sha256=sha(audit_path),
                        manifest_sha256=sha(study / "evidence-manifest.json"),
                        contract_sha256=sha(study / "contract.json"), choices_sha256=sha(study / "choices.json")))


def compose(rescore, *, comparison=None, comparison_audit=None):
    if (comparison is None) != (comparison_audit is None):
        raise ValueError("Comparison path and independent audit must be supplied together")
    result = dict(version=VERSION, rescore=rescore_section(rescore),
        scope="Recorded simulation evidence, re-scored under a new version. Original scores and checkpoints are preserved.",
        comparison=dict(phase="preparing corrected labels", status="pending"))
    if comparison is not None:
        result["comparison"] = comparison_section(comparison, comparison_audit)
    return result


def publish(rescore, display_report, *, comparison=None, comparison_audit=None):
    display_report = Path(display_report)
    temporary = display_report.with_suffix(display_report.suffix + ".board-first.tmp")
    protected_roots = [Path(rescore).resolve()]
    if comparison is not None:
        protected_roots.append(Path(comparison).resolve())
    for target in (display_report, temporary):
        resolved = target.resolve()
        if (any(resolved.is_relative_to(root) for root in protected_roots)
                or (comparison_audit is not None and resolved == Path(comparison_audit).resolve())):
            raise ValueError("Display must be outside immutable study evidence and audits")
    display = read(display_report) if display_report.exists() else {}
    section = compose(rescore, comparison=comparison, comparison_audit=comparison_audit)
    display["board_first_study"] = section
    temporary.write_text(json.dumps(display, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, display_report)
    return section


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rescore", type=Path, required=True)
    parser.add_argument("--display-report", type=Path, required=True)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--comparison-audit", type=Path)
    args = parser.parse_args()
    if (args.comparison is None) != (args.comparison_audit is None):
        parser.error("--comparison and --comparison-audit must be supplied together")
    section = publish(args.rescore, args.display_report, comparison=args.comparison, comparison_audit=args.comparison_audit)
    print(json.dumps(section["rescore"]["counts"]))


if __name__ == "__main__":
    main()
