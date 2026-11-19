"""Read-only, provenance-checked comparison of broad and map-guided PPO pilots.

The native report and media exporter remain shared with the broad pilot. This
wrapper adds a matched-layout comparison without modifying frozen evidence.
Geometry preparation is not a scoring search, and its cost is separate.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

from continuous_pilot_report import export as native_export
from continuous_pilot_report import local_file, read, require, sha, summarize

SCHEMA = "sircus-feasibility-guided-comparison-v1"
BASELINE_RESULT_SHA256 = "cf7e05910a22269330fc3ec89c10ed080ef0c5c67f3528b31c07e1b3735ca614"
GUIDED_STUDY_SCHEMA = "sircus-native-feasibility-guided-pilot-v1"
SPLITS = ("train", "validation", "test")


def compact_counts(row):
    if row is None:
        return None
    attempts, invalid, captures = (row[key] for key in ("attempts", "invalid", "captures"))
    require(all(type(v) is int and v >= 0 for v in (attempts, invalid, captures))
            and invalid <= attempts and captures <= attempts-invalid,
            "Invalid comparison attempt/admission/capture counts")
    released = attempts-invalid
    return dict(attempts=attempts, released=released, rejected=invalid, captures=captures,
                clean=row.get("clean"), capture_rate=captures/attempts if attempts else None,
                capture_rate_per_released=captures/released if released else None,
                update_count=row.get("update_count"), unique_layouts=row.get("unique_layouts"),
                layouts_with_capture=row.get("layouts_with_capture"),
                scope="Every active attempted request is in the denominator; identical nominal repeats are not independent layouts")


def stages(report, split):
    checks = sorted((row for row in report["assessments"] if row["split"] == split),
                    key=lambda row: row["update_count"])
    require(len({row["update_count"] for row in checks}) == len(checks),
            "Ambiguous frozen checkpoint comparison")
    initial = compact_counts(checks[0]) if checks else None
    final = compact_counts(checks[-1]) if len(checks) > 1 and report["status"] == "completed" else None
    delta = (final["capture_rate"]-initial["capture_rate"]
             if initial and final and initial["capture_rate"] is not None
             and final["capture_rate"] is not None else None)
    return dict(initial=initial, final=final, change_percentage_points=delta*100 if delta is not None else None)


def layout_identity(report):
    cells = report["cells"]
    require(len(cells) == 35 and len({cell["id"] for cell in cells}) == 35,
            "Comparison requires all 35 distinct original workcells")
    return {cell["id"]: (cell["split"], tuple(cell["bucket_xy_m"])) for cell in cells}


def comparison(guided, baseline, *, baseline_report_sha256=None, support=None):
    require(guided["study"]["schema"] == GUIDED_STUDY_SCHEMA, "Expected a guided study")
    require(baseline["status"] == "completed", "Broad baseline must remain a completed frozen study")
    require(baseline["source_sha256"].get("result.json") == BASELINE_RESULT_SHA256,
            "Unexpected broad-pilot baseline; do not silently compare a different run")
    require(guided["geometry_fingerprint"] == baseline["geometry_fingerprint"],
            "Geometry fingerprint mismatch")
    require(layout_identity(guided) == layout_identity(baseline), "Bucket coordinates or train/validation/test split changed")
    for field in ("contract_sha256", "camera_sha256"):
        a, b = (report["study"].get("source", {}).get(field) for report in (guided, baseline))
        require(a is not None and a == b, "Comparison changed "+field)
    require(guided["scoring"] == baseline["scoring"], "Scoring definition changed")
    def method(report, label):
        training = compact_counts(report["training"])
        budgets = report["study"].get("training_batches")
        expected = budgets*25 if type(budgets) is int else None
        if report["status"] == "completed":
            require(expected == training["attempts"] == 1600,
                    "Completed comparison must retain the equal 1,600-attempt budget")
        return dict(label=label, status=report["status"], study_fingerprint=report["study_fingerprint"],
                    source_result_sha256=report["source_sha256"].get("result.json"),
                    training=training, planned_attempts=expected, elapsed_s=report["elapsed_s"],
                    frozen={split: stages(report, split) for split in SPLITS},
                    learning_curve=copy.deepcopy(report["learning_curve"]),
                    audit_status={key:value["report"].get("status", "unknown")
                                  for key,value in report.get("audits", {}).items()})
    methods = dict(broad=method(baseline, "Broad continuous action domain"),
                   guided=method(guided, "Geometry-conditioned local continuous domain"))
    matched = {}
    for split in SPLITS:
        left, right = (methods[name]["frozen"][split]["final"] for name in ("broad", "guided"))
        delta = None
        if left is not None and right is not None:
            require(left["attempts"] == right["attempts"], "Frozen evaluation budgets differ")
            delta = 100*(right["capture_rate"]-left["capture_rate"])
        matched[split] = dict(guided_minus_broad_percentage_points=delta)
    prep = dict(original_sobol_requests=baseline["feasibility"]["attempts"],
                original_map_admitted=baseline["feasibility"]["admitted"],
                original_map_scope="Geometry-only endpoint and coarse-clearance labels; no capture outcomes",
                support=support,
                cost_scope="Geometry map/support preparation is additional preprocessing, not part of the matched native shot budget. Native elapsed includes recording/evaluation and is not a headless compute benchmark.")
    require(not support or isinstance(support, dict), "Invalid support receipt")
    return dict(schema=SCHEMA, status=guided["status"], methods=methods,
                final_policy_difference=matched, geometry_preparation=prep,
                baseline_report_sha256=baseline_report_sha256,
                scientific_scope="Single-seed method change on the same inherited 25/5/5 layouts. The new action support uses geometry, including held-out geometry, but no held-out rewards. These coordinates are not globally unseen. The comparison does not isolate PPO from geometry preprocessing or establish table-wide robustness.",
                action_scope="A bounded residual is decoded continuously around a geometry-selected anchor; it is not the previous full-domain policy. Every requested residual still undergoes native IK/path/settling guards. Unresolved support is not proof of impossibility.",
                inference_scope="No capture search or outcome-based action selection during frozen inference; arm paths and initialized ball release remain scripted.")


def verify_comparison_audit(receipt, value):
    if "comparison" in receipt:
        require(receipt.get("status") == "passed"
                and receipt.get("source_result_sha256") == value["methods"]["guided"]["source_result_sha256"],
                "Independent containing audit is not passed or belongs to another guided result")
        receipt = receipt["comparison"]
    require(receipt.get("schema") == SCHEMA and receipt.get("status") == "completed",
            "Independent comparison is incomplete or uses a different schema")
    require(receipt.get("original_result_sha256") == value["methods"]["broad"]["source_result_sha256"]
            and receipt.get("guided_result_sha256") == value["methods"]["guided"]["source_result_sha256"],
            "Independent comparison belongs to different native results")
    def equal_counts(left, right):
        require(left is not None and right is not None
                and all(left[k] == right[k] for k in ("attempts", "captures", "clean"))
                and left["invalid"] == right["rejected"], "Independent comparison count mismatch")
    for source, method in (("original", "broad"), ("guided", "guided")):
        left, right = receipt[source], value["methods"][method]
        equal_counts(left["training"], right["training"])
        for stage in ("initial", "final"):
            for split in SPLITS:
                equal_counts(left["frozen"][stage][split], right["frozen"][split][stage])
    return receipt


def verify_support_exclusion(receipt, summary, baseline, support):
    """Recompute a posthoc restriction diagnostic without changing proposals."""
    require(receipt.get("schema") == "sircus-guided-support-success-exclusion-v1"
            and receipt.get("status") == "completed", "Incomplete support-exclusion diagnostic")
    require(receipt.get("old_source_result_sha256") == BASELINE_RESULT_SHA256
            == baseline["source_sha256"].get("result.json")
            and receipt.get("old_study_fingerprint") == baseline["study_fingerprint"]
            and receipt.get("geometry_fingerprint") == summary["geometry_fingerprint"]
            and receipt.get("support_sha256") == summary["study"]["support_sha256"]
            and receipt.get("support_fingerprint") == support["fingerprint"],
            "Support-exclusion diagnostic belongs to different evidence")
    for label, key in (("native", "original_native_audit_sha256"),
                       ("checkpoint", "original_checkpoint_audit_sha256")):
        require(receipt.get(key) == baseline.get("audits", {}).get(label, {}).get("sha256"),
                "Support-exclusion original audit binding changed")
    trial_hashes = receipt.get("source_trial_sha256", {})
    expected_trials = sorted(k for k in baseline["source_sha256"] if k.startswith("trials/eval-"))[-8:]
    require(len(trial_hashes) == 8 and set(trial_hashes) == set(expected_trials) and all(baseline["source_sha256"].get(k) == v
                                        for k,v in trial_hashes.items()),
            "Support-exclusion original evaluation trial hashes changed")
    known = {}
    for cell in baseline["cells"]:
        if not cell.get("evaluation"):
            continue
        final = max(cell["evaluation"], key=lambda row: row["update_count"])
        if final["captures"]:
            predictions = [row for row in cell.get("frozen_predictions", [])
                           if row["update_count"] == final["update_count"]]
            require(len(predictions) == 1, "Missing original captured prediction")
            known[cell["id"]] = (cell, final, predictions[0]["requested"])
    rows = receipt.get("records", [])
    require(len(rows) == len(known) and {r["cell_id"] for r in rows} == set(known),
            "Support-exclusion successful setting membership changed")
    support_cells = {r["cell_id"]:r for r in support["cells"]}
    names = ("bearing_offset_rad", "radius_m", "height_m", "gap_m", "tilt_rad")
    excluded = retained = captures = excluded_captures = 0
    axes = {name:0 for name in names}
    for row in rows:
        cell, final, predicted = known[row["cell_id"]]
        box = support_cells[row["cell_id"]]
        require(row["split"] == cell["split"] and row["bucket_xy_m"] == cell["bucket_xy_m"]
                and row["old_frozen_captures"] == final["captures"]
                and row["old_frozen_attempts"] == final["attempts"]
                and row["guided_low"] == box["low"] and row["guided_high"] == box["high"],
                "Support-exclusion cell geometry or outcome counts changed")
        action = row["old_requested_action"]
        require(set(action) == set(names)
                and all(math.isclose(action[k], predicted[k], rel_tol=0, abs_tol=1e-12) for k in names),
                "Support-exclusion original action changed")
        outside = [name for i,name in enumerate(names)
                   if not box["low"][i] <= action[name] <= box["high"][i]]
        require(row.get("inside_guided_box") is (not outside)
                and [a["axis"] for a in row["excluded_axes"]] == outside,
                "Support-exclusion geometric classification changed")
        captures += final["captures"]
        if outside:
            excluded += 1; excluded_captures += final["captures"]
            for name in outside:axes[name] += 1
        else:
            retained += 1
    require(receipt.get("old_successful_sites") == len(known)
            and receipt.get("old_successful_repeated_captures") == captures
            and receipt.get("old_successful_settings_retained") == retained
            and receipt.get("old_successful_settings_excluded") == excluded
            and receipt.get("excluded_repeated_successes") == excluded_captures
            and receipt.get("excluded_site_counts_by_axis") == {k:v for k,v in axes.items() if v},
            "Support-exclusion aggregate counts changed")
    return receipt


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def compact_support(raw):
    """Retain action definitions and bind omitted geometry probes by digest.

    This is a derived receipt, not a replacement signed support contract. Its
    source fingerprint continues to name the immutable full geometry file.
    """
    unsigned = dict(raw)
    fp = unsigned.pop("fingerprint", None)
    require(fp == canonical_digest(unsigned), "Guided support fingerprint mismatch")
    require(raw.get("schema") == "sircus-feasibility-guided-support-v1"
            and raw.get("policy_geometry_only") is True,
            "Unsupported or non-geometric guided support")
    top_keys = ("schema", "source_fingerprint", "feasibility_fingerprint", "policy_geometry_only",
        "asset_sha256", "action_names", "global_low", "global_high", "base_half_width", "width_scales",
        "probe_count", "probe_seed", "minimum_probe_acceptance", "anchor_rank", "anchor_path_samples",
        "probe_solver", "seed_source", "elapsed_s", "probe_requests", "probe_accepted", "supported_cells", "scope")
    result = {key:copy.deepcopy(raw[key]) for key in top_keys if key in raw}
    result.update(source_support_fingerprint=fp, compact_receipt=True,
        omission_scope="Full geometry probe actions, qualifications, anchor IK joint fits, canonical joints and path readbacks stay outside this report. Exact omitted decision lists are SHA256-bound; native captures never select these supports.")
    result["cells"] = []
    fields = ("cell_id", "cell_index", "bucket_xy_m", "map_candidates", "status", "reason", "low", "high",
              "anchor_action", "anchor_proposal", "anchor_score", "selected_width_scale", "probe_acceptance", "qualification_target_met")
    for cell in raw["cells"]:
        row = {key:copy.deepcopy(cell[key]) for key in fields if key in cell}
        row["anchor_checks_sha256"] = canonical_digest(cell.get("anchor_checks", []))
        row["anchor_checks_count"] = len(cell.get("anchor_checks", []))
        row["anchor_ik_sha256"] = canonical_digest(cell["anchor_ik"]) if cell.get("anchor_ik") else None
        row["calibration"] = []
        for probe in cell.get("calibration", []):
            decision = {key:copy.deepcopy(probe[key]) for key in ("scale", "low", "high", "requests", "accepted", "acceptance")}
            records = probe["records"]
            require(len(records) == decision["requests"], "Probe decision count mismatch")
            decision.update(records_count=len(records), records_sha256=canonical_digest(records))
            row["calibration"].append(decision)
        result["cells"].append(row)
    return result


def add_comparison(summary, baseline_path, run, *, support_cost_path=None, support_exclusion_path=None):
    baseline_path, run = Path(baseline_path), Path(run)
    baseline = read(baseline_path)
    support_path = local_file(run, "support.json")
    raw = read(support_path)
    support_sha = sha(support_path)
    study = summary["study"]
    require(study.get("support_sha256") == support_sha,
            "Study belongs to different support bytes")
    require(study.get("support_fingerprint") == raw["fingerprint"]
            and study.get("domain", {}).get("support_fingerprint") == raw["fingerprint"],
            "Study support fingerprint mismatch")
    require(study.get("domain", {}).get("decoder") == "geometry-conditioned local affine residual-v1",
            "Unknown guided action decoder")
    require(study.get("feasibility_sha256") == summary["source_sha256"].get("feasibility.json"),
            "Study belongs to a different geometry map")
    summary["source_sha256"]["support.json"] = support_sha
    require(raw.get("source_fingerprint") == summary["geometry_fingerprint"],
            "Support belongs to a different scene")
    support = compact_support(raw)
    support["source_support_sha256"] = support_sha
    summary["comparison"] = comparison(summary, baseline,
        baseline_report_sha256=sha(baseline_path), support=support)
    audit_path = run/"comparison-audit.json"
    if audit_path.is_file():
        audit_path = local_file(run, "comparison-audit.json")
        receipt = read(audit_path)
        comparison_receipt = verify_comparison_audit(receipt, summary["comparison"])
        audit_sha = sha(audit_path)
        summary["source_sha256"]["comparison-audit.json"] = audit_sha
        summary["comparison"]["independent_audit"] = dict(sha256=audit_sha, report=comparison_receipt,
            source_receipt_schema=receipt.get("schema"), source_receipt_status=receipt.get("status"))
    else:
        summary["comparison"]["independent_audit"] = dict(status="not_attached")
    preparation = summary["comparison"]["geometry_preparation"]
    if support_cost_path is not None:
        cost_path = Path(support_cost_path)
        cost = read(cost_path)
        require(cost.get("support_sha256") == support_sha
                and cost.get("support_fingerprint") == raw["fingerprint"],
                "Process resource receipt belongs to another support")
        preparation["process_resource_receipt"] = dict(sha256=sha(cost_path), report=cost,
            scope="Explicit process-resource receipt. A process peak RSS is not simultaneous aggregate RAM across workers; CPU accounting scope follows the source receipt.")
    else:
        preparation["process_resource_receipt"] = dict(status="not_attached",
            scope="Support wall time is recorded; process CPU and RSS remain unavailable in this report.")
    if support_exclusion_path is not None:
        exclusion_path = Path(support_exclusion_path)
        diagnostic = read(exclusion_path)
        verify_support_exclusion(diagnostic, summary, baseline, raw)
        summary["comparison"]["posthoc_support_exclusion"] = dict(sha256=sha(exclusion_path), report=diagnostic,
            scope="Posthoc evaluation only. This diagnostic did not select geometry anchors, policy actions, rewards or training support and must not be used as inference input.")
    summary["exporter_wrapper_sha256"] = sha(__file__)
    return summary


def export(run, output, baseline_report, *, media=False, action_edit=False, support_cost_path=None, support_exclusion_path=None):
    # Fail before encoding expensive native footage if the baseline or split
    # is incompatible. Recheck after the shared exporter binds its sources.
    add_comparison(summarize(run), baseline_report, run, support_cost_path=support_cost_path, support_exclusion_path=support_exclusion_path)
    summary = native_export(run, output, media=media, action_edit=action_edit)
    add_comparison(summary, baseline_report, run, support_cost_path=support_cost_path, support_exclusion_path=support_exclusion_path)
    (Path(output)/"report.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--support-exclusion", type=Path, help="Optional SHA-bound posthoc support restriction diagnostic")
    parser.add_argument("--support-cost", type=Path, help="Optional explicit process-resource JSON receipt")
    parser.add_argument("--media", action="store_true")
    parser.add_argument("--action-edit", action="store_true")
    args = parser.parse_args()
    value = export(args.run, args.output, args.baseline_report,
                   media=args.media, action_edit=args.action_edit, support_cost_path=args.support_cost, support_exclusion_path=args.support_exclusion)
    print(json.dumps(dict(status=value["status"], training=value["training"],
                         comparison=value["comparison"]["final_policy_difference"],
                         movie=value["movie"]), allow_nan=False))
