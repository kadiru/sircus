"""Read-only, provenance-checked comparison of broad and map-guided PPO pilots.

The native report and media exporter remain shared with the broad pilot. This
wrapper adds a matched-layout comparison without modifying frozen evidence.
Geometry preparation is not a scoring search, and its cost is separate.
"""

import argparse
import copy
import json
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


def add_comparison(summary, baseline_path, run):
    baseline_path, run = Path(baseline_path), Path(run)
    baseline = read(baseline_path)
    support_path = run/"support.json"
    support = None
    if support_path.is_file():
        raw = read(local_file(run, "support.json"))
        summary["source_sha256"]["support.json"] = sha(support_path)
        # Keep exact support definitions/provenance auditable. No outcomes may
        # influence this receipt; the independent audit verifies its derivation.
        support = raw
    summary["comparison"] = comparison(summary, baseline,
        baseline_report_sha256=sha(baseline_path), support=support)
    summary["exporter_wrapper_sha256"] = sha(__file__)
    return summary


def export(run, output, baseline_report, *, media=False, action_edit=False):
    # Fail before encoding expensive native footage if the baseline or split
    # is incompatible. Recheck after the shared exporter binds its sources.
    add_comparison(summarize(run), baseline_report, run)
    summary = native_export(run, output, media=media, action_edit=action_edit)
    add_comparison(summary, baseline_report, run)
    (Path(output)/"report.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--media", action="store_true")
    parser.add_argument("--action-edit", action="store_true")
    args = parser.parse_args()
    value = export(args.run, args.output, args.baseline_report,
                   media=args.media, action_edit=args.action_edit)
    print(json.dumps(dict(status=value["status"], training=value["training"],
                         comparison=value["comparison"]["final_policy_difference"],
                         movie=value["movie"]), allow_nan=False))
