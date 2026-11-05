"""Posthoc comparison of frozen successful shots with fixed geometry supports.

This CPU diagnostic never feeds training, selects anchors or changes support.
Excluding known successes is evidence of changed difficulty, not evidence that
all other actions inside a local box fail. Raw inputs remain unchanged.
"""

import argparse
import collections
import json
import math
import re
from pathlib import Path

import numpy as np

from continuous_feasibility import ACTION_NAMES
from continuous_pilot_audit import (digest, evidence_file, read, require, signed,
                                    validate_study)
from feasibility_guided_support import validate_support

SCHEMA = "sircus-guided-support-success-exclusion-v1"
SCOPE = (
    "Requested parameters of independently audited original final-policy captures "
    "are compared with fixed geometry-only local boxes for the same known bucket "
    "coordinates. This is posthoc evaluation of the support change, not input to "
    "training, anchor selection, reward tuning or inference. Excluding demonstrated "
    "settings does not prove other points inside the boxes cannot capture. Eight "
    "nominal repeats are not eight independent layouts; no impossibility claim."
)


def _expected_hash(path, expected, label):
    require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected),
            "Explicit expected " + label + " SHA256 is required")
    require(digest(path) == expected, label + " checksum changed")


def support_exclusion(original_run, support_path, *, expected_result_sha256,
                      expected_support_sha256):
    """Evaluate fixed original final predictions; never run a policy or search."""
    original_run, support_path = Path(original_run), Path(support_path)
    result_path = evidence_file(original_run, "result.json")
    require(support_path.is_file() and not support_path.is_symlink(), "Invalid support input")
    _expected_hash(result_path, expected_result_sha256, "Original result")
    _expected_hash(support_path, expected_support_sha256, "Guided support")
    result = read(result_path)
    study = read(evidence_file(original_run, "study.json"))
    spec = read(evidence_file(original_run, "contract.json"))
    support = read(support_path)
    study_fp, geometry_fp, _ = validate_study(study, spec)
    support_fp = signed(support, "Guided support")
    validate_support(support, spec["cells"])
    require(result.get("status") == "completed"
            and result.get("schema") == study["schema"]
            and result.get("study_fingerprint") == study_fp
            and result.get("geometry_fingerprint") == geometry_fp == support.get("source_fingerprint"),
            "Original result/support geometry association changed")
    native_path = evidence_file(original_run, "native-audit.json")
    checkpoint_path = evidence_file(original_run, "checkpoint-audit.json")
    native, checkpoint = read(native_path), read(checkpoint_path)
    require(native.get("schema") == "sircus-continuous-native-outcome-audit-v1"
            and checkpoint.get("schema") == "sircus-continuous-pilot-checkpoint-audit-v1"
            and all(a.get("status") == "passed"
                    and a.get("source_result_sha256") == expected_result_sha256
                    for a in (native, checkpoint)), "Original independent audit binding changed")
    model = evidence_file(original_run, result["final"]["file"], suffix=".pt")
    require(digest(model) == result["final"]["sha256"], "Original final checkpoint changed")
    repeats = study["assessment_batches"]
    require(len(result["assessments"]) == 2*repeats, "Incomplete original frozen assessments")
    rows, trials = [], {}
    expected_membership = [(c["id"], c["split"]) for c in study["cells"]]
    for summary in result["assessments"][repeats:]:
        path = evidence_file(original_run, summary["file"], trial=True)
        sha = digest(path)
        require(sha == summary["sha256"] == native.get("source_trial_sha256", {}).get(summary["file"])
                == checkpoint.get("trials", {}).get(summary["file"]),
                "Original final trial differs from independent audits")
        trial = read(path)
        require(trial.get("mode") == "eval" and trial.get("checkpoint") == result["final"]
                and trial.get("deterministic") is True
                and [(r["cell_id"], r["split"]) for r in trial.get("rows", [])] == expected_membership,
                "Original frozen checkpoint or membership changed")
        rows.extend(trial["rows"])
        trials[summary["file"]] = sha
    by_id = {row["cell_id"]: row for row in support["cells"]}
    records, axes = [], collections.Counter()
    for cell in study["cells"]:
        group = [r for r in rows if r["cell_id"] == cell["id"]]
        require(len(group) == repeats and len({tuple(r[k] for k in ACTION_NAMES) for r in group}) == 1,
                "Frozen assessment requested actions differ across nominal repeats")
        captures = sum(bool(r["route_success"]) for r in group)
        if not captures:
            continue
        values = np.asarray([group[0][k] for k in ACTION_NAMES], dtype=float)
        local = by_id[cell["id"]]
        lo, hi = np.asarray(local["low"]), np.asarray(local["high"])
        outside = (values < lo-1e-12) | (values > hi+1e-12)
        details = []
        for i, key in enumerate(ACTION_NAMES):
            if not outside[i]:
                continue
            axes[key] += 1
            factor = 180/math.pi if key.endswith("_rad") else 1000
            details.append(dict(axis=key, unit="degrees" if key.endswith("_rad") else "mm",
                original=float(values[i]*factor), minimum=float(lo[i]*factor), maximum=float(hi[i]*factor),
                side="below_minimum" if values[i] < lo[i] else "above_maximum",
                distance_outside=float(max(lo[i]-values[i], values[i]-hi[i])*factor)))
        records.append(dict(cell_id=cell["id"], split=cell["split"], bucket_xy_m=cell["bucket_xy_m"],
            old_frozen_captures=captures, old_frozen_attempts=repeats, support_status=local["status"],
            old_requested_action=dict(zip(ACTION_NAMES, map(float, values))), guided_low=lo.tolist(),
            guided_high=hi.tolist(), inside_guided_box=not bool(outside.any()), excluded_axes=details))
    excluded = [r for r in records if not r["inside_guided_box"]]
    return dict(schema=SCHEMA, status="completed", source_diagnostic_sha256=digest(__file__),
        analysis_kind="Posthoc frozen support-difficulty diagnostic; no training or support tuning",
        old_source_result_sha256=expected_result_sha256, old_checkpoint=result["final"],
        old_study_fingerprint=study_fp, geometry_fingerprint=geometry_fp,
        support_sha256=expected_support_sha256, support_fingerprint=support_fp,
        original_native_audit_sha256=digest(native_path), original_checkpoint_audit_sha256=digest(checkpoint_path),
        source_trial_sha256=trials, old_successful_sites=len(records),
        old_successful_repeated_captures=sum(r["old_frozen_captures"] for r in records),
        old_successful_settings_retained=len(records)-len(excluded), old_successful_settings_excluded=len(excluded),
        excluded_repeated_successes=sum(r["old_frozen_captures"] for r in excluded),
        excluded_site_counts_by_axis=dict(axes), records=records, scope=SCOPE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-run", type=Path, required=True)
    parser.add_argument("--support-json", type=Path, required=True)
    parser.add_argument("--expected-result-sha256", required=True)
    parser.add_argument("--expected-support-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output; prior evidence is preserved")
    result = support_exclusion(args.original_run, args.support_json,
        expected_result_sha256=args.expected_result_sha256, expected_support_sha256=args.expected_support_sha256)
    with args.output.open("x") as stream:
        json.dump(result, stream, allow_nan=False)
    print(json.dumps({k: result[k] for k in ("status", "old_successful_sites",
        "old_successful_settings_retained", "old_successful_settings_excluded", "excluded_repeated_successes")}), flush=True)


if __name__ == "__main__":
    main()
