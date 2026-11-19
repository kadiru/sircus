"""Read-only fixed-bucket learning evidence for the existing mjlab viewer.

The output is a separate derived display report. No checkpoint is selected,
policy activated, experiment mutated, or simulator started by this module.
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

KEY = "window_study_evaluation"
TERMINAL = {"completed", "failed", "cancelled", "blocked_geometry", "blocked_storage", "blocked_resources"}
STATS = ("attempts", "requests", "captures", "clean", "invalid", "native_releases", "board_bounces",
         "mean_reward", "mean_learning_reward", "eligible_attempts", "conditional_capture_rate")
SETTINGS = ("version", "seed", "bucket_xy_m", "board_xy_m", "context_bounds", "gap_bounds", "contexts",
            "budgets", "learning", "sac", "adaptive", "selection", "evaluation", "noise", "gate_scope",
            "reward", "scope", "anchor_selection")


def statistics(raw):
    """Keep requested and actually released denominators separate; never infer releases."""
    value = select(raw, STATS) if isinstance(raw, dict) else {}
    n = value.get("attempts", value.get("requests"))
    k = value.get("captures")
    if isinstance(n, int) and not isinstance(n, bool) and isinstance(k, int) and not isinstance(k, bool) and 0 <= k <= n:
        if n:
            z = 1.959963984540054
            center = (k / n + z*z/(2*n)) / (1 + z*z/n)
            half = z * math.sqrt((k/n)*(1-k/n)/n + z*z/(4*n*n)) / (1 + z*z/n)
            value["descriptive_95pct_wilson"] = [max(0., center-half), min(1., center+half)]
            value["capture_rate"] = k / n
        else:
            value["descriptive_95pct_wilson"] = None
            value["capture_rate"] = None
    value["uncertainty_scope"] = "Descriptive binomial interval; repeated shots share contexts and do not establish independent scene generalization"
    return value


def confirmations(raw, parent_fingerprint):
    """Compact a separate engineering repeat, never merge its counts into learning."""
    if not isinstance(raw, dict):
        return None
    value = select(raw, ("version", "status", "scope", "parent_contract_fingerprint", "reason"))
    value["recipes"] = []
    value["audit"] = select(raw.get("audit", {}), ("status", "checks_count", "reason"))
    if not isinstance(parent_fingerprint, str) or not parent_fingerprint or raw.get("parent_contract_fingerprint") != parent_fingerprint:
        value.update(status="failed", reason="Window confirmation parent fingerprint does not match this learning study")
        value["audit"] = dict(status="failed", reason=value["reason"])
        return value
    for row in raw.get("recipes", []):
        recipe = select(row, ("id", "context_id", "board_height_m", "tilt_deg", "gap_m", "release_center_height_m", "distance_m"))
        recipe.update({stage:statistics(row.get(stage)) for stage in ("nominal", "fresh", "total")})
        value["recipes"].append(recipe)
    return value


def frozen_test_examples(study, raw, audit):
    """Read verified saved traces for a labeled illustration, never rerun a model."""
    import numpy as np
    study = Path(study).resolve()
    if raw.get("status") != "completed" or audit.get("status") != "passed" or not raw.get("contract_fingerprint") or audit.get("contract_fingerprint") != raw.get("contract_fingerprint"):
        return [], ["Frozen-test illustrations await a completed study and matching passed audit."]
    manifest = raw.get("evidence_sha256", {})
    def verified(name):
        path = (study / name).resolve()
        if not path.is_relative_to(study) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get(name):
            raise ValueError("Missing or mismatched evidence hash: " + str(name))
        return path
    try:
        rows = [json.loads(line) for line in verified("records.jsonl").read_text().splitlines() if line.strip()]
        nominal = [r for r in rows if r.get("role") == "test" and r.get("phase") == "nominal" and r.get("repeat_index") == 0 and r.get("native_release") is True and not r.get("invalid")]
        anchor = next((r for r in nominal if r.get("method") == "adaptive_surrogate" and r.get("route_success") is True), None)
        if anchor is None:
            return [], ["No native nominal frozen-surrogate TEST capture is available for the selected illustration."]
        selected = [anchor]
        for method in ("sac", "uniform_bounded"):
            row = next((r for r in nominal if r.get("method") == method and r.get("context_id") == anchor.get("context_id")), None)
            if row is not None:
                selected.append(row)
        examples = []
        for row in selected[:3]:
            with np.load(verified(row["native_cache_file"]), allow_pickle=False) as data:
                points = data["position_m"]
                ids = list(map(str, data["record_ids"]))
                index = row["native_cache_row"]
                if points.shape != (960, len(ids), 3) or not np.isfinite(points).all() or not isinstance(index, int) or not 0 <= index < len(ids) or ids[index] != row["id"]:
                    raise ValueError("Native trace shape, values or record association mismatch")
                trajectory = points[::3, index].tolist()
            examples.append(dict(**select(row, ("context_id", "gap_m", "route_success", "method", "role", "phase")),
                record_id=row["id"], position_m=trajectory, sample_dt_s=3/240,
                selection_scope="Illustrative outcome-selected TEST context: first surrogate nominal capture with matching native SAC/uniform traces; aggregate metrics remain authoritative",
                scope="Saved native positions verified against the original evidence manifest; no new inference or physics"))
        warnings = [] if any(r["method"] == "sac" for r in selected) else ["The matching SAC TEST action was not natively released; no SAC trajectory was fabricated."]
        return examples, warnings
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        return [], ["Frozen-test illustration unavailable: " + str(error)]


def compact(raw):
    if not isinstance(raw, dict):
        return None
    result = select(raw, ("version", "status", "phase", "contract_fingerprint", "counts", "scope", "reason", "elapsed_s"))
    result["settings"] = select(raw.get("settings", {}), SETTINGS)
    result["active_policy"] = "Read-only study; the existing interactive solver and previous policies remain unchanged"
    result["method_scope"] = dict(
        adaptive_surrogate="Supervised binary capture predictor; uncalibrated probability, frozen 129-gap prediction grid",
        sac="One-step terminal SAC; shaped-reward objective plus entropy, frozen mean-gap action; no next-state bootstrap",
        uniform_bounded="No training; bounded random release-gap selection")
    def context(row):
        return select(row, ("id", "context_id", "role", "board_height_m", "tilt_deg", "status", "eligible", "reason", "prepared", "board_admitted"))
    prepared = raw.get("prepared_contexts", [])
    result["prepared_contexts"] = ({key:[context(r) for r in rows] for key,rows in prepared.items() if isinstance(rows,list)}
        if isinstance(prepared,dict) else [context(r) for r in prepared])
    result["learning_curve"] = [dict(method=r.get("method"), milestone=r.get("milestone"),
        validation=statistics(r.get("validation"))) for r in raw.get("learning_curve", [])]
    result["test"] = {name: statistics(value) for name, value in raw.get("test", {}).items()
        if name in result["method_scope"]}
    result["test_contexts"] = []
    for row in raw.get("test_contexts", []):
        value = select(row, ("context_id", "board_height_m", "tilt_deg"))
        value["methods"] = {}
        for name, method in row.get("methods", {}).items():
            if name not in result["method_scope"]:
                continue
            value["methods"][name] = dict(gap_m=method.get("gap_m"),
                **{stage:statistics(method.get(stage)) for stage in ("nominal", "fresh", "total")})
        result["test_contexts"].append(value)
    result["observed_windows"] = [dict(**select(row, ("context_id", "scope")),
        points=[select(p, ("gap_m", "attempts", "captures", "native_releases")) for p in row.get("points", [])],
        observed_hit_runs=[select(r, ("low_observed_gap_m", "high_observed_gap_m", "points", "lower_tested_nonhit_gap_m", "upper_tested_nonhit_gap_m"))
            for r in row.get("observed_hit_runs", [])]) for row in raw.get("observed_windows", [])]
    result["trace_examples"] = []
    for row in raw.get("trace_examples", [])[:12]:
        points = row.get("position_m", [])
        if not isinstance(points, list) or not 2 <= len(points) <= 5000:
            continue
        if not all(isinstance(p, list) and len(p) == 3 and all(isinstance(v, (int,float)) and math.isfinite(v) for v in p) for p in points):
            continue
        result["trace_examples"].append(dict(**select(row, ("context_id", "gap_m", "route_success", "record_id", "method", "role", "phase", "sample_dt_s", "selection_scope")),
            position_m=points, scope="Recorded native ball positions supplied by the study; no analytic reconstruction"))
    result["selected_checkpoints"] = {name:select(value, ("method", "milestone", "path", "sha256", "weight_hash", "training_requests"))
        for name,value in raw.get("selected_checkpoints", {}).items() if name in result["method_scope"] and isinstance(value,dict)}
    result["storage"] = select(raw.get("storage", {}), ("status", "free_gb", "minimum_free_gb", "bytes", "scope"))
    result["audit"] = select(raw.get("audit", {}), ("status", "contract_fingerprint", "checks_count", "reason", "limitations"))
    result["window_confirmations"] = confirmations(raw.get("window_confirmations"), raw.get("contract_fingerprint"))
    return result


def prepare(base_display, study, output, *, confirmation_study=None):
    base_display, study, output = map(Path, (base_display, study, output))
    if output.resolve() == base_display.resolve() or output.resolve().is_relative_to(study.resolve()):
        raise ValueError("Write a separate derived report outside original evidence")
    if confirmation_study is not None:
        confirmation_study = Path(confirmation_study)
        if confirmation_study.resolve().is_relative_to(study.resolve()) or study.resolve().is_relative_to(confirmation_study.resolve()):
            raise ValueError("Confirmation evidence must be separate from the original study")
        if output.resolve().is_relative_to(confirmation_study.resolve()):
            raise ValueError("Derived report must not overwrite confirmation evidence")
    base = optional_json(base_display)
    if not isinstance(base, dict):
        raise ValueError("A valid existing base display report is required")
    result = copy.deepcopy(base)
    raw = optional_json(study / "report.json")
    value = compact(raw)
    result[KEY] = value
    if value is not None:
        if confirmation_study is not None:
            external = optional_json(confirmation_study / "report.json")
            if isinstance(external, dict):
                external = copy.deepcopy(external)
                audit = optional_json(confirmation_study / "audit-window-confirmation.json")
                if audit is not None:
                    external["audit"] = audit
                value["window_confirmations"] = confirmations(external, raw.get("contract_fingerprint"))
        audit = optional_json(study / "audit-window.json") or raw.get("audit", {})
        value["audit"] = select(audit, ("status", "contract_fingerprint", "checks_count", "reason", "limitations"))
        if audit.get("status") == "passed" and audit.get("contract_fingerprint") != raw.get("contract_fingerprint"):
            value["audit"].update(status="failed", reason="Audit fingerprint does not match this study")
        examples, warnings = frozen_test_examples(study, raw, value["audit"])
        ids = {r["record_id"] for r in examples}
        value["trace_examples"] = (examples + [r for r in value["trace_examples"] if r.get("record_id") not in ids])[:12]
        value["trace_warnings"] = warnings
        value["display_updated_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    # Fail before replacing the last valid display if any supplied evidence is nonfinite.
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(payload)
    temporary.replace(output)
    return result


def watch(base_display, study, output, seconds=0, *, confirmation_study=None):
    if not math.isfinite(seconds) or not 0 <= seconds <= 30000:
        raise ValueError("Bounded watch must be 0–30,000 seconds")
    started = time.monotonic()
    while True:
        result = prepare(base_display, study, output, confirmation_study=confirmation_study)
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
    parser.add_argument("--confirmation-study", type=Path)
    args = parser.parse_args()
    watch(args.base_display, args.study, args.output, args.watch_seconds, confirmation_study=args.confirmation_study)
