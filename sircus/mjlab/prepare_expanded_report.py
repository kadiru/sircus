"""Append compact expansion progress to a derived viewer report, CPU/read-only.

This never modifies an experiment, checkpoint, solver route, or active model.
The base display report remains authoritative for all older study evidence.
"""

import argparse
import copy
import datetime
import json
from pathlib import Path
import time

ACTIVE_POLICY = "Previous bank remains active; expanded model not deployed"
RAW_FIELDS = frozenset(
    (
        "test_rows",
        "rows",
        "shots",
        "chunks",
        "chunk_arrays",
        "trajectory",
        "q10",
        "q14",
        "mask",
        "values",
        "candidate_arrays",
    )
)
STATS = (
    "attempts",
    "captures",
    "clean",
    "invalid",
    "board_bounces",
    "mean_reward",
    "native_releases",
    "eligible_attempts",
    "conditional_capture_rate",
    "positions_with_capture",
    "positions_all_repeats_captured",
    "unresolved_positions",
    "layouts",
    "repeats",
    "trials",
    "policy_weight_sha256",
    "action_protocol",
    "search_at_inference",
    "bank_sha256",
    "action_bounds",
    "scope",
)
SEARCH_FIELDS = (
    "status",
    "phase",
    "elapsed_s",
    "timestamp_utc",
    "contract_fingerprint",
    "old_bank_study_fingerprint",
    "old_bank_sha256",
    "native_search_attempts",
    "confirmation_attempts",
    "prior_native_search_attempts",
    "prior_confirmation_attempts",
    "effort",
    "geometry_proposals",
    "additions",
    "original_templates",
    "expanded_templates",
    "expanded_bank_sha256",
    "proposal_bank_sha256",
    "bounds",
    "waiting_for_4090",
    "scope",
)
TRAIN_FIELDS = (
    "status",
    "phase",
    "elapsed_s",
    "timestamp_utc",
    "contract_fingerprint",
    "proposal_bank_sha256",
    "old_proposal_bank_sha256",
    "train_episodes",
    "native_train_releases",
    "updates",
    "selected_update",
    "checkpoint_sha256",
    "policy_weight_sha256",
    "geometry_progress",
    "train_sampling",
    "gpu_queue",
    "scope",
    "development_note",
)


def compact_value(value):
    if isinstance(value, dict):
        return {
            key: compact_value(item)
            for key, item in value.items()
            if key not in RAW_FIELDS
        }
    if isinstance(value, list):
        return [compact_value(item) for item in value]
    return copy.deepcopy(value)


def select(value, keys):
    return {key: compact_value(value[key]) for key in keys if key in value}


def optional_json(path):
    """A missing or transiently unreadable progress report is not a crash."""
    try:
        value = json.loads(Path(path).read_text())
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def compact_search(report):
    if report is None:
        return None
    result = select(report, SEARCH_FIELDS)
    cells = report.get("cells") or report.get("development_layouts") or []
    result["cells"] = []
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        row = select(
            cell,
            ("id", "bucket_xy_m", "original_test_index", "role", "unresolved_reason", "retained_source_captures"),
        )
        row["search"] = select(cell.get("search", {}), STATS)
        row["fresh_confirmation"] = select(cell.get("fresh_confirmation", {}), STATS)
        promoted = cell.get("promotion")
        row["promotion"] = (
            select(
                promoted,
                (
                    "source_trial_id",
                    "source_target_id",
                    "source_phase",
                    "development_phase",
                    "action_bounds_version",
                    "relative_board_xy_m",
                    "height_m",
                    "tilt_deg",
                    "gap_m",
                    "source_route_quality",
                    "confirmation",
                    "qualification",
                ),
            )
            if isinstance(promoted, dict)
            else None
        )
        result["cells"].append(row)
    for key in ("native_search_statistics", "confirmation_statistics"):
        if isinstance(report.get(key), dict):
            result[key] = select(report[key], STATS)
    return result


def compact_training(report, contract=None):
    if report is None:
        return None
    result = select(report, TRAIN_FIELDS)
    result["geometry"] = {
        label: select(
            group,
            (
                "queries",
                "candidates_per_query",
                "eligible_candidates",
                "queries_with_candidate",
                "unresolved_queries",
            ),
        )
        for label, group in report.get("geometry", {}).items()
        if isinstance(group, dict)
    }
    result["learning_curve"] = []
    for update in report.get("learning_curve", []):
        if isinstance(update, dict):
            row = select(update, ("update",) + STATS)
            if isinstance(update.get("validation"), dict):
                row["validation"] = select(update["validation"], STATS)
            result["learning_curve"].append(row)
    if isinstance(report.get("initial_validation"), dict):
        result["initial_validation"] = select(report["initial_validation"], STATS)
    frozen = report.get("frozen_evaluation")
    if isinstance(frozen, dict):
        result["frozen_evaluation"] = select(frozen, STATS)
        result["frozen_evaluation"]["comparison"] = {
            label: select(stats, STATS)
            for label, stats in frozen.get("comparison", {}).items()
            if isinstance(stats, dict)
        }
    if isinstance(report.get("development_evaluation"), dict):
        result["development_evaluation"] = {
            label: select(stats, STATS)
            for label, stats in report["development_evaluation"].items()
            if isinstance(stats, dict)
        }
    result["paired_comparisons"] = compact_value(report.get("paired_comparisons", {}))
    if contract is not None:
        result["study_plan"] = dict(
            **select(
                contract,
                (
                    "batch_size",
                    "ppo_updates",
                    "test_repeats",
                    "checkpoint_selection",
                    "development_note",
                    "fairness",
                    "scope",
                    "evaluation_methods",
                ),
            ),
            **{
                label: len(contract.get(label, []))
                for label in (
                    "train_layouts",
                    "validation_layouts",
                    "test_layouts",
                    "development_layouts",
                )
            },
        )
    return result


def expansion_status(search, training):
    if search is None:
        return "not_started"
    if search.get("status") in ("failed", "error"):
        return "search_failed"
    if search.get("status") != "completed":
        return "search_running"
    if training is None:
        return "awaiting_training"
    if training.get("status") in ("failed", "error"):
        return "training_failed"
    return "completed" if training.get("status") == "completed" else "training_running"


def prepare(base_display, expansion_study, training_study, output):
    base_display, expansion_study, training_study, output = map(
        Path, (base_display, expansion_study, training_study, output)
    )
    protected = (
        base_display,
        expansion_study / "report.json",
        expansion_study / "contract.json",
        training_study / "report.json",
        training_study / "contract.json",
    )
    if output.resolve() in {path.resolve() for path in protected} or any(
        output.resolve().is_relative_to(study.resolve())
        for study in (expansion_study, training_study)
    ):
        raise ValueError(
            "Write a separate derived display report; never overwrite source evidence"
        )
    base = optional_json(base_display)
    if base is None:
        raise ValueError("A valid existing base display report is required")
    search = compact_search(optional_json(expansion_study / "report.json"))
    training = compact_training(
        optional_json(training_study / "report.json"),
        optional_json(training_study / "contract.json"),
    )
    base["expansion_evaluation"] = dict(
        search=search,
        training=training,
        status=expansion_status(search, training),
        active_policy=ACTIVE_POLICY,
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_scope="Old twelve unresolved test positions are now targeted development. New frozen test positions remain separate; no per-scene outcome search at evaluation.",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(base, indent=2, allow_nan=False))
    tmp.replace(output)
    return base


def watch(base_display, expansion_study, training_study, output, seconds=0):
    if not 0 <= seconds <= 1800:
        raise ValueError("Watch duration must be between 0 and 1800 seconds")
    start = time.monotonic()
    while True:
        result = prepare(base_display, expansion_study, training_study, output)
        if result["expansion_evaluation"]["status"] == "completed":
            return result
        remaining = seconds - (time.monotonic() - start)
        if remaining <= 0:
            return result
        time.sleep(min(5, remaining))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("base-display", "expansion-study", "training-study", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--watch-seconds", type=float, default=0)
    args = parser.parse_args()
    if not 0 <= args.watch_seconds <= 1800:
        parser.error("Bounded --watch-seconds must be between 0 and 1800")
    watch(
        args.base_display,
        args.expansion_study,
        args.training_study,
        args.output,
        args.watch_seconds,
    )
