"""Read-only aggregate comparison of completed, matched privileged studies."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
from learning_contract import outcome_summary

LABELS = ("test", "test_warm_fixed", "test_midpoint")


def load_run(path):
    path = Path(path)
    summary = json.loads((path / "summary.json").read_text())
    splits = json.loads((path / "splits.json").read_text())
    expected = summary["contract"]["test_pool"]
    if summary["training_episodes"] != summary["contract"]["run_episodes"]:
        raise ValueError("Incomplete training budget")
    ids = [s["id"] for s in splits["test"]]
    if len(ids) != expected or len(set(ids)) != expected:
        raise ValueError("Incomplete or duplicate test split")
    records = {}
    for label in LABELS:
        rows = json.loads((path / (label + ".json")).read_text())
        if [r["scene_id"] for r in rows] != ids:
            raise ValueError("Evaluation rows do not match the ordered test split")
        computed = outcome_summary(rows)
        recorded = summary["evaluation"][label]
        for key in ["trials", "captures", "invalid_endpoints"]:
            if computed[key] != recorded[key]:
                raise ValueError("Summary disagrees with evaluation rows")
        records[label] = rows
    return summary, splits, records


def diagnose(rows):
    actions = np.array([r["action"] for r in rows])
    groups = {}
    for name in ["near", "medium", "far"]:
        group = [r for r in rows if r["difficulty"] == name]
        groups[name] = {
            **outcome_summary(group),
            "mean_action": np.mean([r["action"] for r in group], axis=0).tolist()
            if group
            else None,
            "failures": {
                "endpoint_ik": sum(bool(r.get("invalid")) for r in group),
                "no_board_bounce": sum(
                    not r.get("invalid") and not r.get("board_bounce") for r in group
                ),
                "no_whole_ball_entry": sum(
                    not r.get("invalid")
                    and r.get("board_bounce", False)
                    and not r.get("whole_ball_entry")
                    for r in group
                ),
                "retention": sum(
                    not r.get("invalid")
                    and r.get("board_bounce", False)
                    and r.get("whole_ball_entry", False)
                    and not r["success"]
                    for r in group
                ),
            },
        }
    return {
        "action_min": actions.min(0).tolist(),
        "action_max": actions.max(0).tolist(),
        "action_std": actions.std(0).tolist(),
        "action_units": ["degrees", "degrees", "metres"],
        "ik_failures_by_arm": {
            name: sum(r.get("invalid_component") == name for r in rows)
            for name in ["Arm10", "Arm14"]
        },
        "groups": groups,
    }


def paired(a, b):
    return {
        "a_only_capture": sum(x["success"] and not y["success"] for x, y in zip(a, b)),
        "b_only_capture": sum(y["success"] and not x["success"] for x, y in zip(a, b)),
        "both_capture": sum(x["success"] and y["success"] for x, y in zip(a, b)),
        "both_miss": sum(not x["success"] and not y["success"] for x, y in zip(a, b)),
    }


def compare(
    paths, recommendation="Review required before starting another training stage."
):
    loaded = [load_run(p) for p in paths]
    first = loaded[0]
    for other in loaded[1:]:
        if other[0]["training_episodes"] != first[0]["training_episodes"]:
            raise ValueError("Unequal training budgets")
        if other[0]["contract"]["fingerprint"] != first[0]["contract"]["fingerprint"]:
            raise ValueError("Incompatible experiment contracts")
        if other[1]["test"] != first[1]["test"]:
            raise ValueError("Test layouts differ")
    runs = []
    for path, (s, _, records) in zip(paths, loaded):
        checkpoint = Path(path) / "selected.pt"
        runs.append(
            {
                "run": s["run"],
                "gpu": s["contract"]["gpu"],
                "training_layouts": s["unique_training_layouts"],
                "training_episodes": s["training_episodes"],
                "selected_episode": s["selected_episode"],
                "elapsed_hours": s["elapsed_s"] / 3600,
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                if checkpoint.exists()
                else None,
                "evaluation": s["evaluation"],
                "learning_curve": s["learning_curve"],
                "diagnostics": diagnose(records["test"]),
                "paired_vs_fixed": paired(records["test"], records["test_warm_fixed"]),
            }
        )
    return {
        "schema": "sircus-learning-comparison-v1",
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "source_runs": [r["run"] for r in runs],
        "contract_fingerprint": first[0]["contract"]["fingerprint"],
        "runs": runs,
        "paired_learned": paired(loaded[0][2]["test"], loaded[1][2]["test"])
        if len(loaded) == 2
        else None,
        "scope": "Frozen privileged-state shots; one seed, different GPUs. No physical or point-cloud policy validation.",
        "recommendation": recommendation,
        "recommendation_status": "Post-study engineering judgment, not a preregistered statistical pass threshold.",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs=2)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--recommendation",
        default="Review required before starting another training stage.",
    )
    args = p.parse_args()
    report = compare(args.runs, args.recommendation)
    target = Path(args.output)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    tmp.replace(target)


if __name__ == "__main__":
    main()
