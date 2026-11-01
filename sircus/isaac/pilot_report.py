"""Validate a completed paired pilot and export aggregates, never raw joint records."""

import argparse
import collections
import hashlib
import json
from pathlib import Path

LABELS = ["pilot_selected", "ppo_100", "ppo_1000", "warm_fixed"]


def finalize(run):
    run = Path(run)
    report = json.loads((run / "summary.json").read_text())
    if report["phase"] != "completed" or report["episodes"] != report["budget"]:
        raise ValueError("Incomplete pilot")
    pools = json.loads((run / "splits.json").read_text())
    ids = [s["id"] for p in pools.values() for s in p]
    if len(ids) != len(set(ids)):
        raise ValueError("Overlapping split IDs")
    test_ids = {s["id"] for s in pools["test"]}
    if len(test_ids) != 36:
        raise ValueError("Expected36 test scenes")
    evaluations = {}
    for label in LABELS:
        rows = json.loads((run / f"test-{label}.json").read_text())
        if len(rows) != 36 or {r["scene_id"] for r in rows} != test_ids:
            raise ValueError("Unmatched paired test layouts")
        evaluations[label] = {r["scene_id"]: r for r in rows}
        stats = report["evaluation"][label]
        if stats["captures"] != sum(r["success"] for r in rows):
            raise ValueError("Capture aggregate does not match saved trials")
        stats["invalid_reasons"] = dict(
            collections.Counter(
                (r.get("invalid") or "") + ":" + str(r.get("invalid_component"))
                for r in rows
                if r.get("invalid")
            )
        )
    training = [
        json.loads(s) for s in (run / "episodes.jsonl").read_text().splitlines()
    ]
    if len(training) != report["budget"]:
        raise ValueError("Truncated training data")
    counts = collections.Counter(r["difficulty"] for r in training)
    if set(counts) != {"near", "medium", "far"} or len(set(counts.values())) != 1:
        raise ValueError("Training was not balanced")
    report["verified_train_balance"] = dict(counts)
    report["paired_comparison"] = {}
    pilot = evaluations["pilot_selected"]
    for label in LABELS[1:]:
        old = evaluations[label]
        report["paired_comparison"][label] = {
            "pilot_only_captures": sum(
                pilot[i]["success"] and not old[i]["success"] for i in test_ids
            ),
            "reference_only_captures": sum(
                old[i]["success"] and not pilot[i]["success"] for i in test_ids
            ),
            "both_capture": sum(
                old[i]["success"] and pilot[i]["success"] for i in test_ids
            ),
            "neither_capture": sum(
                not old[i]["success"] and not pilot[i]["success"] for i in test_ids
            ),
        }
    report["contract"] = json.loads((run / "contract.json").read_text())
    report["artifact_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(run.iterdir())
        if p.is_file() and p.suffix in (".json", ".jsonl", ".pt")
    }
    if report["selected_sha256"] != report["artifact_sha256"]["selected.pt"]:
        raise ValueError("Selected checkpoint hash mismatch")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--output", required=True)
    parser.add_argument("--conclusion", required=True)
    args = parser.parse_args()
    report = finalize(args.run)
    report["conclusion"] = args.conclusion
    path = Path(args.output)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False))
    tmp.replace(path)
