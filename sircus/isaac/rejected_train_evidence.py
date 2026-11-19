"""Read-only extraction of every original seed-22 paid TRAIN rejection.

The unchanged successful-TRAIN collector first verifies the full 289-packet,
10,000-request history and original audit/model bindings. This module then reads
only that exact paid TRAIN namespace again to export rejected continuous actions
and their measured starts. It never imports a policy, solver or simulator and
never consumes held-out packets, ball trajectories, media or reward values.
"""

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys

import candidate_train_evidence as original

VERSION = "sircus-seed22-paid-train-rejections-v1"
NATIVE_REASONS = frozenset(("endpoints_not_settled", "setup_motion_not_quiet",
                            "setup_transit_incomplete"))
ACTION_FIELDS = frozenset(original.ACTION_NAMES + ("id", "bearing_rad", "board_x_m",
    "board_y_m", "tilt_deg", "azimuth_deg", "azimuth_offset_deg"))


def collect(benchmark, *, audit_sha, progress_sha, experiment_sha, split_sha,
            checkpoint_sha):
    """Verify all paid TRAIN evidence and return all original invalid requests.

    Original qualifications/fits are provenance only. The geometry analyzer must
    use measured starts, not supply those saved endpoint answers to its solver.
    A native motion/settling rejection can already be geometrically admitted;
    offline geometry cannot establish that the original dynamic failure is fixed.
    """
    successful = original.collect(benchmark, audit_sha=audit_sha,
        progress_sha=progress_sha, experiment_sha=experiment_sha,
        split_sha=split_sha, checkpoint_sha=checkpoint_sha)
    run = Path(benchmark) / "runs" / f"ppo-seed{original.SEED}"
    cells = {cell["id"]: cell for cell in successful["cells"]}
    rows, reasons, keys = [], Counter(), set()
    for relative, expected in sorted(successful["train_trial_sha256"].items()):
        original.require(original.TRAIN_PATH.fullmatch(relative), "Unexpected paid TRAIN path")
        packet = original.read(original._inside(run, relative), expected)
        original.require(packet["mode"] == "train"
            and packet["source"] == successful["source"]
            and packet["split_manifest_sha256"] == split_sha,
            "Paid TRAIN packet changed during rejection collection")
        for slot, enabled in enumerate(packet["enabled"]):
            if not enabled:
                continue
            row = packet["rows"][slot]
            reason = row["invalid"]
            if reason is None:
                continue
            original.require(isinstance(reason, str) and bool(reason)
                and row["route_success"] is False,
                "A rejected TRAIN request cannot be a capture")
            cell = original._cell(packet["cells"][slot])
            original.require(cell == cells.get(cell["id"])
                and row["cell_id"] == cell["id"] and row["split"] == "train",
                "Rejected request left original TRAIN geometry")
            action = copy.deepcopy(packet["actions"][slot])
            original.require(isinstance(action, dict)
                and set(action) <= ACTION_FIELDS
                and isinstance(action.get("id"), str) and bool(action["id"]),
                "Missing rejected continuous action")
            original._vector([action[name] for name in original.ACTION_NAMES], 5,
                             "rejected continuous action")
            for name in original.ACTION_NAMES:
                original.require(row[name] == action[name], "Rejected row/action settings differ")
            measured = original._joints({arm: packet["home"]["measured_joint_rad"][arm][slot]
                                         for arm in original.ARMS})
            eligible = packet["home"]["path_eligible"][slot]
            original.require(type(eligible) is bool, "Invalid original home path eligibility")
            qualification, path = row.get("qualification"), row.get("path_proxy")
            original.require(qualification is None or isinstance(qualification, dict),
                             "Invalid original qualification provenance")
            original.require(path is None or isinstance(path, dict),
                             "Invalid original path provenance")
            key = relative + "#" + str(slot)
            original.require(key not in keys, "Duplicate rejected paid TRAIN request")
            keys.add(key)
            reasons[reason] += 1
            rows.append(dict(rejection_key=key, trial_file=relative,
                trial_sha256=expected, slot=slot, cell=copy.deepcopy(cell), action=action,
                recorded_starts_rad=measured, original_reason=reason,
                home_path_eligible=eligible, qualification=copy.deepcopy(qualification),
                path_proxy=copy.deepcopy(path), rejection_stage=(
                    "native_motion_or_settling" if reason in NATIVE_REASONS else "geometry_or_path")))
    totals = successful["training_totals"]
    expected_reasons = {name: count for name, count in totals["failures"].items()
                        if name not in ("captured", "external_contact", "released_miss")}
    original.require(len(rows) == len(keys) == totals["rejected"]
        and dict(reasons) == expected_reasons, "Incomplete original TRAIN rejection population")
    successful_raw = json.dumps(successful, separators=(",", ":"), allow_nan=False) + "\n"
    result = dict(schema=VERSION, seed=original.SEED, source=successful["source"],
        source_bindings=copy.deepcopy(successful["source_bindings"]),
        scene=copy.deepcopy(successful["scene"]), cells=copy.deepcopy(successful["cells"]),
        training_totals=copy.deepcopy(totals), train_trial_sha256=copy.deepcopy(successful["train_trial_sha256"]),
        rejected_count=len(rows), rejection_reasons=dict(sorted(reasons.items())),
        rejection_stages=dict(sorted(Counter(row["rejection_stage"] for row in rows).items())),
        rejections=rows,
        successful_train_export_sha256=hashlib.sha256(successful_raw.encode()).hexdigest(),
        successful_train_export_fingerprint=successful["fingerprint"],
        evaluated_trial_files_read=0, new_native_shots=0, optimizer_updates=0,
        historical_endpoint_witnesses_for_solver=False, release_authorized=False,
        scope="Every original invalid paid TRAIN proposal from seed 20261122; exact actions and measured starts, with original geometry/native rejection provenance. No held-out trial, trajectory, reward, media, model inference, geometry solve, training or new outcome consumed. Stored old IK answers are provenance only and must not seed the primary geometry analysis.")
    result["fingerprint"] = original.canonical_hash(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    for name in ("audit", "progress", "experiment", "split", "checkpoint"):
        parser.add_argument("--"+name+"-sha", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = None
    if args.output != "-":
        output = Path(args.output).resolve()
        original.require(not output.is_relative_to(args.benchmark.resolve()),
                         "Collector output must be separate from frozen evidence")
        original.require(not output.exists(), "Refuse to overwrite rejected TRAIN evidence")
    value = collect(args.benchmark, **{name+"_sha": getattr(args, name+"_sha")
                                      for name in ("audit", "progress", "experiment", "split", "checkpoint")})
    text = json.dumps(value, separators=(",", ":"), allow_nan=False)+"\n"
    if output is None:
        sys.stdout.write(text)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as handle:
            handle.write(text)


if __name__ == "__main__":
    main()
