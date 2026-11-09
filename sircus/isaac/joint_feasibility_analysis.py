"""CPU shadow analysis of the original final PPO requests, without new shots.

Replays the saved finite IK answers, independently verifies FK, and runs the
external exact-action preflight. Native outcomes remain historical observations;
this is neither an improved-policy evaluation nor a replacement independent audit.
"""

import argparse
import copy
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np

from continuous_feasibility import canonical_hash
from expanded_pose_menu import endpoint_targets
from joint_feasibility import preflight
from prepare_assets import chain
from shot_benchmark_reach_audit import fit_details

VERSION = "sircus-joint-feasibility-shadow-v1"
SEEDS = (20261121, 20261122, 20261123)
GEOMETRY_RECEIPT = Path(__file__).resolve().parents[2] / "baselines/isaac-reach-ik-audit-20261106/summary.json"
SCOPE = [
    "Exact saved final-checkpoint requests and actual recorded home starts; no actor changes or action replacement.",
    "Saved finite IK answers are replayed, not solved again; independent FK and geometry/path proxy checks are recomputed.",
    "Native release, capture and clean labels remain original audited observations, not predictions for alternative actions.",
    "TEST is retrospective descriptive analysis only; no TEST outcome selects candidates, settings or checkpoints.",
    "All requested positions remain in the overall denominator. Conditional release success is reported separately.",
    "No native simulation, optimizer update, inference-time search, release permission or physical validation.",
]


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path, expected=None):
    raw = Path(path).read_bytes()
    require(expected is None or hashlib.sha256(raw).hexdigest() == expected,
            "Saved evidence SHA mismatch: " + Path(path).name)
    if str(path).endswith(".gz"):
        raw = gzip.decompress(raw)
    return json.loads(raw, parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Nonfinite JSON " + x)))


def contained(root, relative):
    require(isinstance(relative, str) and not Path(relative).is_absolute(), "Expected relative evidence path")
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), "Evidence path escaped its run directory")
    return path


def replay_request(scene, cell, action, row, home, links):
    """No outcomes enter preflight; they are attached only after the decision."""
    starts = {n: np.asarray(home[n], dtype=float) for n in ("Arm10", "Arm14")}
    saved = row["qualification"]
    targets = endpoint_targets(scene, cell, action)
    # Verify successful AND unresolved saved fits. qualify_action independently
    # checks successful endpoints too; numerical failure never proves impossible.
    for name, fit in saved.get("ik", {}).items():
        require(name in starts, "Unknown saved IK arm")
        fit_details(targets[name], fit, links[name], np.eye(4), name == "Arm14")

    calls = []

    def replay(target, robot_links, seed, normal_only):
        name = "Arm14" if normal_only else "Arm10"
        require(name in saved.get("ik", {}), "New layer requested an IK answer absent from original evidence")
        require(np.array_equal(np.asarray(target), targets[name]), "Replay changed endpoint objective")
        calls.append(name)
        return copy.deepcopy(saved["ik"][name])

    decision = preflight(scene, cell, action, links, starts, solver=replay,
                         solver_protocol="original-saved-finite-ik-answers-no-new-search-v1")
    original = bool(saved["accepted"] and row.get("path_proxy") and row["path_proxy"]["accepted"])
    require(bool(decision["accepted"]) == original, "External preflight differs from original admission")
    require(decision["qualification"]["accepted"] == saved["accepted"], "Endpoint qualification changed")
    require(decision["qualification"]["reason"] == saved["reason"], "Endpoint rejection reason changed")
    if saved["accepted"]:
        require(decision["path"]["reasons"] == row["path_proxy"]["reasons"], "Path rejection reasons changed")
    require(calls == list(saved.get("ik", {})), "Saved IK short-circuit sequence changed")
    released = row["invalid"] is None
    capture, clean = bool(row["route_success"]), row["route_quality"] == 2
    require(not released or original, "Original release bypassed geometry/path admission")
    require(not capture or released, "Rejected request labeled a capture")
    require(not clean or capture, "Clean request was not a capture")
    return dict(cell_id=cell["id"], bucket_xy_m=cell["bucket_xy_m"],
                preflight_accepted=original, preflight_status=decision["status"],
                preflight_reason=decision["reason"], native_invalid=row["invalid"],
                released=released, captured=capture, clean=clean,
                exact_action_unchanged=True, parity=True, binding=decision["binding"],
                decision_fingerprint=decision["fingerprint"])


def summarize(records):
    n = len(records)
    require(n > 0, "Empty analysis group")
    admitted = sum(r["preflight_accepted"] for r in records)
    released = sum(r["released"] for r in records)
    captures = sum(r["captured"] for r in records)
    clean = sum(r["clean"] for r in records)
    require(0 <= clean <= captures <= released <= admitted <= n, "Inconsistent admission/release accounting")
    return dict(requests=n, preflight_admitted=admitted, preflight_declined=n-admitted,
                execution_declined_after_preflight=admitted-released, released=released,
                released_failures=released-captures, captures=captures, clean=clean,
                preflight_admission_rate=admitted/n, release_rate=released/n,
                capture_rate=captures/n, clean_rate=clean/n,
                capture_given_release=captures/released if released else None,
                release_given_preflight=released/admitted if admitted else None,
                preflight_reasons=dict(sorted(Counter(r["preflight_reason"] for r in records if not r["preflight_accepted"]).items())),
                native_rejection_reasons=dict(sorted(Counter(r["native_invalid"] for r in records if not r["released"]).items())),
                parity_requests=sum(r["parity"] for r in records))


def analyze_run(root, published, experiment, splits, links):
    seed = published["seed"]
    run = root / "runs" / f"ppo-seed{seed}"
    audit_binding = published["independent_audit"]
    require(audit_binding["status"] == "passed", "Original independent audit has not passed")
    audit = read(Path(str(run) + ".audit.json"), audit_binding["sha256"])
    require(audit["status"] == "passed" and not audit["smoke_only"] and audit["method"] == "ppo"
            and audit["seed"] == seed and audit["paid_train_attempts"] == 10000,
            "Expected original audited production PPO run")
    for key in ("source", "split_manifest_sha256"):
        require(audit[key] == experiment[key], "Original audit experiment binding mismatch")
    progress = read(run / "progress.json", audit["source_progress_sha256"])
    binding = read(run / "experiment-binding.json", audit["source_binding_sha256"])
    contract = read(run / "contract.json", audit["source_contract_sha256"])
    require(binding["experiment_sha256"] == digest(root / "experiment.json"), "Experiment bytes changed")
    require(progress["status"] == "completed" and progress["train"]["attempts"] == 10000,
            "PPO production run incomplete")
    checkpoint = published["checkpoint"]
    checkpoint_path = contained(run, checkpoint["native_file"])
    require(digest(checkpoint_path) == checkpoint["sha256"], "Frozen model bytes changed")
    matches = [c for c in progress["checkpoints"] if c["training_attempts"] == 10000]
    require(len(matches) == 1, "Ambiguous prescribed final checkpoint")
    final = matches[0]
    proof = read(contained(run, final["proof_file"]), checkpoint["proof_sha256"])
    require(proof["no_optimizer_or_flight_search"] is True and proof["training_attempts"] == 10000
            and proof["checkpoint"]["sha256"] == checkpoint["sha256"]
            and proof["state_hash_before"] == proof["state_hash_after"] == proof["checkpoint"]["state_hash"],
            "Final frozen inference proof changed")
    require(proof["source"] == experiment["source"], "Proof source changed")
    records = {s: [] for s in ("train", "validation", "test")}
    receipts = {}
    begin = time.monotonic()
    for relative, expected in proof["source_trial_sha256"].items():
        require(audit["source_trial_sha256"][relative] == expected
                and progress["source_trial_sha256"][relative] == expected, "Final trial receipt differs from audit")
        packet = read(contained(run, relative), expected)
        require(packet["source"] == experiment["source"]
                and packet["split_manifest_sha256"] == experiment["split_manifest_sha256"]
                and packet["checkpoint"] == proof["checkpoint"], "Trial source/checkpoint changed")
        split = packet["mode"].removeprefix("eval-")
        require(split in records and packet["mode"] == "eval-" + split, "Non-final evaluation packet")
        require(len(packet["cells"]) == len(packet["actions"]) == len(packet["rows"]) == len(packet["enabled"]) == 35,
                "Invalid native farm packet length")
        expected_cells = {c["id"]: c for c in splits[split]}
        for i, (cell, action, row, enabled) in enumerate(zip(packet["cells"], packet["actions"], packet["rows"], packet["enabled"])):
            if not enabled:
                continue
            require(enabled is True and cell["split"] == split and cell["id"] in expected_cells,
                    "Unexpected active evaluation context")
            require(cell["bucket_xy_m"] == expected_cells[cell["id"]]["bucket_xy_m"]
                    and row["cell_id"] == cell["id"], "Evaluation position changed")
            home = {n: packet["home"]["measured_joint_rad"][n][i] for n in ("Arm10", "Arm14")}
            require(packet["home"]["path_eligible"][i] is True, "Original common home was not eligible")
            record = replay_request(contract["scene"], cell, action, row, home, links)
            record.update(trial_file=relative, slot=i)
            records[split].append(record)
        receipts[relative] = expected
    groups = {}
    for split, count in (("train", 35), ("validation", 35), ("test", 70)):
        rows = records[split]
        require(len(rows) == len({r["cell_id"] for r in rows}) == count, "Final evaluation layout coverage changed")
        summary = summarize(rows)
        original = final[split]
        for key in ("captures", "clean", "attempts", "rejected"):
            value = summary[{"attempts": "requests", "rejected": "preflight_declined"}.get(key, key)]
            # Native rejection also includes execution/settling after preflight.
            if key == "rejected":
                value += summary["execution_declined_after_preflight"]
            require(value == original[key], "Original final outcome total changed")
        groups[split] = summary
    return dict(seed=seed, method="ppo", checkpoint_sha256=checkpoint["sha256"],
                original_audit_sha256=audit_binding["sha256"],
                source_progress_sha256=audit["source_progress_sha256"],
                source_contract_sha256=audit["source_contract_sha256"],
                proof_sha256=checkpoint["proof_sha256"], trial_sha256=receipts,
                splits=groups, records=records, shadow_wall_s=time.monotonic()-begin,
                original_training_requests=progress["train"]["attempts"],
                original_training_rejected=progress["train"]["rejected"])


def analyze(root, assets, published_path, geometry_receipt=GEOMETRY_RECEIPT):
    root, assets = Path(root), Path(assets)
    published = read(published_path)
    require(published["schema"] == "sircus-shot-benchmark-final-comparison-v1"
            and published["completed_audited_runs"] == 6, "Expected completed frozen comparison package")
    experiment = read(root / "experiment.json", published["experiment_sha256"])
    splits = read(root / "split-manifest.json", published["split_manifest_sha256"])
    require(experiment["source"] == published["source"], "Published experiment source changed")
    asset_files = {n: assets / f for n, f in (("Arm10", "dropper.urdf"), ("Arm14", "board_arm.urdf"))}
    geometry = read(geometry_receipt)
    require(geometry["snapshot"]["source"] == experiment["source"], "Geometry receipt source mismatch")
    for name, path in asset_files.items():
        require(digest(path) == geometry["dependency_sha256"]["assets/" + path.name],
                "URDF differs from frozen geometry receipt: " + name)
    links = {n: chain(ET.fromstring(p.read_bytes())) for n, p in asset_files.items()}
    runs = [r for r in published["runs"] if r["method"] == "ppo"]
    require(sorted(r["seed"] for r in runs) == list(SEEDS), "Expected all three paired PPO seeds")
    require(all(r["independent_audit"]["source_contract_sha256"] == geometry["dependency_sha256"]["contract.json"]
                for r in runs), "Geometry receipt contract differs from PPO scene")
    begin = time.monotonic()
    results = [analyze_run(root, r, experiment, splits, links) for r in runs]
    value = dict(schema=VERSION, status="passed", source=experiment["source"],
                 experiment_sha256=published["experiment_sha256"],
                 split_manifest_sha256=published["split_manifest_sha256"],
                 published_comparison_sha256=digest(published_path),
                 geometry_receipt_sha256=digest(geometry_receipt),
                 assets_sha256={n: digest(p) for n, p in asset_files.items()},
                 diagnostic_source_sha256={p.name: digest(p) for p in (
                     Path(__file__), Path(__file__).with_name("joint_feasibility.py"))},
                 scope=SCOPE, runs=results, requests=sum(sum(s["requests"] for s in r["splits"].values()) for r in results),
                 shadow_wall_s=time.monotonic()-begin,
                 new_native_shots=0, optimizer_updates=0, changed_policy_actions=0,
                 simulator_started=False, release_authorized=False)
    value["fingerprint"] = canonical_hash(value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, type=Path, help="Read-only mirror of original bound evidence")
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--published", required=True, type=Path)
    parser.add_argument("--geometry-receipt", type=Path, default=GEOMETRY_RECEIPT)
    parser.add_argument("--output", required=True, type=Path, help="New analysis file, never inside the frozen experiment")
    args = parser.parse_args()
    require(not args.output.resolve().is_relative_to(args.benchmark.resolve()), "Analysis output must be separate from frozen inputs")
    require(not args.output.exists(), "Refuse to overwrite an analysis receipt")
    value = analyze(args.benchmark, args.assets, args.published, args.geometry_receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=value["status"], requests=value["requests"], fingerprint=value["fingerprint"])))


if __name__ == "__main__":
    main()
