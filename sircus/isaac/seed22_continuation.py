"""Explicitly authorized, bank-free continuation of the original seed-22 PPO.

Importing this entry starts nothing. The parent, optimizer, original physical
helpers, inputs and fresh split are bound before the sole simulator start.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

PARENT_SHA = "57d7658e9ed2f72ad09cf4e8523ab497ccbe0708364f1b8ec4a4c5c1aef1fe9f"
GPU_UUID = "GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4"
SOURCE = Path(__file__).resolve().parent
WORKSPACE = Path("/workspace")


def canonical(path):
    p = Path(path)
    require(p.is_absolute() and str(p.resolve()) == str(p)
        and not any(x.is_symlink() for x in (p, *p.parents)), "Noncanonical or linked path")
    return p


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024**2), b""):
            h.update(part)
    return h.hexdigest()


def read(path):
    def pairs(items):
        result = {}
        for k, v in items:
            require(k not in result, "Duplicate JSON key")
            result[k] = v
        return result
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def inputs(path, expected_sha256):
    require(digest(path) == expected_sha256, "Continuation manifest bytes changed")
    manifest = read(path)
    require(manifest["schema"] == "sircus-seed22-continuation-inputs-v1", "Unknown inputs")
    for role, item in manifest["files"].items():
        p = Path(item["path"])
        require(p.is_absolute() and p.is_file() and not any(x.is_symlink() for x in (p, *p.parents))
                and str(p.resolve()) == str(p) and digest(p) == item["sha256"], "Input changed: " + role)
    source_run = canonical(manifest["source_run"])
    assets = canonical(manifest["assets"])
    require(source_run == WORKSPACE/"output/expanded-training-20261102-v4" and assets == WORKSPACE/"assets",
            "Original native scene/assets source changed")
    for role, name in (("source_contract","contract.json"),("source_result","result.json"),
                       ("source_menu","expanded-menu-qualified.json"),("camera","selected-camera.json")):
        require(Path(manifest["files"][role]["path"]) == source_run/name, "Native source path mismatch")
    require(Path(manifest["files"]["scene"]["path"]) == SOURCE/"scene.json", "Scene input path mismatch")
    require(Path(manifest["files"]["dropper_urdf"]["path"]) == assets/"dropper.urdf"
        and Path(manifest["files"]["board_urdf"]["path"]) == assets/"board_arm.urdf"
        and Path(manifest["files"]["poses"]["path"]) == assets/"poses.json", "URDF/pose path mismatch")
    actual_assets = {str(p.relative_to(assets)) for p in assets.rglob("*") if p.is_file()}
    declared_assets = {k[len("asset:"):] for k in manifest["files"] if k.startswith("asset:")}
    require(actual_assets == declared_assets and all(Path(manifest["files"]["asset:"+name]["path"]) == assets/name
        for name in actual_assets), "Asset membership or path changed")
    target = canonical(manifest["output"])
    require(target.parent == WORKSPACE/"output" and target.name.startswith("seed22-continuation-"), "Unexpected output path")
    protected = [canonical(p) for p in manifest["protected_roots"]]
    require({source_run, assets, SOURCE.parent.parent, WORKSPACE/"output/sweep-ppo-10k-20261106-v2"}.issubset(set(protected)),
            "Missing protected original roots")
    require(all(not target.is_relative_to(p) and not p.is_relative_to(target) for p in protected), "Output overlaps protected evidence")
    expected = manifest["source_sha256"]
    require(set(expected) == {p.name for p in SOURCE.glob("*.py")}, "Source membership changed")
    require(all(digest(SOURCE / name) == sha for name, sha in expected.items()), "Source bytes changed")
    from seed22_continuation_contract import validate_splits, validate_protocol
    from shot_benchmark_contract import validate_fingerprint, make_protocol
    experiment = read(manifest["files"]["experiment"]["path"])
    old_splits = read(manifest["files"]["original_splits"]["path"])
    exclusions = read(manifest["files"]["exclusions"]["path"])
    split = read(manifest["files"]["splits"]["path"])
    validate_splits(split, original_train=old_splits["train"], excluded_coordinates=exclusions["coordinates"])
    protocol = manifest["protocol"]
    validate_protocol(protocol)
    require(protocol["source_sha256"] == expected, "Protocol source changed")
    require(protocol["geometry_layer_sha256"] == expected["continuous_geometry_layer.py"], "Geometry layer/source binding mismatch")
    require(protocol["parent_checkpoint_sha256"] == manifest["files"]["parent"]["sha256"]
        and protocol["original_experiment_sha256"] == manifest["files"]["experiment"]["sha256"]
        and protocol["original_split_sha256"] == manifest["files"]["original_splits"]["sha256"]
        and protocol["exclusions_fingerprint"] == split["exclusions_fingerprint"], "Protocol input bindings changed")
    require(experiment["protocol"] == make_protocol(), "Original experiment changed")
    contract = read(manifest["files"]["source_contract"]["path"])
    require(contract["fingerprint"] == experiment["source"]["geometry_fingerprint"]
        and read(manifest["files"]["scene"]["path"]) == contract["scene"], "Scene or original geometry changed")
    require(manifest["files"]["parent"]["sha256"] == PARENT_SHA, "Wrong parent checkpoint")
    require(manifest["runtime"] == dict(python="3.11.13", torch="2.7.0+cu128", numpy="1.26.0", scipy="1.15.3"),
            "Unreviewed runtime")
    return manifest, experiment, split


def restore_parent(manifest, experiment, device="cpu"):
    import torch
    from continuous_policy import load_checkpoint
    policy, saved = load_checkpoint(manifest["files"]["parent"]["path"],
        study_fingerprint=experiment["protocol"]["fingerprint"],
        geometry_fingerprint=experiment["source"]["geometry_fingerprint"],
        domain_fingerprint=experiment["protocol"]["domain"]["fingerprint"],
        device=device, expected_updates=73, expected_attempted_shots=10000)
    require(saved["seed"] == 20261122 and isinstance(saved.get("optimizer"), dict), "Wrong parent learning state")
    optimizer = torch.optim.Adam(policy.parameters(), lr=.0003)
    optimizer.load_state_dict(saved["optimizer"])
    require(all(g["lr"] == .0003 for g in optimizer.param_groups), "Parent optimizer rate changed")
    require(len(optimizer.state) == len(list(policy.parameters())), "Incomplete parent optimizer")
    for state in optimizer.state.values():
        require(float(state["step"]) == 292, "Parent Adam steps changed")
        require(all(not torch.is_tensor(v) or torch.isfinite(v).all().item() for v in state.values()),
                "Nonfinite parent optimizer state")
    return policy, optimizer, saved


def check_parent_predictions(policy, manifest, split):
    """Reconstruct the old frozen TRAIN readouts; no held-out packet is read."""
    import math
    import numpy as np
    import torch
    from continuous_policy import build_observations, sample
    from continuous_feasibility import LOW, HIGH
    layouts = read(manifest["files"]["parent_progress"]["path"])["checkpoints"][-1]["train"]["layouts"]
    require([x["layout_id"] for x in layouts] == [x["id"] for x in split["train"]], "Original TRAIN prediction membership changed")
    expected = np.asarray([[r["requested"]["bearing_offset_rad"], r["requested"]["radius_m"],
        r["requested"]["height_m"],r["requested"]["gap_m"],math.radians(r["requested"]["tilt_deg"])] for r in layouts])
    expected = 2*(expected-LOW)/(HIGH-LOW)-1
    actual = sample(policy,torch.as_tensor(build_observations(split["train"])),deterministic=True)["actions_normalized"].numpy()
    error = float(np.max(np.abs(actual-expected)))
    require(error <= 5e-6, "Original frozen TRAIN model predictions changed")
    return error


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--expected-manifest-sha256", required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--session-id")
    p.add_argument("--execute-approved", action="store_true")
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args(argv)
    require(args.execute_approved != args.preflight_only, "Choose explicit execution or CPU preflight")
    manifest, experiment, split = inputs(args.manifest, args.expected_manifest_sha256)
    import numpy, scipy, torch
    require(dict(python=sys.version.split()[0], torch=torch.__version__, numpy=numpy.__version__, scipy=scipy.__version__)
            == manifest["runtime"], "Actual dependency versions differ")
    if args.preflight_only:
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Preflight must hide CUDA")
        policy, optimizer, saved = restore_parent(manifest, experiment)
        prediction_error = check_parent_predictions(policy,manifest,split)
        from continuous_policy import weight_hash
        print(json.dumps(dict(status="cpu_preflight_passed", parent_weight_hash=weight_hash(policy),
            parent_updates=saved["update_count"], parent_attempts=saved["attempted_shots"],
            parent_train_predictions_checked=35, max_normalized_prediction_error=prediction_error,
            optimizer_states=len(optimizer.state), simulator_starts=0, training_updates=0)))
        return
    require(args.output and args.output.is_dir() and args.session_id
        and os.environ.get("SIRCUS_CONTINUATION_SESSION") == args.session_id
        and os.environ.get("CUDA_VISIBLE_DEVICES") == GPU_UUID,
        "Watchdog-owned output/session/4090 visibility required")
    require(canonical(args.output) == Path(manifest["output"])
        and os.environ.get("SIRCUS_SEED22_RUNTIME_FINGERPRINT") == manifest["protocol"]["fingerprint"],
        "Watchdog output/protocol mismatch")
    owner_path = args.output/"watchdog-owner.json"
    owner_wait_until = min(time.monotonic()+3, float(os.environ["SIRCUS_CONTINUATION_DEADLINE_MONOTONIC"]))
    while not owner_path.exists() and time.monotonic() < owner_wait_until:
        require(os.getppid() > 1, "Watchdog parent exited before ownership binding")
        time.sleep(.01)
    owner = read(owner_path)
    from seed22_continuation_watchdog import signed
    require(signed({k:v for k,v in owner.items() if k != "fingerprint"}) == owner
        and owner["session_id"] == args.session_id
        and owner["runtime_fingerprint"] == manifest["protocol"]["fingerprint"]
        and owner["deadline_monotonic"] == float(os.environ["SIRCUS_CONTINUATION_DEADLINE_MONOTONIC"])
        and owner["start_monotonic"] == float(os.environ["SIRCUS_CONTINUATION_STARTED_MONOTONIC"])
        and owner["deadline_monotonic"]-owner["start_monotonic"] <= 28800
        and owner["max_requests"] == 30000, "Watchdog owner/deadline mismatch")
    require(owner["native_owner"]["pid"] == os.getpid() and owner["native_owner"]["ppid"] == os.getppid()
        and owner["native_owner"]["cookie"] == args.session_id, "Watchdog child identity mismatch")
    require(owner["shared_lock"] == "/workspace/scene.lock" and owner["gpu_uuid"] == GPU_UUID,
            "Watchdog shared lock or GPU mismatch")
    lock_stat = Path(owner["shared_lock"]).stat()
    require(owner["shared_lock_identity"] == [lock_stat.st_dev,lock_stat.st_ino], "Watchdog lock identity mismatch")
    # Check the actual parent and optimizer before spending the sole App start.
    checked_policy, checked_optimizer, _ = restore_parent(manifest, experiment, device="cpu")
    check_parent_predictions(checked_policy,manifest,split)
    del checked_policy, checked_optimizer
    require(torch.cuda.device_count() == 1 and "4090" in torch.cuda.get_device_name(0), "Wrong GPU")
    require(time.monotonic() < owner["deadline_monotonic"]-2, "Deadline reached before simulator start")
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0")
    try:
        from seed22_continuation_native import run
        return run(manifest, experiment, split, args.output, args.session_id)
    finally:
        app.app.close()


if __name__ == "__main__":
    main()
