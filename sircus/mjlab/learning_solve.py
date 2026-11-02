"""Frozen prediction or explicitly requested bounded per-scene native search.

Artifact and constraint checks are CPU-only. A native environment is imported
and constructed only after those checks and exclusive 4090 ownership succeed.
"""

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

from .interface import DEFAULT_SCENE, study_ready, validate_request
from .learning_contract import (
    HIGH,
    LOW,
    SEED,
    actions,
    constraints,
    digest,
    observations,
    project,
    proposals,
    reward,
    validate_locks,
)

GPU_UUID = "GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4"
CRITICAL_MODULES = (
    "model.py",
    "run.py",
    "contract.py",
    "learning_contract.py",
    "learning_env.py",
    "learning_ik.py",
    "learning_policy.py",
    "learning_score.py",
    "learning_reach.py",
)
SHARED_SOURCES = (
    "sircus/isaac/geometry.py",
    "sircus/isaac/rl_contract.py",
    "sircus/isaac/entry_quality.py",
    "sircus/isaac/board_route.py",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _contained(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise RuntimeError(
            "Artifact manifest paths must remain under their declared root"
        )
    return path


def _verify_files(root, entries, required=()):
    if not isinstance(entries, dict) or not set(required).issubset(entries):
        raise RuntimeError("Artifact provenance omits required source/assets")
    for name, expected in entries.items():
        path = _contained(root, name)
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(
                f"Runtime artifact differs from the evaluated study: {name}"
            )


def required_asset_paths(assets):
    """URDF/seed files and every collision mesh, relative to assets.parent."""
    assets = Path(assets).resolve()
    root = assets.parent
    required = [
        assets / name for name in ("dropper.urdf", "board_arm.urdf", "poses.json")
    ]
    for path in required[:2]:
        tree = ET.parse(path)
        for mesh in tree.findall(".//collision/geometry/mesh"):
            filename = mesh.get("filename", "")
            if filename.startswith("/workspace/"):
                target = root / filename.removeprefix("/workspace/")
            else:
                target = Path(filename)
                if not target.is_absolute():
                    target = assets / target
            required.append(target.resolve())
    try:
        return sorted({str(path.relative_to(root)) for path in required})
    except ValueError:
        raise RuntimeError(
            "Collision mesh lies outside the declared asset tree"
        ) from None


def validate_artifacts(study, assets, *, repository=None, load_checkpoint=None):
    """Fail closed before any CUDA/native environment can be initialized.

    artifact_provenance.json is assembled from the immutable executed snapshot;
    it supplements the study's mjlab-module hashes with shared scorer/geometry
    and generated URDF/mesh hashes. Neither report nor provenance is rewritten.
    """
    study, assets = Path(study), Path(assets)
    repository = Path(repository) if repository else Path(__file__).resolve().parents[2]
    report = json.loads((study / "report.json").read_text())
    if not study_ready(report):
        raise RuntimeError("Study and frozen evaluation must complete first")
    checkpoint_path = study / "selected.pt"
    actual = sha256(checkpoint_path)
    if actual != report.get("checkpoint_sha256"):
        raise RuntimeError("Selected checkpoint hash changed")
    provenance_path = study / "artifact_provenance.json"
    if not provenance_path.is_file():
        raise RuntimeError(
            "Completed study still needs verified artifact_provenance.json"
        )
    provenance = json.loads(provenance_path.read_text())
    if provenance.get("version") != "sircus-mjlab-artifacts-v1":
        raise RuntimeError("Unsupported runtime artifact provenance")
    if (
        provenance.get("contract_fingerprint") != report.get("contract_fingerprint")
        or provenance.get("checkpoint_sha256") != actual
    ):
        raise RuntimeError("Runtime provenance belongs to another study/checkpoint")
    source_hashes = report.get("source_sha256", {})
    _verify_files(
        repository / "sircus/mjlab",
        {
            name: source_hashes[name]
            for name in CRITICAL_MODULES
            if name in source_hashes
        },
        CRITICAL_MODULES,
    )
    _verify_files(repository, provenance.get("source_sha256"), SHARED_SOURCES)
    _verify_files(
        assets.parent,
        provenance.get("asset_sha256"),
        required_asset_paths(assets),
    )
    benchmark_path = repository / "baselines/mjlab-comparison-20261102/benchmark.json"
    benchmark = json.loads(benchmark_path.read_text())
    benchmark_payload = dict(benchmark)
    benchmark_fp = benchmark_payload.pop("fingerprint")
    if digest(benchmark_payload) != benchmark_fp:
        raise RuntimeError("Frozen benchmark fingerprint changed")
    for module, expected in benchmark["scorer_sha256"].items():
        if sha256(repository / "sircus/isaac" / (module + ".py")) != expected:
            raise RuntimeError(
                "Shared capture/quality/route scorer differs from the benchmark"
            )
    # All dependency checks occur before importing the CUDA-capable environment.
    versions = report.get("versions", {})
    for name in ("mjlab", "mujoco", "mujoco-warp", "torch", "warp-lang"):
        if not versions.get(name) or importlib.metadata.version(name) != versions[name]:
            raise RuntimeError(
                f"Runtime dependency differs from the evaluated study: {name}"
            )
    if load_checkpoint is None:
        import torch

        load_checkpoint = lambda path: torch.load(
            path, map_location="cpu", weights_only=True
        )
    checkpoint = load_checkpoint(checkpoint_path)
    contract = dict(checkpoint["contract"])
    fingerprint = contract.pop("fingerprint")
    if (
        digest(contract) != fingerprint
        or report["contract_fingerprint"] != fingerprint
        or digest(contract["scene"]) != digest(benchmark["scene"])
    ):
        raise RuntimeError("Study scene/contract fingerprint mismatch")
    disk_contract = json.loads((study / "contract.json").read_text())
    if disk_contract != checkpoint["contract"]:
        raise RuntimeError("Checkpoint and saved scene contracts differ")
    frozen = report["frozen_evaluation"]
    if frozen.get("policy_weight_sha256") != checkpoint.get("weight_sha256"):
        raise RuntimeError(
            "Frozen evaluation does not identify the selected policy weights"
        )
    return report, checkpoint, actual, fingerprint


def structural_constraint_error(bucket, locks, radius):
    """Reject impossible locks, not merely an invalid learned proposal.

    Within the declared vertical-center-impact family these checks are exact
    action-bound contradictions. They do not claim general robotic infeasibility.
    """
    validate_locks(locks, 1)
    board, dropper = locks["board"][0], locks["dropper"][0]
    b, d = np.isfinite(board).all(), np.isfinite(dropper).all()
    if b and ((board < LOW[:3] - 1e-9).any() or (board > HIGH[:3] + 1e-9).any()):
        return "Locked board center lies outside this study's action bounds"
    if d and (
        (dropper[:2] < LOW[:2] - 1e-9).any() or (dropper[:2] > HIGH[:2] + 1e-9).any()
    ):
        return "Locked release XY lies outside this study's action bounds"
    if b and d and np.linalg.norm(board[:2] - dropper[:2]) > 1e-9:
        return "Both locked XY positions conflict with the vertical center-impact shot family"
    if d:
        lo = max(LOW[2], dropper[2] - radius - HIGH[4])
        hi = min(HIGH[2], dropper[2] - radius - LOW[4])
        if lo > hi + 1e-9 or (b and not lo - 1e-9 <= board[2] <= hi + 1e-9):
            return "Locked release height cannot satisfy the configured board-height/drop-gap bounds"
    fixed_xy = board[:2] if b else dropper[:2] if d else None
    if (
        fixed_xy is not None
        and np.linalg.norm(np.asarray(bucket)[0] - fixed_xy) <= 0.15
    ):
        return "Locked impact XY leaves less than the study's 150 mm bucket separation"
    return None


def search_candidates(
    bucket, locks, rng, *, count=128, policy_values=None, radius=0.0375
):
    """Generate candidates while preserving structurally feasible release locks."""
    xy = np.repeat(bucket, count, axis=0)
    lk = {key: np.repeat(value, count, axis=0) for key, value in locks.items()}
    values = proposals(xy, rng)
    board, dropper = locks["board"][0], locks["dropper"][0]
    if np.isfinite(board).all():
        values[:, :3] = board
    if np.isfinite(dropper).all():
        values[:, :2] = dropper[:2]
        lo = max(LOW[2], dropper[2] - radius - HIGH[4])
        hi = min(HIGH[2], dropper[2] - radius - LOW[4])
        if not np.isfinite(board).all():
            values[:, 2] = rng.uniform(lo, hi, count)
    if policy_values is not None:
        projected, valid = project(policy_values, locks, radius)
        if valid[0]:
            values[0] = projected[0]
    return xy, values, lk


def _write(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, allow_nan=False, indent=2))
    tmp.replace(path)


def _failure(request, message, fingerprint=None):
    result = dict(
        request_id=request["request_id"],
        bucket_xy_m=request.get("bucket_xy"),
        method="frozen_policy" if request["mode"] == "policy" else "search",
        success=False,
        message=message,
        trajectory=[],
        constraints=request["constraints"],
    )
    if fingerprint:
        result["provenance"] = dict(
            contract_fingerprint=fingerprint,
            search_at_inference=request["mode"] == "search",
            proposal_bank_sha256=None,
            proposal_method="none; no native attempt",
        )
    return result


def _assert_gpu_free():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_UUID:
        raise RuntimeError(
            "This solver requires explicit selection of only the RTX4090 UUID"
        )
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
    )
    for line in processes.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if (
            fields
            and fields[0] == GPU_UUID
            and (len(fields) != 2 or int(fields[1]) != os.getpid())
        ):
            raise RuntimeError("4090 has another workload; it will not be interrupted")


def _bound_child_lifetime():
    """A killed HTTP server must not leave its isolated native worker running."""
    if sys.platform.startswith("linux"):
        import ctypes

        parent = os.getppid()
        if parent <= 1:
            raise RuntimeError("Native solve parent has already exited")
        libc = ctypes.CDLL(None, use_errno=True)
        # Linux PR_SET_PDEATHSIG; only this fresh child changes its own setting.
        if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "Cannot bound native worker lifetime")
        if os.getppid() != parent:
            raise RuntimeError("Native solve parent exited during startup")
    signal.alarm(240)


def run(a):
    report, cp, checkpoint_sha, fingerprint = validate_artifacts(a.study, a.assets)
    request = validate_request(json.loads(a.request.read_text()), DEFAULT_SCENE)
    mode = request["mode"]
    xy = np.asarray([request["bucket_xy"]], dtype=float)
    locks = constraints(1)
    for api, key in (("board_position", "board"), ("dropper_position", "dropper")):
        value = request["constraints"].get(api)
        if value is not None:
            locks[key][0] = value
    error = structural_constraint_error(
        xy, locks, cp["contract"]["scene"]["ball"]["radius_m"]
    )
    if error:
        return _write(a.output, _failure(request, error, fingerprint))
    bank = None
    relative_candidates, relative_metadata = np.empty((0, 5)), []
    free_positions = all(np.isnan(value).all() for value in locks.values())
    if mode == "search" and free_positions:
        from .learning_search import bank_candidates, load_training_bank

        bank = load_training_bank(a.study, fingerprint)
        relative_candidates, relative_metadata = bank_candidates(
            bank, xy[0], cp["contract"]["scene"]
        )
    import torch
    from .learning_policy import Policy, weight_hash

    torch.set_num_threads(2)
    policy = Policy()
    policy.load_state_dict(cp["state"])
    policy.eval()
    if weight_hash(policy) != cp["weight_sha256"]:
        raise RuntimeError("Policy tensor hash changed")
    with torch.no_grad():
        values = actions(policy.actor(torch.tensor(observations(xy, locks))).numpy())
    _, valid = project(values, locks)
    if mode == "policy" and not valid[0]:
        return _write(
            a.output,
            _failure(
                request,
                "Frozen policy proposal violates action bounds under these locks; explicit search may still find a solution",
                fingerprint,
            ),
        )
    with open(a.gpu_lock, "r") as gpu_lock:
        try:
            fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                "4090 is busy with another simulation; try again after it finishes"
            ) from None
        _assert_gpu_free()
        from .learning_env import ShotEnvironment

        n = 1 if mode == "policy" else 128
        env = ShotEnvironment(cp["contract"]["scene"], a.assets, n)
        rows = []
        if mode == "policy":
            rows, _ = env.run(xy, values, locks, record=True)
        else:
            rng = np.random.default_rng(SEED + 41)
            bank_offset = 0
            for batch in range(4):
                bucket, candidates, lk = search_candidates(
                    xy,
                    locks,
                    rng,
                    count=n,
                    policy_values=values,
                    radius=cp["contract"]["scene"]["ball"]["radius_m"],
                )
                bank_count = min(n, len(relative_candidates) - bank_offset)
                batch_metadata = relative_metadata[
                    bank_offset : bank_offset + bank_count
                ]
                if bank_count:
                    candidates[:bank_count] = relative_candidates[
                        bank_offset : bank_offset + bank_count
                    ]
                    bank_offset += bank_count
                if batch and rows:
                    best_rows = sorted(rows, key=reward, reverse=True)[:8]
                    for i in range(bank_count, min(n, bank_count + n // 2)):
                        candidates[i] = np.asarray(
                            best_rows[i % len(best_rows)]["values"]
                        ) + rng.normal(0, [0.015, 0.015, 0.018, 2.0, 0.025])
                    candidates = np.clip(candidates, LOW, HIGH)
                result, _ = env.run(bucket, candidates, lk, record=False)
                for i, row in enumerate(result):
                    row["search_proposal"] = (
                        dict(method="relative_train_bank", **batch_metadata[i])
                        if i < bank_count
                        else dict(method="continuous_search")
                    )
                rows.extend(result)
                if any(r["route_success"] and not r["invalid"] for r in result):
                    break
            selected = max(rows, key=reward)
            # The search outcome is not enough: score a separate fresh replay.
            verified, _ = env.run(
                xy, np.asarray([selected["values"]]), locks, record=True
            )
            verified[0]["search_proposal"] = selected.get("search_proposal")
            rows.extend(verified)
        best = rows[-1] if mode == "search" else rows[0]
        if not best.get("scalar_score_audit_passed", False):
            raise RuntimeError("Native replay lacks a passing scalar scoring audit")
        trajectory = [
            dict(time=(i + 1) * best.get("trajectory_dt_s", 1 / 60), ball=p)
            for i, p in enumerate(best.get("trajectory_m", []))
        ]
        config = dict(
            board_center=best["board_center_m"],
            board_normal=best["board_normal"],
            board_quaternion_wxyz=best["board_quaternion_wxyz"],
            release_center=best["release_ball_center_m"],
            dropper_center=best["release_ball_center_m"],
            q10=best["q10"],
            q14=best["q14"],
            link_positions10=best["link_positions10"],
            link_positions14=best["link_positions14"],
            tilt_deg=best["values"][3],
            gap_m=best["values"][4],
        )
        success = bool(best["route_success"] and not best["invalid"])
        message = (
            "Measured simulated capture: board rebound and one continuous second fully inside"
            if success
            else "No qualified capture in this bounded attempt; relax a lock or request explicit search"
        )
        _write(
            a.output,
            dict(
                request_id=request["request_id"],
                bucket_xy_m=request["bucket_xy"],
                method="frozen_policy" if mode == "policy" else "search",
                success=success,
                message=message,
                configuration=config,
                trajectory=trajectory,
                metrics=dict(
                    board_bounces=best["board_contact_episodes"],
                    retention_s=best["max_retention_s"],
                    quality=best["route_quality"],
                    invalid=best["invalid"],
                    external_contacts=best["external_contact_steps"],
                    attempts=len(rows),
                    scalar_score_audit_passed=True,
                    search_proposal=(
                        best.get("search_proposal") if mode == "search" else None
                    ),
                ),
                provenance=dict(
                    contract_fingerprint=fingerprint,
                    checkpoint_sha256=checkpoint_sha,
                    policy_weight_sha256=cp["weight_sha256"],
                    engine="mjlab MuJoCo Warp",
                    device="RTX4090",
                    positional_locks=request["constraints"],
                    search_at_inference=mode == "search",
                    endpoint_configuration_only=True,
                    artifact_provenance_verified=True,
                    proposal_bank_sha256=bank["sha256"] if bank is not None else None,
                    proposal_method=(
                        "relative_train_bank_plus_continuous_search"
                        if bank is not None
                        else (
                            "continuous_search_with_positional_locks"
                            if mode == "search"
                            else "frozen_policy; no search bank"
                        )
                    ),
                    proposal_bank_qualification=(
                        bank["qualification"] if bank is not None else None
                    ),
                    proposal_bank_geometric_candidates=(
                        len(relative_candidates) if bank is not None else None
                    ),
                ),
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("study", "assets", "gpu-lock", "request", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    _bound_child_lifetime()
    run(args)
