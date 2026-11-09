"""Two frozen TRAIN cohorts in the original native farm, without a learner.

Called only through the explicit film entry-point option. All actions and IK
seeds are fixed before launch; native qualification and release guards remain
mandatory. No policy model or optimizer is constructed by this module.
"""

import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import time
import xml.etree.ElementTree as ET

import numpy as np

from candidate_selector import validate_comparison
from continuous_policy import build_observations
from continuous_feasibility import ACTION_NAMES, HIGH, LOW, canonical_hash
from presentation_recording import digest, save_physics
from settled_transit import protocol as setup_protocol
from shot_benchmark_native import compressed_json, summary
from shot_benchmark_reward import build_features, reward


def atomic(path, value):
    from parallel_runtime import atomic as write
    write(path, value)


def run_comparison(env, bundle, plan_path, *, process_started):
    import torch
    from prepare_assets import chain

    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text())
    validate_comparison(plan)
    binding = plan["native_binding"]
    original = binding["original_source"]
    if binding["split_manifest_sha256"] != plan["input_bindings"]["original"]["split_manifest_sha256"]:
        raise ValueError("Frozen split identity changed")
    if env.n != 35 or bundle["spec"]["fingerprint"] != original["geometry_fingerprint"]:
        raise ValueError("Candidate comparison source geometry changed")
    if digest(env.dest/"selected-camera.json") != original["camera_sha256"]:
        raise ValueError("Frozen comparison camera bytes changed")
    checkpoint = Path(binding["checkpoint_path"])
    checkpoint_before = digest(checkpoint)
    if checkpoint_before != binding["checkpoint_sha256"]:
        raise ValueError("Frozen seed-22 model bytes changed")
    if canonical_hash(env.cfg) != binding["source_scene_sha256"]:
        raise ValueError("Candidate comparison physics/material/geometry scene changed")
    if shutil.disk_usage(env.dest).free < 6*1024**3:
        raise RuntimeError("Less than 6GiB free; original evidence is never deleted")
    owner = env.dest.parent.stat()
    os.chown(env.dest, owner.st_uid, owner.st_gid)
    (env.dest/"trials").mkdir()
    source_hashes = {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    controls = {name: digest(env.dest/name) for name in ("probes.json", "rim-rays.json")}
    atomic(env.dest/"comparison-plan.json", plan)
    atomic(env.dest/"experiment-binding.json", dict(
        comparison_sha256=digest(plan_path), comparison_fingerprint=plan["fingerprint"],
        source_sha256=source_hashes, original_source=original,
        setup_controller=setup_protocol(), source_commit=binding["source_commit"],
        source_archive_sha256=binding["source_archive_sha256"],
        checkpoint_sha256=checkpoint_before, control_sha256=controls))
    env.continuous_links = {n: chain(ET.parse(Path(env.args.assets)/file).getroot())
        for n, file in (("Arm10", "dropper.urdf"), ("Arm14", "board_arm.urdf"))}
    model = ET.parse(Path(env.args.assets)/"dropper.urdf")
    env.transit_limits = torch.tensor([float(model.find(f".//joint[@name='joint{i+1}']/limit").get("effort"))
                                      for i in range(6)], device=env.sim.device)
    env.focus = 1
    env.support_missing = [False]*35
    torch.manual_seed(20261122)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    originals = copy.deepcopy(bundle["spec"]["cells"])
    trials, outcomes = [], {}
    initialization_wall_s = time.monotonic()-process_started
    started = time.monotonic()

    def progress(status, **extra):
        atomic(env.dest/"progress.json", dict(schema="sircus-candidate-native-progress-v1",
            status=status, comparison_fingerprint=plan["fingerprint"],
            completed_requests=sum(t["requests"] for t in trials), planned_requests=70,
            trials=trials, methods=outcomes, optimizer_updates=0,
            initialization_shared_wall_s=initialization_wall_s,
            elapsed_s=time.monotonic()-started, **extra))

    progress("running")
    for batch, cohort in enumerate(plan["cohorts"]):
        method = cohort["method"]
        if shutil.disk_usage(env.dest).free < 6*1024**3:
            raise RuntimeError("Candidate comparison storage floor reached")
        torch.cuda.reset_peak_memory_stats()
        begin, cpu = time.monotonic(), time.process_time()
        env.capture_trial = env.film_trial = False
        env.phase = None
        env.base_cells = [dict(copy.deepcopy(originals[i]), **copy.deepcopy(c)) for i, c in enumerate(plan["cells"])]
        env.spec["cells"] = copy.deepcopy(env.base_cells)
        env.bucket_xy = np.asarray([c["bucket_xy_m"] for c in plan["cells"]])
        env.home_readback = env.home()
        env.active_mask = np.asarray(cohort["native_enabled"], dtype=bool)
        env.proposal_ik_seeds = copy.deepcopy(cohort["ik_seeds"])
        env.trial_mode, env.trial_batch = method, batch
        env.frames, env.physics_samples = [], []
        env.capture_trial, env.film_trial, env.phase = True, False, "setup"
        env.trial_start_step = env.steps
        actions = copy.deepcopy(cohort["actions"])
        rows, raw = env.shot_batch(actions, record_video=False, enabled=env.active_mask)
        env.capture_trial = env.film_trial = False
        env.phase = None
        physics, features = None, [None]*35
        if env.physics_samples:
            path = env.dest/f"native-{method}.npz"
            save_physics(path, env.physics_samples)
            physics = dict(file=path.name, sha256=digest(path), samples=len(env.physics_samples))
            pos = np.asarray([s["position_m"] for s in env.physics_samples])-env.origins[None]
            vel = np.asarray([s["velocity_m_s"] for s in env.physics_samples])
            forces = np.asarray([s["forces_N"] for s in env.physics_samples])
            for i, row in enumerate(rows):
                if not row["invalid"]:
                    features[i] = build_features(env.cfg, plan["cells"][i]["bucket_xy_m"],
                        pos[:, i], vel[:, i], forces[:, i], env.dt, env.scoring_normals[i])
        returns = []
        for i, row in enumerate(rows):
            if cohort["decline_reasons"][i] is not None:
                row["invalid"] = cohort["decline_reasons"][i]
            row.update(split="train", qualification=env.qualifications[i],
                path_rejection=env.path_reasons[i], path_proxy=env.path_checks[i],
                scoring_board_normal=env.scoring_normals[i].tolist())
            returns.append(reward(row, features[i]))
        normalized = 2*(np.asarray([[a[k] for k in ACTION_NAMES] for a in actions])-LOW)/(HIGH-LOW)-1
        packet = dict(schema="sircus-shot-benchmark-native-trial-v1", mode=method, batch=batch,
            comparison_fingerprint=plan["fingerprint"], optimizer_updates=0,
            checkpoint=dict(sha256=checkpoint_before, training_attempts=10000),
            source=original, split_manifest_sha256=binding["split_manifest_sha256"],
            cells=copy.deepcopy(plan["cells"]), enabled=list(map(bool, env.active_mask)),
            rows=rows, actions=actions, observations=build_observations(plan["cells"]).tolist(),
            actions_normalized=normalized.tolist(), learner_packet=None, proposal_metadata=None,
            proposal_ik_seeds=copy.deepcopy(cohort["ik_seeds"]),
            rewards=returns, reward_features=features, training_request_order=None,
            home=env.home_readback, release_step=env.release_step, score_end_step=env.steps,
            physics=physics, origins_m=env.origins.tolist(), setup_controller=setup_protocol(),
            settling_history=raw["settling"], transit_history=env.transit_history,
            correction_history=env.correction_history, trajectory_m=raw["trajectory_m"])
        relative = f"trials/{method}-0000{batch}.json.gz"
        receipt = compressed_json(env.dest/relative, packet)
        atomic(env.dest/(relative+".receipt.json"), receipt)
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        metrics = dict(wall_s=time.monotonic()-begin, cpu_s=time.process_time()-cpu,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            torch_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
            torch_peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
            sampled_device_occupied_mib=(total-free)/1024**2,
            setup_wall_s=env.setup_wall_s,
            score_wall_s=max(0., raw["wall_s"]-env.setup_wall_s))
        outcomes[method] = summary(rows, returns)
        trials.append(dict(method=method, file=relative, sha256=receipt["sha256"], requests=35,
                           summary=outcomes[method], metrics=metrics))
        progress("running" if batch == 0 else "native_complete")
    checkpoint_after = digest(checkpoint)
    if checkpoint_after != checkpoint_before:
        raise RuntimeError("Frozen seed-22 checkpoint changed during comparison")
    result = dict(schema="sircus-candidate-native-result-v1", status="completed",
        comparison_fingerprint=plan["fingerprint"], comparison_sha256=digest(plan_path),
        source_sha256=source_hashes, original_source=original, trials=trials,
        optimizer_updates=0, native_requests=70, checkpoint_sha256_before=checkpoint_before,
        checkpoint_sha256_after=checkpoint_after, control_files=controls,
        initialization_shared_wall_s=initialization_wall_s,
        comparison_wall_s=time.monotonic()-started,
        resource_scope="Native cohort wall/CPU include canonical home, setup, flight and serialization. RSS is process-lifetime high-water, not an isolated phase peak. Torch CUDA peaks include persistent scene tensors; sampled device occupancy includes PhysX/renderer and is not isolated model memory. Initialization/control probes are shared and separate. CPU bank preprocessing and selection are separate preparation costs.",
        scope="Two fresh frozen TRAIN-only cohorts, same 35 coordinates; no optimizer, held-out search or claim of generalization. Original benchmark unchanged.")
    atomic(env.dest/"result.json", result)
    progress("native_complete")
