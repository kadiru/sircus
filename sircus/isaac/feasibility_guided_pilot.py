"""Geometry-map-conditioned local continuous PPO pilot; original study is frozen.

Geometry-only anchors condition an affine five-action residual decoder. Every
sampled attempt remains in accounting and the native gates are unchanged.
"""
import json
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from continuous_feasibility import LOW, HIGH, decode_action, canonical_hash
from continuous_pilot import make_study as original_study, SEED, FOCUS
from presentation_recording import digest, save_physics
SCHEMA = "sircus-native-feasibility-guided-pilot-v1"


def make_study(bundle, batches, support):
    from feasibility_guided_policy import ARCHITECTURE, POLICY_VERSION
    from feasibility_guided_support import validate_support
    validate_support(support, bundle["spec"]["cells"])
    if support["source_fingerprint"] != bundle["spec"]["fingerprint"]:
        raise ValueError("Support belongs to a different geometry")
    study = original_study(bundle, batches)
    study.update(schema=SCHEMA, policy_version=POLICY_VERSION, architecture=ARCHITECTURE,
        support_fingerprint=support["fingerprint"],
        support_scope="Geometry-only per-context local continuous residual boxes; not successful-shot menus or certified feasible interiors. Empty support remains a rejected attempt.",
        inference_scope="Geometry-assisted/transductive use of the known map at held-out coordinates; no capture labels or reward-based search at inference. Not globally unseen position generalization.",
        scope="Single-seed geometry-map-conditioned local continuous-action terminal PPO pilot; learned shot parameters, scripted arm travel and initialized release")
    study["domain"].update(decoder="geometry-conditioned local affine residual-v1", support_fingerprint=support["fingerprint"],
        residual_bounds=[-1,1], global_bounds_scope="Original physical envelope; actual per-context bounds are in support.json")
    study["domain"]["fingerprint"] = canonical_hash({k:v for k,v in study["domain"].items() if k != "fingerprint"})
    study["fingerprint"] = canonical_hash({k:v for k,v in study.items() if k != "fingerprint"})
    return study


def run_study(env, bundle, batches=64, feasibility_path=None, support_path=None):
    import torch
    from continuous_policy import (build_observations, sample, weight_hash, actor_weight_hash, update)
    from feasibility_guided_policy import FeasibilityGuidedPolicy, checkpoint_payload
    from feasibility_guided_support import physical_actions, ik_seeds, validate_support
    from farm_contract import reward, statistics
    from parallel_runtime import atomic

    support = json.loads(Path(support_path).read_text())
    validate_support(support)
    study = make_study(bundle, batches, support)
    feasibility = json.loads(Path(feasibility_path).read_text())
    if feasibility.get("source_fingerprint") != study["geometry_fingerprint"]:
        raise ValueError("Feasibility map belongs to a different farm geometry")
    unsigned = {k:v for k,v in feasibility.items() if k != "fingerprint"}
    if canonical_hash(unsigned) != feasibility.get("fingerprint"):
        raise ValueError("Feasibility map fingerprint changed")
    for filename, expected in feasibility["asset_sha256"].items():
        if digest(Path(env.args.assets)/filename) != expected:
            raise ValueError("Feasibility assets differ from native scene")
    if support["feasibility_fingerprint"] != feasibility["fingerprint"]:
        raise ValueError("Guided support belongs to a different feasibility map")
    shutil.copyfile(support_path, env.dest/"support.json")
    study["support_sha256"] = digest(support_path)
    shutil.copyfile(feasibility_path, env.dest/"feasibility.json")
    study["feasibility_sha256"] = digest(feasibility_path)
    study["coverage_batches"] = 4
    study["fingerprint"] = canonical_hash({k:v for k,v in study.items() if k != "fingerprint"})
    atomic(env.dest/"study.json", study)
    env.study = study
    env.focus = FOCUS
    env.proposal_ik_seeds = ik_seeds(support, bundle["spec"]["cells"])
    env.support_missing = [fit is None for fit in env.proposal_ik_seeds]
    env.active_mask = np.ones(35, dtype=bool)
    from prepare_assets import chain
    env.continuous_links = {name: chain(ET.parse(Path(env.args.assets)/filename).getroot())
        for name, filename in (("Arm10", "dropper.urdf"), ("Arm14", "board_arm.urdf"))}
    model = ET.parse(Path(env.args.assets)/"dropper.urdf")
    env.transit_limits = torch.tensor([float(model.find(f".//joint[@name='joint{i+1}']/limit").get("effort")) for i in range(6)], device=env.sim.device)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    policy = FeasibilityGuidedPolicy().to(env.sim.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=study["ppo"]["lr"])
    observations = torch.as_tensor(build_observations(bundle["spec"]["cells"]), device=env.sim.device)
    train_mask = np.array([c["split"] == "train" for c in study["cells"]])
    if int(train_mask.sum()) != 25 or any(sum(c["split"] == label for c in study["cells"]) != 5 for label in ("validation", "test")):
        raise ValueError("Continuous pilot split changed")
    train_ids = torch.tensor(np.flatnonzero(train_mask), device=env.sim.device)
    initial_hash = weight_hash(policy)
    changes, summaries, assessments, coverages, clips, buffer = [], [], [], [], [], []
    (env.dest/"trials").mkdir()
    started = time.monotonic()

    def checkpoint(name, after_batches):
        path = env.dest/name
        payload = checkpoint_payload(policy, optimizer,
            study_fingerprint=study["fingerprint"], geometry_fingerprint=study["geometry_fingerprint"],
            domain_fingerprint=study["domain"]["fingerprint"], support_fingerprint=support["fingerprint"], update_count=len(changes),
            attempted_shots=after_batches*int(train_mask.sum()), seed=SEED)
        torch.save(payload, path)
        return dict(file=name, sha256=digest(path), policy_hash=weight_hash(policy), actor_hash=actor_weight_hash(policy))

    def trial(mode, batch, cp, deterministic=False, film=False, coverage=None):
        if shutil.disk_usage(env.dest).free < 4*1024**3:
            raise RuntimeError("Less than 4GiB free during continuous pilot")
        env.active_mask = train_mask.copy() if mode == "train" else np.ones(35, dtype=bool)
        with torch.no_grad():
            picked = sample(policy, observations, deterministic=deterministic)
        physical = physical_actions(picked["actions_normalized"], support, bundle["spec"]["cells"])
        actions = [decode_action(c, a, f"{mode}-{batch}-{i}") for i,(c,a) in enumerate(zip(bundle["spec"]["cells"], physical))]
        if coverage is not None:
            actions = coverage["actions"]
            env.active_mask = np.asarray(coverage["enabled"], dtype=bool)
            env.coverage_fits = coverage["fits"]
        env.capture_trial, env.film_trial, env.phase = False, False, None
        home=env.home()
        env.home_readback=home
        env.trial_mode, env.trial_batch = mode, batch
        env.frames, env.physics_samples = [], []
        env.capture_trial, env.film_trial, env.phase = True, film, "setup"
        env.trial_start_step = env.steps
        first = env.next_frame
        rows, raw = env.shot_batch(actions, record_video=film, enabled=env.active_mask)
        env.capture_trial, env.film_trial, env.phase = False, False, None
        active = [i for i in range(35) if env.active_mask[i]]
        for i,r in enumerate(rows):
            r["split"] = study["cells"][i]["split"]
            r["qualification"] = env.qualifications[i]
            r["path_rejection"] = env.path_reasons[i]
            r["path_proxy"] = env.path_checks[i]
            r["scoring_board_normal"] = env.scoring_normals[i].tolist()
        returns = torch.tensor([reward(r) for r in rows], dtype=torch.float32, device=env.sim.device)
        name = f"{mode}-{len(summaries)+len(assessments)+len(coverages):04d}"
        physics = None
        if env.physics_samples:
            path = env.dest/f"{name}-physics.npz"
            save_physics(path, env.physics_samples)
            physics = dict(file=path.name, sha256=digest(path), samples=len(env.physics_samples))
        payload = dict(schema=SCHEMA, mode=mode, batch=batch, update_count=len(changes),
            study_fingerprint=study["fingerprint"], checkpoint=cp, policy_hash=weight_hash(policy),
            observations=observations.cpu().tolist(), latent=picked["latent"].cpu().tolist(),
            actions_normalized=picked["actions_normalized"].cpu().tolist(), logp=picked["logp"].cpu().tolist(),
            actions=actions, rows=rows, rewards=returns.cpu().tolist(),
            active_cell_ids=[study["cells"][i]["id"] for i in active],
            enabled=env.active_mask.tolist(), deterministic=deterministic,
            optimizer_samples=len(active) if mode == "train" else 0,
            policy_action_used=mode != "coverage",
            support_fingerprint=support["fingerprint"],
            home=home,
            release_step=env.release_step, score_end_step=env.steps, physics=physics,
            trajectory_m=raw["trajectory_m"], wall_s=raw["wall_s"],
            transit_joint_error_rad=env.transit_joint_error_rad)
        filename=f"trials/{name}.json"
        atomic(env.dest/filename, payload)
        summary = dict(mode=mode, batch=batch, update_count=len(changes), file=filename,
                       sha256=digest(env.dest/filename), **statistics([rows[i] for i in active]))
        if film:
            clip = dict(mode=mode, batch=batch, update_count=len(changes), trial_file=filename,
                frames=env.frames, actions=actions, rows=rows, fps=30,
                start_frame=first, end_frame=env.next_frame,
                release_step=env.release_step, camera_sha256=bundle["source"]["camera_sha256"])
            clips.append(clip)
            atomic(env.dest/"clips.json", dict(clips=clips))
        print(json.dumps(summary), flush=True)
        return summary, (observations[train_ids], picked["latent"][train_ids],
                         picked["logp"][train_ids], returns[train_ids])

    initial = checkpoint("policy-initial.pt", 0)
    from continuous_feasibility import geometry_coverage
    for batch, plan in enumerate(geometry_coverage(feasibility, bundle["spec"]["cells"], batches=4)):
        row,_ = trial("coverage", batch, initial, film=True, coverage=plan)
        coverages.append(row)
    for repeat in range(study["assessment_batches"]):
        row,_ = trial("eval", repeat, initial, deterministic=True)
        assessments.append(row)
    for batch in range(batches):
        cp = checkpoint(f"policy-before-{batch:03d}.pt", batch)
        row, data = trial("train", batch, cp, film=batch in study["filmed_training_batches"])
        summaries.append(row)
        buffer.append(data)
        if len(buffer) == study["batches_per_update"]:
            values = update(policy, optimizer, *[torch.cat(group) for group in zip(*buffer)], study["ppo"])
            values.update(update_count=len(changes)+1, training_batches=list(range(batch-3,batch+1)),
                          sample_count=4*int(train_mask.sum()))
            changes.append(values)
            buffer.clear()
        atomic(env.dest/"progress.json", dict(schema=SCHEMA, status="training", study_fingerprint=study["fingerprint"],
            batches=summaries, assessments=assessments, coverage=coverages, updates=changes, frames=env.next_frame))
    final=checkpoint("policy-final.pt", batches)
    for repeat in range(study["assessment_batches"]):
        row,_=trial("eval", repeat, final, deterministic=True)
        assessments.append(row)
    result=dict(schema=SCHEMA, status="completed", study_fingerprint=study["fingerprint"],
        geometry_fingerprint=study["geometry_fingerprint"], batches=summaries, assessments=assessments,
        updates=changes, coverage=coverages, initial=initial, final=final, initial_hash=initial_hash,
        final_hash=weight_hash(policy), seconds=time.monotonic()-started, frames=env.next_frame,
        origins_m=env.origins.tolist(), camera=env.saved_camera, stage=env.stage_metadata,
        source_sha256={p.name:digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        gpu=torch.cuda.get_device_name(), workcells=35, articulations=70,
        peak_bucket_drift_m=env.peak_bucket_drift, scope=study["scope"])
    atomic(env.dest/"result.json", result)
    return result
