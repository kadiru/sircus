"""Production-only original native physics plus fixed continuous geometry wrapper.

The PPO likelihood always belongs to the RAW proposal, never the adjusted shot.
Every paid request is reserved durably before geometry or native dispatch.
"""
import copy
import json
import os
from pathlib import Path
import resource
import shutil
import time
import xml.etree.ElementTree as ET

import numpy as np

from seed22_continuation import require, digest, restore_parent
from candidate_continuation_journal import write_new_json, _sync_directory
from shot_benchmark_contract import canonical_hash


def signed(value):
    return dict(value, fingerprint=canonical_hash(value))


def aggregate(rows, returns):
    from shot_benchmark_native import summary
    return summary(rows, returns)


def run(manifest, experiment, split, output, session):
    import torch
    from candidate_continuation_scene import construct_scene, StartupRecorder
    from continuous_policy import build_observations, sample, physical_actions, checkpoint_payload, weight_hash, update
    from continuous_feasibility import LOW, HIGH, decode_action
    from prepare_assets import chain
    from presentation_recording import save_physics
    from seed22_continuation_contract import Ledger, batch_schedule, MONITOR_MILESTONES
    from seed22_continuation_geometry import install
    from shot_benchmark_native import atomic, compressed_json
    from shot_benchmark_reward import build_features, reward
    from settled_transit import protocol as setup_protocol

    output = Path(output)
    protocol = manifest["protocol"]
    deadline = float(os.environ["SIRCUS_CONTINUATION_DEADLINE_MONOTONIC"])
    started = float(os.environ["SIRCUS_CONTINUATION_STARTED_MONOTONIC"])
    ledger = Ledger(session)
    for name in ("requests", "checkpoints", "events"):
        (output / name).mkdir()
    write_new_json(output / "inputs.json", manifest)
    write_new_json(output / "splits.json", split)
    atomic(output / "learning-curve.json", dict(curve=[]))
    receipts, curve, assessments, changes = {}, [], [], []
    train_rows, train_rewards = [], []
    additional = completed = 0
    native = None
    last_input_check = 0.
    costs = dict(proposal_wall_s=0., update_wall_s=0., checkpoint_wall_s=0.,
        geometry_wall_s=0., geometry_process_cpu_s=0., train_episode_wall_s=0.,
        assessment_episode_wall_s=0., peak_sampled_gpu_used_mib=0.,
        scope="Call scopes; geometry is included in episode wall. Episode includes home/setup/scoring/serialization; proposal/update/checkpoint are outside it. GPU usage is sampled whole-device occupancy, not policy-only memory.")

    def progress(status):
        value = dict(schema="sircus-seed22-progress-v1", session_id=session,
            runtime_fingerprint=protocol["fingerprint"], status=status,
            train_requested=additional, train_completed=completed,
            parent_attempts=10000, parent_updates=73, new_updates=len(changes),
            elapsed_s=time.monotonic()-started,
            train=aggregate(train_rows, train_rewards), assessments=assessments,
            costs=costs,
            latest_curve=curve[-1:] if curve else [], ledger=ledger.snapshot())
        from seed22_continuation_watchdog import signed as progress_signed
        atomic(output / "progress.json", progress_signed(value))

    def guard():
        nonlocal last_input_check
        if time.monotonic() >= deadline-2:
            raise TimeoutError("Eight-hour continuation ceiling")
        if shutil.disk_usage(output).free < 6*1024**3:
            raise RuntimeError("Continuation 6GiB evidence-preserving storage floor")
        memory = {line.split(":")[0]: int(line.split()[1])*1024
                  for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:")}
        require(memory["MemAvailable"] >= 1024**3, "Host available RAM below 1GiB")
        if native is not None:
            free, total = torch.cuda.mem_get_info()
            require(free >= 512*1024**2, "4090 free memory below 512MiB")
            costs["peak_sampled_gpu_used_mib"] = max(costs["peak_sampled_gpu_used_mib"], (total-free)/1024**2)
        if time.monotonic()-last_input_check >= 60:
            from seed22_continuation import SOURCE, canonical
            require(set(manifest["source_sha256"]) == {p.name for p in SOURCE.glob("*.py")}, "Runtime source membership changed")
            assets = Path(manifest["assets"])
            require({str(p.relative_to(assets)) for p in assets.rglob("*") if p.is_file()} ==
                {k[len("asset:"):] for k in manifest["files"] if k.startswith("asset:")}, "Runtime asset membership changed")
            require(all(canonical(item["path"]).is_file() for item in manifest["files"].values()), "Runtime input link or membership changed")
            require(all(digest(SOURCE/name) == sha for name,sha in manifest["source_sha256"].items()), "Runtime source changed")
            require(all(digest(item["path"]) == item["sha256"] for item in manifest["files"].values()), "Runtime input bytes changed")
            last_input_check = time.monotonic()

    def checkpoint(label, policy, optimizer, parent=False):
        began = time.monotonic()
        guard()
        target = output / "checkpoints" / (label + ".pt")
        require(not target.exists(), "Checkpoint overwrite")
        payload = checkpoint_payload(policy, optimizer,
            study_fingerprint=protocol["fingerprint"],
            geometry_fingerprint=experiment["source"]["geometry_fingerprint"],
            domain_fingerprint=experiment["protocol"]["domain"]["fingerprint"],
            update_count=73+(0 if parent else len(changes)),
            attempted_shots=10000+(0 if parent else completed), seed=20261122)
        payload.update(parent_checkpoint_sha256=manifest["files"]["parent"]["sha256"],
            additional_attempts=0 if parent else completed, additional_updates=0 if parent else len(changes),
            split_fingerprint=split["fingerprint"], exploration_seed=20261130,
            continuation_rng_state=generator.get_state().cpu(), torch_rng_state=torch.get_rng_state(),
            cuda_rng_state=torch.cuda.get_rng_state())
        torch.save(payload, target)
        with target.open("rb") as stream:
            os.fsync(stream.fileno())
        _sync_directory(target.parent)
        costs["checkpoint_wall_s"] += time.monotonic()-began
        return dict(file=str(target.relative_to(output)), sha256=digest(target),
            state_hash=weight_hash(policy), additional_attempts=0 if parent else completed,
            additional_updates=0 if parent else len(changes))

    try:
        progress("initializing")
        guard()
        paths = {k: Path(v["path"]) for k, v in manifest["files"].items()}
        env, bundle = construct_scene(output=output/"native", assets=manifest["assets"],
            source_run=manifest["source_run"], camera=paths["camera"], device="cuda:0")
        require(bundle["spec"]["fingerprint"] == experiment["source"]["geometry_fingerprint"]
            and env.cfg == json.loads(paths["scene"].read_text()), "Constructed native scene changed")
        native = env.dest
        for name in ("trials", "media"):
            (native / name).mkdir()
        env.continuous_links = {name:chain(ET.parse(paths[role]).getroot())
            for name,role in (("Arm10","dropper_urdf"),("Arm14","board_urdf"))}
        model = ET.parse(paths["dropper_urdf"])
        env.transit_limits = torch.tensor([float(model.find(f".//joint[@name='joint{i+1}']/limit").get("effort"))
            for i in range(6)], device=env.sim.device)
        env.support_missing = [False]*35
        env.proposal_ik_seeds = [None]*35
        env.focus = 1
        original_tick = env.tick
        def tick():
            if time.monotonic() >= deadline-2:
                raise TimeoutError("Eight-hour physics ceiling")
            original_tick()
        startup = StartupRecorder(env, native/"startup").run()
        require(startup["status"] == "completed", "Startup checks incomplete")
        env.tick = tick
        install(env)
        policy, optimizer, saved = restore_parent(manifest, experiment, device=env.sim.device)
        parent, _, _ = restore_parent(manifest, experiment, device=env.sim.device)
        torch.manual_seed(20261130)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        generator = torch.Generator(device=env.sim.device).manual_seed(20261130)
        originals = copy.deepcopy(bundle["spec"]["cells"])
        parent_cp = checkpoint("parent", parent, None, parent=True)
        initial_cp = checkpoint("initial", policy, optimizer)
        require(parent_cp["state_hash"] == initial_cp["state_hash"] == saved["weight_hash"], "Parent changed before learning")
        write_new_json(output/"parent-restoration.json", dict(parent_sha256=manifest["files"]["parent"]["sha256"],
            restored_weight_hash=weight_hash(policy), optimizer_restored=True, parent_steps=292,
            parent_attempts=10000,parent_updates=73, exploration_seed=20261130,
            optimizer_state_sha256=__import__("candidate_continuation_checkpoint").state_hash(optimizer.state_dict())))
        trial_index = 0

        def trial(cells, enabled, picked, cp, method, order=None, film=False):
            nonlocal trial_index, additional
            guard()
            require(len(cells) == len(enabled) == 35, "Native cohort must contain 35 slots")
            paid = sum(enabled)
            role = cells[0]["split"]
            require(role == "train" if method == "train" else role in ("validation", "test"), "Wrong paid split")
            reservation = ledger.reserve(trial_index, role, paid)
            if role == "train":
                additional += paid
            stem = f"{method}-{trial_index:05d}"
            request = dict(index=trial_index, role=role, method=method, checkpoint=cp,
                cells=cells, enabled=enabled, request_order=order, learner_packet=picked,
                reservation=reservation, protocol_fingerprint=protocol["fingerprint"],
                additional_requested=additional, elapsed_s=time.monotonic()-started)
            write_new_json(output/"requests"/(stem+".json"), request)
            progress("training" if role == "train" else "assessing")
            begin, cpu = time.monotonic(), time.process_time()
            env.capture_trial = env.film_trial = False
            env.phase = None
            env.base_cells = [dict(copy.deepcopy(originals[i]), **copy.deepcopy(c)) for i,c in enumerate(cells)]
            env.spec["cells"] = copy.deepcopy(env.base_cells)
            env.bucket_xy = np.asarray([c["bucket_xy_m"] for c in cells])
            env.home_readback = env.home()
            env.active_mask = np.asarray(enabled, dtype=bool)
            env.trial_mode, env.trial_batch = method, trial_index
            env.frames, env.physics_samples = [], []
            env.capture_trial, env.film_trial, env.phase = True, film, "setup"
            env.trial_start_step = env.steps
            normalized = np.asarray(picked["actions_normalized"])
            physical = physical_actions(normalized, LOW, HIGH)
            actions = [decode_action(c,a,f"{method}-{trial_index}-{i}") for i,(c,a) in enumerate(zip(cells,physical))]
            rows, raw = env.shot_batch(actions, record_video=film, enabled=env.active_mask)
            env.capture_trial = env.film_trial = False
            env.phase = None
            physics, features = None, [None]*35
            if env.physics_samples:
                target = native/f"native-{trial_index:05d}.npz"
                save_physics(target,env.physics_samples)
                with target.open("rb") as stream:
                    os.fsync(stream.fileno())
                _sync_directory(target.parent)
                physics = dict(file=target.name,sha256=digest(target),samples=len(env.physics_samples))
                pos = np.asarray([s["position_m"] for s in env.physics_samples])-env.origins[None]
                vel = np.asarray([s["velocity_m_s"] for s in env.physics_samples])
                forces = np.asarray([s["forces_N"] for s in env.physics_samples])
                for i,row in enumerate(rows):
                    if not row["invalid"]:
                        features[i] = build_features(env.cfg,cells[i]["bucket_xy_m"],pos[:,i],vel[:,i],forces[:,i],env.dt,env.scoring_normals[i])
            returns = []
            for i,row in enumerate(rows):
                row.update(split=cells[i]["split"],qualification=env.qualifications[i],
                    path_rejection=env.path_reasons[i],path_proxy=env.path_checks[i],
                    scoring_board_normal=env.scoring_normals[i].tolist())
                returns.append(reward(row,features[i]))
            packet = dict(schema="sircus-seed22-native-trial-v1", mode=method,batch=trial_index,
                checkpoint=cp,cells=cells,enabled=list(map(bool,enabled)),rows=rows,actions=actions,
                observations=build_observations(cells).tolist(),actions_normalized=normalized.tolist(),
                learner_packet={k:v for k,v in picked.items() if k != "actions_normalized"},
                proposed_actions=env.proposed_actions, geometry_records=env.geometry_records,
                geometry_starts=env.geometry_starts, geometry_enabled=env.geometry_enabled_this_batch,
                geometry_filter_synthetic=env.geometry_filter_synthetic,
                proposal_ik_seeds=copy.deepcopy(env.proposal_ik_seeds),preflight_records=copy.deepcopy(env.preflight_records),
                rewards=returns,reward_features=features,training_request_order=order,
                home=env.home_readback,release_step=env.release_step,score_end_step=env.steps,physics=physics,
                origins_m=env.origins.tolist(),setup_controller=setup_protocol(),
                settling_history=raw["settling"],transit_history=env.transit_history,
                correction_history=env.correction_history,trajectory_m=raw["trajectory_m"],
                source=experiment["source"],protocol_fingerprint=protocol["fingerprint"],
                reservation_sha256=digest(output/"requests"/(stem+".json")),
                wall_s=time.monotonic()-begin,process_cpu_s=time.process_time()-cpu,
                frames=env.frames,filming=film)
            packet.update(geometry_wall_s=env.geometry_wall_s,geometry_process_cpu_s=env.geometry_process_cpu_s,
                setup_wall_s=env.setup_wall_s)
            name = "trials/"+stem+".json.gz"
            receipt = compressed_json(native/name,packet)
            with (native/name).open("rb") as stream:
                os.fsync(stream.fileno())
            _sync_directory((native/name).parent)
            write_new_json(native/(name+".receipt.json"),receipt)
            receipts[name] = receipt["sha256"]
            costs["geometry_wall_s"] += env.geometry_wall_s
            costs["geometry_process_cpu_s"] += env.geometry_process_cpu_s
            costs["train_episode_wall_s" if role == "train" else "assessment_episode_wall_s"] += time.monotonic()-begin
            completion = ledger.complete(trial_index,role,paid)
            write_new_json(output/"events"/(stem+".json"),dict(index=trial_index,receipt=receipt,
                file=name,ledger_event=completion,ledger=ledger.snapshot(),elapsed_s=time.monotonic()-started))
            trial_index += 1
            return rows, returns, name

        def pick(model, cells, deterministic):
            began = time.monotonic()
            obs = torch.as_tensor(build_observations(cells),device=env.sim.device)
            p = sample(model,obs,generator=generator,deterministic=deterministic)
            plain = {k:p[k].detach().cpu().tolist() for k in ("latent","logp","actions_normalized")}
            plain["deterministic"] = deterministic
            costs["proposal_wall_s"] += time.monotonic()-began
            return obs,p,plain

        def assess(cp, model, cells, method, film=False):
            before = weight_hash(model)
            rows, returns, files = [], [], []
            for begin in range(0,len(cells),35):
                subset = cells[begin:begin+35]
                _, _, proposal = pick(model,subset,True)
                r,v,file = trial(subset,[True]*35,proposal,cp,method,film=film and begin==0)
                rows.extend(r);returns.extend(v);files.append(file)
            require(before == weight_hash(model), "Assessment changed policy")
            record = dict(method=method,split=cells[0]["split"],additional_attempts=completed,
                checkpoint=cp,counts=aggregate(rows,returns),files=files,
                elapsed_s=time.monotonic()-started)
            assessments.append(record)
            atomic(output/"assessments.json",dict(assessments=assessments))
            progress("assessing")

        assess(parent_cp,parent,split["validation"],"raw_parent")
        assess(initial_cp,policy,split["validation"],"monitor")
        buffer, buffer_files = [], []
        stop = "request_cap"
        def learn():
            if not buffer:
                return
            guard()
            began = time.monotonic()
            before = weight_hash(policy)
            values = update(policy,optimizer,*[torch.cat(group) for group in zip(*buffer)],
                dict(epochs=4,clip=.2,value_coef=.5,entropy_coef=.005,grad_norm=1.))
            record = dict(values,additional_attempts=completed,source_files=list(buffer_files),
                update_index=len(changes), before_weight_hash=before,
                elapsed_s=time.monotonic()-started)
            changes.append(record)
            costs["update_wall_s"] += time.monotonic()-began
            write_new_json(output/"events"/f"update-{len(changes):05d}.json",record)
            buffer.clear();buffer_files.clear()

        for batch in batch_schedule():
            # Keep final evaluation within the same eight-hour native ceiling.
            if time.monotonic() >= deadline-1200:
                stop = "time_budget_reserved_final_assessment"
                break
            guard()
            cp = checkpoint(f"before-{batch['batch_index']:05d}",policy,optimizer)
            obs,p,proposal = pick(policy,split["train"],False)
            rows, returns, file = trial(split["train"],batch["enabled"],proposal,cp,"train",
                order=batch["site_indices"],film=completed in (0,1000,5000,10000,20000))
            indices = np.asarray(batch["site_indices"],dtype=int)
            train_rows.extend(rows[i] for i in indices)
            train_rewards.extend(returns[i] for i in indices)
            completed += len(indices)
            buffer.append((obs[indices],p["latent"][indices],p["logp"][indices],
                torch.tensor([returns[i] for i in indices],device=env.sim.device)))
            buffer_files.append(file)
            if len(buffer)>=4 or completed in MONITOR_MILESTONES:
                learn()
            curve.append(dict(aggregate(train_rows,train_rewards),additional_attempts=completed,
                elapsed_s=time.monotonic()-started,new_updates=len(changes)))
            atomic(output/"learning-curve.json",dict(curve=curve))
            if completed in MONITOR_MILESTONES:
                cp = checkpoint(f"milestone-{completed:05d}",policy,optimizer)
                assess(cp,policy,split["validation"],"monitor")
            progress("training")
        learn()
        final_cp = checkpoint("final",policy,optimizer)
        if completed not in MONITOR_MILESTONES:
            assess(final_cp,policy,split["validation"],"monitor-final")
        # Final TEST outcomes never influence learning or checkpoint selection.
        assess(parent_cp,parent,split["test"],"raw_parent")
        assess(parent_cp,parent,split["test"],"parent_geometry")
        assess(final_cp,policy,split["test"],"continued_geometry",film=True)
        finish_event = ledger.finish(stop)
        write_new_json(output/"events"/"ledger-finish.json",finish_event)
        write_new_json(output/"ledger-history.json",dict(events=ledger.history()))
        result = dict(schema="sircus-seed22-continuation-result-v1",status="native_complete_audit_pending",
            session_id=session,protocol_fingerprint=protocol["fingerprint"],stop_reason=stop,
            train_requested=additional,train_completed=completed,train=aggregate(train_rows,train_rewards),
            parent_attempts=10000,parent_updates=73,new_updates=len(changes),new_optimizer_steps=4*len(changes),
            final_checkpoint=final_cp,assessments=assessments,trial_sha256=receipts,
            update_events=len(changes),elapsed_s=time.monotonic()-started,
            lifetime_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            costs=costs,
            ledger=ledger.snapshot(),model_promoted=False)
        write_new_json(output/"result.json",signed(result))
        progress("native_complete_audit_pending")
        return result
    except BaseException as error:
        if native is not None:
            try:
                env.capture_trial = env.film_trial = False
                if env.physics_samples:
                    p = native / "incomplete-last-physics.npz"
                    if not p.exists():
                        save_physics(p,env.physics_samples)
            except Exception:
                pass
        if not ledger.closed:
            write_new_json(output/"events"/"ledger-failure.json",ledger.fail(type(error).__name__+": "+str(error)))
        if not (output/"ledger-history.json").exists():
            write_new_json(output/"ledger-history.json",dict(events=ledger.history()))
        write_new_json(output/"failure.json",signed(dict(status="incomplete_no_retry",
            error=type(error).__name__+": "+str(error),train_requested=additional,
            train_completed=completed,elapsed_s=time.monotonic()-started,ledger=ledger.snapshot())))
        progress("incomplete_no_retry")
        raise
