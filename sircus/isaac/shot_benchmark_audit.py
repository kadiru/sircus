"""Independent CPU audit of the versioned native sweep/PPO benchmark.

No simulator, CUDA model, optimizer update, service or hardware is started.
Original scorer audits receive a temporary legacy-reward copy; original trial
bytes retain their new reward identity. Prefix features and returns are rebuilt
separately from full-rate measured traces. Smoke receipts remain smoke-only.
"""
import argparse
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import continuous_native_audit as native
from farm_contract import reward as historical_reward
from shot_benchmark_contract import batch_plan, validate_protocol, validate_splits
from shot_benchmark_report import safe_path, source_identity
from shot_benchmark_reward import build_features, reward, protocol as reward_protocol

SCHEMA = "sircus-shot-benchmark-independent-audit-v1"
MAX_JSON_BYTES = 256*1024**2
LOGP_TOLERANCE = 5e-5
SOURCE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    def reject(value):
        raise ValueError("Nonfinite JSON: "+value)
    return json.loads(Path(path).read_bytes(), parse_constant=reject)


def close(actual, expected, label, tolerance=1e-7):
    require(native.same(actual, expected, tolerance), label+" changed")


def read_trial(run, name, expected_sha256):
    """Verify both compressed source bytes and the producer's lossless receipt."""
    run = Path(run).resolve()
    path = safe_path(run, name)
    require(path.name.endswith(".json.gz") and path.parent == Path(run)/"trials",
            "Unsafe compressed trial filename")
    require(digest(path) == expected_sha256, "Compressed trial checksum changed")
    receipt_path = safe_path(run, name+".receipt.json")
    receipt = read(receipt_path)
    require(receipt.get("file") == path.name and receipt.get("sha256") == expected_sha256
            and receipt.get("bytes") == path.stat().st_size,
            "Compressed trial receipt differs from actual bytes")
    with gzip.open(path, "rb") as stream:
        raw = stream.read(MAX_JSON_BYTES+1)
    require(len(raw) <= MAX_JSON_BYTES, "Compressed trial exceeds bounded JSON size")
    uncompressed_sha = hashlib.sha256(raw).hexdigest()
    require(receipt.get("uncompressed_sha256") == uncompressed_sha,
            "Uncompressed trial checksum changed")
    def reject(value):
        raise ValueError("Nonfinite compressed JSON: "+value)
    value = json.loads(raw, parse_constant=reject)
    return value, dict(sha256=expected_sha256, bytes=path.stat().st_size,
        uncompressed_sha256=uncompressed_sha, receipt_sha256=digest(receipt_path))


def verify_native_trial(run, trial, scene, clip=None):
    """Audit unchanged physical definitions, then independently rebuild shaping."""
    cells, origins = trial["cells"], np.asarray(trial["origins_m"])
    original = copy.deepcopy(trial)
    legacy = copy.deepcopy(trial)
    legacy["rewards"] = [historical_reward(row) for row in trial["rows"]]
    physical = native.verify_trial(run, legacy, scene, cells, origins, clip)
    # The legacy scorer adapter is in-memory only, never written or published as
    # the benchmark return. Its native scorer and all release gates stay intact.
    physics = trial.get("physics")
    arrays = None
    if physics is not None:
        with np.load(safe_path(run, physics["file"]), allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in ("position_m", "velocity_m_s", "forces_N")}
    require(len(trial["reward_features"]) == len(trial["rows"]) == len(trial["rewards"]),
            "New reward evidence count changed")
    rebuilt, values = [], []
    for i, row in enumerate(trial["rows"]):
        feature = None
        if not row["invalid"]:
            require(arrays is not None, "Admitted release lacks reward source trace")
            feature = build_features(scene, cells[i]["bucket_xy_m"],
                arrays["position_m"][:,i]-origins[i], arrays["velocity_m_s"][:,i],
                arrays["forces_N"][:,i], scene["simulation"]["physics_dt_s"], row["scoring_board_normal"])
        close(trial["reward_features"][i], feature, "Native preexternal feature packet")
        values.append(reward(row, feature));rebuilt.append(feature)
    close(trial["rewards"], values, "New benchmark reward")
    require(trial == original, "Scorer adapter mutated original benchmark trial")
    physical.update(reward_features_rebuilt=len(rebuilt), new_rewards_recomputed=len(values),
        legacy_rewards_scope="Temporary in-memory compatibility input to unchanged native scorer only")
    return physical


def summary(rows, values):
    failures = {}
    for row in rows:
        reason = row["invalid"] or ("captured" if row["route_success"] else
            "external_contact" if row["external_contact_steps"] else "released_miss")
        failures[reason] = failures.get(reason,0)+1
    return dict(attempts=len(rows), captures=sum(bool(row["route_success"]) for row in rows),
        clean=sum(row["route_quality"] == 2 for row in rows),
        rejected=sum(bool(row["invalid"]) for row in rows), failures=failures,
        mean_reward=math.fsum(values)/len(values) if values else None)


def expected_sequence(protocol, splits, seed, smoke):
    sequence = []
    def evaluations(attempts, final=False):
        for role in ["train", "validation"]+(["test"] if final else []):
            for begin in range(0,len(splits[role]),35):
                sequence.append(dict(mode="eval-"+role, cells=splits[role][begin:begin+35],
                    enabled=[True]*35, attempts_before=attempts, milestone=attempts))
    if not smoke:
        evaluations(0)
    plan = batch_plan(protocol,seed)
    for batch in plan[:1] if smoke else plan:
        sequence.append(dict(mode="train", cells=splits["train"], enabled=batch["enabled"],
            attempts_before=batch["attempt_start"], training_plan=batch))
        if not smoke and batch["milestone_after"] is not None:
            evaluations(batch["attempt_stop"],final=batch["attempt_stop"] == 10000)
    if smoke:
        sequence.append(dict(mode="smoke-validation",cells=splits["validation"],enabled=[True]*35,
            attempts_before=35,milestone="smoke"))
    return sequence


def verify_source(binding, result, progress):
    recorded = binding.get("source_sha256")
    actual = {path.name:digest(path) for path in sorted(SOURCE.glob("*.py"))}
    require(isinstance(recorded,dict) and recorded == actual,
            "Pinned executed Python source bytes differ from auditor source")
    for payload in (result, progress):
        if "source_sha256" in payload:
            require(payload["source_sha256"] == recorded, "Producer source hash bindings differ")
    return recorded


def verify_historical_pool(experiment, root, splits):
    path = safe_path(root,experiment["historical_exclusion_file"])
    require(digest(path) == experiment["historical_exclusion_sha256"],
            "Historical coordinate exclusion bytes changed")
    pool = read(path)
    coordinates = native.finite(pool["coordinates"],(len(pool["coordinates"]),2),"historical coordinates")
    require(len({tuple(xy) for xy in coordinates}) == len(coordinates)
        and isinstance(pool.get("scope"),str) and bool(pool["scope"])
        and isinstance(pool.get("sources"),list) and bool(pool["sources"]),
        "Historical exclusion discovery receipt missing or duplicated")
    require(sum(source["added_coordinates"] for source in pool["sources"]) == len(coordinates),
            "Historical discovery counts differ from saved coordinates")
    for source in pool["sources"]:
        blob = source.get("blob")
        require(isinstance(blob,str) and len(blob) == 40 and set(blob).issubset("0123456789abcdef")
            and isinstance(source.get("path"),str) and source["path"].startswith("baselines/")
            and isinstance(source.get("ref"),str) and source["ref"]
            and type(source["added_coordinates"]) is int and source["added_coordinates"] >= 0,
            "Malformed historical discovery provenance")
    validate_splits(splits,excluded_coordinates=coordinates)
    points=np.asarray([cell["bucket_xy_m"] for role in ("train","validation","test") for cell in splits[role]])
    minimum = float(np.linalg.norm(points[:,None]-coordinates[None,:],axis=2).min())
    return dict(file=experiment["historical_exclusion_file"],sha256=digest(path),
        historical_coordinates=len(coordinates),published_source_blobs=len(pool["sources"]),
        minimum_distance_m=minimum,discovery_scope=pool["scope"],
        audit_scope="Saved published-coordinate pool and discovery receipt bytes verified; no claim of unpublished/global novelty or re-enumeration of Git blobs")


def verify_home(trial):
    from continuous_pilot_audit import HOME_PROTOCOL
    home=trial.get("home",{})
    require(home.get("protocol") == HOME_PROTOCOL and home.get("episode_reset") is True
        and home.get("path_eligible") == [True]*35 and len(home.get("path_checks",[])) == 35
        and all(check.get("accepted") is True for check in home["path_checks"])
        and type(home.get("physics_step")) is int
        and home["physics_step"] == trial["transit_history"][0]["step"] <= trial["release_step"],
        "Canonical home reset/visible driven setup association changed")
    result={}
    for key in ("target_joint_rad","measured_joint_rad","measured_joint_velocity_rad_s"):
        require(set(home.get(key,{})) == {"Arm10","Arm14"},"Missing canonical home readbacks")
        result[key]=np.stack([native.finite(home[key][arm],(35,6),"home "+key) for arm in ("Arm10","Arm14")])
    return result


def verify_policy_sample(policy, trial, *, deterministic):
    """CPU likelihood and affine-action checks; no optimizer or CUDA is run."""
    import torch
    from continuous_policy import build_observations
    packet = trial.get("learner_packet")
    require(isinstance(packet,dict) and packet.get("deterministic") is deterministic,
            "Policy sampling mode changed")
    observations = torch.as_tensor(build_observations(trial["cells"]),dtype=torch.float32)
    latent = native.finite(packet.get("latent"),(35,5),"saved policy latent")
    with torch.no_grad():
        distribution = policy.distribution(observations)
        mean = distribution.loc.numpy().astype(float)
        scale = distribution.scale.numpy().astype(float)
        actual = policy.log_prob(observations,torch.tensor(latent,dtype=torch.float32)).numpy()
    saved = native.finite(packet.get("logp"),(35,),"saved policy logp")
    error = float(np.max(np.abs(saved-actual)))
    require(error <= LOGP_TOLERANCE,"Saved policy likelihood differs from frozen checkpoint")
    # Independent float64 Gaussian/Jacobian arithmetic, using actual CPU actor
    # means/scales. This checks the density convention beyond reusing log_prob.
    gaussian = -.5*np.square((latent-mean)/scale)-np.log(scale)-.5*math.log(2*math.pi)
    log_jacobian = 2*(math.log(2)-latent-np.logaddexp(0.,-2*latent))
    independent = np.sum(gaussian-log_jacobian,axis=1)
    independent_error = float(np.max(np.abs(independent-saved)))
    require(independent_error <= LOGP_TOLERANCE,"Independent tanh-Gaussian likelihood differs")
    require(np.allclose(np.tanh(latent),trial["actions_normalized"],atol=2e-7,rtol=0),
            "Saved action differs from sampled latent")
    if deterministic:
        require(np.allclose(latent,mean,atol=2e-6,rtol=0),
                "Frozen evaluation did not use direct deterministic policy mean")
    return max(error,independent_error)


def audit_run(path, plan_path):
    """Return an evidence-bound passed receipt, or raise before publication."""
    from continuous_feasibility import decode_action
    from continuous_policy import build_observations, physical_actions
    from diverse_pilot_audit import verify_settling_history
    from settled_transit import protocol as setup_protocol
    from shot_benchmark_methods import GuidedSweep

    run, plan_path = Path(path).resolve(), Path(plan_path).resolve()
    experiment, progress, result, binding, spec = (read(p) for p in (
        plan_path,run/"progress.json",run/"result.json",run/"experiment-binding.json",run/"contract.json"))
    protocol = experiment["protocol"];validate_protocol(protocol)
    require(experiment["schema"] == "sircus-shot-benchmark-experiment-v1"
        and experiment["reward"] == reward_protocol(), "Experiment/reward identity changed")
    source_identity(experiment["source"])
    splits_path = safe_path(plan_path.parent,experiment["split_manifest_file"])
    require(digest(splits_path) == experiment["split_manifest_sha256"],"Split source bytes changed")
    splits = read(splits_path);validate_splits(splits)
    historical_pool = verify_historical_pool(experiment,plan_path.parent,splits)
    require(binding["experiment_sha256"] == digest(plan_path)
        and binding["split_sha256"] == digest(splits_path)
        and binding["source"] == experiment["source"] and binding["setup"] == setup_protocol(),
        "Native experiment/controller binding changed")
    require(progress.get("schema") == "sircus-shot-benchmark-progress-v1"
        and progress.get("experiment_id") == experiment["experiment_id"]
        and result.get("status") == progress.get("status") == "completed",
        "Completed benchmark run required")
    require(result["source"] == progress["source"] == experiment["source"]
        and result["study_fingerprint"] == protocol["fingerprint"]
        and result["split_fingerprint"] == splits["fingerprint"]
        and progress["split_manifest_sha256"] == digest(splits_path), "Run source/split changed")
    source_hashes = verify_source(binding,result,progress)
    geometry = native.signed(spec,"Original geometry")
    require(geometry == experiment["source"]["geometry_fingerprint"],"Original scene geometry changed")
    camera_path = safe_path(run,"selected-camera.json")
    require(digest(camera_path) == experiment["source"]["camera_sha256"],"Selected camera bytes changed")
    camera = read(camera_path)
    method, seed, smoke = progress["method"], progress["seed"], progress["smoke"]
    require(method in protocol["methods"] and seed in protocol["seeds"] and type(smoke) is bool
        and result["smoke"] is smoke, "Unplanned method, seed or smoke identity")
    sequence = expected_sequence(protocol,splits,seed,smoke)
    receipts = progress["source_trial_sha256"]
    require(isinstance(receipts,dict) and len(receipts) == len(sequence),"Incomplete native trial sequence")
    names = sorted(receipts,key=lambda name: int(Path(name).name.split("-")[-1].split(".")[0]))
    require(set(names) == {str(p.relative_to(run)) for p in (run/"trials").glob("*.json.gz")},
            "Unlisted or missing compressed trial")
    clips_payload = read(run/"clips.json")
    clips = {clip["trial_file"]:clip for clip in clips_payload["clips"]}
    require(len(clips) == len(clips_payload["clips"]) and set(clips).issubset(receipts),"Duplicate/unbound video trial")
    sweep = GuidedSweep(splits["train"],seed,protocol["fingerprint"]) if method == "sweep" else None
    updates = read(run/"updates.json")["updates"] if method == "ppo" else []
    checkpoints, audits, train_rows, train_returns, homes = {}, [], [], [], []
    evaluation_rows, evaluation_returns, evaluation_files = {}, {}, {}
    evaluation_sources, evaluation_references = {}, {}
    frame_ids=[]
    curve_index = update_index = pending_samples = pending_batches = 0
    next_policy_hash = pending_policy_hash = next_actor_hash = pending_actor_hash = None
    maximum_logp_error = 0.

    def checkpoint(reference, attempts):
        nonlocal next_policy_hash, next_actor_hash
        artifact = safe_path(run,reference["file"])
        require(artifact.parent == run/"checkpoints" and digest(artifact) == reference["sha256"]
            and reference["training_attempts"] == attempts,"Sampling checkpoint source/counter changed")
        if method == "sweep":
            loaded = GuidedSweep.load(artifact,protocol_fingerprint=protocol["fingerprint"],train_cells=splits["train"])
            require(reference["state_hash"] == loaded.state()["fingerprint"]
                and loaded.state() == sweep.state(),"Sweep checkpoint differs from paid TRAIN replay")
            value = loaded
        else:
            from continuous_policy import load_checkpoint, weight_hash, actor_weight_hash
            from shot_benchmark_methods import make_cpu_ppo
            value, saved = load_checkpoint(artifact,study_fingerprint=protocol["fingerprint"],
                geometry_fingerprint=geometry,domain_fingerprint=protocol["domain"]["fingerprint"],
                device="cpu",expected_updates=update_index,expected_attempted_shots=attempts)
            require(saved["seed"] == seed and saved.get("benchmark_split_fingerprint") == splits["fingerprint"]
                and saved.get("benchmark_reward_version") == experiment["reward"]["version"]
                and reference["state_hash"] == weight_hash(value),"Policy checkpoint benchmark identity changed")
            if next_policy_hash is None:
                require(attempts == 0 and weight_hash(make_cpu_ppo(seed)) == weight_hash(value),
                        "Policy was not freshly initialized from the declared seed")
                next_policy_hash = weight_hash(value)
                next_actor_hash = actor_weight_hash(value)
            require(weight_hash(value) == next_policy_hash and actor_weight_hash(value) == next_actor_hash,
                    "Unexpected policy/actor mutation or update")
            require(isinstance(saved.get("optimizer"),dict),"PPO optimizer state missing")
            # Hash and sample/update chaining is audited; CUDA Adam arithmetic
            # is not claimed to be independently rerun bit for bit on CPU.
        checkpoints[reference["file"]] = reference["sha256"]
        return value

    for index,(name,expected) in enumerate(zip(names,sequence)):
        require(name == f"trials/{expected['mode']}-{index:05d}.json.gz","Native trial identity/order changed")
        trial, receipt = read_trial(run,name,receipts[name])
        close(trial.get("cells"),expected["cells"],"Fresh trial context",0.)
        require(trial.get("schema") == "sircus-shot-benchmark-native-trial-v1"
            and trial["mode"] == expected["mode"] and trial["batch"] == index
            and trial["enabled"] == expected["enabled"] and all(type(x) is bool for x in trial["enabled"])
            and trial["source"] == experiment["source"]
            and trial["split_manifest_sha256"] == digest(splits_path),"Native trial/source membership changed")
        close(trial["observations"],build_observations(expected["cells"]).tolist(),"Privileged observations")
        require(all(row["split"] == cell["split"] for row,cell in zip(trial["rows"],expected["cells"])),
                "Native row split association changed")
        normalized = native.finite(trial["actions_normalized"],(35,5),"normalized action")
        physical = physical_actions(normalized,protocol["domain"]["low"],protocol["domain"]["high"])
        decoded = [decode_action(cell,action,f"{trial['mode']}-{index}-{i}")
                   for i,(cell,action) in enumerate(zip(expected["cells"],physical))]
        close(trial["actions"],decoded,"Native physical action decoding")
        model = checkpoint(trial["checkpoint"],expected["attempts_before"])
        train = trial["mode"] == "train"
        if train:
            order=expected["training_plan"]["site_indices"]
            require(trial.get("training_request_order") == order,
                    "Paid TRAIN permutation/request order changed")
            active=np.asarray(order,dtype=int)
        else:
            require(trial.get("training_request_order") is None,"Evaluation entered TRAIN request order")
            active=np.flatnonzero(expected["enabled"])
        if method == "ppo":
            maximum_logp_error = max(maximum_logp_error,verify_policy_sample(model,trial,deterministic=not train))
            require(trial.get("proposal_metadata") is None,"PPO unexpectedly used sweep proposals")
        elif train:
            proposed = sweep.propose([expected["cells"][i] for i in active])
            close(trial["proposal_metadata"],proposed["proposal_metadata"],"Sweep paid proposal",1e-12)
            require(np.allclose(normalized[active],proposed["actions_normalized"],atol=1e-12,rtol=0),
                    "Sweep paid action changed")
            require(trial["learner_packet"] is None,"Sweep unexpectedly used PPO samples")
        else:
            predicted = model.predict(expected["cells"])
            require(np.allclose(normalized,predicted["actions_normalized"],atol=1e-12,rtol=0),
                    "Frozen sweep prediction changed")
            close(trial["learner_packet"],dict(deterministic=True,prediction_metadata=predicted["prediction_metadata"]),
                  "Frozen sweep provenance",1e-12)
        clip = clips.get(name)
        planned_film = train and (expected["attempts_before"] in (0,1000,5000)
            or expected["training_plan"]["call_index"] == len(batch_plan(protocol,seed))-1)
        require((clip is not None) == planned_film,"Prescheduled training recording changed")
        if clip is not None:
            require(clip["trial_sha256"] == receipts[name] and clip["camera_sha256"] == digest(camera_path)
                and clip["mode"] == trial["mode"] and clip["batch"] == index and clip["fps"] == 30,
                "Video provenance differs from trial/camera")
            require(clip["frames"] and [f["frame"] for f in clip["frames"]] == list(range(clip["start_frame"],clip["end_frame"])),
                    "Video source frame coverage changed")
            for frame in clip["frames"]:
                native.verify_camera_frame(frame,camera)
                require(frame["phase"] in ("setup","release") and type(frame["physics_step"]) is int
                    and trial["home"]["physics_step"] <= frame["physics_step"] <= trial["score_end_step"]
                    and ((frame["physics_step"] <= trial["release_step"]) if frame["phase"] == "setup"
                         else frame["physics_step"] > trial["release_step"]),
                    "Recorded frame lies outside its actual setup/release interval")
                native.finite(frame["all_ball_world_m"],(35,3),"recorded ball states")
                for arm in ("Arm10","Arm14"):
                    native.finite(frame["all_joints_rad"][arm],(35,6),"recorded joint states")
                image = safe_path(run,f"frames/{frame['frame']:05d}.jpg")
                require(digest(image) == frame["jpeg_sha256"] and frame["jpeg_quality"] == 82,
                        "Native JPEG source bytes changed")
                frame_ids.append(frame["frame"])
        checked = verify_native_trial(run,trial,spec["scene"],clip)
        homes.append(verify_home(trial))
        checked["settlement"] = verify_settling_history(trial,spec["scene"],expected["cells"],
            dict(settling_protocol=setup_protocol()))
        checked.update(trial_file=name,mode=trial["mode"],**receipt)
        audits.append(checked)
        rows = [trial["rows"][i] for i in active];values = [trial["rewards"][i] for i in active]
        if train:
            train_rows.extend(rows);train_returns.extend(values)
            expected_counter = summary(train_rows,train_returns)
            curve = progress["learning_curve"][curve_index]
            close({k:curve[k] for k in expected_counter},expected_counter,"Paid learning curve")
            require(type(curve["wall_s"]) in (int,float) and math.isfinite(curve["wall_s"])
                and (curve_index == 0 or curve["wall_s"] >= progress["learning_curve"][curve_index-1]["wall_s"]),
                "Training wall clock regressed")
            curve_index += 1
            if sweep is not None:
                sweep.observe([dict(metadata,split="train",reward=trial["rewards"][i])
                    for metadata,i in zip(trial["proposal_metadata"],active)])
            else:
                if pending_policy_hash is None:
                    from continuous_policy import actor_weight_hash
                    pending_policy_hash=trial["checkpoint"]["state_hash"]
                    pending_actor_hash=actor_weight_hash(model)
                require(pending_policy_hash == trial["checkpoint"]["state_hash"],"PPO batch used stale/mixed policy")
                pending_samples += len(active);pending_batches += 1
                plan = expected["training_plan"]
                if pending_batches == 4 or plan["milestone_after"] is not None or smoke:
                    change = updates[update_index]
                    require(change["before"] == pending_policy_hash and change["actor_before"] == pending_actor_hash
                        and change["samples"] == pending_samples
                        and change["attempts"] == len(train_rows)
                        and change["source_batches"] == [plan["call_index"]-pending_batches+1,plan["call_index"]]
                        and len(change["epochs"]) == 4
                        and all(epoch["samples"] == pending_samples for epoch in change["epochs"]),
                        "PPO optimizer sample/update chain changed")
                    require(change["actor_before"] != change["actor_after"],"PPO actor did not update")
                    next_policy_hash = change["after"]
                    next_actor_hash = change["actor_after"]
                    update_index += 1;pending_samples=pending_batches=0;pending_policy_hash=pending_actor_hash=None
        else:
            key = expected["milestone"]
            role = "validation" if smoke else trial["mode"][5:]
            evaluation_rows.setdefault(key,{}).setdefault(role,[]).extend(rows)
            evaluation_returns.setdefault(key,{}).setdefault(role,[]).extend(values)
            evaluation_files.setdefault(key,{})[name] = receipts[name]
            evaluation_sources.setdefault(key,{}).setdefault(role,[]).extend(
                [dict(trial_file=name,trial_sha256=receipts[name],checkpoint_sha256=trial["checkpoint"]["sha256"],
                      inference_searches=0) for _ in rows])
            if key in evaluation_references:
                require(evaluation_references[key] == trial["checkpoint"],"Frozen evaluation mixed checkpoint references")
            else:evaluation_references[key]=trial["checkpoint"]
    require(curve_index == len(progress["learning_curve"]),"Unassociated learning curve point")
    require(not pending_batches and update_index == len(updates) == result["updates"] == progress["updates"],
            "Unconsumed or missing optimizer updates")
    counters = summary(train_rows,train_returns)
    close(progress["train"],counters,"Completed TRAIN counters")
    close(result["train"],counters,"Completed result counters")
    require(counters["attempts"] == (35 if smoke else 10000),"Incomplete paid TRAIN sample budget")
    require(frame_ids == list(range(result["frames"])),"Recorded frame sequence missing, duplicated or unassociated")
    close(result["checkpoints"],progress["checkpoints"],"Frozen checkpoint summaries")
    if smoke:
        require(not progress["checkpoints"] and set(evaluation_rows) == {"smoke"}
            and set(evaluation_rows["smoke"]) == {"validation"},"Smoke accessed TEST or claimed a full milestone study")
    else:
        require([entry["training_attempts"] for entry in progress["checkpoints"]] == protocol["milestones"],
                "Missing or reordered frozen milestones")
        for entry in progress["checkpoints"]:
            attempts = entry["training_attempts"]
            artifact = safe_path(run,entry["checkpoint_file"])
            require(digest(artifact) == entry["checkpoint_sha256"],"Milestone artifact changed")
            proof_path = safe_path(run,entry["proof_file"])
            require(digest(proof_path) == entry["proof_sha256"],"Frozen proof bytes changed")
            proof = read(proof_path)
            require(proof["source_trial_sha256"] == evaluation_files[attempts]
                and proof["training_attempts"] == attempts and proof["source"] == experiment["source"]
                and proof["no_optimizer_or_flight_search"] is True
                and proof["checkpoint"] == evaluation_references[attempts]
                and proof["checkpoint"]["file"] == entry["checkpoint_file"]
                and proof["checkpoint"]["sha256"] == entry["checkpoint_sha256"]
                and proof["state_hash_before"] == proof["state_hash_after"] == proof["checkpoint"]["state_hash"],
                "Frozen evaluation state/provenance changed")
            roles = ["train","validation"]+(["test"] if attempts == 10000 else [])
            require(set(evaluation_rows[attempts]) == set(roles) and ("test" in entry) == (attempts == 10000),
                    "Held-out TEST was accessed before final frozen checkpoint")
            for role in roles:
                values = summary(evaluation_rows[attempts][role],evaluation_returns[attempts][role])
                close({k:entry[role][k] for k in values},values,"Frozen evaluation counts/rewards")
                require(len(entry[role]["layouts"]) == len(splits[role]),"Missing frozen per-layout result")
                for saved,cell,row,provenance in zip(entry[role]["layouts"],splits[role],
                        evaluation_rows[attempts][role],evaluation_sources[attempts][role]):
                    require(saved["layout_id"] == cell["id"] and saved["bucket_xy_m"] == cell["bucket_xy_m"]
                        and saved["outcome"] == dict(captured=bool(row["route_success"]),clean=row["route_quality"]==2,
                            rejected=bool(row["invalid"]),reason=row["invalid"])
                        and saved["provenance"] == provenance,
                        "Frozen per-layout outcome/provenance changed")
                    close(saved["requested"],{k:row[k] for k in (
                        "board_x_m","board_y_m","height_m","gap_m","tilt_deg","radius_m","bearing_offset_rad")},
                        "Frozen requested pose")
                    close(saved["measured"],dict(board_center_m=row["settled_board_center_m"],
                        board_normal=row["settled_board_normal"],settled=row["settled"]),"Frozen measured pose")
    timings = progress["batches"]
    require(len(timings) == len(sequence),"Missing native timing entries")
    native_training_wall = 0.
    for i,(timing,expected) in enumerate(zip(timings,sequence)):
        require(timing["index"] == i and timing["mode"] == expected["mode"],"Timing/trial association changed")
        require(type(timing["filming"]) is bool and timing["filming"] == (names[i] in clips),
                "Rendering/headless timing identity changed")
        require(all(type(timing[k]) in (int,float) and math.isfinite(timing[k]) and timing[k] >= 0
            for k in ("setup_wall_s","score_wall_s","wall_s")),"Invalid native timing observation")
        require(timing["setup_wall_s"]+timing["score_wall_s"] <= timing["wall_s"]+1e-6,
                "Native subphase timing exceeds complete episode interval")
        if expected["mode"] == "train":native_training_wall+=timing["wall_s"]
    require(progress["training_wall_s"]+1e-6 >= native_training_wall
        and progress["elapsed_s"]+1e-6 >= progress["training_wall_s"]
        and abs(progress["training_wall_s"]-progress["learning_curve"][-1]["wall_s"]) < 1e-6,
        "Training timing scope inconsistent")
    for phase in ("headless","render"):
        resource=progress["resources"][phase]
        wall=math.fsum(t["wall_s"] for t in timings if t["filming"] == (phase == "render"))
        require(abs(resource["wall_s"]-wall) < 1e-6,"Rendering/headless episode aggregate differs")
        require(all(type(resource[k]) in (int,float) and math.isfinite(resource[k]) and resource[k] >= 0
            for k in ("wall_s","cpu_s","peak_rss_mib","peak_gpu_mib")),"Invalid measured resource observation")
    home_targets=np.stack([home["target_joint_rad"] for home in homes])
    home_measured=np.stack([home["measured_joint_rad"] for home in homes])
    home_velocity=np.stack([home["measured_joint_velocity_rad_s"] for home in homes])
    require(np.ptp(home_targets,axis=0).max() <= 1e-7,"Canonical home targets changed between fresh contexts")
    return dict(schema=SCHEMA,status="passed",smoke_only=smoke,method=method,seed=seed,
        source=experiment["source"],experiment_sha256=digest(plan_path),split_manifest_sha256=digest(splits_path),
        source_result_sha256=digest(run/"result.json"),source_progress_sha256=digest(run/"progress.json"),
        source_binding_sha256=digest(run/"experiment-binding.json"),source_contract_sha256=digest(run/"contract.json"),
        historical_exclusion=historical_pool,
        source_sha256=source_hashes,audit_source_sha256=digest(__file__),
        source_trial_sha256=receipts,checkpoint_sha256=checkpoints,trials=audits,
        paid_train_attempts=counters["attempts"],evaluation_attempts=sum(len(rows) for by_role in evaluation_rows.values() for rows in by_role.values()),
        updates=update_index,max_logp_error=maximum_logp_error if method == "ppo" else None,
        native_steps=sum(value["native_steps"] for value in audits),
        rescored_releases=sum(value["rescored_releases"] for value in audits),
        frame_readbacks=sum(value["frame_readbacks"] for value in audits),
        home_readbacks=dict(trials=len(homes),commanded_targets_unchanged=True,
            maximum_joint_component_range_rad=float(np.ptp(home_measured,axis=0).max()),
            maximum_measured_speed_rad_s=float(np.abs(home_velocity).max()),
            scope="Measured canonical-home repeatability described without dropping attempts or assuming exact state equality"),
        scope="Full-rate original physical scorer and all-DOF setup readbacks, lossless bytes, new preexternal reward, paid split membership, sweep proposal/state replay or frozen CPU policy likelihood and update/sample hash chain. No CPU bitwise rerun of CUDA Adam, collision certification or physical validation.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():parser.error("Choose a new audit output; previous receipts are preserved")
    result = audit_run(args.run,args.plan)
    with args.output.open("x") as file:json.dump(result,file,allow_nan=False)
    print(json.dumps(dict(status=result["status"],method=result["method"],seed=result["seed"],
        smoke_only=result["smoke_only"],paid_train_attempts=result["paid_train_attempts"],
        trials=len(result["trials"]),output=str(args.output),sha256=digest(args.output))),flush=True)


if __name__ == "__main__":main()
