"""Opt-in, paid-attempt Isaac sweep/PPO benchmark on fresh bucket layouts.

The existing physical scorer and measured driven controller are inherited
unchanged. This module changes the experiment, reward and learner, not physics.
"""
import copy
import gzip
import hashlib
import json
import os
import resource
import shutil
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
from continuous_feasibility import LOW, HIGH, decode_action
from continuous_policy import build_observations
from presentation_recording import digest, save_physics
from settled_transit import SettledTransitShots
from shot_benchmark_contract import batch_plan, validate_protocol, validate_splits
from shot_benchmark_reward import build_features, reward, protocol as reward_protocol


def atomic(path, value):
    from parallel_runtime import atomic as write
    write(path, value)


def compressed_json(path, value):
    """Losslessly retain full native histories without the old JSON disk cost."""
    raw = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
    temp = Path(str(path)+".tmp")
    with temp.open("wb") as file:
        with gzip.GzipFile(fileobj=file, mode="wb", compresslevel=3, mtime=0) as out:
            out.write(raw)
    temp.replace(path)
    return dict(file=str(Path(path).name), sha256=digest(path),
                uncompressed_sha256=hashlib.sha256(raw).hexdigest(), bytes=Path(path).stat().st_size)


def summary(rows, returns):
    n = len(rows)
    return dict(attempts=n, captures=sum(bool(r["route_success"]) for r in rows),
        clean=sum(r["route_quality"] == 2 for r in rows),
        rejected=sum(bool(r["invalid"]) for r in rows),
        failures=dict(Counter(r["invalid"] or ("captured" if r["route_success"] else
                         "external_contact" if r["external_contact_steps"] else "released_miss") for r in rows)),
        mean_reward=float(np.mean(returns)) if n else None)


class BenchmarkShots(SettledTransitShots):
    """Current controller, fixed selected camera, lower-storage JPEG source."""
    def write_rgb(self, eye, target, elapsed_steps):
        import torch
        from PIL import Image
        from scipy.spatial.transform import Rotation
        self.camera.set_world_poses_from_view(
            torch.as_tensor(eye[None], dtype=torch.float32, device=self.sim.device),
            torch.as_tensor(target[None], dtype=torch.float32, device=self.sim.device))
        for _ in range(3): self.sim.render()
        self.camera.update(elapsed_steps*self.dt, force_recompute=True)
        data = self.camera.data
        pos = data.pos_w[0].cpu().numpy()
        quat = data.quat_w_ros[0].cpu().numpy()
        optical = Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()[:,2]
        aim = (target-eye)/np.linalg.norm(target-eye)
        if np.linalg.norm(pos-eye) > .0002 or optical@aim < .99999:
            raise RuntimeError("Benchmark camera readback differs from selected fixed view")
        rgb = data.output["rgb"][0,:,:,:3].cpu().numpy().astype(np.uint8)
        p = self.frames_path/f"{self.next_frame:05d}.jpg"
        Image.fromarray(rgb).save(p, quality=82)
        self.frames.append(dict(frame=self.next_frame, physics_step=self.steps,
            eye_m=pos.tolist(), target_m=target.tolist(), jpeg_sha256=digest(p),
            resolution=[int(rgb.shape[1]), int(rgb.shape[0])], jpeg_quality=82))
        self.next_frame += 1

    def prepare(self, actions):
        start = time.monotonic()
        result = super().prepare(actions)
        self.setup_wall_s = time.monotonic()-start
        return result


def run_benchmark(env, bundle, plan_path, method, seed, *, smoke=False):
    import torch
    from continuous_policy import (ContinuousPolicy, sample, physical_actions,
        weight_hash, actor_weight_hash, checkpoint_payload, update)
    from prepare_assets import chain
    from shot_benchmark_methods import GuidedSweep, geometry_prior
    from settled_transit import protocol as setup_protocol

    plan_path = Path(plan_path)
    experiment = json.loads(plan_path.read_text())
    protocol = experiment["protocol"]
    validate_protocol(protocol)
    splits_path = plan_path.parent/experiment["split_manifest_file"]
    if digest(splits_path) != experiment["split_manifest_sha256"]:
        raise ValueError("Benchmark split bytes changed")
    splits = json.loads(splits_path.read_text()); validate_splits(splits)
    if method not in protocol["methods"] or seed not in protocol["seeds"]:
        raise ValueError("Undeclared method/seed")
    if experiment["reward"] != reward_protocol():
        raise ValueError("Benchmark reward version changed")
    if digest(env.dest/"selected-camera.json") != experiment["source"]["camera_sha256"]:
        raise ValueError("Benchmark camera bytes changed")
    source = experiment["source"]
    source_hashes={p.name:digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    if source["geometry_fingerprint"] != bundle["spec"]["fingerprint"]:
        raise ValueError("Benchmark source geometry changed")
    # Only this newly created run and derived-media directory receive the owner
    # of the user-created experiment parent. Raw files keep normal permissions.
    parent_owner=env.dest.parent.stat()
    media_dir=env.dest/"media";media_dir.mkdir()
    for directory in (env.dest,media_dir):
        os.chown(directory,parent_owner.st_uid,parent_owner.st_gid)
    for p in (env.dest/"trials", env.dest/"checkpoints"): p.mkdir()
    atomic(env.dest/"experiment-binding.json", dict(experiment_sha256=digest(plan_path),
        split_sha256=digest(splits_path), source=source, setup=setup_protocol(),
        source_sha256=source_hashes,
        scope="Fresh contexts replace table-local bucket poses only at canonical episode reset; geometry/scoring/impact materials unchanged"))
    env.continuous_links = {n:chain(ET.parse(Path(env.args.assets)/f).getroot())
        for n,f in (("Arm10","dropper.urdf"),("Arm14","board_arm.urdf"))}
    model = ET.parse(Path(env.args.assets)/"dropper.urdf")
    env.transit_limits = torch.tensor([float(model.find(f".//joint[@name='joint{i+1}']/limit").get("effort"))
                                     for i in range(6)], device=env.sim.device)
    env.support_missing = [False]*35
    env.focus = 1
    torch.manual_seed(seed); torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False; torch.set_float32_matmul_precision("highest")
    policy = ContinuousPolicy().to(env.sim.device) if method == "ppo" else None
    optimizer = torch.optim.Adam(policy.parameters(), lr=.0003) if policy is not None else None
    sweep = GuidedSweep(splits["train"], seed, protocol["fingerprint"]) if method == "sweep" else None
    counters = dict(attempts=0, captures=0, clean=0, rejected=0, failures={}, mean_reward=None)
    changes, curve, checks, clip_manifest, timings, receipts, buffer = [], [], [], [], [], {}, []
    total_return = 0.; trial_index=0; train_wall=0.
    overhead=dict(proposal_wall_s=0.,update_wall_s=0.,checkpoint_wall_s=0.,
        scope="Measured call wall times; checkpoint time includes TRAIN/initial/milestone artifacts. Already included in training_wall_s when inside a paid TRAIN iteration; do not add twice.")
    resources = {k:dict(wall_s=0., cpu_s=0., peak_rss_mib=0., peak_gpu_mib=0.,
        scope="Native episode wall/CPU intervals by filming flag, including home/evidence serialization but excluding learner/proposal/checkpoint calls; evaluation included. RSS process lifetime high-water and sampled whole-device occupied GPU memory, not isolated phase peaks. Full TRAIN iteration timing is separately logged.")
        for k in ("headless","render")}
    originals = copy.deepcopy(bundle["spec"]["cells"])
    started = time.monotonic()

    def progress(status, error=None):
        p = dict(schema="sircus-shot-benchmark-progress-v1", experiment_id=experiment["experiment_id"],
            method=method, seed=seed, status=status, source=source,source_sha256=source_hashes,
            split_manifest_sha256=experiment["split_manifest_sha256"], train=counters,
            learning_curve=curve, checkpoints=checks, resources=resources, batches=timings,
            source_trial_sha256=receipts,
            videos=(json.loads((env.dest/"video-index.json").read_text())["videos"]
                    if (env.dest/"video-index.json").exists() else []), training_wall_s=train_wall,
            elapsed_s=time.monotonic()-started, updates=len(changes), smoke=smoke,
            algorithm_overhead=overhead)
        if error:p["error"]=error
        atomic(env.dest/"progress.json", p)

    def checkpoint(label):
        begin=time.monotonic()
        name=f"checkpoints/{label}"+(".pt" if policy is not None else ".json")
        path=env.dest/name
        if policy is not None:
            payload=checkpoint_payload(policy,optimizer,study_fingerprint=protocol["fingerprint"],
                geometry_fingerprint=source["geometry_fingerprint"],domain_fingerprint=protocol["domain"]["fingerprint"],
                update_count=len(changes),attempted_shots=counters["attempts"],seed=seed)
            payload.update(torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state(),
                benchmark_split_fingerprint=splits["fingerprint"],benchmark_reward_version=experiment["reward"]["version"])
            torch.save(payload,path)
            state_hash=weight_hash(policy)
        else:
            sweep.save(path);state_hash=sweep.state()["fingerprint"]
        result=dict(file=name,sha256=digest(path),state_hash=state_hash,training_attempts=counters["attempts"])
        overhead["checkpoint_wall_s"]+=time.monotonic()-begin
        return result

    def install_contexts(cells):
        if len(cells)!=35:raise ValueError("Every farm cohort needs 35 distinct tables")
        if len({tuple(c["bucket_xy_m"]) for c in cells})!=35:raise ValueError("Duplicate farm bucket positions")
        env.capture_trial=env.film_trial=False;env.phase=None
        env.base_cells=[dict(copy.deepcopy(originals[i]),**copy.deepcopy(c)) for i,c in enumerate(cells)]
        env.spec["cells"]=copy.deepcopy(env.base_cells)
        env.bucket_xy=np.asarray([c["bucket_xy_m"] for c in cells])
        # Bucket moves only as part of the canonical simulation reset.
        env.home_readback=env.home()

    def trial(cells, enabled, normalized, picked, mode, cp, film=False, proposal_meta=None,request_order=None):
        nonlocal trial_index
        if shutil.disk_usage(env.dest).free < 6*1024**3:
            raise RuntimeError("Benchmark storage guard: less than 6GiB free; prior data never deleted")
        begin=time.monotonic();cpu=time.process_time()
        install_contexts(cells)
        env.active_mask=np.asarray(enabled,dtype=bool)
        env.trial_mode=mode;env.trial_batch=trial_index
        env.frames=[];env.physics_samples=[]
        env.capture_trial=True;env.film_trial=film;env.phase="setup"
        env.trial_start_step=env.steps;first=env.next_frame
        physical=physical_actions(normalized,LOW,HIGH)
        actions=[decode_action(c,a,f"{mode}-{trial_index}-{i}") for i,(c,a) in enumerate(zip(cells,physical))]
        rows,raw=env.shot_batch(actions,record_video=film,enabled=env.active_mask)
        env.capture_trial=env.film_trial=False;env.phase=None
        physics=None;features=[None]*35
        if env.physics_samples:
            p=env.dest/f"native-{trial_index:05d}.npz";save_physics(p,env.physics_samples)
            physics=dict(file=p.name,sha256=digest(p),samples=len(env.physics_samples))
            pos=np.asarray([s["position_m"] for s in env.physics_samples])-env.origins[None]
            vel=np.asarray([s["velocity_m_s"] for s in env.physics_samples])
            force=np.asarray([s["forces_N"] for s in env.physics_samples])
            for i,r in enumerate(rows):
                if not r["invalid"]:
                    features[i]=build_features(env.cfg,cells[i]["bucket_xy_m"],pos[:,i],vel[:,i],force[:,i],env.dt,env.scoring_normals[i])
        returns=[]
        for i,r in enumerate(rows):
            r.update(split=cells[i]["split"],qualification=env.qualifications[i],path_rejection=env.path_reasons[i],
                path_proxy=env.path_checks[i],scoring_board_normal=env.scoring_normals[i].tolist())
            returns.append(reward(r,features[i]))
        name=f"trials/{mode}-{trial_index:05d}.json.gz"
        packet=dict(schema="sircus-shot-benchmark-native-trial-v1",mode=mode,batch=trial_index,
            checkpoint=cp,source=source,split_manifest_sha256=experiment["split_manifest_sha256"],
            cells=cells,enabled=list(map(bool,enabled)),rows=rows,actions=actions,
            observations=build_observations(cells).tolist(),actions_normalized=np.asarray(normalized).tolist(),
            learner_packet=picked,proposal_metadata=proposal_meta,rewards=returns,reward_features=features,
            training_request_order=request_order,
            home=env.home_readback,release_step=env.release_step,score_end_step=env.steps,
            physics=physics,origins_m=env.origins.tolist(),
            setup_controller=setup_protocol(),settling_history=raw["settling"],
            transit_history=env.transit_history,correction_history=env.correction_history,
            trajectory_m=raw["trajectory_m"])
        saved=compressed_json(env.dest/name,packet);receipts[name]=saved["sha256"]
        atomic(env.dest/(name+".receipt.json"),saved)
        if film:
            clip=dict(mode=mode,batch=trial_index,training_attempts=counters["attempts"],method=method,seed=seed,
                trial_file=name,trial_sha256=receipts[name],start_frame=first,end_frame=env.next_frame,
                frames=env.frames,fps=30,release_step=env.release_step,camera_sha256=source["camera_sha256"],
                rows=rows,actions=actions,
                scope="Actual prescheduled native trials, scripted driven setup and initialized release; fixed camera, no outcome selection")
            clip_manifest.append(clip);atomic(env.dest/"clips.json",dict(clips=clip_manifest))
        elapsed=time.monotonic()-begin
        use=resources["render" if film else "headless"]
        use["wall_s"]+=elapsed;use["cpu_s"]+=time.process_time()-cpu
        use["peak_rss_mib"]=max(use["peak_rss_mib"],resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        free,total=torch.cuda.mem_get_info()
        use["peak_gpu_mib"]=max(use["peak_gpu_mib"],(total-free)/1024**2)
        timings.append(dict(index=trial_index,mode=mode,filming=film,setup_wall_s=env.setup_wall_s,
            score_wall_s=max(0.,raw["wall_s"]-env.setup_wall_s),wall_s=elapsed))
        trial_index+=1
        return rows,returns,elapsed,name

    def predict(cells):
        if policy is not None:
            with torch.no_grad():p=sample(policy,torch.tensor(build_observations(cells),device=env.sim.device),deterministic=True)
            return p["actions_normalized"].cpu().numpy(),dict(deterministic=True,latent=p["latent"].cpu().tolist(),logp=p["logp"].cpu().tolist())
        p=sweep.predict(cells)
        return p["actions_normalized"],dict(deterministic=True,prediction_metadata=p["prediction_metadata"])

    def evaluate(cp, final=False):
        progress("evaluating")
        before=weight_hash(policy) if policy is not None else sweep.state()["fingerprint"]
        proof=dict(schema="sircus-shot-benchmark-frozen-proof-v1",checkpoint=cp,
            no_optimizer_or_flight_search=True,source=source,source_trial_sha256={})
        result=dict(training_attempts=counters["attempts"],checkpoint_file=cp["file"],checkpoint_sha256=cp["sha256"])
        roles=["train","validation"]+(["test"] if final else [])
        for role in roles:
            totals=[];all_returns=[];layout_rows=[]
            for start in range(0,len(splits[role]),35):
                cells=splits[role][start:start+35]
                normalized,picked=predict(cells)
                rows,returns,_,file=trial(cells,[True]*35,normalized,picked,"eval-"+role,cp)
                proof["source_trial_sha256"][file]=receipts[file]
                totals.extend(rows);all_returns.extend(returns)
                for c,r in zip(cells,rows):
                    layout_rows.append(dict(layout_id=c["id"],bucket_xy_m=c["bucket_xy_m"],
                        requested={k:r[k] for k in ("board_x_m","board_y_m","height_m","gap_m","tilt_deg","radius_m","bearing_offset_rad")},
                        measured=dict(board_center_m=r["settled_board_center_m"],board_normal=r["settled_board_normal"],settled=r["settled"]),
                        outcome=dict(captured=bool(r["route_success"]),clean=r["route_quality"]==2,rejected=bool(r["invalid"]),reason=r["invalid"]),
                        provenance=dict(trial_file=file,trial_sha256=receipts[file],checkpoint_sha256=cp["sha256"],inference_searches=0)))
            result[role]=dict(summary(totals,all_returns),layouts=layout_rows)
        after=weight_hash(policy) if policy is not None else sweep.state()["fingerprint"]
        if after!=before:raise RuntimeError("Evaluation mutated the frozen learner")
        proof.update(state_hash_before=before,state_hash_after=after,training_attempts=counters["attempts"])
        file=f"checkpoints/proof-{counters['attempts']:05d}.json";atomic(env.dest/file,proof)
        result.update(proof_file=file,proof_sha256=digest(env.dest/file));checks.append(result)
        progress("training")

    try:
        progress("running")
        initial=checkpoint("milestone-00000")
        if not smoke:evaluate(initial)
        plan=batch_plan(protocol,seed)
        if smoke:plan=plan[:1]
        for b in plan:
            iteration_begin=time.monotonic()
            cells=splits["train"];enabled=b["enabled"];indices=np.asarray(b["site_indices"],dtype=int)
            cp=checkpoint(f"before-{b['call_index']:04d}")
            proposal_begin=time.monotonic()
            if policy is not None:
                observations=torch.tensor(build_observations(cells),device=env.sim.device)
                with torch.no_grad():p=sample(policy,observations)
                normalized=p["actions_normalized"].cpu().numpy()
                picked=dict(deterministic=False,latent=p["latent"].cpu().tolist(),logp=p["logp"].cpu().tolist())
                metadata=None
            else:
                p=sweep.propose([cells[i] for i in indices]);metadata=p["proposal_metadata"]
                normalized=geometry_prior(cells)["actions_normalized"].copy()
                normalized[indices]=p["actions_normalized"];picked=None
            overhead["proposal_wall_s"]+=time.monotonic()-proposal_begin
            film=counters["attempts"] in (0,1000,5000) or b["call_index"]==len(batch_plan(protocol,seed))-1
            rows,returns,wall,_=trial(cells,enabled,normalized,picked,"train",cp,film=film,
                                    proposal_meta=metadata,request_order=indices.tolist())
            active=[rows[i] for i in indices];ret=[returns[i] for i in indices]
            stats=summary(active,ret)
            for k in ("attempts","captures","clean","rejected"):counters[k]+=stats[k]
            for k,v in stats["failures"].items():counters["failures"][k]=counters["failures"].get(k,0)+v
            total_return+=sum(ret);counters["mean_reward"]=total_return/counters["attempts"]
            update_begin=time.monotonic()
            if sweep is not None:
                sweep.observe([dict(m,split="train",reward=returns[i]) for m,i in zip(metadata,indices)])
            else:
                buffer.append((observations[indices],p["latent"][indices],p["logp"][indices],torch.tensor(ret,device=env.sim.device)))
                if len(buffer)>=4 or b["milestone_after"] is not None or smoke:
                    values=update(policy,optimizer,*[torch.cat(group) for group in zip(*buffer)],
                        dict(epochs=4,clip=.2,value_coef=.5,entropy_coef=.005,grad_norm=1.))
                    values.update(attempts=counters["attempts"],source_batches=[b["call_index"]-len(buffer)+1,b["call_index"]])
                    changes.append(values);buffer.clear();atomic(env.dest/"updates.json",dict(updates=changes))
            overhead["update_wall_s"]+=time.monotonic()-update_begin
            train_wall+=time.monotonic()-iteration_begin
            curve.append(dict(copy.deepcopy(counters),wall_s=train_wall))
            if b["milestone_after"] is not None and not smoke:
                cp=checkpoint(f"milestone-{counters['attempts']:05d}")
                evaluate(cp,final=counters["attempts"]==10000)
            progress("training")
        if smoke:
            # Independent validation contexts only; TEST never inspected in smoke.
            cp=checkpoint("smoke-final")
            normalized,picked=predict(splits["validation"])
            trial(splits["validation"],[True]*35,normalized,picked,"smoke-validation",cp)
            progress("completed")
        else:
            if counters["attempts"]!=10000:raise RuntimeError("Incomplete paid TRAIN budget")
            progress("completed")
        atomic(env.dest/"result.json",dict(status="completed",smoke=smoke,train=counters,
            source=source,source_sha256=source_hashes,study_fingerprint=protocol["fingerprint"],split_fingerprint=splits["fingerprint"],
            checkpoints=checks,updates=len(changes),frames=env.next_frame,seconds=time.monotonic()-started))
    except BaseException as exc:
        progress("failed",str(exc));raise
