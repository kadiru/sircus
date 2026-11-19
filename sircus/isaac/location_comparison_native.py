"""Explicit bridge to the unchanged PPO and native shot helpers.

Importing this module does not import Torch/Isaac or start an app. A caller must
provide an already-owned scene, a guard and authorization for real execution.
Tests inject synthetic dependencies and an invented environment instead.
"""

import copy
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time

SLOTS = 35
PARENT_SHA = "57d7658e9ed2f72ad09cf4e8523ab497ccbe0708364f1b8ec4a4c5c1aef1fe9f"
PREPARED_FINGERPRINT = "440cbcca29612820d3b36005010edd90fd4f8172646bd8ea95e57f154775f4e6"
EXPERIMENT_SHA = "77afe173a07917dd87ce43f37809a9ac6ef99beebd7e8e35f703c0e5cf1e3a26"
SCENE_SHA = "27ebf306e5a24a2702f13fa88e100cb107520c56688c15e9a8868350d4b5bf97"
SOURCE = Path(__file__).resolve().parent
CRITICAL_SOURCES = {
    "continuous_policy.py": "66fb4815ed76275d778b46c8cb8fba86a21bf175557e7c64460cecbbe4373613",
    "continuous_geometry_layer.py": "c6489a44e268610470edd2a4c37bd4cfb1a78c89989883113f76996bd8fbad23",
    "seed22_continuation_geometry.py": "b42e7800b82e7f2245e0deefe878c3293b9a774bf22a193081f951e2c6476a03",
    "settled_transit.py": "08e37329978f3acbe4476ebaf5513cfa28539985c806daa4a071596d381dd0a7",
    "continuous_pilot.py": "d9e247d4cd708b67c59091250f11f0a772b92e45ebfc5aed56c2efd0847b9b21",
    "parallel_runtime.py": "33261c6eadbc7d0f4f823caaedb80f8e95f08229da5d545c688043d945b1b25d",
    "shot_benchmark_reward.py": "4aad0f1c5d8f721e01376bdfd56e19c1d998a139c97d3a5cf89dd3fe8d359eb5",
    "prepare_assets.py": "d7517e4f1f350bd7f48e5c23e48fc4dc4c63882b6b18ce4502bc8c5fd1999d93",
    "continuous_feasibility.py": "4a7ce280ae65f1c08bad711abefab03c5cc949736bc47c8058b2798ac0acb5e4",
    "joint_feasibility.py": "7fdf85de0d86fbb3bba161fd280d9e2939983e8d81687a42cdf40c5921803b70",
    "robust_ik.py": "c3e5ce8673042c2fefbb784e825b34b30bb334a1ea3f9896cf0de0f975c71ee9",
    "rl_contract.py": "077471e80a63eba605db24a8c928c4fd1073636aafae3d9d9d44c59d3f2abcfd",
    "board_route.py": "6409889aed6b0fd6a288eb9494b954ec5c3233ec41fdcf146940fc23276fb2e9",
    "presentation_recording.py": "56666d2a6311dfdfc50eddde6e804ab9fc1e688cee844ed59e39ced58a2f1912",
    "feasibility_runtime.py": "679636df4143239b1aa22a9355a8d6e62cc95e4f11b4eee208c5bf8b5e0b8213",
    "expanded_pose_native.py": "b5092c3003419c04f569fe4475064a0ea92f907efe5a96cbcbb0872898690d37",
    "settling_contract.py": "d2586973fa21527f552de9c39569a88f31d4a9214c085d0756867869f572b9b1",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def _plain(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _hash(value):
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    raw = json.dumps(_plain(value), sort_keys=True, allow_nan=False).encode()
    with Path(path).open("xb") as stream:
        stream.write(gzip.compress(raw, mtime=0) if str(path).endswith(".gz") else raw)
        stream.flush()
        os.fsync(stream.fileno())
    _sync_parent(path)


def _sync_parent(path):
    descriptor = os.open(str(Path(path).parent), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir(path):
    path.mkdir(exist_ok=True)
    _sync_parent(path)


def _ids(cell):
    return cell.get("request_id", cell["id"]), cell.get("target_id", cell.get("original_id", cell["id"]))


def validate_native_inputs(manifest, experiment, plan, *, env_cfg, env_geometry_fingerprint):
    """Byte checks before Torch loading; this does not authorize or start an app.

    The external owner must additionally pin this new manifest and supervise the
    process. Source/asset checks here are not a substitute for native trace audit.
    """
    require(plan.get("fingerprint") == PREPARED_FINGERPRINT
            and _hash({k:v for k,v in plan.items() if k != "fingerprint"}) == PREPARED_FINGERPRINT,
            "Different prepared comparison package")
    files = manifest["files"]
    for name, item in files.items():
        path = Path(item["path"])
        require(path.is_absolute() and path.is_file() and str(path.resolve()) == str(path)
                and not any(p.is_symlink() for p in (path, *path.parents))
                and _file_hash(path) == item["sha256"], "Changed native input: " + name)
    for role, expected in (("parent", PARENT_SHA), ("experiment", EXPERIMENT_SHA), ("scene", SCENE_SHA)):
        require(files[role]["sha256"] == expected, "Wrong immutable native input: " + role)
    require(json.loads(Path(files["experiment"]["path"]).read_bytes()) == experiment,
            "Experiment object differs from original bytes")
    require(json.loads(Path(files["scene"]["path"]).read_bytes()) == env_cfg
            and env_geometry_fingerprint == experiment["source"]["geometry_fingerprint"],
            "Constructed native scene or source geometry differs")
    declared = manifest["source_sha256"]
    require(set(declared) == {path.name for path in SOURCE.glob("*.py")}, "Native source membership changed")
    require(all(_file_hash(SOURCE/name) == checksum for name, checksum in declared.items()),
            "Native source bytes changed")
    require(all(declared.get(name) == checksum for name, checksum in CRITICAL_SOURCES.items()),
            "Original policy/geometry/native/scoring helper changed")
    require(plan["bindings"]["continuous_geometry_layer"] == declared["continuous_geometry_layer.py"],
            "Prepared geometry layer binding changed")
    return True


def validate_source_imports(manifest, names=()):
    """Reject source helpers resolved or already loaded from another checkout."""
    declared = manifest["source_sha256"]
    for name in names:
        spec = importlib.util.find_spec(name)
        require(spec is not None and spec.origin
                and Path(spec.origin).resolve() == SOURCE/(name+".py"),
                "Native helper import resolves outside verified source: " + name)
    for filename, checksum in declared.items():
        module = sys.modules.get(Path(filename).stem)
        if module is not None:
            origin = getattr(module, "__file__", None)
            require(origin is not None and Path(origin).resolve() == SOURCE/filename
                    and _file_hash(origin) == checksum,
                    "Loaded native helper differs from verified source: " + filename)


class ProductionDependencies:
    """Lazy production calls, isolated from the standard-library mock runner."""

    synthetic = False

    def __init__(self, manifest, experiment, device):
        validate_source_imports(manifest, (
            "continuous_policy", "continuous_feasibility", "shot_benchmark_reward",
            "candidate_continuation_checkpoint", "seed22_continuation",
            "seed22_continuation_geometry", "presentation_recording", "settled_transit"))
        import numpy as np
        import torch
        import continuous_policy as policy
        import continuous_feasibility as geometry
        import shot_benchmark_reward as feedback
        from candidate_continuation_checkpoint import state_hash
        from seed22_continuation import restore_parent
        from seed22_continuation_geometry import install
        from presentation_recording import save_physics
        from settled_transit import protocol
        validate_source_imports(manifest)
        self.np, self.torch, self.policy = np, torch, policy
        self.geometry, self.feedback, self.state_hash = geometry, feedback, state_hash
        self.restore_parent, self.install = restore_parent, install
        self.save_physics, self.setup_protocol = save_physics, protocol
        self.manifest, self.experiment, self.device = manifest, experiment, device
        self.devices = [torch.device(device).index or 0] if str(device).startswith("cuda") else []
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    def prepare_environment(self, env):
        import xml.etree.ElementTree as ET
        validate_source_imports(self.manifest, ("prepare_assets",))
        from prepare_assets import chain
        validate_source_imports(self.manifest)
        paths = {name: Path(item["path"]) for name, item in self.manifest["files"].items()}
        require(_file_hash(paths["parent"]) == PARENT_SHA, "Parent checkpoint bytes changed")
        require(env.cfg == json.loads(paths["scene"].read_text()), "Constructed physical scene differs")
        env.continuous_links = {name: chain(ET.parse(paths[role]).getroot())
            for name, role in (("Arm10", "dropper_urdf"), ("Arm14", "board_urdf"))}
        model = ET.parse(paths["dropper_urdf"])
        env.transit_limits = self.torch.tensor([
            float(model.find(f".//joint[@name='joint{i+1}']/limit").get("effort")) for i in range(6)],
            device=self.device)
        env.support_missing, env.proposal_ik_seeds, env.focus = [False]*SLOTS, [None]*SLOTS, 1

    def _rng(self):
        return dict(cpu=self.torch.get_rng_state().clone(),
                    cuda=[self.torch.cuda.get_rng_state(i).clone() for i in self.devices])

    def restore(self, action_seed, optimizer_seed):
        # Entropy sampling inside the original update uses global Torch RNG.
        # Each model retains its own stream; fork_rng restores caller globals.
        with self.torch.random.fork_rng(devices=self.devices):
            model, optimizer, saved = self.restore_parent(self.manifest, self.experiment, self.device)
            self.torch.manual_seed(optimizer_seed)
            rng = self._rng()
        return dict(policy=model, optimizer=optimizer,
                    generator=self.torch.Generator(device=self.device).manual_seed(action_seed),
                    optimizer_rng=rng, saved=saved)

    def state(self, handle):
        steps = {int(float(state["step"])) for state in handle["optimizer"].state.values()}
        require(len(steps) == 1, "Missing or inconsistent Adam step counts")
        return dict(weight_hash=self.policy.weight_hash(handle["policy"]),
            optimizer_hash=self.state_hash(handle["optimizer"].state_dict()),
            optimizer_steps=steps.pop(), action_rng_hash=self.state_hash(handle["generator"].get_state()),
            optimizer_rng_hash=self.state_hash(handle["optimizer_rng"]))

    def observations(self, cells):
        return self.policy.build_observations(cells).tolist()

    def sample(self, handle, cells, deterministic):
        observation = self.torch.as_tensor(self.policy.build_observations(cells), device=self.device)
        packet = self.policy.sample(handle["policy"], observation,
                                   generator=handle["generator"], deterministic=deterministic)
        return {key: _plain(packet[key]) for key in ("latent", "logp", "actions_normalized")}

    def update(self, handle, examples):
        t = self.torch
        values = [t.tensor([example["proposal"][key] for example in examples], device=self.device, dtype=t.float32)
                  for key in ("observation", "latent", "logp")]
        returns = t.tensor([example["outcome"]["reward"] for example in examples], device=self.device, dtype=t.float32)
        with t.random.fork_rng(devices=self.devices):
            t.set_rng_state(handle["optimizer_rng"]["cpu"])
            for index, state in zip(self.devices, handle["optimizer_rng"]["cuda"]):
                t.cuda.set_rng_state(state, index)
            result = self.policy.update(handle["policy"], handle["optimizer"], *values, returns,
                dict(epochs=4, clip=.2, value_coef=.5, entropy_coef=.005, grad_norm=1.))
            handle["optimizer_rng"] = self._rng()
        return result

    def save_checkpoint(self, handle, path, metadata):
        value = self.policy.checkpoint_payload(handle["policy"], handle["optimizer"],
            study_fingerprint=metadata["comparison_fingerprint"],
            geometry_fingerprint=self.experiment["source"]["geometry_fingerprint"],
            domain_fingerprint=self.experiment["protocol"]["domain"]["fingerprint"],
            update_count=metadata["state"]["updates"], attempted_shots=metadata["state"]["attempts"],
            seed=20261122)
        value.update(metadata, action_rng_state=handle["generator"].get_state().cpu(),
                     optimizer_rng_state=handle["optimizer_rng"])
        with Path(path).open("xb") as stream:
            self.torch.save(value, stream)
            stream.flush(); os.fsync(stream.fileno())

    def actions(self, cells, normalized, label):
        physical = self.policy.physical_actions(normalized, self.geometry.LOW, self.geometry.HIGH)
        return [self.geometry.decode_action(cell, value, f"{label}-{i}")
                for i, (cell, value) in enumerate(zip(cells, physical))]

    def features(self, env, cells, rows):
        result = [None] * SLOTS
        if not env.physics_samples:
            require(all(row["invalid"] for row in rows), "Released native trial lacks full-rate evidence")
            return result
        samples = env.physics_samples
        pos = self.np.asarray([s["position_m"] for s in samples]) - env.origins[None]
        vel = self.np.asarray([s["velocity_m_s"] for s in samples])
        force = self.np.asarray([s["forces_N"] for s in samples])
        for i, row in enumerate(rows):
            if not row["invalid"]:
                result[i] = self.feedback.build_features(env.cfg, cells[i]["bucket_xy_m"],
                    pos[:, i], vel[:, i], force[:, i], env.dt, env.scoring_normals[i])
        return result

    def reward(self, row, features):
        return self.feedback.reward(row, features)

    def resource_sample(self):
        import resource
        import sys
        raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        value = dict(lifetime_peak_rss_mib=raw_rss / (1024**2 if sys.platform == "darwin" else 1024),
                     memory_scope="Lifetime process RSS peak; sampled whole-device GPU occupancy, not policy-only or phase-local peak")
        if self.devices:
            free, total = self.torch.cuda.mem_get_info(self.device)
            value["sampled_gpu_used_mib"] = (total-free) / 1024**2
        return value


class NativeEngine:
    """One already-owned scene and three independently restored policy copies."""

    def __init__(self, env, manifest, experiment, output, plan, *, guard,
                 dependencies=None, execution_authorized=False, clock=time.monotonic):
        require(callable(guard), "An external bound/resource guard is required")
        require(execution_authorized is True or getattr(dependencies, "synthetic", False) is True,
                "Native bridge is not execution authorization")
        require(manifest["files"]["parent"]["sha256"] == PARENT_SHA, "Wrong parent identity")
        self.env, self.manifest, self.experiment, self.plan = env, manifest, experiment, plan
        self.output, self.guard, self.clock = Path(output), guard, clock
        require(self.output.is_dir(), "Caller must own an existing fresh output directory")
        require(all(not (self.output/name).exists() for name in ("comparison-native", "checkpoints", "updates")),
                "Bridge output paths must be untouched")
        if not getattr(dependencies, "synthetic", False):
            require(dependencies is None, "Production dependencies cannot be substituted")
            validate_native_inputs(manifest, experiment, plan, env_cfg=env.cfg,
                                   env_geometry_fingerprint=env.spec.get("fingerprint"))
        self.deps = dependencies or ProductionDependencies(manifest, experiment, env.sim.device)
        self.synthetic = bool(self.deps.synthetic)
        self.models, self.trial_index, self.used_proposals, self.used_examples = {}, 0, set(), set()
        self.sampling_checkpoints = {}
        self.dispatched, self.learning_examples = set(), {}
        self.originals = copy.deepcopy(env.spec["cells"])
        require(env.n == len(self.originals) == SLOTS, "Expected the original 35 native slots")
        self.deps.prepare_environment(env)
        self.cleanup_geometry = self.deps.install(env)
        self.closed, self.failed = False, False

    def _guard(self):
        require(not self.closed and not self.failed, "Engine is closed or failed; no retry")
        require(self.guard() is True, "External native bound/resource guard did not pass")

    def restore(self, name, action_seed, optimizer_seed):
        self._guard()
        require(type(name) is str and re.fullmatch(r"[A-Za-z0-9_-]{1,96}", name), "Unsafe model identity")
        require(name not in self.models, "A model cannot be restored twice")
        handle = self.deps.restore(action_seed, optimizer_seed)
        self.models[name] = dict(handle=handle, updates=73, attempts=10000,
                                 action_seed=action_seed, optimizer_seed=optimizer_seed)
        state = self.state(name)
        require(state["optimizer_steps"] == 292, "Wrong inherited optimizer state")
        return state

    def state(self, name):
        model = self.models[name]
        return dict(self.deps.state(model["handle"]), model_id=name,
            updates=model["updates"], attempts=model["attempts"],
            action_seed=model["action_seed"], optimizer_seed=model["optimizer_seed"],
            parent_checkpoint_sha256=PARENT_SHA)

    def propose(self, name, cells, deterministic):
        self._guard()
        require(type(deterministic) is bool and 1 <= len(cells) <= SLOTS, "Invalid proposal cohort")
        # Persist every distinct learned state before sampling. Milestone-only
        # saves cannot support independent likelihood checks for intervening
        # TRAIN batches. These evidence snapshots perform no learning.
        state = self.state(name)
        key = (name, state["updates"])
        if key not in self.sampling_checkpoints:
            self.sampling_checkpoints[key] = self.checkpoint(name, f"sampling-{state['updates']:05d}")
        sampling_checkpoint = self.sampling_checkpoints[key]
        require(all(sampling_checkpoint["state"][field] == state[field] for field in
                    ("model_id", "weight_hash", "optimizer_hash", "optimizer_steps", "updates", "attempts")),
                "Sampling state changed without a recorded PPO update")
        observations = self.deps.observations(cells)
        result = self.deps.sample(self.models[name]["handle"], cells, deterministic)
        rows = []
        for i, cell in enumerate(cells):
            request_id, target_id = _ids(cell)
            row = dict(request_id=request_id, target_id=target_id, observation=observations[i],
                       latent=result["latent"][i], logp=result["logp"][i],
                       actions_normalized=result["actions_normalized"][i])
            require(all(len(row[key]) == 5 for key in ("observation", "latent", "actions_normalized"))
                    and all(math.isfinite(float(x)) for key in ("observation", "latent", "actions_normalized") for x in row[key])
                    and math.isfinite(float(row["logp"])) and all(abs(x) <= 1 for x in row["actions_normalized"]),
                    "Malformed policy proposal")
            rows.append(row)
        return dict(model_id=name, deterministic=deterministic, cells_sha256=_hash(cells), rows=rows,
                    weight_hash=self.state(name)["weight_hash"], checkpoint=copy.deepcopy(sampling_checkpoint))

    def execute(self, cells, proposal, film=False):
        self._guard()
        require(1 <= len(cells) <= SLOTS and proposal["cells_sha256"] == _hash(cells)
                and len(proposal["rows"]) == len(cells), "Proposal/target association changed")
        require([_ids(c) for c in cells] == [(p["request_id"], p["target_id"]) for p in proposal["rows"]],
                "Proposal request identities changed")
        require([row["observation"] for row in proposal["rows"]] == self.deps.observations(cells),
                "Policy observations disagree with assigned bucket targets")
        request_ids = [_ids(c)[0] for c in cells]
        require(len(set(request_ids)) == len(cells) and not set(request_ids) & self.dispatched,
                "Native requests cannot be repeated")
        require(proposal["weight_hash"] == self.state(proposal["model_id"])["weight_hash"],
                "Proposal belongs to an earlier model state")
        require(all(c["split"] == "train" for c in cells) if not proposal["deterministic"]
                else all(c["split"] in ("validation", "test") for c in cells), "Wrong policy/split mode")
        identity = _hash(proposal)
        require(identity not in self.used_proposals, "Proposal dispatch cannot be repeated")
        self.used_proposals.add(identity)
        self.dispatched.update(request_ids)
        env, count, begin = self.env, len(cells), self.clock()
        cpu_begin = time.process_time()
        resource_before = self.deps.resource_sample()
        # Inactive padding remains a valid placement but is never a request.
        padded = copy.deepcopy(cells)
        for i in range(count, SLOTS):
            padded.append(dict(copy.deepcopy(cells[0]), id=f"inactive-{self.trial_index}-{i}",
                               request_id=f"inactive-{self.trial_index}-{i}", target_id="inactive"))
        native_cells = [{**copy.deepcopy(self.originals[i]), **copy.deepcopy(cell), "id": _ids(cell)[0]}
                        for i, cell in enumerate(padded)]
        enabled = [i < count for i in range(SLOTS)]
        label = f"trial-{self.trial_index:05d}"
        self.trial_index += 1
        native_dir = self.output / "comparison-native"
        _mkdir(native_dir)
        physics_dir = self.output / "native"
        _mkdir(physics_dir)
        original_tick = env.tick
        def tick():
            self._guard()
            original_tick()
        env.tick = tick
        try:
            import numpy as np
            env.capture_trial = env.film_trial = False
            env.phase = None
            env.base_cells, env.spec["cells"] = copy.deepcopy(native_cells), copy.deepcopy(native_cells)
            env.bucket_xy = np.asarray([c["bucket_xy_m"] for c in native_cells])
            env.active_mask = np.asarray(enabled, dtype=bool)
            env.home_readback = env.home()  # unchanged reset writes actual bucket root states
            env.assert_buckets_fixed()
            env.trial_mode = "train" if not proposal["deterministic"] else "assessment"
            env.trial_batch = self.trial_index - 1
            env.frames, env.physics_samples = [], []
            env.capture_trial, env.film_trial, env.phase = True, bool(film), "setup"
            env.trial_start_step = env.steps
            normalized = [p["actions_normalized"] for p in proposal["rows"]] + [[0.] * 5 for _ in range(SLOTS-count)]
            actions = self.deps.actions(native_cells, normalized, label)
            rows, raw = env.shot_batch(actions, record_video=bool(film), enabled=env.active_mask)
            require(len(rows) == SLOTS, "Native result lost physical slots")
            require(env.geometry_enabled_this_batch is True
                    and bool(env.geometry_filter_synthetic) is self.synthetic,
                    "Geometry layer bypassed or synthetic filter entered production")
            env.assert_buckets_fixed()
            features = self.deps.features(env, native_cells, rows)
            returns = [self.deps.reward(row, feature) for row, feature in zip(rows, features)]
            # The original recorder owns image creation. Bind and sync its
            # prescheduled images before the terminal receipt claims them.
            for frame in env.frames:
                image = physics_dir / "frames" / f"{frame['frame']:05d}.jpg"
                require(image.is_file() and not image.is_symlink()
                        and _file_hash(image) == frame["jpeg_sha256"], "Native frame bytes changed")
                with image.open("rb") as stream:
                    os.fsync(stream.fileno())
                _sync_parent(image)
            physics = None
            if env.physics_samples:
                target = physics_dir / (label + ".npz")
                require(not target.exists(), "Native trace overwrite")
                self.deps.save_physics(target, env.physics_samples)
                with target.open("rb") as stream:
                    os.fsync(stream.fileno())
                _sync_parent(target)
                physics = dict(file=target.name, sha256=_file_hash(target), samples=len(env.physics_samples))
            selected = []
            for cell, row, value in zip(cells, rows, returns):
                request_id, target_id = _ids(cell)
                invalid = row["invalid"]
                captured = bool(row["route_success"]) and not invalid
                selected.append(dict(request_id=request_id, target_id=target_id,
                    outcome="rejected" if invalid else "capture" if captured else "miss",
                    clean=bool(captured and row["route_quality"] == 2),
                    rejection_reason=str(invalid) if invalid else None, reward=float(value)))
            packet = dict(schema="sircus-location-comparison-native-trial-v1", synthetic=self.synthetic,
                evidence_root="native",
                comparison_fingerprint=self.plan["fingerprint"], cells=native_cells, enabled=enabled,
                proposal=proposal, raw_rows=rows, actions=actions, reward_features=features, rewards=returns,
                normalized_results=selected, home=env.home_readback, physics=physics,
                origins_m=_plain(env.origins), dt_s=env.dt, source=self.experiment["source"],
                setup_controller=self.deps.setup_protocol(), frames=env.frames, filming=bool(film),
                release_step=env.release_step, score_end_step=env.steps,
                geometry_records=env.geometry_records, geometry_starts=env.geometry_starts,
                proposed_actions=env.proposed_actions, proposal_ik_seeds=env.proposal_ik_seeds,
                preflight_records=env.preflight_records, qualifications=env.qualifications,
                path_reasons=env.path_reasons, path_checks=env.path_checks,
                scoring_normals=_plain(env.scoring_normals), transit_history=env.transit_history,
                correction_history=env.correction_history, settling_history=raw["settling"],
                trajectory_m=raw["trajectory_m"], geometry_enabled=env.geometry_enabled_this_batch,
                geometry_filter_synthetic=env.geometry_filter_synthetic)
            target = native_dir / (label + ".json.gz")
            _write(target, packet)
            if not proposal["deterministic"]:
                for raw_proposal, outcome in zip(proposal["rows"], selected):
                    self.learning_examples[outcome["request_id"]] = dict(
                        model_id=proposal["model_id"], weight_hash=proposal["weight_hash"],
                        example=copy.deepcopy(dict(proposal=raw_proposal, outcome=outcome)))
            costs = dict(episode_wall_s=self.clock()-begin,
                         episode_process_cpu_s=time.process_time()-cpu_begin,
                         geometry_wall_s=env.geometry_wall_s, geometry_process_cpu_s=env.geometry_process_cpu_s,
                         geometry_nested_in_episode=True,
                         resources_before=resource_before, resources_after=self.deps.resource_sample())
            return dict(rows=selected, evidence=dict(file=str(target.relative_to(self.output)),
                        sha256=_file_hash(target), physics=physics, evidence_root="native",
                        synthetic=self.synthetic), costs=costs)
        except BaseException as error:
            self.failed = True
            # Preserve any observed partial native trace; no failed request is
            # unmarked or eligible for another dispatch.
            if getattr(env, "physics_samples", None):
                partial = physics_dir / (label + "-incomplete.npz")
                if not partial.exists():
                    self.deps.save_physics(partial, env.physics_samples)
                    with partial.open("rb") as stream:
                        os.fsync(stream.fileno())
                    _sync_parent(partial)
            _write(native_dir/(label+"-failure.json.gz"), dict(
                status="incomplete_no_retry", error=type(error).__name__+": "+str(error),
                cells=native_cells, enabled=enabled, proposal=proposal, synthetic=self.synthetic))
            raise
        finally:
            env.capture_trial = env.film_trial = False
            env.phase = None
            env.tick = original_tick

    def learn(self, name, examples):
        self._guard()
        require(len(examples) == 140, "Exactly 140 charged requests per PPO update")
        identifiers = [e["proposal"]["request_id"] for e in examples]
        require(len(set(identifiers)) == 140 and not set(identifiers) & self.used_examples,
                "Training examples cannot repeat")
        require(all(e["proposal"]["request_id"] == e["outcome"]["request_id"]
                    and e["proposal"]["target_id"] == e["outcome"]["target_id"] for e in examples),
                "Reward/proposal join changed")
        state_before = self.state(name)
        for example, identity in zip(examples, identifiers):
            recorded = self.learning_examples.get(identity)
            require(recorded is not None and recorded["model_id"] == name
                    and recorded["weight_hash"] == state_before["weight_hash"]
                    and recorded["example"] == example,
                    "PPO buffer differs from this model's unchanged executed proposals/results")
        self.used_examples.update(identifiers)
        model = self.models[name]
        try:
            statistics = self.deps.update(model["handle"], examples)
        except BaseException:
            self.failed = True
            raise
        model["updates"] += 1
        model["attempts"] += 140
        state = self.state(name)
        require(state["optimizer_steps"] == 292 + 4 * (state["updates"]-73), "Adam did not perform four passes")
        directory = self.output/"updates"
        _mkdir(directory)
        _write(directory/f"{name}-{state['updates']:05d}.json", dict(state_before=state_before,
            state_after=state, statistics=_plain(statistics), examples_sha256=_hash(examples),
            request_ids=identifiers, synthetic=self.synthetic))
        return state

    def checkpoint(self, name, label):
        self._guard()
        require(re.fullmatch(r"[A-Za-z0-9_-]{1,96}", label) is not None, "Unsafe checkpoint label")
        state = self.state(name)
        directory = self.output / "checkpoints"
        _mkdir(directory)
        target = directory / f"{name}-{label}.pt"
        require(not target.exists() and re.fullmatch(r"[A-Za-z0-9_-]{1,96}", name), "Checkpoint overwrite/unsafe model")
        self.deps.save_checkpoint(self.models[name]["handle"], target,
            dict(comparison_fingerprint=self.plan["fingerprint"], state=state,
                 parent_checkpoint_sha256=PARENT_SHA, synthetic=self.synthetic))
        _sync_parent(target)
        return dict(file=str(target.relative_to(self.output)), sha256=_file_hash(target), state=state)

    def close(self):
        if not self.closed:
            self.cleanup_geometry()
            self.closed = True
