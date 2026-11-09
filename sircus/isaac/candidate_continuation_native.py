"""Reserved TRAIN requests through the unchanged native guard/controller path.

No simulator is constructed, no training method is called, and no launch CLI is
provided. Explicit future callers must supply an initialized guarded simulator
and a durable reservation. Prepared fits are warm starts, never release permits.
"""

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET

import numpy as np

from candidate_continuation_journal import read_events, write_new_json, _sync_directory
from candidate_continuation_ledger import Ledger, PARENT_CHECKPOINT_SHA256, canonical_hash
from candidate_continuation_policy import FrozenCandidateSupport
from continuous_feasibility import ACTION_NAMES, LOW, HIGH, decode_action, canonical_hash as geometry_hash
from presentation_recording import save_physics
from settled_transit import protocol as setup_protocol
from settled_transit import release_ready
from shot_benchmark_native import compressed_json
from shot_benchmark_reward import build_features, reward

SCHEMA = "sircus-candidate-continuation-native-binding-v1"
PACKET_SCHEMA = "sircus-candidate-continuation-native-trial-v1"
SOURCE = Path(__file__).resolve().parent
GUARD_FILES = ("candidate_continuation_native.py", "joint_feasibility.py", "settled_transit.py",
    "continuous_pilot.py", "parallel_runtime.py", "feasibility_runtime.py", "shot_benchmark_native.py", "continuous_feasibility.py",
    "robust_ik.py", "expanded_pose_native.py", "expanded_pose_menu.py", "prepare_assets.py",
    "feasibility_contract.py", "settling_contract.py", "shot_benchmark_reward.py", "rl_contract.py",
    "entry_quality.py", "board_route.py", "farm_contract.py")


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sync_file(path):
    with Path(path).open("rb") as stream:
        os.fsync(stream.fileno())
    _sync_directory(Path(path).parent)


def _raw_incomplete(value):
    """Preserve even invalid numeric samples with explicit nonfinite tags."""
    if isinstance(value, np.ndarray): return _raw_incomplete(value.tolist())
    if isinstance(value, np.generic): return _raw_incomplete(value.item())
    if isinstance(value, dict): return {str(k):_raw_incomplete(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [_raw_incomplete(v) for v in value]
    if type(value) is float and not math.isfinite(value): return dict(nonfinite_numeric_repr=repr(value))
    return value


def source_fingerprint():
    return canonical_hash({p.name:digest(p) for p in sorted(SOURCE.glob("*.py"))})


def guard_sources():
    return {name:digest(SOURCE/name) for name in GUARD_FILES}


def _frozen_identity():
    """Original seed-22 scene/assets/camera identities, never caller overrides.

    Geometry/camera/assets are in the audited compact bank/comparison receipts;
    scene identity is the canonical JSON hash of their hash-bound source scene.
    """
    return dict(scene_sha256="9a50295d9908be8634c2df1c0b460ec6011710cf8e14ce33cbc67a75749ff1c8",
        geometry_fingerprint="63362ef37a93881f769b39d1ee7456335f1cc7bf5db04ffa7e548ce9697455bf",
        camera_sha256="a7967c072b8f949318b43a510235d503c81c21e05f433670a0ce92a7af385ede",
        poses_sha256="5847c2793606860a82cc74030c3fe529767318f189c5d48e3d66c0dc7b9f776a",
        assets_sha256=dict(Arm10="3f40e56e46b040fb3339c1235ea85872f46511c45db2d37f7661db94a470bae3",
            Arm14="9981178abd0d87db43f7cca9fe26b0a74618aaa75ea10709194bd23acf59a340"))


def make_binding(*, scene, geometry_fingerprint, camera_sha256, assets_sha256, poses_sha256,
                 cells, canonical_home_rad, supports, prepared_records, startup_control_sha256):
    """Produce identities for separately verified inputs; this grants no launch."""
    value = dict(schema=SCHEMA, scene_sha256=geometry_hash(scene),
        geometry_fingerprint=geometry_fingerprint, camera_sha256=camera_sha256,
        poses_sha256=poses_sha256,
        assets_sha256=copy.deepcopy(assets_sha256), parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
        cells_sha256=canonical_hash(cells), canonical_home_rad=copy.deepcopy(canonical_home_rad),
        support_sha256=canonical_hash([s.to_dict() for s in supports]),
        prepared_records_sha256=canonical_hash(prepared_records), source_sha256=source_fingerprint(),
        guard_source_sha256=guard_sources(), setup_controller=setup_protocol(),
        startup_control_sha256=copy.deepcopy(startup_control_sha256),
        execution_authorized=False, release_authorized=False)
    value["fingerprint"] = canonical_hash(value)
    return value


def _trusted_methods():
    """Lazy imports inspect existing classes only; no app/runtime is started."""
    from continuous_pilot import ContinuousShots
    from feasibility_runtime import FeasibilityShots
    from parallel_runtime import ParallelShots
    from settled_transit import SettledTransitShots
    from shot_benchmark_native import BenchmarkShots
    return dict(__prepare_chain__=(BenchmarkShots.prepare, SettledTransitShots.prepare),
        __shot_batch_chain__=(FeasibilityShots.shot_batch, ParallelShots.shot_batch),
        prepare=BenchmarkShots.prepare, home=ContinuousShots.home,
        shot_batch=FeasibilityShots.shot_batch, tick=ContinuousShots.tick,
        render=ContinuousShots.render, approach_endpoints=SettledTransitShots.approach_endpoints,
        _wait_stable=SettledTransitShots._wait_stable, _ik=SettledTransitShots._ik,
        _measure=SettledTransitShots._measure, _joint_status=SettledTransitShots._joint_status,
        _state=SettledTransitShots._state, board_poses=SettledTransitShots.board_poses,
        assert_buckets_fixed=ParallelShots.assert_buckets_fixed, set_balls=ParallelShots.set_balls,
        forces=ParallelShots.forces)


def validate_environment(env, binding):
    """Reject substituted methods/instance overrides before home or shot calls."""
    for name, expected in _trusted_methods().items():
        if name in ("__prepare_chain__", "__shot_batch_chain__"):
            attribute = "prepare" if name == "__prepare_chain__" else "shot_batch"
            owners = tuple(base.__dict__[attribute] for base in type(env).__mro__ if attribute in base.__dict__)
            require(owners[:len(expected)] == expected, "Native "+attribute+" delegation bypassed measured controller")
            continue
        require(name not in vars(env) and getattr(type(env), name, None) is expected,
                "Native guarded method resolution changed: "+name)
    require(env.n == 35 and geometry_hash(env.cfg) == binding["scene_sha256"]
            and env.spec.get("fingerprint") == binding["geometry_fingerprint"],
            "Native source scene or geometry changed")
    require(math.isclose(env.dt, 1/240, rel_tol=0, abs_tol=1e-15), "Original physics timestep changed")
    require(set(env.continuous_links) == {"Arm10", "Arm14"} and hasattr(env, "transit_limits"),
            "Native arm chains/effort caps must be configured from verified assets")


def prepare_packet(choices, ledger, cells, supports, prepared_records, original_cells):
    """Pure slot mapping; active actions stay exact and inactive slots stay parked."""
    require(type(ledger) is Ledger and ledger.phase == "train" and ledger.stopped_reason is None
            and ledger.pending is not None and not ledger.pending.outcomes, "Expected fresh reserved TRAIN batch")
    requests = ledger.pending.requests
    require(type(choices) in (list, tuple) and len(choices) == len(requests), "Reserved choice count changed")
    by_cell = {c["id"]:c for c in cells}
    by_support = {s.context_id:s for s in supports}
    require(len(original_cells) == 35 and set(by_cell) == set(by_support) == set(ledger.train_context_ids),
            "Original TRAIN coordinate/support coverage changed")
    actions, native_cells, seeds, selected = [], [], [], []
    for slot in range(35):
        template = copy.deepcopy(original_cells[slot])
        if slot >= len(requests):
            template.update(id=f"inactive-{ledger.pending.batch_index:04d}-{slot:02d}", split="inactive")
            native_cells.append(template)
            actions.append(decode_action(template, ((LOW+HIGH)/2).tolist(), template["id"]))
            seeds.append(None); selected.append(None)
            continue
        request, choice = requests[slot], choices[slot]
        support = by_support[request.context_id]
        index = choice["index"]
        require(choice["request_id"] == request.request_id and choice["context_id"] == request.context_id
            and type(index) is int and 0 <= index < len(support.candidate_ids)
            and choice["candidate_id"] == support.candidate_ids[index]
            and choice["support_fingerprint"] == support.support_fingerprint
            and choice["action"] == support.action(index) and choice["latent"] == list(support.latents[index]),
            "Reserved exact candidate/action/support changed")
        record = prepared_records[choice["candidate_id"]]
        require(record["candidate_id"] == choice["candidate_id"] and record["cell_id"] == request.context_id
            and record["action"] == choice["action"] and record["accepted"] is True,
            "Selected candidate lacks matching admitted preparation")
        decision = record["decision"]
        require(decision["fingerprint"] == geometry_hash({k:v for k,v in decision.items() if k != "fingerprint"})
            and decision["accepted"] is True and decision["release_authorized"] is False,
            "Prepared decision integrity changed")
        template.update(copy.deepcopy(by_cell[request.context_id]))
        require(template.get("split") == "train", "TRAIN native adapter cannot open held-out contexts")
        native_cells.append(template); actions.append(copy.deepcopy(choice["action"])); selected.append(copy.deepcopy(choice))
        # The exact parent follows original actual-home seeding. Other fits only
        # seed the unchanged fresh actual-state preflight in SettledTransitShots.
        seeds.append(None if index == support.parent_index else copy.deepcopy(decision["qualification"]["ik"]))
    return dict(cells=native_cells, actions=actions, ik_seeds=seeds, selected=selected,
        enabled=[i < len(requests) for i in range(35)], batch_index=ledger.pending.batch_index,
        requests=len(requests), request_ids=[r.request_id for r in requests])


class GuardedNativeProducer:
    """One call per durable reservation; no restart, retry or optimizer callback."""

    def __init__(self, env, binding, cells, supports, prepared_records, *, parent_checkpoint_path,
                 camera_path, poses_path, assets_paths, frozen_experiment_root):
        self.env, self.binding = env, copy.deepcopy(binding)
        self.cells, self.supports = copy.deepcopy(cells), tuple(supports)
        self.prepared_records = copy.deepcopy(prepared_records)
        self.parent_checkpoint_path, self.camera_path = Path(parent_checkpoint_path), Path(camera_path)
        self.poses_path = Path(poses_path)
        self.assets_paths = {name:Path(path) for name,path in assets_paths.items()}
        self.frozen_root = Path(frozen_experiment_root).resolve()
        self.original_cells = copy.deepcopy(env.base_cells)
        self.claimed = set()
        self._validate_binding()

    def _validate_binding(self):
        b = self.binding
        require(b.get("schema") == SCHEMA and b.get("fingerprint") == canonical_hash({k:v for k,v in b.items() if k != "fingerprint"})
            and b.get("execution_authorized") is False and b.get("release_authorized") is False,
            "Native binding schema/fingerprint changed")
        require(all(b.get(name) == expected for name,expected in _frozen_identity().items()),
                "Original frozen scene/geometry/camera/asset identity changed")
        require(b["parent_checkpoint_sha256"] == PARENT_CHECKPOINT_SHA256
            and digest(self.parent_checkpoint_path) == PARENT_CHECKPOINT_SHA256, "Frozen parent checkpoint changed")
        require(digest(self.poses_path) == b["poses_sha256"], "Frozen canonical pose bytes changed")
        poses = json.loads(self.poses_path.read_bytes(), parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite poses")))
        canonical = {name:poses[key] for name,key in (("Arm10","dropper_joint_rad"),("Arm14","board_joint_rad"))}
        require(b["canonical_home_rad"] == canonical and all(_raw_incomplete(self.env.poses[key]) == poses[key]
                for key in ("dropper_joint_rad", "board_joint_rad")), "Native canonical pose definition changed")
        require(digest(self.camera_path) == b["camera_sha256"]
            and set(self.assets_paths) == set(b["assets_sha256"]) == {"Arm10", "Arm14"}
            and all(digest(path) == b["assets_sha256"][name] for name,path in self.assets_paths.items()),
            "Frozen camera or asset bytes changed")
        from prepare_assets import chain
        from joint_feasibility import _chains
        chains = {name:chain(ET.fromstring(path.read_bytes())) for name,path in self.assets_paths.items()}
        require(_chains(self.env.continuous_links) == _chains(chains), "Native chains differ from verified asset bytes")
        caps = self.env.transit_limits
        if hasattr(caps, "detach"): caps = caps.detach().cpu().numpy()
        model = ET.fromstring(self.assets_paths["Arm10"].read_bytes())
        expected_caps = [float(model.find(f".//joint[@name='joint{i+1}']/limit").get("effort")) for i in range(6)]
        require(np.asarray(caps).shape == (6,) and np.array_equal(np.asarray(caps), expected_caps),
                "Native transit effort caps differ from original URDF")
        require(b["source_sha256"] == source_fingerprint() and b["guard_source_sha256"] == guard_sources()
            and b["setup_controller"] == setup_protocol(), "Original native guards/controller source changed")
        require(b["support_sha256"] == canonical_hash([s.to_dict() for s in self.supports])
            and b["cells_sha256"] == canonical_hash(self.cells)
            and b["prepared_records_sha256"] == canonical_hash(self.prepared_records), "Prepared bank bytes changed")
        require(set(b["canonical_home_rad"]) == set(self.env.arms) == {"Arm10", "Arm14"}, "Canonical home arm association changed")
        for name,arm in self.env.arms.items():
            home = arm.data.default_joint_pos
            if hasattr(home, "detach"): home = home.detach().cpu().numpy()
            expected = np.asarray(b["canonical_home_rad"][name], dtype=float)
            require(expected.shape == (6,) and np.isfinite(expected).all()
                and np.asarray(home).ndim == 2 and np.asarray(home).shape[0] == 35
                and np.asarray(home).shape[1] >= 6
                and np.allclose(np.asarray(home)[:,:6],expected[None],atol=1e-6,rtol=0),
                "Native canonical default home differs from frozen poses")
        validate_environment(self.env, b)

    def execute_reserved(self, runner, choices, *, expected_journal_head):
        """Explicit future native work; callers separately record returned results."""
        self._validate_binding()
        require(not runner.failed and not runner.finished and runner.engine.stage == "ppo",
                "TRAIN runner is stopped or out of phase")
        require(runner.bindings["source_sha256"] == self.binding["source_sha256"]
            and runner.bindings["support_sha256"] == self.binding["support_sha256"]
            and runner.last_weight == runner.engine.state_dict()["weight_hash"], "Reserved model/source changed")
        require(runner.journal.head == expected_journal_head and expected_journal_head not in self.claimed,
                "Stale or duplicate native reservation")
        events = read_events(runner.journal.directory)
        require(events and events[-1]["fingerprint"] == expected_journal_head and events[-1]["kind"] == "reserve"
            and events[-1]["payload"]["ledger"] == runner.ledger.to_dict()
            and events[-1]["payload"]["choices"] == list(choices)
            and events[-1]["payload"]["policy_weight_hash"] == runner.last_weight,
            "Reservation is not the exact durable journal head")
        directory = Path(runner.directory).resolve()
        require(not directory.is_relative_to(self.frozen_root) and Path(self.env.dest).resolve() == directory,
                "Native evidence must use the new run, outside the frozen experiment")
        require(shutil.disk_usage(directory).free >= 6*1024**3, "Original 6 GiB storage floor reached")
        from candidate_native_audit import verify_controls
        controls = verify_controls(directory, self.env.cfg, self.binding["startup_control_sha256"])
        packet = prepare_packet(choices, runner.ledger, self.cells, self.supports, self.prepared_records, self.original_cells)
        native = directory/"native"
        if not native.exists():
            native.mkdir(); _sync_directory(directory)
        stem = f"train-{packet['batch_index']:06d}"
        target, physics_path = native/(stem+".json.gz"), native/(stem+".npz")
        require(not target.exists() and not physics_path.exists(), "Native output already exists")
        # The durable dispatch claim is intentionally outside the learning
        # journal. It prevents the same reservation executing after a crash.
        write_new_json(native/(stem+".dispatch.json"), dict(reservation_sha256=expected_journal_head,
            ledger_fingerprint=runner.ledger.to_dict()["fingerprint"], request_ids=packet["request_ids"],
            native_binding_fingerprint=self.binding["fingerprint"], status="claimed_before_native_call"))
        self.claimed.add(expected_journal_head)
        env = self.env
        began, cpu = time.monotonic(), time.process_time()
        try:
            env.capture_trial = env.film_trial = False; env.phase = None
            env.base_cells = copy.deepcopy(packet["cells"]); env.spec["cells"] = copy.deepcopy(packet["cells"])
            env.bucket_xy = np.asarray([c["bucket_xy_m"] for c in packet["cells"]])
            env.home_readback = env.home()
            env.active_mask = np.asarray(packet["enabled"], dtype=bool)
            env.support_missing = [False]*35
            env.proposal_ik_seeds = copy.deepcopy(packet["ik_seeds"])
            env.trial_mode, env.trial_batch = "train", packet["batch_index"]
            env.frames, env.physics_samples = [], []
            env.capture_trial, env.film_trial, env.phase = True, False, "setup"
            env.trial_start_step = env.steps
            rows, raw = env.shot_batch(copy.deepcopy(packet["actions"]), record_video=False, enabled=env.active_mask)
            env.capture_trial = env.film_trial = False; env.phase = None
            require(len(rows) == 35 and all(rows[i]["invalid"] for i in range(packet["requests"],35)),
                    "Inactive native slot was released or row coverage changed")
            for i,row in enumerate(rows[:packet["requests"]]):
                require(type(row.get("route_success")) is bool and type(row.get("route_quality")) is int
                    and row["route_quality"] in ((1,2) if row["route_success"] else (0,))
                    and (not row["invalid"] or not row["route_success"]), "Native outcome flags contradict guard rejection")
                if not row["invalid"]:
                    require(env.preflight_records[i]["accepted"] is True and env.qualifications[i]["accepted"] is True
                        and env.path_checks[i]["accepted"] is True and env.path_reasons[i] is None
                        and release_ready(row["settled"]), "Native release lacks unchanged geometry/path/settling gates")
            physics, features = None, [None]*35
            if env.physics_samples:
                save_physics(physics_path, env.physics_samples)
                _sync_file(physics_path)
                physics = dict(file=physics_path.name, sha256=digest(physics_path), samples=len(env.physics_samples))
                position = np.asarray([s["position_m"] for s in env.physics_samples])-env.origins[None]
                velocity = np.asarray([s["velocity_m_s"] for s in env.physics_samples])
                forces = np.asarray([s["forces_N"] for s in env.physics_samples])
                for i in range(packet["requests"]):
                    if not rows[i]["invalid"]:
                        features[i] = build_features(env.cfg, packet["cells"][i]["bucket_xy_m"],
                            position[:,i], velocity[:,i], forces[:,i], env.dt, env.scoring_normals[i])
            require(physics is not None or all(r["invalid"] for r in rows[:packet["requests"]]),
                    "Released request lacks full-rate physics evidence")
            returns = []
            for i,row in enumerate(rows):
                row.update(split=packet["cells"][i]["split"], qualification=env.qualifications[i],
                    path_rejection=env.path_reasons[i], path_proxy=env.path_checks[i],
                    scoring_board_normal=env.scoring_normals[i].tolist())
                returns.append(reward(row, features[i]) if i < packet["requests"] else None)
            native_packet = dict(schema=PACKET_SCHEMA, mode="train", batch=packet["batch_index"],
                synthetic=runner.synthetic,
                reservation_sha256=expected_journal_head, native_binding=self.binding,
                policy_weight_hash=runner.last_weight, cells=packet["cells"], actions=packet["actions"],
                choices=packet["selected"], request_ids=packet["request_ids"], enabled=packet["enabled"],
                proposal_ik_seeds=packet["ik_seeds"], rows=rows, rewards=returns, reward_features=features,
                home=env.home_readback, preflight_records=copy.deepcopy(env.preflight_records),
                release_step=env.release_step, score_end_step=env.steps, physics=physics,
                origins_m=env.origins.tolist(), setup_controller=setup_protocol(), startup_controls=controls,
                settling_history=raw["settling"], transit_history=env.transit_history,
                correction_history=env.correction_history, trajectory_m=raw["trajectory_m"],
                wall_s=time.monotonic()-began, process_cpu_s=time.process_time()-cpu,
                scope="Exact reserved TRAIN actions through unchanged native guards; independent native evidence audit still required.")
            receipt = compressed_json(target, native_packet)
            _sync_file(target)
            write_new_json(native/(stem+".receipt.json"), receipt)
            self._validate_binding()
            return [dict(request_id=choice["request_id"], released=not bool(row["invalid"]),
                decline_reason=row["invalid"] or None, source_sha256=receipt["sha256"],
                captured=bool(row["route_success"]), clean=row["route_quality"] == 2,
                **{"return":returns[i]}) for i,(choice,row) in enumerate(zip(choices, rows))]
        except BaseException as error:
            env.capture_trial = env.film_trial = False; env.phase = None
            partial = None
            if env.physics_samples and not physics_path.exists():
                try:
                    save_physics(physics_path, env.physics_samples); _sync_file(physics_path)
                except Exception as telemetry_error:
                    path = native/(stem+".incomplete-physics.json")
                    write_new_json(path, dict(status="invalid_unscored_raw_evidence",
                        error=type(telemetry_error).__name__+": "+str(telemetry_error),
                        samples=_raw_incomplete(env.physics_samples)))
                    partial = dict(file=path.name, sha256=digest(path))
            write_new_json(native/(stem+".failure.json"), dict(status="incomplete_no_retry",
                reservation_sha256=expected_journal_head, error=type(error).__name__+": "+str(error),
                physics_sha256=digest(physics_path) if physics_path.exists() else None, partial_physics=partial))
            raise
