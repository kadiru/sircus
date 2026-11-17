"""Independent read-only model and physical-evidence audit of the comparison.

Import is standard-library only. Production readers load checkpoints on CPU,
recompute likelihoods, geometry/FK and recorded 240 Hz scores; they never step
an optimizer, solve a new shot or start a simulator. Injectable readers are
permitted only for explicitly synthetic fixtures, which cannot pass natively.
"""

from collections import Counter
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re

from location_comparison_audit import audit_trace, digest as signed_hash, CORE_STATE

SCHEMA = "sircus-location-comparison-independent-audit-v1"
INPUT_SCHEMA = "sircus-location-comparison-production-inputs-v1"
SOURCE = Path(__file__).resolve().parent
CHECKS = {"inputs", "parent", "checkpoint", "policy", "geometry", "native", "startup"}
MAX_JSON_BYTES = 256 * 1024**2


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024**2), b""):
            result.update(part)
    return result.hexdigest()


def read(path):
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, "Duplicate evidence JSON key")
            result[key] = value
        return result
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as stream:
        raw = stream.read(MAX_JSON_BYTES + 1)
    require(len(raw) <= MAX_JSON_BYTES, "Evidence JSON exceeds fixed bound")
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite evidence JSON")))


def artifact(output, relative, checksum=None):
    require(type(relative) is str and relative and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts, "Escaped evidence path")
    root = Path(output).resolve()
    path = root / relative
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents))
            and path.resolve().is_relative_to(root), "Missing or linked evidence: " + relative)
    if checksum is not None:
        require(re.fullmatch(r"[0-9a-f]{64}", checksum or "") and sha(path) == checksum,
                "Evidence bytes changed: " + relative)
    return path


def _finite_tree(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(v) for v in value)
    return True


def verify_inputs(manifest, plan):
    from location_comparison_native import validate_native_inputs, validate_source_imports
    from seed22_continuation_audit import (PARENT_AUDIT_SHA256, ORIGINAL_MODULES,
        PINNED_GUARD_ADAPTERS, verify_guard_adapters)
    from seed22_continuation_contract import ORIGINAL_SPLIT_SHA256
    from shot_benchmark_contract import make_protocol
    require(manifest.get("schema") == INPUT_SCHEMA and manifest.get("synthetic", False) is False,
            "Real comparison input manifest required")
    require(manifest.get("fingerprint") == signed_hash({k: v for k, v in manifest.items() if k != "fingerprint"}),
            "Input manifest fingerprint changed")
    files = manifest["files"]
    require({"parent", "parent_audit", "experiment", "original_splits", "scene", "camera", "poses",
             "dropper_urdf", "board_urdf", "original_film_source", "original_settled_source", "source_contract"}.issubset(files),
            "Missing independent original input bindings")
    experiment, scene = read(files["experiment"]["path"]), read(files["scene"]["path"])
    validate_native_inputs(manifest, experiment, plan, env_cfg=scene,
                           env_geometry_fingerprint=experiment["source"]["geometry_fingerprint"])
    validate_source_imports(manifest)
    require(files["parent_audit"]["sha256"] == PARENT_AUDIT_SHA256
            and files["original_splits"]["sha256"] == ORIGINAL_SPLIT_SHA256,
            "Wrong independently audited parent or original split")
    original = read(files["parent_audit"]["path"])
    require(original.get("status") == "passed" and original.get("seed") == 20261122
            and original.get("method") == "ppo" and original.get("paid_train_attempts") == 10000
            and original.get("updates") == 73, "Original seed22 audit did not pass")
    sources = manifest["source_sha256"]
    require(all(original["source_sha256"].get(name) == sources.get(name)
                for name in ORIGINAL_MODULES if name not in PINNED_GUARD_ADAPTERS),
            "Original physics, policy, scoring or controller changed")
    adapters = verify_guard_adapters(original["source_sha256"], sources, files)
    require(experiment["protocol"] == make_protocol(), "Original benchmark protocol changed")
    return dict(experiment=experiment, scene=scene, files=files, guard_adapters=adapters,
                source_sha256=copy.deepcopy(sources))


def load_parent(path, experiment):
    """Read the original parent, including Adam, without constructing an optimizer."""
    import torch
    from continuous_policy import load_checkpoint
    from candidate_continuation_checkpoint import state_hash
    _, saved = load_checkpoint(path, study_fingerprint=experiment["protocol"]["fingerprint"],
        geometry_fingerprint=experiment["source"]["geometry_fingerprint"],
        domain_fingerprint=experiment["protocol"]["domain"]["fingerprint"], device="cpu",
        expected_updates=73, expected_attempted_shots=10000)
    require(saved.get("seed") == 20261122, "Wrong original parent seed")
    require(saved.get("optimizer") is not None, "Original Adam state missing")
    require(all(torch.isfinite(value).all().item() for value in saved["state_dict"].values()),
            "Nonfinite original parent")
    return dict(weight_hash=saved["weight_hash"], optimizer_hash=state_hash(saved["optimizer"]),
                optimizer_config=copy.deepcopy(saved["optimizer"]["param_groups"]))


def load_model(path, receipt, plan, experiment):
    import torch
    from continuous_policy import load_checkpoint
    from candidate_continuation_checkpoint import state_hash
    from location_comparison_native import PARENT_SHA
    state = receipt["state"]
    additional = state["updates"] - 73
    require(type(additional) is int and 0 <= additional <= 64
            and state["attempts"] == 10000 + 140 * additional
            and state["optimizer_steps"] == 292 + 4 * additional,
            "Checkpoint learning counters changed")
    model, saved = load_checkpoint(path, study_fingerprint=plan["fingerprint"],
        geometry_fingerprint=experiment["source"]["geometry_fingerprint"],
        domain_fingerprint=experiment["protocol"]["domain"]["fingerprint"], device="cpu",
        expected_updates=state["updates"], expected_attempted_shots=state["attempts"])
    require(saved.get("synthetic") is False and saved.get("seed") == 20261122
            and saved.get("comparison_fingerprint") == plan["fingerprint"]
            and saved.get("parent_checkpoint_sha256") == PARENT_SHA and saved.get("state") == state,
            "Checkpoint comparison, parent or recorded state changed")
    require(saved["weight_hash"] == state["weight_hash"]
            and all(torch.isfinite(value).all().item() for value in saved["state_dict"].values()),
            "Checkpoint weights differ from recorded state")
    optimizer = saved.get("optimizer")
    require(type(optimizer) is dict and type(optimizer.get("state")) is dict
            and len(optimizer["state"]) == len(list(model.parameters()))
            and optimizer.get("param_groups")
            and all(group.get("lr") == .0003 for group in optimizer["param_groups"]),
            "Checkpoint Adam membership or learning rate changed")
    for value in optimizer["state"].values():
        require(float(value.get("step", -1)) == state["optimizer_steps"]
                and all(not torch.is_tensor(item) or torch.isfinite(item).all().item()
                        for item in value.values()), "Adam counters or finite tensors changed")
    require(state_hash(optimizer) == state["optimizer_hash"], "Saved Adam hash differs")
    require(state_hash(saved.get("action_rng_state")) == state.get("action_rng_hash")
            and state_hash(saved.get("optimizer_rng_state")) == state.get("optimizer_rng_hash"),
            "Saved random stream bytes differ")
    return dict(policy=model, state=copy.deepcopy(state), actor_hash=saved["actor_weight_hash"],
                optimizer_config=copy.deepcopy(optimizer["param_groups"]))


def verify_policy(model, proposal, cells):
    """Check every paid action, including shorter assessment batches, on CPU."""
    import numpy as np
    import torch
    from continuous_policy import build_observations
    from shot_benchmark_audit import LOGP_TOLERANCE
    rows, count = proposal["rows"], len(cells)
    observations = build_observations(cells)
    require(np.array_equal(np.asarray([r["observation"] for r in rows]), observations),
            "Recorded policy inputs differ from original features")
    latent, normalized = (np.asarray([r[k] for r in rows], dtype=float)
                          for k in ("latent", "actions_normalized"))
    logp = np.asarray([r["logp"] for r in rows], dtype=float)
    require(latent.shape == normalized.shape == (count, 5) and logp.shape == (count,)
            and all(np.isfinite(v).all() for v in (latent, normalized, logp)), "Invalid policy sample")
    observation_tensor = torch.as_tensor(observations, dtype=torch.float32)
    with torch.no_grad():
        distribution = model.distribution(observation_tensor)
        mean = distribution.loc.numpy().astype(float)
        scale = distribution.scale.numpy().astype(float)
        actual = model.log_prob(observation_tensor, torch.tensor(latent, dtype=torch.float32)).numpy()
    require(np.isfinite(mean).all() and np.isfinite(scale).all() and (scale > 0).all(),
            "Invalid policy distribution")
    gaussian = -.5 * np.square((latent - mean) / scale) - np.log(scale) - .5 * math.log(2 * math.pi)
    jacobian = 2 * (math.log(2) - latent - np.logaddexp(0., -2 * latent))
    independent = np.sum(gaussian - jacobian, axis=1)
    error = float(max(np.max(np.abs(actual - logp)), np.max(np.abs(independent - logp))))
    require(error <= LOGP_TOLERANCE, "Original checkpoint likelihood differs from sampled latent")
    require(np.allclose(np.tanh(latent), normalized, atol=2e-7, rtol=0), "Sampled action changed")
    if proposal["deterministic"]:
        require(np.allclose(latent, mean, atol=2e-6, rtol=0), "Assessment did not use direct model mean")
    return dict(maximum_logp_error=error, paid_samples=count)


def guard_view(packet, stem):
    """Pure shape adapter to unchanged independent geometry/physics readers."""
    value = copy.deepcopy(packet)
    require(re.fullmatch(r"trial-[0-9]{5}", stem), "Wrong native trial stem")
    value["rows"] = value["raw_rows"]
    value["mode"], value["batch"] = "trial", stem.removeprefix("trial-")
    paid = value["proposal"]["rows"]
    value["actions_normalized"] = [r["actions_normalized"] for r in paid] + [[0.] * 5 for _ in range(35-len(paid))]
    return value


def geometry_reader(trial, scene, links):
    from continuous_policy import build_observations
    from seed22_continuation_audit import verify_geometry
    view = copy.deepcopy(trial)
    # Padding has no policy query. Its zero-action shape exists solely for the
    # original 35-slot guard reader; paid observations are checked separately.
    view["observations"] = build_observations(view["cells"]).tolist()
    return verify_geometry(view, scene, links)


def _default_checks():
    from seed22_continuation_audit import verify_native
    from candidate_continuation_scene import audit_startup
    return dict(inputs=verify_inputs, parent=load_parent, checkpoint=load_model,
                policy=verify_policy, geometry=geometry_reader, native=verify_native, startup=audit_startup)


def verify_exit(output, manifest, plan, trace, *, synthetic):
    from candidate_continuation_ledger import canonical_hash
    from location_comparison_watchdog import SCHEMA as WATCHDOG_SCHEMA
    trace_path = artifact(output, "comparison-trace.json")
    require(read(trace_path) == trace, "Final trace differs from the audited file")
    child = read(artifact(output, "child-exit.json"))
    require(child.get("status") == ("synthetic_complete_pending_audit" if synthetic else "native_complete_pending_audit")
            and child.get("synthetic") is synthetic and child.get("app_closed") is True
            and child.get("error") is None and child.get("runtime_fingerprint") == manifest["fingerprint"]
            and child.get("comparison_fingerprint") == plan["fingerprint"]
            and child.get("trace_sha256") == sha(trace_path), "Native child did not close with this exact complete trace")
    watchdog = read(artifact(output, "comparison-watchdog-exit.json"))
    require(watchdog.get("schema") == WATCHDOG_SCHEMA
            and watchdog.get("fingerprint") == canonical_hash({k:v for k,v in watchdog.items() if k != "fingerprint"})
            and watchdog.get("runtime_fingerprint") == manifest["fingerprint"]
            and watchdog.get("comparison_fingerprint") == plan["fingerprint"]
            and watchdog.get("session_id") == child.get("session_id")
            and watchdog.get("status") == "native_exit_zero_independent_audit_required"
            and watchdog.get("exit_code") == 0 and watchdog.get("reason") is None
            and watchdog.get("lock_release_safe") is True and watchdog.get("native_application_starts") == 1,
            "Watchdog did not confirm the exact one-start clean native exit")
    final = trace["events"][-1]["payload"]["bounds"]
    require(all(watchdog.get("final_snapshot", {}).get(key) == final[key] for key in
                ("train_requested", "train_completed", "assessment_requested", "assessment_completed", "pending", "stopped")),
            "Watchdog final accounting differs from completed trace")
    return dict(native_exit_fingerprint=watchdog["fingerprint"], trace_sha256=sha(trace_path),
                child_exit_sha256=sha(output/"child-exit.json"))


def audit(output, manifest, plan, trace, *, synthetic=False, checks=None):
    """Fail closed unless all original model/guard/score evidence is present.

    This does not certify OS ownership: the outer production supervisor must
    also independently audit its watchdog receipt before publishing completion.
    """
    require(type(synthetic) is bool, "Explicit evidence mode required")
    require(checks is None or synthetic is True, "Injected checks cannot certify native evidence")
    require(trace.get("mode") == ("synthetic" if synthetic else "native"), "Trace evidence mode mismatch")
    if not synthetic:
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Independent audit must be CPU-only")
    checks = _default_checks() if checks is None else checks
    require(set(checks) == CHECKS and all(callable(v) for v in checks.values()), "Incomplete verifier set")
    output = Path(output).resolve()
    accounting = audit_trace(plan, trace)
    require(accounting["status"] in ("synthetic_verified", "native_accounting_verified_physics_pending"),
            "Incomplete comparison cannot pass final independent audit")
    exit_proof = verify_exit(output, manifest, plan, trace, synthetic=synthetic)
    verified = checks["inputs"](manifest, plan)
    experiment, scene, files = verified["experiment"], verified["scene"], verified["files"]
    original = checks["parent"](Path(files["parent"]["path"]), experiment)
    contract = read(artifact(output, "native/contract.json"))
    require(contract.get("fingerprint") == experiment["source"]["geometry_fingerprint"]
            and contract.get("scene") == scene and len(contract.get("cells", [])) == 35
            and sha(output/"native/contract.json") == files["source_contract"]["sha256"],
            "Constructed native scene differs from immutable source")
    from camera_options import preview_origins
    origins = preview_origins()
    if synthetic:
        links = {arm: None for arm in ("Arm10", "Arm14")}
    else:
        import xml.etree.ElementTree as ET
        from prepare_assets import chain
        links = {arm: chain(ET.fromstring(Path(files[role]["path"]).read_bytes()))
                 for arm, role in (("Arm10", "dropper_urdf"), ("Arm14", "board_urdf"))}
    startup = checks["startup"](output/"native/startup", scene=scene, cells=contract["cells"], origins=origins)
    require(startup.get("status") == "passed", "Independent startup physics audit failed")
    poses, camera = read(files["poses"]["path"]), read(files["camera"]["path"])
    checkpoints, receipts, trials, physics, frames, updates = {}, {}, set(), set(), set(), set()
    model_states, actor_states, update_records = {}, {}, []
    def checkpoint(receipt):
        require(type(receipt) is dict and set(receipt) == {"file", "sha256", "state"}
                and Path(receipt["file"]).parts[:1] == ("checkpoints",), "Malformed checkpoint receipt")
        filename, state = receipt["file"], receipt["state"]
        if filename in checkpoints:
            require(receipts[filename] == receipt, "Conflicting checkpoint identity")
            return checkpoints[filename]
        path = artifact(output, filename, receipt["sha256"])
        result = checks["checkpoint"](path, receipt, plan, experiment)
        require(result["state"] == state, "Independent model state differs")
        require("optimizer_config" in result and result["optimizer_config"] == original.get("optimizer_config"),
                "Adam settings differ from inherited parent configuration")
        key = (state["model_id"], state["updates"])
        core = {field: state[field] for field in CORE_STATE}
        require(key not in model_states or model_states[key] == core, "One update has multiple model states")
        model_states[key] = core
        actor = result.get("actor_hash")
        require(type(actor) is str and re.fullmatch(r"[0-9a-f]{64}", actor), "Missing independent actor identity")
        require(key not in actor_states or actor_states[key] == actor, "One state has multiple actors")
        actor_states[key] = actor
        if state["updates"] == 73:
            require(state["weight_hash"] == original["weight_hash"]
                    and state["optimizer_hash"] == original["optimizer_hash"],
                    "Restored copy differs from original seed22 weights or Adam")
        checkpoints[filename], receipts[filename] = result, copy.deepcopy(receipt)
        return result
    reservation, proposal = None, None
    batches, paid_samples, maximum_error = 0, 0, 0.
    geometry_counts = Counter()
    for event in trace["events"]:
        kind, payload = event["kind"], event["payload"]
        if kind == "checkpoint":
            checkpoint(payload["receipt"])
        elif kind == "reserve":
            reservation = payload
        elif kind == "proposal":
            proposal = payload["packet"]
            cp = checkpoint(proposal.get("checkpoint"))
            require(cp["state"]["model_id"] == proposal["model_id"]
                    and cp["state"]["weight_hash"] == proposal["weight_hash"], "Proposal checkpoint changed")
        elif kind == "terminal":
            receipt = payload["evidence"]
            filename = f"comparison-native/trial-{batches:05d}.json.gz"
            require(receipt.get("file") == filename and receipt.get("evidence_root") == "native"
                    and receipt.get("synthetic") is synthetic and filename not in trials,
                    "Native batch evidence identity changed")
            packet = read(artifact(output, filename, receipt["sha256"]))
            require(packet.get("schema") == "sircus-location-comparison-native-trial-v1"
                    and packet.get("synthetic") is synthetic and packet.get("evidence_root") == "native"
                    and packet.get("comparison_fingerprint") == plan["fingerprint"]
                    and packet.get("source") == experiment["source"] and packet.get("proposal") == proposal
                    and packet.get("filming") is reservation["film"], "Raw native trial binding changed")
            cp = checkpoint(proposal["checkpoint"])
            require(all(cp["state"][field] == payload["state_before"][field] for field in CORE_STATE),
                    "Trial checkpoint is stale or belongs to another state")
            cells = copy.deepcopy(reservation["cells"])
            count = len(cells)
            for index in range(count, 35):
                cells.append(dict(copy.deepcopy(cells[0]), id=f"inactive-{batches}-{index}",
                    request_id=f"inactive-{batches}-{index}", target_id="inactive"))
            expected = [{**copy.deepcopy(contract["cells"][i]), **cell, "id": cell["request_id"]}
                        for i, cell in enumerate(cells)]
            require(packet.get("cells") == expected and packet.get("enabled") == [i < count for i in range(35)]
                    and all(type(v) is bool for v in packet["enabled"]), "Native assigned targets or paid mask changed")
            require(packet.get("origins_m") == origins.tolist()
                    and packet.get("dt_s") == scene["simulation"]["physics_dt_s"],
                    "Native world origins or full-rate clock changed")
            require(len(packet.get("raw_rows", [])) == len(packet.get("rewards", [])) == 35
                    and packet.get("normalized_results") == payload["rows"], "Incomplete raw outcome coverage")
            reconstructed = []
            for cell, row, reward in zip(cells[:count], packet["raw_rows"], packet["rewards"]):
                rejected, captured = bool(row["invalid"]), bool(row["route_success"]) and not row["invalid"]
                reconstructed.append(dict(request_id=cell["request_id"], target_id=cell["target_id"],
                    outcome="rejected" if rejected else "capture" if captured else "miss",
                    clean=bool(captured and row["route_quality"] == 2),
                    rejection_reason=str(row["invalid"]) if rejected else None, reward=float(reward)))
            require(reconstructed == payload["rows"], "Published capture/rejection/reward differs from raw native rows")
            measure = checks["policy"](cp["policy"], proposal, reservation["cells"])
            require(measure.get("paid_samples") == count and type(measure.get("maximum_logp_error")) in (int, float)
                    and math.isfinite(measure["maximum_logp_error"]) and measure["maximum_logp_error"] >= 0,
                    "Independent policy check did not cover the paid batch")
            paid_samples += count
            maximum_error = max(maximum_error, measure["maximum_logp_error"])
            native_physics = packet.get("physics")
            require(receipt.get("physics") == native_physics, "Native physics receipt differs")
            if native_physics is not None:
                name = native_physics["file"]
                require(Path(name).name == name and name == f"trial-{batches:05d}.npz" and name not in physics,
                        "Repeated or escaped full-rate trace")
                artifact(output, "native/"+name, native_physics["sha256"])
                physics.add(name)
            view = guard_view(packet, f"trial-{batches:05d}")
            geometry = checks["geometry"](view, scene, links)
            require(type(geometry) is dict, "Independent geometry audit missing")
            for key in ("unchanged", "adjusted", "rejected"):
                geometry_counts[key] += geometry.get(key, 0)
            measured = checks["native"](output/"native", view, scene, links, poses, camera)
            require(type(measured) is dict and measured, "Independent full-rate physical audit missing")
            for frame in packet.get("frames", []):
                name = f"frames/{frame['frame']:05d}.jpg"
                require(name not in frames, "Repeated scheduled native frame")
                artifact(output, "native/"+name, frame["jpeg_sha256"])
                frames.add(name)
            trials.add(filename)
            batches += 1
        elif kind == "update":
            filename = f"updates/{payload['model']}-{payload['after']['updates']:05d}.json"
            update = read(artifact(output, filename))
            require(filename not in updates and update.get("synthetic") is synthetic
                    and update.get("state_before") == payload["before"]
                    and update.get("state_after") == payload["after"]
                    and update.get("examples_sha256") == payload["examples_sha256"]
                    and update.get("request_ids") == payload["request_ids"]
                    and type(update.get("statistics")) is dict and _finite_tree(update["statistics"]),
                    "Persisted PPO update evidence differs from accounting")
            updates.add(filename)
            statistics = update["statistics"]
            epochs = statistics.get("epochs")
            require(statistics.get("before") == payload["before"]["weight_hash"]
                    and statistics.get("after") == payload["after"]["weight_hash"]
                    and statistics.get("samples") == 140 and type(epochs) is list and len(epochs) == 4
                    and all(type(epoch) is dict and epoch.get("samples") == 140
                        and all(type(v) in (int,float) and math.isfinite(v) for v in epoch.values()) for epoch in epochs),
                    "PPO four-pass statistics or sample/weight chain changed")
            update_records.append((payload, statistics))
    def inventory(directory, patterns):
        path = output/directory
        return {str(p.relative_to(output)) for pattern in patterns for p in path.glob(pattern)}
    require(set(receipts) == inventory("checkpoints", ["*"]), "Unlisted or missing checkpoint artifact")
    require(trials == inventory("comparison-native", ["*"]), "Unlisted, partial or missing native trial")
    require(updates == inventory("updates", ["*"]), "Unlisted or missing PPO update")
    require({"native/"+name for name in physics} == inventory("native", ["trial-*.npz"]),
            "Unlisted, partial or missing full-rate physics")
    require({"native/"+name for name in frames} == inventory("native/frames", ["*.jpg"]),
            "Unlisted or missing prescheduled native frames")
    require(sorted(int(Path(name).stem) for name in frames) == list(range(len(frames))),
            "Recorded native frame sequence has a missing index")
    for payload, statistics in update_records:
        require(statistics.get("actor_before") == actor_states[(payload["model"], payload["before"]["updates"])]
                and statistics.get("actor_after") == actor_states[(payload["model"], payload["after"]["updates"])],
                "Saved PPO actor chain differs from checkpoint bytes")
    require(paid_samples == accounting["completed"] == 18520 and batches == 539 and len(updates) == 128,
            "Independent model/physical coverage incomplete")
    report = dict(schema=SCHEMA, status="synthetic_verified" if synthetic else "native_verified",
        synthetic=synthetic, comparison_fingerprint=plan["fingerprint"], trace_head=accounting["trace_head"],
        manifest_fingerprint=manifest.get("fingerprint"), trace_canonical_sha256=signed_hash(trace),
        exit_proof=exit_proof,
        accounting=accounting, checkpoint_sha256={name: value["sha256"] for name, value in receipts.items()},
        checkpoint_bytes_independently_verified=not synthetic, model_likelihoods_independently_verified=not synthetic,
        native_physics_verified=not synthetic, checkpoint_count=len(receipts), paid_policy_samples=paid_samples,
        terminal_batches=batches, update_artifacts=len(updates), geometry=dict(geometry_counts),
        maximum_logp_error=maximum_error, startup=startup,
        limitations="CPU checkpoint/Adam counters and original likelihoods; saved geometry/FK and measured motion/release/full-rate scores. No bitwise replay of CUDA Adam arithmetic, collision-free continuous path certificate or physical validation. OS process ownership requires the separate watchdog audit.")
    report["fingerprint"] = signed_hash(report)
    return report
