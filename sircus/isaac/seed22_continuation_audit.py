"""Independent, CPU-only reader for the separate direct seed-22 continuation.

This reader never launches an app, optimizer, physics probe or geometry search.
It authenticates original inputs and raw evidence before reconstructing request
accounting, policy likelihoods, geometry/FK, measured motion guards and 240 Hz
capture/reward outcomes. It does not bitwise replay CUDA Adam arithmetic.
"""

import argparse
import ast
from collections import Counter
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from seed22_continuation_contract import (
    ASSESSMENT_REQUEST_CAP, GEOMETRY_LAYER_SHA256, Ledger, MONITOR_MILESTONES,
    ORIGINAL_EXPERIMENT_SHA256, ORIGINAL_SPLIT_SHA256, PARENT_CHECKPOINT_SHA256,
    TRAIN_REQUEST_CAP, WALL_SECONDS, batch_schedule, canonical_hash, signed,
    validate_fingerprint, validate_protocol, validate_splits,
)

SCHEMA = "sircus-seed22-continuation-independent-audit-v1"
SOURCE = Path(__file__).resolve().parent
PARENT_AUDIT_SHA256 = "cec9cff0a6de59b07dd35de3342ee00bc4106d4801edc0237c570e006aedc1c8"
# These exact guard-recording adapters predate this continuation. They are not
# byte-identical to the original benchmark producer, so do not claim that they
# are. The two inherited controller modules additionally require original-source
# AST comparison below; no changed threshold or unreviewed source is admitted.
PINNED_GUARD_ADAPTERS = {
    "film_cylinder_farm.py": "849b1af800a0f13a73af8845d39ec30117f8f01648effeb9698b5891af4efe8d",
    "settled_transit.py": "08e37329978f3acbe4476ebaf5513cfa28539985c806daa4a071596d381dd0a7",
    "joint_feasibility.py": "7fdf85de0d86fbb3bba161fd280d9e2939983e8d81687a42cdf40c5921803b70",
}
ORIGINAL_ADAPTER_SOURCES = {
    "film_cylinder_farm.py": ("original_film_source", "8ddc3a123b0abc3c6beca1e9053b4c189c31951481bd033da3bb65bf63ff0e2f"),
    "settled_transit.py": ("original_settled_source", "2bff0597861b629c589174c2bb14c39bd439efdf9036c0972e862a0204a78bdc"),
}
ORIGINAL_MODULES = (
    "continuous_policy.py", "continuous_feasibility.py", "robust_ik.py",
    "joint_feasibility.py", "expanded_pose_menu.py", "prepare_assets.py",
    "film_cylinder_farm.py", "parallel_runtime.py", "continuous_pilot.py",
    "settled_transit.py", "settling_contract.py", "feasibility_runtime.py",
    "feasibility_contract.py", "rl_contract.py", "board_route.py",
    "entry_quality.py", "farm_contract.py", "shot_benchmark_reward.py",
    "shot_benchmark_native.py", "cylinder_farm.py", "presentation_recording.py",
)
ARMS = ("Arm10", "Arm14")
CHECK_NAMES = {"inputs", "checkpoint", "policy", "geometry", "native", "startup"}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024**2), b""):
            result.update(part)
    return result.hexdigest()


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate independent-audit JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite audit JSON")))


def evidence(root, relative):
    require(type(relative) is str and relative and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts, "Escaped evidence path")
    root = Path(root).resolve()
    path = root / relative
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents))
            and path.resolve().is_relative_to(root), "Missing or linked evidence: " + relative)
    return path


def _same(actual, expected, label, tolerance=1e-7):
    from continuous_native_audit import same
    require(same(actual, expected, tolerance), label + " changed")


def verify_controller_definitions(name, original_text, current_text):
    """Compare every class/function definition except two pinned adapter bodies.

    Function signatures and decorators still compare. Imports/global setup and
    the two excluded bodies are bound by exact full-file hashes, not by a broad
    rule accepting arbitrary edits to main or prepare.
    """
    exclusions = {"film_cylinder_farm.py": ("main",),
                  "settled_transit.py": ("SettledTransitShots", "prepare")}
    require(name in exclusions, "Unknown reviewed controller adapter")
    wanted = exclusions[name]
    def definitions(text):
        tree = ast.parse(text)
        found = []
        def normalize(node, scope=()):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                scope = scope + (node.name,)
                if scope == wanted:
                    found.append(scope)
                    node.body = [ast.Pass()]
                    return node
            for field, child in ast.iter_fields(node):
                if isinstance(child, list):
                    setattr(node, field, [normalize(item, scope) if isinstance(item, ast.AST) else item
                                          for item in child])
                elif isinstance(child, ast.AST):
                    setattr(node, field, normalize(child, scope))
            return node
        nodes = [normalize(node) for node in tree.body
                 if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
        require(found == [wanted], "Reviewed adapter body missing or duplicated")
        return [ast.dump(node, include_attributes=False) for node in nodes]
    require(definitions(original_text) == definitions(current_text),
            "Original controller class/function body or signature changed")
    return dict(module=name, all_other_class_function_definitions_unchanged=True,
                excluded_body=".".join(wanted), excluded_body_bound_by_full_source_sha256=True)


def verify_guard_adapters(original_sources, current_sources, files):
    require(all(current_sources.get(name) == expected for name, expected in PINNED_GUARD_ADAPTERS.items()),
            "Pinned pre-existing guard adapter bytes changed")
    require("joint_feasibility.py" not in original_sources,
            "Reviewed new preflight helper was unexpectedly present in original source")
    proof = []
    for name, (role, expected) in ORIGINAL_ADAPTER_SOURCES.items():
        require(original_sources.get(name) == expected and files.get(role, {}).get("sha256") == expected,
                "Original controller adapter source identity changed")
        path = Path(files[role]["path"])
        require(digest(path) == expected, "Original controller source bytes changed")
        proof.append(verify_controller_definitions(name, path.read_text(), (SOURCE / name).read_text()))
    return dict(pinned_current_sha256=copy.deepcopy(PINNED_GUARD_ADAPTERS),
                original_source_sha256={name: item[1] for name, item in ORIGINAL_ADAPTER_SOURCES.items()},
                controller_definitions=proof,
                scope="Exact pre-existing CLI/guard-recording adapters; original non-adapted controller definitions and all other physics/scoring/policy helpers remain bound")


def verify_inputs(manifest):
    """Authenticate every declared input and unchanged original physical helper."""
    require(manifest.get("schema") == "sircus-seed22-continuation-inputs-v1"
            and manifest.get("synthetic", False) is False, "Real bound manifest required")
    protocol = manifest["protocol"]
    validate_protocol(protocol)
    members = {path.name: digest(path) for path in SOURCE.glob("*.py")}
    require(manifest.get("source_sha256") == protocol["source_sha256"] == members,
            "Executed/auditor Python source membership changed")
    require(members.get("continuous_geometry_layer.py") == GEOMETRY_LAYER_SHA256,
            "Frozen continuous geometry layer changed")
    files = manifest.get("files")
    required = {"parent", "parent_audit", "experiment", "original_splits", "exclusions", "splits",
                "camera", "poses", "dropper_urdf", "board_urdf", "original_film_source", "original_settled_source"}
    require(type(files) is dict and required.issubset(files), "Missing bound original inputs")
    for role, record in files.items():
        require(type(record) is dict and set(record) == {"path", "sha256"}, "Malformed input role")
        path = Path(record["path"])
        require(path.is_absolute() and path.is_file()
                and not any(p.is_symlink() for p in (path, *path.parents))
                and str(path.resolve()) == str(path) and digest(path) == record["sha256"],
                "Bound input bytes changed: " + role)
    for role, expected in (("parent", PARENT_CHECKPOINT_SHA256), ("parent_audit", PARENT_AUDIT_SHA256),
                           ("experiment", ORIGINAL_EXPERIMENT_SHA256), ("original_splits", ORIGINAL_SPLIT_SHA256)):
        require(files[role]["sha256"] == expected, "Wrong original " + role)
    original = read(files["parent_audit"]["path"])
    require(original.get("status") == "passed" and original.get("seed") == 20261122
            and original.get("method") == "ppo" and original.get("paid_train_attempts") == 10000
            and original.get("updates") == 73, "Original parent independent audit failed")
    require(all(original["source_sha256"].get(name) == members.get(name)
                for name in ORIGINAL_MODULES if name not in PINNED_GUARD_ADAPTERS),
            "Original physics, controller, policy or scoring helper changed")
    guard_adapters = verify_guard_adapters(original["source_sha256"], members, files)
    experiment = read(files["experiment"]["path"])
    from shot_benchmark_contract import make_protocol
    require(experiment["protocol"] == make_protocol(), "Original experiment protocol changed")
    old = read(files["original_splits"]["path"])
    exclusions = read(files["exclusions"]["path"])
    splits = read(files["splits"]["path"])
    validate_splits(splits, original_train=old["train"], excluded_coordinates=exclusions["coordinates"])
    require(splits["exclusions_fingerprint"] == protocol["exclusions_fingerprint"],
            "Outcome-blind exclusion history changed")
    return dict(experiment=experiment, splits=splits, original=original, guard_adapters=guard_adapters,
                files=files, source_sha256=members)


def load_model(path, cp, protocol, experiment):
    """Read model/Adam state on CPU; perform no optimizer step or CUDA call."""
    import torch
    from continuous_policy import load_checkpoint
    from candidate_continuation_checkpoint import state_hash
    new_attempts, new_updates = cp["additional_attempts"], cp["additional_updates"]
    require(type(new_attempts) is type(new_updates) is int and 0 <= new_attempts <= TRAIN_REQUEST_CAP
            and 0 <= new_updates <= new_attempts, "Invalid continuation checkpoint counters")
    model, saved = load_checkpoint(path, study_fingerprint=protocol["fingerprint"],
        geometry_fingerprint=experiment["source"]["geometry_fingerprint"],
        domain_fingerprint=experiment["protocol"]["domain"]["fingerprint"], device="cpu",
        expected_updates=73 + new_updates, expected_attempted_shots=10000 + new_attempts)
    require(saved.get("seed") == 20261122 and saved.get("parent_checkpoint_sha256") == PARENT_CHECKPOINT_SHA256
            and saved.get("additional_attempts") == new_attempts
            and saved.get("additional_updates") == new_updates
            and saved.get("exploration_seed") == 20261130
            and saved["weight_hash"] == cp["state_hash"], "Checkpoint parent or state binding changed")
    require(all(torch.isfinite(value).all().item() for value in saved["state_dict"].values()),
            "Nonfinite checkpoint weights")
    optimizer = saved.get("optimizer")
    optimizer_sha = None
    if Path(cp["file"]).name == "parent.pt":
        require(optimizer is None and new_attempts == new_updates == 0, "Frozen parent state changed")
    else:
        require(type(optimizer) is dict and type(optimizer.get("state")) is dict
                and len(optimizer["state"]) == len(list(model.parameters()))
                and optimizer.get("param_groups")
                and all(group.get("lr") == .0003 for group in optimizer["param_groups"]),
                "Checkpoint optimizer incomplete or learning rate changed")
        for state in optimizer["state"].values():
            require(float(state.get("step", -1)) == 292 + 4 * new_updates
                    and all(not torch.is_tensor(value) or torch.isfinite(value).all().item()
                            for value in state.values()), "Adam counter or finite state changed")
        optimizer_sha = state_hash(optimizer)
    return dict(policy=model, state_hash=saved["weight_hash"], actor_hash=saved["actor_weight_hash"],
                optimizer_sha256=optimizer_sha, additional_attempts=new_attempts,
                additional_updates=new_updates, split_fingerprint=saved.get("split_fingerprint"))


def verify_geometry(trial, scene, links):
    """Recheck saved geometric proofs without running a solver or ball flight."""
    import numpy as np
    from continuous_feasibility import LOW, HIGH, decode_action
    from continuous_geometry_audit import audit_result
    from continuous_policy import build_observations, physical_actions
    active, cells, raw = trial["enabled"], trial["cells"], trial["proposed_actions"]
    require(all(type(trial.get(name)) is list and len(trial[name]) == 35
                for name in ("actions", "proposed_actions", "geometry_records", "geometry_starts", "proposal_ik_seeds")),
            "Missing full raw/executed geometry coverage")
    require(trial.get("geometry_filter_synthetic") is False, "Synthetic geometry filter cannot certify native execution")
    enabled = trial["mode"] != "raw_parent"
    require(trial.get("geometry_enabled") is enabled, "Geometry wrapper mode changed")
    require(np.array_equal(trial.get("observations"), build_observations(cells)),
            "Saved network inputs differ from the original bucket features")
    physical = physical_actions(trial["actions_normalized"], LOW, HIGH)
    counts = Counter()
    for index, cell in enumerate(cells):
        proposal = decode_action(cell, physical[index], f"{trial['mode']}-{trial['batch']}-{index}")
        _same(raw[index], proposal, "Raw continuous proposal", 1e-12)
        record, starts, seeds = (trial[name][index] for name in
                                ("geometry_records", "geometry_starts", "proposal_ik_seeds"))
        if not enabled or not active[index]:
            require(record is starts is seeds is None, "Bypassed or unpaid slot ran the geometry filter")
            _same(trial["actions"][index], raw[index], "Bypassed raw action", 0)
            continue
        expected_starts = {arm: trial["home"]["measured_joint_rad"][arm][index] for arm in ARMS}
        require(starts == expected_starts, "Geometry starts differ from actual measured home")
        classification = audit_result(record, scene, cell, raw[index], links, starts)
        require(record["binding"]["protocol"]["max_nfev"] == 24
                and record["binding"]["protocol"]["max_residual_evaluations"] == 512
                and record["binding"]["protocol"]["max_normalized_adjustment"] == .10,
                "Geometry work/adjustment limits changed")
        counts[classification] += 1
        if classification == "rejected":
            require(seeds is None and trial["actions"][index] == raw[index]
                    and trial["rows"][index]["invalid"] == "no_geometry_support"
                    and trial["rows"][index]["qualification"] == dict(accepted=False, reason="no_geometry_support")
                    and trial["preflight_records"][index] is None,
                    "Paid geometry decline was released, relabelled or given a fabricated native preflight")
        else:
            require(trial["actions"][index] == record["action"]
                    and seeds == record["final_preflight"]["qualification"]["ik"],
                    "Executed action or native IK seeds differ from admitted geometric proof")
    return dict(counts, raw_latent_likelihood_preserved=True, no_geometry_search=True)


def padding_guard_view(trial):
    """Adapt only unpaid parked native labels for the old first-cause reader."""
    guarded = copy.deepcopy(trial)
    normalized = []
    for index, enabled in enumerate(trial["enabled"]):
        if enabled:
            continue
        row = trial["rows"][index]
        require(row.get("invalid") == "no_selected_setting"
                and row.get("qualification") == dict(accepted=False, reason="inactive_split")
                and row.get("route_success") is False and row.get("success") is False
                and row.get("route_quality") == 0 and row.get("path_proxy") is None
                and row.get("path_rejection") is None
                and trial["preflight_records"][index] is None
                and trial["proposal_ik_seeds"][index] is None
                and trial["geometry_records"][index] is None
                and trial["geometry_starts"][index] is None
                and trial["reward_features"][index] is None and trial["rewards"][index] == -1.
                and trial["trajectory_m"][index] == [], "Unpaid padding was executed, rewarded or relabelled")
        guarded["rows"][index]["invalid"] = "inactive_split"
        normalized.append(index)
    require(all(guarded["rows"][index] == row for index, row in enumerate(trial["rows"])
                if trial["enabled"][index]), "Padding adapter changed a paid row")
    return guarded, normalized


def verify_native(native, trial, scene, links, poses, camera):
    """Reconstruct actual native motion, first-cause rejections and all scores."""
    import numpy as np
    from candidate_native_audit import verify_initial_geometry
    from candidate_continuation_native_guards import (
        _verify_preflight_records, _verify_corrections, _verify_rejections,
    )
    from continuous_native_audit import verify_camera_frame
    from diverse_pilot_audit import verify_settling_history
    from settled_transit import protocol
    from shot_benchmark_audit import verify_home, verify_native_trial
    verify_home(trial)
    home = {"Arm10": poses["dropper_joint_rad"], "Arm14": poses["board_joint_rad"]}
    for arm in ARMS:
        require(np.allclose(trial["home"]["target_joint_rad"][arm], np.tile(home[arm], (35, 1)), atol=1e-6, rtol=0),
                "Original canonical home target changed")
        require(np.array_equal(trial["transit_history"][0]["state"][arm]["position_rad"],
                               trial["home"]["measured_joint_rad"][arm]), "Native transit start changed")
        require(np.array_equal(trial["transit_history"][0]["state"][arm]["velocity_rad_s"],
                               trial["home"]["measured_joint_velocity_rad_s"][arm]), "Native start velocity changed")
    # These paid declines have independently checked layer proofs. They never
    # invoked native IK. Mask only the IK reader's view; paid accounting, rows,
    # first-cause labels, settling and physical scoring remain original.
    ik_view = copy.deepcopy(trial)
    for index, row in enumerate(trial["rows"]):
        if row["qualification"].get("reason") == "no_geometry_support":
            ik_view["enabled"][index] = False
    _verify_preflight_records(ik_view, scene, trial["cells"], trial["proposal_ik_seeds"])
    geometry = verify_initial_geometry(ik_view, scene, trial["cells"], links, trial["proposal_ik_seeds"])
    settling = verify_settling_history(trial, scene, trial["cells"], dict(settling_protocol=protocol()))
    corrections = _verify_corrections(trial, scene, trial["cells"], links)
    guarded, padding = padding_guard_view(trial)
    rejections = _verify_rejections(guarded, trial["cells"])
    frames = trial.get("frames")
    require(type(frames) is list and (bool(frames) if trial["filming"] else not frames),
            "Scheduled native frames missing or unexpectedly recorded")
    for frame in frames:
        verify_camera_frame(frame, camera)
        require(frame.get("phase") in ("setup", "release")
                and trial["home"]["physics_step"] <= frame["physics_step"] <= trial["score_end_step"]
                and frame.get("jpeg_quality") == 82, "Native camera phase or quality changed")
        image = evidence(native, f"frames/{frame['frame']:05d}.jpg")
        require(digest(image) == frame["jpeg_sha256"], "Native JPEG source changed")
    clip = dict(rows=trial["rows"], actions=trial["actions"], release_step=trial["release_step"], frames=frames) if frames else None
    score = verify_native_trial(native, trial, scene, clip)
    return dict(geometry=geometry, settling=settling, corrections=corrections, rejections=rejections,
                physical=score, unpaid_padding_normalized=padding,
                geometry_declines_skipped_only_in_native_ik_view=True)


def expected_trials(splits, completed):
    """Reconstruct all native cohort order, masks and monitoring boundaries."""
    require(type(completed) is int and 0 <= completed <= TRAIN_REQUEST_CAP, "Invalid completed TRAIN count")
    schedule = [batch for batch in batch_schedule() if batch["attempt_stop"] <= completed]
    require((schedule[-1]["attempt_stop"] if schedule else 0) == completed,
            "TRAIN count is not a completed fixed-schedule prefix")
    result = []
    def add(method, role, cells, attempts, batch=None, film=False):
        result.append(dict(method=method, role=role, cells=cells,
            enabled=batch["enabled"] if batch else [True] * 35,
            order=batch["site_indices"] if batch else None,
            attempts=attempts, schedule=batch, filming=film))
    add("raw_parent", "validation", splits["validation"], 0)
    add("monitor", "validation", splits["validation"], 0)
    for batch in schedule:
        add("train", "train", splits["train"], batch["attempt_start"], batch,
            film=batch["attempt_start"] in (0, 1000, 5000, 10000, 20000))
        if batch["attempt_stop"] in MONITOR_MILESTONES:
            add("monitor", "validation", splits["validation"], batch["attempt_stop"])
    if completed not in MONITOR_MILESTONES:
        add("monitor-final", "validation", splits["validation"], completed)
    for method in ("raw_parent", "parent_geometry", "continued_geometry"):
        for begin in (0, 35):
            add(method, "test", splits["test"][begin:begin + 35], completed,
                film=method == "continued_geometry" and begin == 0)
    return result


def verify_updates(train, updates, final_checkpoint, *, initial_checkpoint, states_by_update):
    """Every fresh paid TRAIN sample appears in exactly one bounded update."""
    groups, pending = [], []
    for item in train:
        pending.append(item)
        if len(pending) == 4 or item["completed"] in MONITOR_MILESTONES:
            groups.append(pending)
            pending = []
    if pending:
        groups.append(pending)
    require(len(updates) == len(groups), "Missing or extra PPO update")
    previous, previous_actor = initial_checkpoint["state_hash"], initial_checkpoint["actor_hash"]
    require(set(states_by_update) == set(range(len(groups) + 1)), "Missing actual model state at a learning boundary")
    for index, (group, record) in enumerate(zip(groups, updates)):
        count = sum(item["paid"] for item in group)
        require(record.get("update_index") == index and record.get("source_files") == [item["file"] for item in group]
                and record.get("samples") == count and record.get("additional_attempts") == group[-1]["completed"]
                and record.get("before") == record.get("before_weight_hash") == previous
                and all(item["state_hash"] == previous and item["actor_hash"] == previous_actor
                        and item["additional_updates"] == index for item in group),
                "PPO raw-sample source or weight update chain changed")
        actual_after = states_by_update[index + 1]
        require(record.get("actor_before") == previous_actor
                and record.get("actor_after") == actual_after[1]
                and record.get("after") == actual_after[0],
                "PPO actor update identity changed")
        epochs = record.get("epochs")
        require(type(epochs) is list and len(epochs) == 4
                and all(epoch.get("samples") == count and all(type(value) in (int, float)
                    and not isinstance(value, bool) and math.isfinite(value) for value in epoch.values()) for epoch in epochs),
                "PPO optimizer epochs, sample count or finite metrics changed")
        previous, previous_actor = record["after"], record["actor_after"]
    require(final_checkpoint["state_hash"] == previous
            and final_checkpoint["actor_hash"] == previous_actor
            and final_checkpoint["additional_updates"] == len(groups), "Final checkpoint is not the last learned state")
    return dict(updates=len(groups), optimizer_steps=4 * len(groups), samples=sum(x["paid"] for x in train),
                original_latent_probabilities_checked=True, cpu_bitwise_adam_replay=False)


def verify_watchdog(output, result, wall_limit=WALL_SECONDS):
    from candidate_continuation_ledger import canonical_hash as compact_hash
    require(type(wall_limit) in (int, float) and math.isfinite(wall_limit)
            and 0 < wall_limit <= WALL_SECONDS, "Invalid remaining native wall budget")
    owner = read(evidence(output, "watchdog-owner.json"))
    exit_record = read(evidence(output, "watchdog-exit.json"))
    for record in (owner, exit_record):
        require(record.get("fingerprint") == compact_hash({k: v for k, v in record.items() if k != "fingerprint"})
                and record.get("schema") == "sircus-seed22-continuation-watchdog-v1"
                and record.get("session_id") == result["session_id"]
                and record.get("runtime_fingerprint") == result["protocol_fingerprint"],
                "Watchdog receipt identity or fingerprint changed")
    require(owner.get("shared_lock") == "/workspace/scene.lock"
            and owner.get("gpu_uuid") == "GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4"
            and owner.get("max_requests") == TRAIN_REQUEST_CAP
            and owner.get("storage_floor_bytes", 0) >= 6 * 1024**3
            and math.isclose(owner.get("deadline_monotonic", 0) - owner.get("start_monotonic", 0),
                             wall_limit, rel_tol=0, abs_tol=1e-6),
            "Watchdog GPU, shared lock, wall or request limits changed")
    require(exit_record.get("status") == "native_exit_zero_independent_audit_required"
            and exit_record.get("native_application_starts") == 1
            and exit_record.get("exit_code") == 0 and exit_record.get("reason") is None
            and exit_record.get("lock_release_safe") is True
            and exit_record.get("native_owner") == owner.get("native_owner")
            and exit_record.get("train_requested") == result["train_requested"]
            and exit_record.get("train_completed") == result["train_completed"]
            and result["elapsed_s"] <= exit_record.get("wall_s", -1) <= wall_limit,
            "Watchdog did not confirm a bounded clean native exit")
    return dict(owner_sha256=digest(output / "watchdog-owner.json"),
                exit_sha256=digest(output / "watchdog-exit.json"), wall_s=exit_record["wall_s"],
                native_wall_limit_s=wall_limit)


def _default_checks():
    from shot_benchmark_audit import verify_policy_sample
    from candidate_continuation_scene import audit_startup
    return dict(inputs=verify_inputs, checkpoint=load_model, policy=verify_policy_sample,
                geometry=verify_geometry, native=verify_native, startup=audit_startup)


def audit(output, manifest, expected_result_sha256, *, synthetic=False, checks=None):
    """Verify a complete run against an externally pinned result and inputs.

    Dependency injection is available only for explicitly synthetic fixtures.
    Such a fixture can never receive the production ``passed`` status.
    """
    output = Path(output).resolve()
    require(type(synthetic) is bool and type(manifest) is dict
            and manifest.get("synthetic", False) is synthetic,
            "Synthetic provenance must be explicit and match the manifest")
    require(checks is None or synthetic, "Injected checks cannot certify real native evidence")
    if not synthetic:
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Independent audit must hide CUDA")
    checks = _default_checks() if checks is None else checks
    require(type(checks) is dict and set(checks) == CHECK_NAMES and all(callable(x) for x in checks.values()),
            "Complete independent check set required")
    result_path = evidence(output, "result.json")
    require(digest(result_path) == expected_result_sha256, "Externally pinned result bytes changed")
    result = read(result_path)
    validate_fingerprint(result)
    require(result.get("schema") == "sircus-seed22-continuation-result-v1"
            and result.get("status") == "native_complete_audit_pending"
            and result.get("parent_attempts") == 10000 and result.get("parent_updates") == 73
            and result.get("model_promoted") is False and not (output / "failure.json").exists(),
            "Complete unpromoted continuation required")
    protocol = manifest["protocol"]
    validate_protocol(protocol)
    require(result["protocol_fingerprint"] == protocol["fingerprint"], "Result protocol changed")
    verified = checks["inputs"](manifest)
    wall_limit = WALL_SECONDS
    if not synthetic:
        from seed22_continuation import native_budget
        wall_limit = native_budget(manifest)
    experiment, splits = verified["experiment"], verified["splits"]
    validate_splits(splits)
    require(read(evidence(output, "inputs.json")) == manifest
            and read(evidence(output, "splits.json")) == splits, "Saved inputs or fresh layouts changed")
    completed = result.get("train_completed")
    require(type(completed) is int and completed == result.get("train_requested") and 0 <= completed <= TRAIN_REQUEST_CAP
            and type(result.get("elapsed_s")) in (int, float) and 0 <= result["elapsed_s"] <= wall_limit,
            "Completed request count or eight-hour wall bound changed")
    reason = result.get("stop_reason")
    require((reason == "request_cap" and completed == TRAIN_REQUEST_CAP)
            or (reason == "time_budget_reserved_final_assessment" and completed < TRAIN_REQUEST_CAP
                and result["elapsed_s"] >= wall_limit - 1200), "Run stopped outside its prespecified bounds")
    watchdog = verify_watchdog(output, result, wall_limit=wall_limit)
    native = output / "native"
    require(native.is_dir() and not native.is_symlink() and not (native / "incomplete-last-physics.npz").exists(),
            "Native evidence missing or partial")
    spec = read(evidence(native, "contract.json"))
    validate_fingerprint(spec)
    require(spec["fingerprint"] == experiment["source"]["geometry_fingerprint"], "Original native scene changed")
    scene, files = spec["scene"], verified["files"]
    from camera_options import preview_origins
    origins = preview_origins()
    startup = checks["startup"](native / "startup", scene=scene, cells=spec["cells"], origins=origins)
    require(startup.get("status") == "passed", "Independent measured startup checks failed")
    links = {}
    if not synthetic:
        from prepare_assets import chain
        for arm, role in (("Arm10", "dropper_urdf"), ("Arm14", "board_urdf")):
            links[arm] = chain(ET.fromstring(Path(files[role]["path"]).read_bytes()))
    else:
        links = {arm: None for arm in ARMS}
    poses, camera = read(files["poses"]["path"]), read(files["camera"]["path"])
    expected = expected_trials(splits, completed)
    expected_files = [f"trials/{entry['method']}-{index:05d}.json.gz" for index, entry in enumerate(expected)]
    receipts = result.get("trial_sha256")
    require(type(receipts) is dict and set(receipts) == set(expected_files)
            and set(expected_files) == {str(p.relative_to(native)) for p in (native / "trials").glob("*.json.gz")},
            "Missing, repeated or extra native trial")
    expected_requests = {Path(name).name[:-8] + ".json" for name in expected_files}
    require(expected_requests == {p.name for p in (output / "requests").iterdir()},
            "Missing, linked or extra durable request reservation")
    cp_cache, cp_receipts, states_by_update = {}, {}, {}
    def checkpoint(cp):
        require(type(cp) is dict and set(cp) == {"file", "sha256", "state_hash", "additional_attempts", "additional_updates"}
                and Path(cp["file"]).parts[:1] == ("checkpoints",), "Invalid checkpoint receipt")
        path = evidence(output, cp["file"])
        require(digest(path) == cp["sha256"], "Checkpoint bytes changed")
        if cp["file"] in cp_cache:
            require(cp_receipts[cp["file"]] == cp, "One checkpoint has conflicting identities")
        else:
            cp_cache[cp["file"]] = checks["checkpoint"](path, cp, protocol, experiment)
            cp_receipts[cp["file"]] = copy.deepcopy(cp)
            require(cp_cache[cp["file"]]["split_fingerprint"] == splits["fingerprint"],
                    "Checkpoint fresh split changed")
            if cp["file"] != "checkpoints/parent.pt":
                model = cp_cache[cp["file"]]
                identity = (model["state_hash"], model["actor_hash"], model["optimizer_sha256"])
                count = model["additional_updates"]
                if count in states_by_update:
                    require(states_by_update[count] == identity,
                            "Model or optimizer changed without a recorded learning update")
                else:
                    states_by_update[count] = identity
        return cp_cache[cp["file"]]
    from shot_benchmark_audit import read_trial, summary
    ledger, train_rows, train_returns, train_details, trial_audits = Ledger(result["session_id"]), [], [], [], []
    evaluations, update_records, frame_files, physics_files = {}, [], set(), set()
    expected_curves, buffered_batches = [], 0
    previous_elapsed, maximum_logp = 0., 0.
    for index, (name, exp) in enumerate(zip(expected_files, expected)):
        stem = Path(name).name[:-8]
        request_path = evidence(output, "requests/" + stem + ".json")
        request = read(request_path)
        paid = sum(exp["enabled"])
        reservation = ledger.reserve(index, exp["role"], paid)
        additional = exp["attempts"] + (paid if exp["role"] == "train" else 0)
        require(request.get("index") == index and request.get("role") == exp["role"]
                and request.get("method") == exp["method"] and request.get("reservation") == reservation
                and request.get("cells") == exp["cells"] and request.get("enabled") == exp["enabled"]
                and request.get("request_order") == exp["order"]
                and request.get("additional_requested") == additional
                and request.get("protocol_fingerprint") == protocol["fingerprint"],
                "Durable request, seeded schedule, split or reservation changed")
        require(type(request.get("elapsed_s")) in (int, float)
                and previous_elapsed <= request["elapsed_s"] <= result["elapsed_s"], "Request clock regressed")
        previous_elapsed = request["elapsed_s"]
        trial, raw_receipt = read_trial(native, name, receipts[name])
        require(trial.get("schema") == "sircus-seed22-native-trial-v1" and trial.get("mode") == exp["method"]
                and trial.get("batch") == index and trial.get("cells") == exp["cells"]
                and trial.get("enabled") == exp["enabled"] and all(type(v) is bool for v in trial["enabled"])
                and trial.get("training_request_order") == exp["order"]
                and trial.get("reservation_sha256") == digest(request_path)
                and trial.get("protocol_fingerprint") == protocol["fingerprint"]
                and trial.get("source") == experiment["source"] and trial.get("filming") is exp["filming"],
                "Raw trial association, mask, source or filming schedule changed")
        require(type(trial.get("rows")) is list and len(trial["rows"]) == 35
                and len(trial.get("rewards", [])) == 35, "Incomplete native outcome slots")
        require(trial.get("checkpoint") == request.get("checkpoint")
                and trial.get("actions_normalized") == request["learner_packet"]["actions_normalized"]
                and trial.get("learner_packet") == {k: v for k, v in request["learner_packet"].items()
                                                    if k != "actions_normalized"},
                "Native execution differs from reserved original policy sample")
        cp = checkpoint(trial["checkpoint"])
        if exp["role"] == "train":
            require(cp["additional_attempts"] == exp["attempts"], "TRAIN used a stale checkpoint counter")
        elif exp["method"] in ("raw_parent", "parent_geometry"):
            require(trial["checkpoint"]["file"] == "checkpoints/parent.pt", "Parent baseline changed model")
        else:
            require(cp["additional_attempts"] == exp["attempts"], "Assessment used a different training milestone")
        maximum_logp = max(maximum_logp, checks["policy"](cp["policy"], trial,
            deterministic=exp["role"] != "train"))
        geometry = checks["geometry"](trial, scene, links)
        measured = checks["native"](native, trial, scene, links, poses, camera)
        for frame in trial.get("frames", []):
            relative = f"frames/{frame['frame']:05d}.jpg"
            require(relative not in frame_files, "Repeated native frame identity")
            frame_files.add(relative)
        if trial.get("physics") is not None:
            require(trial["physics"]["file"] not in physics_files, "Repeated native physics packet")
            physics_files.add(trial["physics"]["file"])
        done = ledger.complete(index, exp["role"], paid)
        terminal = read(evidence(output, "events/" + stem + ".json"))
        require(terminal.get("index") == index and terminal.get("file") == name
                and terminal.get("receipt") == {k: raw_receipt[k] for k in ("sha256", "bytes", "uncompressed_sha256")}
                | {"file": Path(name).name}, "Terminal receipt association changed")
        require(terminal.get("ledger_event") == done and terminal.get("ledger") == ledger.snapshot(),
                "Terminal request completion or ledger changed")
        active_indices = exp["order"] if exp["role"] == "train" else list(range(35))
        rows = [{key: trial["rows"][i][key] for key in ("invalid", "route_success", "route_quality", "external_contact_steps")}
                for i in active_indices]
        values = [trial["rewards"][i] for i in active_indices]
        if exp["role"] == "train":
            train_rows.extend(rows)
            train_returns.extend(values)
            train_details.append(dict(file=name, paid=paid, completed=additional,
                state_hash=cp["state_hash"], actor_hash=cp["actor_hash"], additional_updates=cp["additional_updates"]))
            buffered_batches += 1
            flush = buffered_batches == 4 or additional in MONITOR_MILESTONES
            expected_curves.append(dict(summary(train_rows, train_returns), additional_attempts=additional,
                new_updates=cp["additional_updates"] + int(flush)))
            if flush:
                buffered_batches = 0
        else:
            key = (exp["method"], exp["role"], exp["attempts"])
            group = evaluations.setdefault(key, dict(rows=[], values=[], files=[], checkpoint=trial["checkpoint"], cells=[]))
            require(group["checkpoint"] == trial["checkpoint"], "Assessment cohort changed frozen model")
            group["rows"].extend(rows)
            group["values"].extend(values)
            group["files"].append(name)
            group["cells"].extend(exp["cells"])
        trial_audits.append(dict(file=name, **raw_receipt, geometry=geometry, native=measured))
    finish = ledger.finish(reason)
    require(read(evidence(output, "events/ledger-finish.json")) == finish
            and read(evidence(output, "ledger-history.json")) == dict(events=ledger.history())
            and result.get("ledger") == ledger.snapshot(), "Closed independent ledger history changed")
    require(ledger.train_reserved == ledger.train_terminal == completed
            and ledger.assessment_reserved == ledger.assessment_terminal <= ASSESSMENT_REQUEST_CAP,
            "Unresolved or over-budget requests cannot pass")
    _same(result["train"], summary(train_rows, train_returns), "Complete TRAIN outcome counts")
    recorded = result.get("assessments")
    require(type(recorded) is list and len(recorded) == len(evaluations), "Missing or extra final/monitoring assessment")
    for saved, ((method, role, attempts), group) in zip(recorded, evaluations.items()):
        require(saved.get("method") == method and saved.get("split") == role
                and saved.get("additional_attempts") == attempts and saved.get("files") == group["files"]
                and saved.get("checkpoint") == group["checkpoint"], "Frozen assessment membership or milestone changed")
        _same(saved.get("counts"), summary(group["rows"], group["values"]), "Frozen assessment outcomes")
    require(read(evidence(output, "assessments.json")) == dict(assessments=recorded), "Assessment publication differs from result")
    final_receipt = result["final_checkpoint"]
    require(final_receipt.get("file") == "checkpoints/final.pt"
            and final_receipt.get("additional_attempts") == completed, "Final artifact was not the last checkpoint")
    final = checkpoint(final_receipt)
    updates_files = sorted((output / "events").glob("update-*.json"))
    require([p.name for p in updates_files] == [f"update-{i + 1:05d}.json" for i in range(len(updates_files))],
            "PPO update receipt gap or repeated update")
    update_records = [read(evidence(output, "events/" + p.name)) for p in updates_files]
    learning = verify_updates(train_details, update_records, final,
        initial_checkpoint=cp_cache["checkpoints/initial.pt"], states_by_update=states_by_update)
    require(result.get("new_updates") == result.get("update_events") == learning["updates"]
            and result.get("new_optimizer_steps") == learning["optimizer_steps"], "Reported learning counters changed")
    curves = read(evidence(output, "learning-curve.json")).get("curve")
    require(type(curves) is list and len(curves) == len(expected_curves), "Incomplete TRAIN learning curve")
    previous_curve_time = 0.
    for saved, expected_curve in zip(curves, expected_curves):
        _same({key: saved.get(key) for key in expected_curve}, expected_curve, "Cumulative TRAIN curve")
        require(type(saved.get("elapsed_s")) in (int, float)
                and previous_curve_time <= saved["elapsed_s"] <= result["elapsed_s"], "Learning curve clock regressed")
        previous_curve_time = saved["elapsed_s"]
    expected_events = {name[:-8] + ".json" for name in map(lambda x: Path(x).name, expected_files)}
    expected_events |= {p.name for p in updates_files} | {"ledger-finish.json"}
    require(expected_events == {p.name for p in (output / "events").iterdir()}, "Unlisted or missing execution event")
    require(set(cp_receipts) == {str(p.relative_to(output)) for p in (output / "checkpoints").iterdir()},
            "Unlisted or missing saved checkpoint")
    require(physics_files == {p.name for p in native.glob("native-*.npz")}, "Unlisted or missing native physics")
    require(frame_files == {str(p.relative_to(native)) for p in (native / "frames").glob("*.jpg")},
            "Unlisted or missing recorded JPEG frame")
    initial, parent = cp_cache["checkpoints/initial.pt"], cp_cache["checkpoints/parent.pt"]
    restoration = read(evidence(output, "parent-restoration.json"))
    require(restoration.get("parent_sha256") == PARENT_CHECKPOINT_SHA256
            and restoration.get("parent_steps") == 292 and restoration.get("parent_attempts") == 10000
            and restoration.get("parent_updates") == 73 and restoration.get("optimizer_restored") is True
            and restoration.get("exploration_seed") == 20261130
            and restoration.get("restored_weight_hash") == initial["state_hash"] == parent["state_hash"]
            and restoration.get("optimizer_state_sha256") == initial["optimizer_sha256"],
            "Parent weights or optimizer restoration changed")
    if not synthetic:
        import torch
        from candidate_continuation_checkpoint import state_hash
        original_parent = torch.load(files["parent"]["path"], map_location="cpu", weights_only=True)
        require(original_parent["weight_hash"] == parent["state_hash"]
                and state_hash(original_parent["optimizer"]) == initial["optimizer_sha256"],
                "Continuation did not start from exact saved parent learning state")
    # All TEST cohorts follow the final update and use the prescribed models.
    comparison = {}
    for method in ("raw_parent", "parent_geometry", "continued_geometry"):
        group = evaluations[(method, "test", completed)]
        require(group["cells"] == splits["test"] and len(group["rows"]) == 70,
                "Final TEST is not the complete common 70-position set")
        expected_cp = final_receipt if method == "continued_geometry" else cp_receipts["checkpoints/parent.pt"]
        require(group["checkpoint"] == expected_cp, "Final TEST selected another model")
        comparison[method] = summary(group["rows"], group["values"])
    return signed(dict(schema=SCHEMA, status="synthetic_passed" if synthetic else "passed", synthetic=synthetic,
        result_sha256=expected_result_sha256, protocol_fingerprint=protocol["fingerprint"],
        split_fingerprint=splits["fingerprint"], parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
        final_checkpoint=final_receipt, train=summary(train_rows, train_returns), learning=learning,
        assessments=recorded, final_test=comparison, trials=trial_audits, startup=startup,
        ledger=ledger.snapshot(), maximum_policy_logp_error=maximum_logp, watchdog=watchdog,
        native_wall_seconds=wall_limit, previous_startup_wall_s=WALL_SECONDS-wall_limit,
        audit_source_sha256=digest(__file__), guard_adapters=verified.get("guard_adapters"), model_promoted=False,
        scope="Independent raw-request/seeded-schedule/parent/optimizer-counter/likelihood chain; geometry saved-fit FK and sampled paths; measured movement/settling/release gates and all full-rate 240Hz ball/contact scores. No CPU bitwise CUDA Adam replay, mesh-path safety certificate, physical validation or repeated training-seed uncertainty."))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-result-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require(digest(args.manifest) == args.expected_manifest_sha256, "Externally pinned manifest changed")
    require(not args.output.exists(), "Independent audit cannot replace old evidence")
    value = audit(args.run, read(args.manifest), args.expected_result_sha256)
    from candidate_continuation_journal import write_new_json
    write_new_json(args.output, value)
    print(json.dumps(dict(status=value["status"], output=str(args.output), sha256=digest(args.output))))


if __name__ == "__main__":
    main()
