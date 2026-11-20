"""Separate CPU saved-data audit; the original shutdown failure stays failed.

Original scientific helper readers are imported statically from the pinned
producer source. The private scientific function below is an AST-checked copy
of its audit body; only exit-proof/report binding and status differ. This module
starts no process, simulator, optimizer or shot, and writes no evidence.
"""

import ast
from collections import Counter
import copy
from datetime import datetime
import hashlib
import inspect
import math
import os
from pathlib import Path
import re
import stat

import location_comparison_verify as original_verifier
from candidate_continuation_ledger import canonical_hash as process_digest
from location_comparison_audit import audit_trace, digest as signed_hash, CORE_STATE
from location_comparison_verify import (
    CHECKS, _default_checks, _finite_tree, artifact, guard_view, read, require, sha,
)

SCHEMA = "sircus-location-comparison-recovery-audit-v1"
REQUEST_SCHEMA = "sircus-location-comparison-recovery-request-v1"
RETIREMENT_SCHEMA = "sircus-location-comparison-retirement-proof-v1"
INVENTORY_SCHEMA = "sircus-comparison-closed-evidence-inventory-v1"
STATUS = "data_verified_shutdown_incomplete"
SOURCE_COMMIT = "300ab5c6d161b51577d37d2a70f7499fc3f782e9"
ORIGINAL_VERIFIER_SHA256 = "0e196852b48070d423d4db61b894b93b36edc655bc5f92c77adbf6b3815eccdd"
BINDINGS = ("manifest_fingerprint", "comparison_fingerprint", "native_session_id",
            "native_exit_fingerprint", "native_exit_sha256", "inventory_sha256",
            "retirement_proof_sha256", "source_commit")


def _sha(value):
    require(type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value), "Invalid recovery SHA256")
    return value


def _path(value):
    path = Path(value)
    require(path.is_absolute() and path == path.resolve()
            and not any(p.is_symlink() for p in (path, *path.parents)),
            "Recovery path must be canonical and unlinked")
    return path


def _pinned_external(path, checksum, output):
    path = _path(path)
    require(path.is_file() and not path.is_relative_to(output), "Recovery proof must be outside original evidence")
    require(sha(path) == _sha(checksum), "Pinned recovery proof bytes changed")
    return read(path)


def _instant(value):
    require(type(value) is str, "Missing retirement timestamp")
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("Invalid retirement timestamp") from error
    require(stamp.tzinfo is not None and stamp.utcoffset().total_seconds() == 0,
            "Retirement timestamp must be UTC")
    return stamp


def verify_scientific_body():
    """Prove the copied scientific body and imported original source are fixed."""
    source_path = Path(original_verifier.__file__).resolve()
    require(sha(source_path) == ORIGINAL_VERIFIER_SHA256, "Original independent verifier source changed")
    original = ast.parse(inspect.getsource(original_verifier.audit)).body[0]
    copied = ast.parse(inspect.getsource(_audit_scientific)).body[0]
    require(copied.name == "_audit_scientific" and copied.args.args[-1].arg == "recovery_exit_proof",
            "Recovery scientific signature changed")
    copied.name = original.name
    copied.args.args.pop()
    exit_changes = status_changes = 0
    for before, after in zip(original.body, copied.body):
        if isinstance(before, ast.Assign) and isinstance(before.targets[0], ast.Name):
            if before.targets[0].id == "exit_proof":
                require(isinstance(after, ast.Assign)
                        and [ast.dump(n) for n in after.targets] == [ast.dump(n) for n in before.targets],
                        "Recovery exit-proof assignment changed")
                require(isinstance(after.value, ast.Name) and after.value.id == "recovery_exit_proof",
                        "Recovery exit-proof substitution changed")
                after.value = copy.deepcopy(before.value)
                exit_changes += 1
            if before.targets[0].id == "report":
                require(isinstance(after, ast.Assign) and isinstance(after.value, ast.Call),
                        "Recovery scientific report changed")
                for old_key, new_key in zip(before.value.keywords, after.value.keywords):
                    if old_key.arg == "status":
                        expected = ast.parse('"synthetic_verified" if synthetic else STATUS', mode="eval").body
                        require(ast.dump(new_key.value) == ast.dump(expected), "Recovery native status changed")
                        new_key.value = copy.deepcopy(old_key.value)
                        status_changes += 1
    require(exit_changes == status_changes == 1 and ast.dump(copied) == ast.dump(original),
            "Original scientific audit body changed")
    # Source bytes give a stable pin across Python 3.11/3.12/3.14 AST display
    # differences. Semantic parity above is checked by each actual interpreter.
    canonical = inspect.getsource(original_verifier.audit).encode()
    return dict(scientific_body_preserved=True, original_verifier_sha256=sha(source_path),
                recovery_scientific_body_sha256=hashlib.sha256(canonical).hexdigest())


def verify_recovery_exit(output, manifest, plan, trace, request):
    """Check the new failure-preserving proof; never synthesize original receipts."""
    from location_comparison_watchdog import SCHEMA as WATCHDOG_SCHEMA
    output = _path(output)
    require(output.is_dir() and type(request) is dict and request.get("schema") == REQUEST_SCHEMA,
            "Explicit recovery request required")
    require(request.get("fingerprint") == process_digest({k:v for k,v in request.items() if k != "fingerprint"}),
            "Recovery request fingerprint changed")
    require(request.get("native_output") == str(output) and request.get("source_commit") == SOURCE_COMMIT,
            "Recovery output/source identity differs")
    require(request.get("manifest_fingerprint") == manifest.get("fingerprint")
            and request.get("comparison_fingerprint") == plan.get("fingerprint"), "Recovery input fingerprints differ")
    for name in BINDINGS:
        if name not in ("native_session_id", "source_commit"):
            _sha(request.get(name))
    require(type(request.get("native_session_id")) is str
            and re.fullmatch(r"[a-f0-9]{32}", request["native_session_id"]), "Invalid original native session")
    _sha(request.get("manifest_sha256"))
    for name in ("child-exit.json", "child-failure.json"):
        require(not (output/name).exists() and not (output/name).is_symlink(),
                "Recovery cannot replace an existing child completion/failure receipt")
    trace_path = artifact(output, "comparison-trace.json")
    require(read(trace_path) == trace, "Recovery trace differs from saved evidence")
    native_exit_path = artifact(output, "comparison-watchdog-exit.json", request["native_exit_sha256"])
    native_exit = read(native_exit_path)
    require(native_exit.get("schema") == WATCHDOG_SCHEMA
            and native_exit.get("fingerprint") == request["native_exit_fingerprint"]
            == process_digest({k:v for k,v in native_exit.items() if k != "fingerprint"}),
            "Original failed watchdog identity changed")
    require(native_exit.get("status") == "stopped_no_retry"
            and native_exit.get("reason") == "independent watchdog wall deadline reached"
            and native_exit.get("termination") == "stopped" and native_exit.get("lock_release_safe") is True
            and all(type(native_exit.get(k)) is int and native_exit[k] == 1 for k in
                    ("native_application_starts", "native_child_start_attempts", "native_child_process_starts")),
            "Recovery requires the preserved one-start deadline failure and safe cleanup")
    require(native_exit.get("session_id") == request["native_session_id"]
            and native_exit.get("runtime_fingerprint") == manifest["fingerprint"]
            and native_exit.get("comparison_fingerprint") == plan["fingerprint"], "Failed watchdog belongs to another run")
    final = trace["events"][-1]["payload"]["bounds"]
    require(trace["events"][-1]["kind"] == "finish"
            and trace["events"][-1]["payload"].get("status") == "complete_audit_pending"
            and trace["events"][-1]["payload"].get("reason") is None, "Saved work is incomplete")
    require(all(native_exit.get("final_snapshot", {}).get(k) == final[k] for k in
                ("train_requested", "train_completed", "assessment_requested", "assessment_completed", "pending", "stopped")),
            "Failed watchdog accounting differs from completed trace")
    require(final.get("train_requested") == final.get("train_completed") == {"original_usable":8960,"broader":8960}
            and final.get("assessment_requested") == final.get("assessment_completed") == 600
            and final.get("pending") is None and final.get("stopped") is None
            and final.get("phase_index") == 3 and final.get("active_phase") == "shared", "Recovery accounting is incomplete")
    inventory = _pinned_external(request["inventory_path"], request["inventory_sha256"], output)
    require(inventory.get("schema") == INVENTORY_SCHEMA
            and inventory.get("status") == "hash_inventory_only_not_a_scientific_audit"
            and inventory.get("session_id") == request["native_session_id"]
            and inventory.get("source_commit") == SOURCE_COMMIT
            and inventory.get("manifest_sha256") == request["manifest_sha256"], "Sealed inventory identity differs")
    retirement = _pinned_external(request["retirement_proof_path"], request["retirement_proof_sha256"], output)
    require(retirement.get("schema") == RETIREMENT_SCHEMA
            and retirement.get("status") == "owned_native_processes_retired"
            and all(retirement.get(k) == request[k] for k in BINDINGS
                    if k not in ("inventory_sha256", "retirement_proof_sha256")), "Retirement proof identity differs")
    require(retirement.get("container_id") == inventory.get("container_id")
            and type(retirement.get("container_id")) is str
            and re.fullmatch(r"[a-f0-9]{64}", retirement["container_id"]), "Retired container differs")
    state = retirement.get("container_state", {})
    require(state.get("Status") == "exited" and type(state.get("Pid")) is int and state["Pid"] == 0
            and type(state.get("ExitCode")) is int and state["ExitCode"] == 1 and state.get("OOMKilled") is False
            and all(state.get(k) == v for k,v in inventory.get("closed_container_state", {}).items()),
            "Native container is not the preserved stopped container")
    require(_instant(retirement.get("observed_at")) >= _instant(state.get("FinishedAt"))
            >= _instant(state.get("StartedAt")), "Retirement proof predates native exit")
    require(retirement.get("owner") == native_exit.get("owner")
            and retirement.get("native_owner") == native_exit.get("native_owner")
            and all(type(native_exit.get(k)) is dict and native_exit[k].get("cookie") == request["native_session_id"]
                    for k in ("owner", "native_owner")), "Retired process ownership differs")
    require(retirement.get("owned_processes_alive") == []
            and retirement.get("owned_process_identity_check") == "verified_absent"
            and retirement.get("original_lock_released") is True
            and retirement.get("new_lock_owner_not_touched") is True, "Safe owned-process retirement is not established")
    proof = dict({k:request[k] for k in BINDINGS}, manifest_sha256=request["manifest_sha256"], trace_sha256=sha(trace_path),
                 recovery_request_fingerprint=request["fingerprint"], original_audit_passed=False,
                 clean_shutdown=False, model_promoted=False)
    return proof, inventory


def verify_inventory(output, inventory):
    """Read all sealed bytes; reject symlinks/special files or changed membership."""
    output = _path(output)
    files = inventory.get("files")
    require(type(files) is dict and bool(files) and type(inventory.get("file_count")) is int
            and inventory["file_count"] == len(files), "Invalid sealed inventory count")
    actual = {}
    for directory, names, filenames in os.walk(output, followlinks=False):
        for name in names:
            require(not (Path(directory)/name).is_symlink(), "Linked evidence directory")
        for name in filenames:
            path = Path(directory)/name
            value = path.lstat()
            require(stat.S_ISREG(value.st_mode), "Linked or special evidence file")
            relative = str(path.relative_to(output))
            require(relative in files, "Unlisted evidence file")
            expected = files[relative]
            require(type(expected) is dict and set(expected) == {"bytes", "sha256"}
                    and type(expected["bytes"]) is int and expected["bytes"] >= 0
                    and value.st_size == expected["bytes"] and sha(path) == _sha(expected["sha256"]),
                    "Sealed evidence bytes changed: " + relative)
            actual[relative] = dict(bytes=value.st_size, sha256=expected["sha256"])
    require(actual == files and sum(v["bytes"] for v in actual.values()) == inventory.get("total_bytes"),
            "Missing evidence or sealed byte total differs")
    return dict(file_count=len(actual), total_bytes=sum(v["bytes"] for v in actual.values()),
                files_fingerprint=signed_hash(actual))


def verify_journal(output, trace):
    output = _path(output)
    events = trace.get("events")
    require(type(events) is list and bool(events), "Missing recovery trace events")
    directory = output/"events"
    require(directory.is_dir() and not directory.is_symlink(), "Missing persisted journal")
    expected = {f"event-{index:06d}.json" for index in range(len(events))}
    require({p.name for p in directory.iterdir()} == expected, "Persisted journal membership differs")
    hashes = {}
    for index, event in enumerate(events):
        name = f"events/event-{index:06d}.json"
        path = artifact(output, name)
        require(read(path) == event, "Persisted journal differs from final trace")
        hashes[name] = sha(path)
    return dict(event_count=len(events), file_count=len(hashes), trace_head=events[-1]["fingerprint"],
                journal_sha256=signed_hash(hashes))


def audit_recovery(output, manifest, plan, trace, recovery_request, *, synthetic=False, checks=None):
    """Audit saved data only, retaining the original failed shutdown and guards."""
    require(type(synthetic) is bool and (checks is None or synthetic is True),
            "Injected checks cannot certify native recovery evidence")
    if not synthetic:
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Independent recovery audit must be CPU-only")
        require(str(_path(output)) == manifest.get("output"), "Native output differs from original manifest")
    parity = verify_scientific_body()
    proof, inventory = verify_recovery_exit(output, manifest, plan, trace, recovery_request)
    before = verify_inventory(output, inventory)
    journal = verify_journal(output, trace)
    report = _audit_scientific(output, manifest, plan, trace, proof, synthetic=synthetic, checks=checks)
    require(report["paid_policy_samples"] == 18520 and report["terminal_batches"] == 539
            and report["update_artifacts"] == 128 and report["checkpoint_count"] == 144,
            "Recovery scientific coverage differs")
    after_proof, after_inventory = verify_recovery_exit(output, manifest, plan, trace, recovery_request)
    after = verify_inventory(output, after_inventory)
    require(after_proof == proof and after_inventory == inventory and after == before
            and verify_journal(output, trace) == journal, "Original evidence changed during recovery audit")
    proof.update(parity, inventory_before_after_verified=True, inventory=before, journal=journal)
    report.update({k: recovery_request[k] for k in BINDINGS},
                  manifest_sha256=recovery_request["manifest_sha256"],
                  recovery_request_fingerprint=recovery_request["fingerprint"],
                  original_audit_passed=False, clean_shutdown=False, model_promoted=False,
                  recovery_proof=proof, exit_proof=proof)
    report["fingerprint"] = signed_hash({k:v for k,v in report.items() if k != "fingerprint"})
    return report


def _audit_scientific(output, manifest, plan, trace, recovery_exit_proof, *, synthetic=False, checks=None):
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
    exit_proof = recovery_exit_proof
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
    report = dict(schema=SCHEMA, status="synthetic_verified" if synthetic else STATUS,
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
