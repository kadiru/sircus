"""Read-only presentation of a completed independent location-comparison audit.

This is a certificate consumer, not another physics/model auditor. The caller
must obtain expected_audit_sha256 from the independently verified, completed
production audit, not from an arbitrary audit.json discovered by this module.
The original CPU auditor binds all native/checkpoint/physics bytes. We verify
its pinned certificate, both process receipts, the exact complete trace, final
checkpoint bytes and the ten scheduled recording packets/JPEGs before exposing
results. No model, optimizer, simulator or subprocess is imported or started.

Without a completed audit-process receipt AND that external certificate pin,
only allowlisted progress counters are read. TRAIN/monitor/TEST outcomes are
not opened. A malformed or mismatched completed certificate raises ValueError;
it never falls back to a partial result or silently labels an audit as passed.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

SCHEMA = "sircus-location-comparison-publication-v1"
MODELS = ("parent", "original_usable", "broader")
LABELS = {"parent": "Original seed 22", "original_usable": "Continued on original locations",
          "broader": "Continued on broader locations"}
PHASES = ("train", "monitor", "test_broad", "test_local")
TIME_FIELDS = ("episode_wall_s", "episode_process_cpu_s", "geometry_wall_s",
               "geometry_process_cpu_s", "proposal_wall_s", "update_wall_s",
               "checkpoint_wall_s", "restore_wall_s")
MAX_JSON_BYTES = 256 * 1024**2
MEMORY_SCOPE = ("Lifetime process RSS peak; sampled whole-device GPU occupancy. "
                "Neither is a policy-only or phase-local memory peak.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value, *, compact=False):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
        **({"separators": (",", ":")} if compact else {})).encode()).hexdigest()


def _hash(value):
    require(type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value), "Invalid SHA256 identity")
    return value


def _fingerprint(value, *, compact=False):
    require(type(value) is dict and _hash(value.get("fingerprint")) ==
        digest({k: v for k, v in value.items() if k != "fingerprint"}, compact=compact),
        "Changed certificate fingerprint")


def _file(root, name):
    require(type(name) is str and name and not Path(name).is_absolute()
        and ".." not in Path(name).parts, "Escaped publication evidence path")
    root = Path(root).absolute()
    path = root / name
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents))
        and path.resolve().is_relative_to(root.resolve()), "Missing or linked publication evidence: " + name)
    return path


def _file_hash(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def _read(root, name, checksum=None):
    path = _file(root, name)
    if checksum is not None:
        require(_file_hash(path) == _hash(checksum), "Changed evidence bytes: " + name)
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate evidence JSON key")
            result[key] = value
        return result
    opener = gzip.open if name.endswith(".gz") else open
    with opener(path, "rb") as stream:
        raw = stream.read(MAX_JSON_BYTES + 1)
    require(len(raw) <= MAX_JSON_BYTES, "Publication evidence exceeds size limit")
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite evidence JSON")))


def _integer(value, maximum=18520):
    require(type(value) is int and 0 <= value <= maximum, "Invalid request count")
    return value


def _number(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            "Invalid recorded resource measurement")
    return value


def _summary(rows):
    counts = Counter(row["outcome"] for row in rows)
    require(set(counts) <= {"capture", "miss", "rejected"}, "Unknown audited outcome")
    require(all(type(row["clean"]) is bool and (not row["clean"] or row["outcome"] == "capture")
                and type(row["reward"]) in (int, float) and math.isfinite(row["reward"])
                for row in rows), "Invalid audited clean/reward value")
    return dict(requests=len(rows), captures=counts["capture"], clean=sum(row["clean"] for row in rows),
        rejected=counts["rejected"], misses=counts["miss"],
        rejection_reasons=dict(sorted(Counter(row["rejection_reason"] for row in rows
                                            if row["outcome"] == "rejected").items())))


def _metrics(summary):
    result = {key: _integer(summary[key]) for key in ("requests", "captures", "clean", "rejected", "misses")}
    n = result["requests"]
    require(n > 0 and result["captures"] + result["misses"] + result["rejected"] == n
        and result["clean"] <= result["captures"], "Invalid audited outcome totals")
    reasons = summary["rejection_reasons"]
    require(type(reasons) is dict and all(type(k) is str and k and len(k) <= 512
        and _integer(v) > 0 for k, v in reasons.items())
        and sum(reasons.values()) == result["rejected"], "Invalid rejection-reason totals")
    return dict(result, rejection_reasons=dict(reasons), capture_rate=result["captures"] / n,
        clean_rate=result["clean"] / n, rejection_rate=result["rejected"] / n)


def _identity(value, expected):
    require(all(value.get(key) == wanted for key, wanted in expected.items()), "Cross-run publication identity")


def recovery_publication_verified(report):
    """Presentation/media status check; never replaces the pinned recovery gate."""
    try:
        audit = report["audit"]
        require(report.get("schema") == SCHEMA and report.get("synthetic") is False
            and report.get("status") == audit.get("status") == "data_verified_shutdown_incomplete"
            and report.get("model_promoted") is False and audit.get("complete") is True
            and audit.get("synthetic") is False and audit.get("original_audit_status") == "not_passed"
            and audit.get("native_shutdown_status") == "failed" and audit.get("native_clean_shutdown") is False,
            "Missing separate recovery/failure status")
        for key in ("manifest_fingerprint", "comparison_fingerprint", "audit_sha256", "audit_exit_sha256",
                    "native_exit_sha256", "native_exit_fingerprint", "audit_exit_fingerprint", "trace_sha256",
                    "recovery_request_fingerprint", "inventory_sha256", "retirement_proof_sha256"):
            _hash(audit.get(key))
        require(all(type(audit.get(k)) is str and re.fullmatch(r"[a-f0-9]{32}", audit[k])
            for k in ("session_id", "recovery_session_id"))
            and audit["session_id"] != audit["recovery_session_id"], "Recovery session is not distinct")
        failure = report["costs"]["shutdown_failure"]
        require(failure.get("status") == "failed" and failure.get("original_audit_status") == "not_passed"
            and failure.get("clean_shutdown") is False and type(failure.get("scope")) is str and failure["scope"]
            and failure.get("reason") == "independent watchdog wall deadline reached"
            and failure.get("shared_phase_limit_s") == 3600,
            "Shutdown failure scope is missing")
        for key in ("native_watchdog_wall_s", "last_progress_shared_phase_s", "after_last_progress_wall_s"):
            _number(failure.get(key))
        require(failure["after_last_progress_wall_s"] <= failure["native_watchdog_wall_s"]
            and failure["native_watchdog_wall_s"] == report["costs"].get("native_watchdog_wall_s"),
            "Shutdown elapsed scope differs")
        return True
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def _pending(output, audit_output, identity):
    result = dict(schema=SCHEMA, generated_at=datetime.now(timezone.utc).isoformat(),
        status="unknown", synthetic=False, audit=dict(status="pending", complete=False, synthetic=False),
        progress={}, model_promoted=False)
    if (Path(output) / "comparison-progress.json").exists():
        progress = _read(output, "comparison-progress.json")
        _fingerprint(progress, compact=True)
        _identity(progress, identity)
        require(progress.get("schema") == "sircus-location-comparison-process-progress-v1", "Wrong progress schema")
        snapshot = progress["snapshot"]
        for model in MODELS[1:]:
            requested = _integer(snapshot["train_requested"][model], 8960)
            completed = _integer(snapshot["train_completed"][model], requested)
            result["progress"][model] = dict(requested=requested, completed=completed)
        requested = _integer(snapshot["assessment_requested"], 600)
        result["progress"]["assessment"] = dict(requested=requested,
            completed=_integer(snapshot["assessment_completed"], requested))
        phase = snapshot.get("active_phase")
        require(phase in ("shared", *MODELS[1:]), "Unknown progress phase")
        result["phase"] = phase
        if progress.get("running") is True:
            result["status"] = "running"
    for root, filename in ((output, "comparison-watchdog-exit.json"), (audit_output, "comparison-audit-exit.json")):
        if (Path(root) / filename).exists():
            receipt = _read(root, filename)
            _fingerprint(receipt, compact=True)
            _identity(receipt, identity)
            if receipt.get("status") in ("stopped_no_retry", "audit_stopped_no_retry"):
                result["status"] = "stopped_no_retry"
                result["audit"]["status"] = "not_passed"
            elif result["status"] != "stopped_no_retry":
                result["status"] = "awaiting_audit"
    return result


def _gate(output, audit_output, identity, expected_audit_sha256):
    """Do not open the outcome-bearing audit envelope until its outer gate passes."""
    audit_exit = _read(audit_output, "comparison-audit-exit.json")
    _fingerprint(audit_exit, compact=True)
    _identity(audit_exit, identity)
    require(audit_exit.get("schema") == "sircus-location-comparison-audit-watchdog-v1"
        and audit_exit.get("status") == "audit_exit_zero_report_verified"
        and audit_exit.get("exit_code") == 0 and audit_exit.get("reason") is None
        and audit_exit.get("cleanup_safe") is True and audit_exit.get("independent_report_verified") is True
        and audit_exit.get("cuda_visible_devices") == "" and audit_exit.get("model_promoted") is False
        and audit_exit.get("audit_process_starts") == audit_exit.get("audit_child_start_attempts") == 1
        and audit_exit.get("native_application_starts") == 0,
        "Independent audit process did not complete safely")
    native = _read(output, "comparison-watchdog-exit.json")
    _fingerprint(native, compact=True)
    _identity(native, identity)
    require(native.get("schema") == "sircus-location-comparison-watchdog-v1"
        and native.get("status") == "native_exit_zero_independent_audit_required"
        and native.get("exit_code") == 0 and native.get("reason") is None
        and native.get("lock_release_safe") is True and native.get("native_application_starts") == 1
        and audit_exit.get("native_exit_fingerprint") == native["fingerprint"], "Native completion gate failed")
    child = _read(output, "child-exit.json")
    _identity(child, identity)
    require(child.get("status") == "native_complete_pending_audit" and child.get("synthetic") is False
        and child.get("app_closed") is True and child.get("error") is None, "Native child did not close successfully")
    envelope = _read(audit_output, "audit.json", expected_audit_sha256)
    _fingerprint(envelope)
    _identity(envelope, {"session_id": identity["session_id"],
        "manifest_fingerprint": identity["runtime_fingerprint"], "comparison_fingerprint": identity["comparison_fingerprint"]})
    require(envelope.get("schema") == "sircus-location-comparison-audit-envelope-v1"
        and envelope.get("status") == "native_verified" and envelope.get("synthetic") is False
        and envelope.get("native_exit_fingerprint") == native["fingerprint"], "Invalid completed audit envelope")
    report = envelope["report"]
    _fingerprint(report)
    _identity(report, {"manifest_fingerprint": identity["runtime_fingerprint"],
                       "comparison_fingerprint": identity["comparison_fingerprint"]})
    require(report.get("schema") == "sircus-location-comparison-independent-audit-v1"
        and report.get("status") == "native_verified" and report.get("synthetic") is False
        and all(report.get(k) is True for k in ("checkpoint_bytes_independently_verified",
            "model_likelihoods_independently_verified", "native_physics_verified"))
        and report.get("paid_policy_samples") == 18520 and report.get("terminal_batches") == 539
        and report.get("update_artifacts") == 128 and report.get("checkpoint_count") == 144
        and report.get("startup", {}).get("status") == "passed",
        "Complete independent native audit required")
    require(envelope.get("trace_sha256") == child.get("trace_sha256"), "Trace/child certificate mismatch")
    trace = _read(output, "comparison-trace.json", envelope["trace_sha256"])
    require(trace.get("schema") == "sircus-location-comparison-trace-v1" and trace.get("mode") == "native"
        and trace.get("plan_fingerprint") == identity["comparison_fingerprint"]
        and digest(trace) == report.get("trace_canonical_sha256"), "Changed audited trace")
    previous = "0" * 64
    for index, event in enumerate(trace["events"]):
        _fingerprint(event)
        require(event.get("seq") == index and event.get("previous") == previous, "Broken audited trace chain")
        previous = event["fingerprint"]
    require(trace["events"][-1]["kind"] == "finish"
        and trace["events"][-1]["payload"]["status"] == "complete_audit_pending"
        and report.get("trace_head") == previous, "Trace is incomplete")
    require(report.get("exit_proof") == dict(native_exit_fingerprint=native["fingerprint"],
        trace_sha256=envelope["trace_sha256"], child_exit_sha256=_file_hash(_file(output, "child-exit.json"))),
        "Independent exit proof differs")
    final_bounds = trace["events"][-1]["payload"]["bounds"]
    require(all(native.get("final_snapshot", {}).get(k) == final_bounds[k] for k in
        ("train_requested", "train_completed", "assessment_requested", "assessment_completed", "pending", "stopped")),
        "Native final accounting differs")
    accounting = report["accounting"]
    _fingerprint(accounting)
    require(accounting.get("schema") == "sircus-location-comparison-accounting-audit-v1"
        and accounting.get("status") == "native_accounting_verified_physics_pending"
        and accounting.get("synthetic") is False and accounting.get("plan_fingerprint") == identity["comparison_fingerprint"]
        and accounting.get("trace_head") == previous and accounting.get("event_count") == len(trace["events"])
        and accounting.get("requested") == accounting.get("completed") == 18520 and accounting.get("unresolved") == 0
        and accounting.get("updates") == {"original_usable": 64, "broader": 64}, "Incomplete audited accounting")
    return envelope, trace, native, audit_exit


def _recording(output, reservation, evidence):
    packet = _read(output, evidence["file"], evidence["sha256"])
    require(packet.get("schema") == "sircus-location-comparison-native-trial-v1"
        and packet.get("synthetic") is False and packet.get("filming") is True, "Recording is not audited native filming")
    frames = packet.get("frames")
    require(type(frames) is list and frames, "Scheduled recording has no frames")
    frame_ids = []
    for frame in frames:
        index = _integer(frame["frame"], 10000000)
        require(_file_hash(_file(output, f"native/frames/{index:05d}.jpg")) == _hash(frame["jpeg_sha256"]),
                "Scheduled JPEG bytes changed")
        frame_ids.append(index)
    require(frame_ids == list(range(frame_ids[0], frame_ids[0] + len(frames))), "Recording frame order changed")
    model, phase, before = reservation["model"], reservation["phase"], reservation["milestone"]
    count = len(reservation["cells"])
    if phase == "train":
        label = f"{LABELS[model]} · TRAIN requests {before + 1}–{before + count} · {count} active slots"
    else:
        label = f"{LABELS[model]} · {'Broad' if phase == 'test_broad' else 'Nearby'} TEST subset · {count} active slots"
    return dict(id=reservation["batch_id"], model=model, phase=phase,
        train_requests_before=before if phase == "train" else None, active_slots=count, parked_slots=35-count,
        label=label, packet_file=evidence["file"], packet_sha256=evidence["sha256"],
        frame_sequence_sha256=digest(frames), frames=len(frames), fps=30,
        chapters=[dict(start_s=0, label=label, phase=phase, train_requests_before=before if phase == "train" else None)],
        media=None, media_status="verified_frames_not_encoded")


def _build_publication(output, audit_output, *, expected_manifest_fingerprint,
                      expected_comparison_fingerprint, expected_session_id, expected_audit_sha256=None):
    """Return status only, or fully audit-gated allowlisted UI data; write nothing.

    The required identities refer to one prepared run. The optional audit SHA
    is an external trust pin supplied ONLY after the unchanged independent
    auditor and production audit watchdog pass. Its absence withholds outcomes.
    Encoded media is deliberately absent; a separate verified encoding step
    may attach same-origin URLs bound to the packet/frame-sequence identities.
    """
    identity = dict(runtime_fingerprint=_hash(expected_manifest_fingerprint),
        comparison_fingerprint=_hash(expected_comparison_fingerprint), session_id=expected_session_id)
    require(type(expected_session_id) is str and re.fullmatch(r"[0-9a-f]{32}", expected_session_id), "Invalid run session")
    pending = _pending(output, audit_output, identity)
    audit_path = Path(audit_output) / "comparison-audit-exit.json"
    if expected_audit_sha256 is None or not audit_path.exists() or pending["status"] == "stopped_no_retry":
        return pending
    receipt = _read(audit_output, "comparison-audit-exit.json")
    if receipt.get("status") != "audit_exit_zero_report_verified":
        return pending
    envelope, trace, native, audit_exit = _gate(output, audit_output, identity, _hash(expected_audit_sha256))
    return _present_verified(output, identity, envelope, trace, native, audit_exit, expected_audit_sha256)


def _present_verified(output, identity, envelope, trace, native, audit_exit, expected_audit_sha256, *, recovery=None):
    """Render already verified scientific evidence; process semantics stay explicit."""
    accounting = envelope["report"]["accounting"]
    all_rows, monitors, train_curves = defaultdict(list), {}, {model: [] for model in MODELS[1:]}
    memory, videos, checkpoints, reserved = [], [], {}, {}
    for event in trace["events"]:
        kind, payload = event["kind"], event["payload"]
        if kind == "reserve":
            require(payload["batch_id"] not in reserved, "Repeated reserved batch")
            reserved[payload["batch_id"]] = payload
        elif kind == "frozen":
            require(not checkpoints, "Repeated final model selection")
            for model in MODELS:
                cp = payload["checkpoints"][model]
                require(_file_hash(_file(output, cp["file"])) == _hash(cp["sha256"])
                    == envelope["report"]["checkpoint_sha256"].get(cp["file"]), "Final checkpoint bytes changed")
                state = cp["state"]
                checkpoints[model] = dict(file=cp["file"], sha256=cp["sha256"], weight_hash=_hash(state["weight_hash"]),
                    parent_checkpoint_sha256=_hash(state["parent_checkpoint_sha256"]),
                    attempted_train_requests=_integer(state["attempts"], 30000), ppo_updates=_integer(state["updates"], 1000))
        elif kind == "terminal":
            reservation = reserved.pop(payload["batch_id"])
            model, phase = reservation["model"], reservation["phase"]
            require(model in MODELS and phase in PHASES, "Unknown model or phase")
            rows = payload["rows"]
            require(len(rows) == len(reservation["cells"]) and all(row["request_id"] == cell["request_id"]
                and row["target_id"] == cell["target_id"] for row, cell in zip(rows, reservation["cells"])),
                "Outcome/request association changed")
            all_rows[(model, phase)].extend(rows)
            summary = _metrics(_summary(rows))
            if phase == "train":
                train_curves[model].append(dict(summary, train_requests=len(all_rows[(model, phase)]),
                    mean_reward=sum(row["reward"] for row in rows) / len(rows)))
            elif phase == "monitor":
                key = (model, reservation["milestone"])
                require(key not in monitors, "Repeated monitoring point")
                monitors[key] = dict(summary, train_requests=reservation["milestone"], source_model=model)
            for when in ("before", "after"):
                resources = payload["costs"].get("resources_" + when, {})
                sample = {key: _number(resources[key]) for key in ("lifetime_peak_rss_mib", "sampled_gpu_used_mib") if key in resources}
                if sample:
                    memory.append(dict(batch_id=payload["batch_id"], model=model, phase=phase, when=when, **sample))
            if reservation["film"]:
                videos.append(_recording(output, reservation, payload["evidence"]))
    require(not reserved and len(checkpoints) == 3, "Incomplete final evidence")
    train, final_tests = {}, {}
    for model in MODELS[1:]:
        actual = _summary(all_rows[(model, "train")])
        require(actual == accounting["train"][model] and actual["requests"] == 8960
            and len(train_curves[model]) == 256, "TRAIN curve/accounting mismatch")
        train[model] = _metrics(actual)
    for model in MODELS:
        final_tests[model] = {}
        for phase, count in (("test_broad", 70), ("test_local", 25)):
            actual = _summary(all_rows[(model, phase)])
            require(actual == accounting["final_tests"][model][phase] and actual["requests"] == count,
                    "Final TEST/accounting mismatch")
            final_tests[model][phase] = _metrics(actual)
    require(set(monitors) == {("parent", 0)} | {(model, n) for model in MODELS[1:] for n in (2240, 4480, 6720, 8960)}
        and all(point["requests"] == 35 for point in monitors.values()), "Fixed monitoring schedule changed")
    for assessment in accounting["assessments"]:
        if assessment["phase"] == "monitor":
            point = monitors[(assessment["model"], assessment["milestone"])]
            require(all(point[k] == assessment[k] for k in ("requests", "captures", "clean", "rejected", "misses", "rejection_reasons")),
                    "Monitoring/accounting mismatch")
    expected_films = {(m, "train", n, 35) for m in MODELS[1:] for n in (0, 2240, 4480, 6720)} | {
        ("parent", "test_broad", None, 24), ("parent", "test_local", None, 9)}
    require(len(videos) == 10 and {(v["model"], v["phase"], v["train_requests_before"], v["active_slots"]) for v in videos}
        == expected_films, "Prescheduled recording coverage changed")
    scalar = lambda values: {key: _number(values[key]) for key in TIME_FIELDS if key in values}
    memory_groups = defaultdict(list)
    for sample in memory:
        memory_groups[(sample["model"], sample["phase"])].append(sample)
    memory_summary = [dict(model=m, phase=p, samples=len(samples), **{
        key: max(s[key] for s in samples if key in s) for key in ("lifetime_peak_rss_mib", "sampled_gpu_used_mib")
        if any(key in s for s in samples)}) for (m, p), samples in memory_groups.items()]
    result = dict(schema=SCHEMA, generated_at=datetime.now(timezone.utc).isoformat(), status="completed_audited",
        synthetic=False, model_promoted=False, audit=dict(status="passed", complete=True, synthetic=False,
            session_id=identity["session_id"], manifest_fingerprint=identity["runtime_fingerprint"],
            comparison_fingerprint=identity["comparison_fingerprint"], audit_sha256=expected_audit_sha256,
            native_exit_fingerprint=native["fingerprint"], audit_exit_fingerprint=audit_exit["fingerprint"],
            trace_sha256=envelope["trace_sha256"]),
        protocol=dict(models=LABELS, train_requests_per_copy=8960, total_requests=18520,
            broad_positions=70, nearby_positions=25, monitor_positions=35,
            capture_denominator="All requested positions, including rejections; clean capture is a subset of capture.",
            interpretation="One seed-22 parent continued twice; not three independent seeds. Broad and nearby cohorts stay separate. TRAIN is exploratory; monitoring and TEST use frozen direct inference. No model selection from monitoring or TEST."),
        progress={**{model: dict(requested=8960, completed=8960) for model in MODELS[1:]},
                  "assessment": dict(requested=600, completed=600)},
        train=train, train_curves=train_curves,
        monitor_curves={model: [monitors[("parent", 0)]] + [monitors[(model, n)] for n in (2240, 4480, 6720, 8960)] for model in MODELS[1:]},
        final_tests=final_tests, checkpoints=checkpoints, videos=videos,
        costs=dict(recorded_totals=scalar(accounting["costs"]),
            per_model_phase={m: {p: scalar(v) for p, v in phases.items()} for m, phases in accounting["per_model_phase_costs"].items()},
            observed_phase_wall_s={k: _number(v) for k, v in accounting["observed_phase_wall_s"].items()},
            native_watchdog_wall_s=_number(native["wall_s"]), audit_watchdog_wall_s=_number(audit_exit["wall_s"]),
            memory_samples=memory, memory_by_model_phase=memory_summary, memory_scope=MEMORY_SCOPE,
            geometry_nested_in_episode=True,
            scope="Recorded call and watchdog wall scopes only. Geometry is already inside episode wall; do not add it twice. Watchdog wall also includes startup/cleanup. Audit CPU and audit memory are not measured here."),
        limitations=["No native performance claim before the unchanged independent audit passes.",
            "Single paired continuation; descriptive differences do not establish generalization beyond the held-out cohorts.",
            "Eight TRAIN recordings and two parent-only TEST subsets were scheduled; no final-model TEST recordings exist.",
            "Verified source JPEGs require a separate bound encoding step for playable media.",
            "No continuous-path collision certificate, physical-robot validation, retraining or model promotion is implied."])
    if recovery is not None:
        result["status"] = "data_verified_shutdown_incomplete"
        result["audit"].update(recovery["audit"])
        result["costs"].update(recovery["costs"])
        result["limitations"][0] = "Saved data passed a separate CPU recovery audit; simulator shutdown failed and the original final audit remains not passed."
    return result


def build_publication(output, audit_output, *, expected_manifest_fingerprint,
                      expected_comparison_fingerprint, expected_session_id, expected_audit_sha256=None):
    """Consume externally pinned final evidence; otherwise expose safe status only.

    See the module trust-boundary documentation. This function writes nothing;
    missing evidence does not imply that a run was started or is still alive.
    """
    try:
        return _build_publication(output, audit_output,
            expected_manifest_fingerprint=expected_manifest_fingerprint,
            expected_comparison_fingerprint=expected_comparison_fingerprint,
            expected_session_id=expected_session_id, expected_audit_sha256=expected_audit_sha256)
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("Malformed publication evidence") from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--expected-manifest-fingerprint", required=True)
    parser.add_argument("--expected-comparison-fingerprint", required=True)
    parser.add_argument("--expected-session-id", required=True)
    parser.add_argument("--expected-audit-sha256")
    args = vars(parser.parse_args(argv))
    print(json.dumps(build_publication(**args), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
