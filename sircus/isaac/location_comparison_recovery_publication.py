"""Explicit publication gate for separately verified saved data after failed shutdown.

This never grants the original clean-exit gate, runs an audit, edits evidence,
or promotes a model. Three external byte pins bind the successful CPU recovery
report/process and the original failed native watchdog receipt. Outcomes are
opened only after the new process proof and preserved failure are checked.
"""

import argparse
import json
from pathlib import Path
import re

from location_comparison_publication import (
    _fingerprint, _hash, _identity, _number, _present_verified,
    _read, digest, require,
)

STATUS = "data_verified_shutdown_incomplete"
REPORT_SCHEMA = "sircus-location-comparison-recovery-audit-v1"
EXIT_SCHEMA = "sircus-location-comparison-recovery-watchdog-v1"
FAILURE_REASON = "independent watchdog wall deadline reached"
PIN_FIELDS = ("native_exit_fingerprint", "native_exit_sha256", "inventory_sha256",
              "retirement_proof_sha256")
OWNER_FIELDS = ("manifest_sha256", "native_session_id", "runtime_fingerprint",
    "comparison_fingerprint", "recovery_request_fingerprint", *PIN_FIELDS, "source_commit",
    "session_id", "owner", "child_owner", "command", "limits", "start_monotonic",
    "deadline_monotonic", "storage_root", "cuda_visible_devices", "visible_gpu_devices")


def _session(value):
    require(type(value) is str and re.fullmatch(r"[a-f0-9]{32}", value), "Invalid recovery session")
    return value


def _recovery_gate(output, recovery_output, original_audit_output, identity, recovery_session,
                   report_sha, receipt_sha, failed_sha):
    receipt = _read(recovery_output, "recovery-audit-exit.json", receipt_sha)
    _fingerprint(receipt, compact=True)
    _identity(receipt, dict(runtime_fingerprint=identity["runtime_fingerprint"],
        comparison_fingerprint=identity["comparison_fingerprint"],
        native_session_id=identity["session_id"], session_id=recovery_session))
    require(receipt.get("schema") == EXIT_SCHEMA
        and receipt.get("status") == "recovery_exit_zero_report_verified"
        and type(receipt.get("exit_code")) is int and receipt["exit_code"] == 0
        and receipt.get("reason") is None and receipt.get("error_type") is None
        and receipt.get("termination") is None and receipt.get("cleanup_safe") is True
        and receipt.get("independent_report_verified") is True
        and receipt.get("report_status") == STATUS
        and receipt.get("report_sha256") == report_sha
        and receipt.get("cuda_visible_devices") == ""
        and receipt.get("model_promoted") is False,
        "Recovery audit process did not complete safely")
    require(all(type(receipt.get(k)) is int and receipt[k] == v for k, v in {
        "audit_process_starts": 1, "audit_child_start_attempts": 1,
        "visible_gpu_devices": 0, "native_application_starts": 0,
        "optimizer_steps": 0, "new_shots": 0}.items()), "Recovery audit execution scope changed")
    require(receipt.get("measurement_scope") == "wait4_exact_audit_child_including_reaped_descendants",
            "Recovery audit resource scope changed")
    limits = receipt.get("limits", {})
    require(limits == dict(wall_s=7200, min_free_bytes=6*2**30, cpu_cores=4,
        memory_bytes=8*2**30, cpu_seconds=28800), "Recovery audit limits changed")
    require(_number(receipt.get("wall_s")) < limits["wall_s"], "Recovery audit exceeded wall limit")
    for key in ("cpu_user_s", "cpu_system_s", "peak_rss_bytes"):
        _number(receipt.get(key))
    require(receipt["cpu_user_s"] + receipt["cpu_system_s"] <= limits["cpu_seconds"]
        and receipt["peak_rss_bytes"] <= limits["memory_bytes"], "Recovery audit exceeded resource limits")
    owner, child = receipt.get("owner"), receipt.get("child_owner")
    require(type(owner) is dict and type(child) is dict
        and owner.get("cookie") == child.get("cookie") == recovery_session
        and child.get("ppid") == owner.get("pid") and child.get("sid") == child.get("pid")
        and all(type(p.get("pid")) is int and p["pid"] > 1
                and type(p.get("start_token")) is str and p["start_token"] for p in (owner, child)),
        "Recovery process ancestry incomplete")
    owner_receipt = _read(recovery_output, "recovery-audit-owner.json", _hash(receipt.get("owner_receipt_sha256")))
    _fingerprint(owner_receipt, compact=True)
    require(owner_receipt.get("schema") == EXIT_SCHEMA
        and all(k in owner_receipt and owner_receipt[k] == receipt.get(k) for k in OWNER_FIELDS),
        "Recovery owner/exit binding differs")
    controls = receipt.get("cgroup_limits", {})
    require(type(controls.get("cgroup_memory_limit_bytes")) is int
        and 0 < controls["cgroup_memory_limit_bytes"] <= limits["memory_bytes"]
        and 0 < _number(controls.get("cgroup_cpu_cores")) <= limits["cpu_cores"]
        and _number(receipt.get("measured_cgroup_cpu_s")) <= limits["cpu_seconds"]
        and type(receipt.get("sampled_peak_cgroup_memory_bytes")) is int
        and 0 <= receipt["sampled_peak_cgroup_memory_bytes"] <= limits["memory_bytes"]
        and type(receipt.get("peak_rss_bytes")) is int
        and receipt.get("deadline_monotonic") == _number(receipt.get("start_monotonic")) + limits["wall_s"],
        "Recovery resource controls or measurements differ")

    native = _read(output, "comparison-watchdog-exit.json", failed_sha)
    _fingerprint(native, compact=True)
    _identity(native, identity)
    require(native.get("schema") == "sircus-location-comparison-watchdog-v1"
        and native.get("status") == "stopped_no_retry" and native.get("reason") == FAILURE_REASON
        and native.get("lock_release_safe") is True and native.get("termination") == "stopped"
        and type(native.get("native_application_starts")) is int and native["native_application_starts"] == 1,
        "Original failed shutdown was changed or hidden")
    require(not (Path(output) / "child-exit.json").exists()
        and not any((Path(original_audit_output) / n).exists()
                    for n in ("audit.json", "comparison-audit-exit.json")),
        "Original clean-exit or canonical-audit receipt was added")
    require(receipt.get("native_exit_sha256") == failed_sha
        and receipt.get("native_exit_fingerprint") == native["fingerprint"], "Recovery failed-exit binding differs")

    # The externally pinned scientific report is read only after its successful
    # CPU process and the original failed native exit have both been checked.
    report = _read(recovery_output, "recovery-audit.json", report_sha)
    _fingerprint(report)
    _identity(report, dict(manifest_fingerprint=identity["runtime_fingerprint"],
        comparison_fingerprint=identity["comparison_fingerprint"], native_session_id=identity["session_id"]))
    require(report.get("schema") == REPORT_SCHEMA and report.get("status") == STATUS
        and report.get("synthetic") is False and report.get("model_promoted") is False
        and report.get("original_audit_passed") is False and report.get("clean_shutdown") is False
        and report.get("fingerprint") == receipt.get("report_fingerprint")
        and _hash(report.get("manifest_sha256")) == _hash(receipt.get("manifest_sha256"))
        and report.get("recovery_request_fingerprint") == _hash(receipt.get("recovery_request_fingerprint"))
        and type(report.get("source_commit")) is str
        and re.fullmatch(r"[a-f0-9]{40}", report["source_commit"])
        and report["source_commit"] == receipt.get("source_commit"),
        "Invalid recovery data-audit report")
    require(all(report.get(k) is True for k in ("checkpoint_bytes_independently_verified",
        "model_likelihoods_independently_verified", "native_physics_verified"))
        and report.get("paid_policy_samples") == 18520 and report.get("terminal_batches") == 539
        and report.get("update_artifacts") == 128 and report.get("checkpoint_count") == 144
        and report.get("startup", {}).get("status") == "passed", "Incomplete recovery scientific coverage")
    proof = report["recovery_proof"]
    require(type(proof) is dict and proof.get("inventory_before_after_verified") is True,
            "Recovery evidence inventory was not verified unchanged")
    require(proof.get("scientific_body_preserved") is True
        and proof.get("original_verifier_sha256") == "0e196852b48070d423d4db61b894b93b36edc655bc5f92c77adbf6b3815eccdd"
        and proof.get("recovery_scientific_body_sha256") == "6bef2f77aa1800047d4c1d4a0416a2cca1c3d2cb475cb7895fae470a5f1f8947"
        and all(proof.get(k) == report.get(k) for k in ("manifest_sha256", "manifest_fingerprint",
            "comparison_fingerprint", "native_session_id", "source_commit", "recovery_request_fingerprint",
            "original_audit_passed", "clean_shutdown", "model_promoted")),
        "Recovery scientific parity or proof bindings differ")
    for field in PIN_FIELDS:
        require(_hash(report.get(field)) == _hash(receipt.get(field)) == _hash(proof.get(field)),
                "Recovery proof identity differs: " + field)
    require(report["native_exit_sha256"] == failed_sha and report["native_exit_fingerprint"] == native["fingerprint"],
            "Recovery proof refers to another failed attempt")
    require(report.get("exit_proof") == proof and "child_exit_sha256" not in proof,
            "Recovery cannot invent original clean-exit proof")
    trace_sha = _hash(proof.get("trace_sha256"))
    trace = _read(output, "comparison-trace.json", trace_sha)
    require(trace.get("schema") == "sircus-location-comparison-trace-v1" and trace.get("mode") == "native"
        and trace.get("plan_fingerprint") == identity["comparison_fingerprint"]
        and digest(trace) == report.get("trace_canonical_sha256"), "Changed recovery-audited trace")
    previous = "0" * 64
    for index, event in enumerate(trace["events"]):
        _fingerprint(event)
        require(event.get("seq") == index and event.get("previous") == previous, "Broken recovery trace chain")
        previous = event["fingerprint"]
    require(trace["events"][-1]["kind"] == "finish"
        and trace["events"][-1]["payload"]["status"] == "complete_audit_pending"
        and report.get("trace_head") == previous, "Recovery trace is incomplete")
    final = trace["events"][-1]["payload"]["bounds"]
    require(all(native.get("final_snapshot", {}).get(k) == final[k] for k in
        ("train_requested", "train_completed", "assessment_requested", "assessment_completed", "pending", "stopped")),
        "Original failed receipt and final recorded accounting differ")
    accounting = report["accounting"]
    _fingerprint(accounting)
    require(accounting.get("schema") == "sircus-location-comparison-accounting-audit-v1"
        and accounting.get("status") == "native_accounting_verified_physics_pending"
        and accounting.get("synthetic") is False and accounting.get("plan_fingerprint") == identity["comparison_fingerprint"]
        and accounting.get("trace_head") == previous and accounting.get("event_count") == len(trace["events"])
        and accounting.get("requested") == accounting.get("completed") == 18520
        and accounting.get("unresolved") == 0
        and accounting.get("updates") == {"original_usable": 64, "broader": 64}, "Incomplete recovery accounting")
    journal = proof.get("journal", {})
    require(journal.get("event_count") == len(trace["events"])
        and journal.get("trace_head") == previous
        and type(journal.get("file_count")) is int and journal["file_count"] > 0,
        "Persisted journal proof differs")
    _hash(journal.get("journal_sha256"))
    return report, trace, native, receipt


def build_recovery_publication(output, recovery_output, *, original_audit_output,
        expected_manifest_fingerprint, expected_comparison_fingerprint, expected_session_id,
        expected_recovery_session_id, expected_recovery_audit_sha256,
        expected_recovery_exit_sha256, expected_native_exit_sha256):
    """Return separately verified results with an unavoidable failed-shutdown status."""
    try:
        identity = dict(runtime_fingerprint=_hash(expected_manifest_fingerprint),
            comparison_fingerprint=_hash(expected_comparison_fingerprint), session_id=_session(expected_session_id))
        recovery_session = _session(expected_recovery_session_id)
        require(recovery_session != identity["session_id"], "Recovery needs its own process session")
        report_sha, receipt_sha, native_sha = map(_hash, (expected_recovery_audit_sha256,
            expected_recovery_exit_sha256, expected_native_exit_sha256))
        report, trace, native, receipt = _recovery_gate(output, recovery_output, original_audit_output,
            identity, recovery_session, report_sha, receipt_sha, native_sha)
        snapshot = native["final_snapshot"]
        started, wall, last = (_number(native["start_monotonic"]), _number(native["wall_s"]),
            _number(snapshot["observed_monotonic"]))
        require(started <= last <= started + wall, "Failed-shutdown clocks differ")
        failure = dict(status="failed", original_audit_status="not_passed", clean_shutdown=False,
            reason=native["reason"], native_watchdog_wall_s=wall,
            last_progress_shared_phase_s=_number(snapshot["phase_elapsed_s"]["shared"]),
            after_last_progress_wall_s=started + wall - last, shared_phase_limit_s=3600,
            scope="Native watchdog total includes all phases and failed shutdown. Last-progress shared time is stale. After-last-progress elapsed includes shutdown wait; it is not isolated compute or pure shutdown time.")
        recovery = dict(audit=dict(status=STATUS, original_audit_status="not_passed",
            native_shutdown_status="failed", native_clean_shutdown=False,
            recovery_session_id=recovery_session, audit_exit_sha256=receipt_sha,
            native_exit_sha256=native_sha, recovery_request_fingerprint=report["recovery_request_fingerprint"],
            inventory_sha256=report["inventory_sha256"], retirement_proof_sha256=report["retirement_proof_sha256"]),
            costs=dict(shutdown_failure=failure,
                recovery_audit=dict(wall_s=receipt["wall_s"], cpu_user_s=receipt["cpu_user_s"],
                    cpu_system_s=receipt["cpu_system_s"], peak_rss_bytes=receipt["peak_rss_bytes"],
                    scope=receipt["measurement_scope"]),
                scope="Recorded call and watchdog wall scopes overlap. Geometry is nested in episode wall. Original simulator shutdown failed; its wait is shown separately. Recovery audit CPU/RSS are measured for the exact audit child including reaped descendants."))
        # Only the common presentation extraction is reused. No clean-exit
        # receipt or native_verified report is synthesized or accepted here.
        return _present_verified(output, identity, dict(report=report,
            trace_sha256=report["recovery_proof"]["trace_sha256"]), trace, native, receipt,
            report_sha, recovery=recovery)
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError("Malformed recovery publication evidence") from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "recovery-output", "original-audit-output"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("manifest-fingerprint", "comparison-fingerprint", "session-id", "recovery-session-id",
                 "recovery-audit-sha256", "recovery-exit-sha256", "native-exit-sha256"):
        parser.add_argument("--expected-" + name, required=True)
    print(json.dumps(build_recovery_publication(**vars(parser.parse_args(argv))), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
