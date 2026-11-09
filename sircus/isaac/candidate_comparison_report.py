"""Publish compact UI evidence for one frozen TRAIN-only native comparison."""

import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import re

from candidate_selector import validate_comparison
from continuous_feasibility import canonical_hash

SCHEMA = "sircus-candidate-native-comparison-report-v1"
AUDIT_SCHEMA = "sircus-candidate-native-independent-audit-v1"
RESULT_SCHEMA = "sircus-candidate-native-result-v1"
METHODS = (("parent", "parent"), ("candidate_bank", "bank"))
SCORE_FILES = ("rl_contract.py", "board_route.py", "entry_quality.py", "farm_contract.py",
               "settling_contract.py", "shot_benchmark_reward.py")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    raw = Path(path).read_bytes()
    return json.loads(gzip.decompress(raw) if str(path).endswith(".gz") else raw,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Nonfinite JSON " + x)))


def require(value, message):
    if not value:
        raise ValueError(message)


def verify_fingerprint(value, label):
    payload = copy.deepcopy(value)
    fingerprint = payload.pop("fingerprint", None)
    require(fingerprint == canonical_hash(payload), label + " fingerprint changed")


def verify_final_binding(plan, plan_sha256, run, audit):
    """Verify portable receipts against their copied producer evidence.

    Source hashes refer to the audited snapshot, not this publisher's checkout.
    The report joins that receipt to immutable files; it does not rerun physics.
    """
    require(audit.get("schema") == AUDIT_SCHEMA, "Unknown final native audit schema")
    verify_fingerprint(audit, "Final native audit")
    native = plan.get("native_binding")
    require(isinstance(native, dict), "Final native source binding missing")
    expected = plan["input_bindings"]["original"]["checkpoint_sha256"]
    require(audit.get("status") == "passed"
            and audit.get("comparison_fingerprint") == plan["fingerprint"]
            and audit.get("comparison_sha256") == plan_sha256
            and audit.get("native_requests") == 70
            and audit.get("optimizer_updates") == 0
            and audit.get("held_out_shot_files_read") == 0
            and audit.get("checkpoint_sha256") == expected,
            "Final native audit binding mismatch")
    for filename, field in (("result.json", "source_result_sha256"),
                            ("contract.json", "source_contract_sha256"),
                            ("experiment-binding.json", "source_binding_sha256")):
        require(digest(run/filename) == audit.get(field), "Audited native " + filename + " bytes changed")
    result, contract, binding = (read(run/name) for name in
                               ("result.json", "contract.json", "experiment-binding.json"))
    verify_fingerprint(contract, "Audited geometry contract")
    require(contract["fingerprint"] == native["geometry_fingerprint"]
            == native["original_source"]["geometry_fingerprint"]
            and canonical_hash(contract["scene"]) == native["source_scene_sha256"],
            "Audited geometry contract identity changed")
    require(result.get("schema") == RESULT_SCHEMA and result.get("status") == "completed"
            and result.get("comparison_fingerprint") == plan["fingerprint"]
            and result.get("comparison_sha256") == plan_sha256
            and result.get("optimizer_updates") == 0 and result.get("native_requests") == 70
            and result.get("checkpoint_sha256_before") == result.get("checkpoint_sha256_after") == expected
            and result.get("original_source") == native["original_source"],
            "Audited native result identity changed")
    require(binding.get("comparison_sha256") == plan_sha256
            and binding.get("comparison_fingerprint") == plan["fingerprint"]
            and binding.get("checkpoint_sha256") == expected
            and binding.get("original_source") == native["original_source"]
            and binding.get("source_commit") == native["source_commit"]
            and binding.get("source_archive_sha256") == native["source_archive_sha256"]
            and binding.get("control_sha256") == result.get("control_files"),
            "Audited producer binding identity changed")
    sources = audit.get("source_sha256")
    scoring = native.get("original_scoring_source_sha256", {})
    require(isinstance(sources, dict) and sources
            and all(isinstance(name, str) and isinstance(sha, str)
                    and re.fullmatch(r"[a-f0-9]{64}", sha) for name, sha in sources.items())
            and sources == result.get("source_sha256") == binding.get("source_sha256")
            and audit.get("audit_source_sha256") == sources.get("candidate_native_audit.py")
            and isinstance(scoring, dict) and set(scoring) == set(SCORE_FILES)
            and all(sources.get(name) == sha for name, sha in scoring.items()),
            "Audited producer/auditor source identity changed")
    require(set(audit.get("methods", {})) == {method for method, _ in METHODS}
            and isinstance(audit.get("trials"), list) and len(audit["trials"]) == 2
            and [t.get("method") for t in audit["trials"]] == [m for m, _ in METHODS]
            and isinstance(result.get("trials"), list) and len(result["trials"]) == 2
            and [t.get("method") for t in result["trials"]] == [m for m, _ in METHODS],
            "Audited cohort identities changed")
    return result


def verify_outcomes(packet, cohort, plan, method, counts):
    require(packet.get("schema") == "sircus-shot-benchmark-native-trial-v1"
            and packet.get("mode") == method
            and packet.get("comparison_fingerprint") == plan["fingerprint"]
            and packet.get("cells") == plan["cells"]
            and packet.get("actions") == cohort["actions"]
            and packet.get("enabled") == cohort["native_enabled"]
            and packet.get("proposal_ik_seeds") == cohort["ik_seeds"]
            and isinstance(packet.get("rows"), list) and len(packet["rows"]) == 35,
            "Audited trial action/case mapping changed")
    rows = packet["rows"]
    require(all(type(row.get("route_success")) is bool
                and type(row.get("route_quality")) is int and row["route_quality"] in (0, 1, 2)
                and (row.get("invalid") is None or isinstance(row["invalid"], str) and row["invalid"])
                and (not row["invalid"] or not row["route_success"] and row["route_quality"] == 0)
                and (row["route_quality"] != 2 or row["route_success"]) for row in rows),
            "Invalid audited outcome mapping")
    reasons = {}
    for row in rows:
        if row["invalid"]:
            reasons[row["invalid"]] = reasons.get(row["invalid"], 0) + 1
    actual = dict(requests=35, captures=sum(row["route_success"] for row in rows),
                  clean=sum(row["route_quality"] == 2 for row in rows), rejected=sum(reasons.values()))
    actual["released"] = 35-actual["rejected"]
    require(all(type(counts.get(name)) is int and counts[name] == value for name, value in actual.items()),
            "Audited counts disagree with bound trial outcomes")
    return reasons


def build_report(plan, *, plan_sha256, run=None, audit_path=None, status="prepared", queue_reason=None, error=None):
    validate_comparison(plan)
    native = plan.get("native_binding", {})
    report = dict(schema=SCHEMA, status=status, audit=dict(status="pending"),
        source_binding=dict(checkpoint_sha256=plan["input_bindings"]["original"]["checkpoint_sha256"],
            candidate_bank_sha256=plan["input_bindings"]["bank_artifact_sha256"]["qualification.json"],
            plan_sha256=plan_sha256, comparison_fingerprint=plan["fingerprint"],
            original_source=native.get("original_source"), producer_commit=native.get("source_commit")),
        methods={k:dict(requests=0, planned_requests=35, captures=None, clean=None, rejected=None,
                        released=None, compute={}) for k in ("parent", "bank")},
        paired={}, cases=[], optimizer_updates=0,
        preprocessing=copy.deepcopy(native.get("preprocessing", {})),
        compute={},
        scope=["35 known TRAIN positions requested once per frozen method; every rejection stays in the denominator.",
               "The candidate bank includes historical successful TRAIN shots. This is a TRAIN-informed selector comparison, not unseen-position generalization.",
               "Frozen seed-22 weights; no optimizer update, held-out search, inference-time ball-flight search or benchmark change.",
               "Unchanged parent actions use identical native IK seeding in both cohorts; changed bank actions can use prepared IK warm starts.",
               "Scripted driven transit and initialized release; no learned motor control, collision certificate or physical validation."],
        resource_scope="Native cohort wall/CPU include canonical home, setup, flight and serialization. Shared initialization/control probes and inherited bank preparation are separate. RSS is a lifetime process high-water; Torch CUDA peaks include persistent tensors, not model-only memory. The current selection and bank-load timings are CPU preparation, outside native execution.")
    if queue_reason:
        report["queue"] = dict(reason=queue_reason)
    if error:
        report["error"] = str(error)
    if run is not None:
        run = Path(run)
        progress_path = run/"progress.json"
        if progress_path.exists():
            progress = read(progress_path)
            require(progress["comparison_fingerprint"] == plan["fingerprint"], "Native progress plan mismatch")
            for method, counts in progress.get("methods", {}).items():
                require(method in {m for m, _ in METHODS}, "Unknown native progress method")
                require(type(counts.get("attempts")) is int and 0 <= counts["attempts"] <= 35,
                        "Native progress request count changed")
                key = "bank" if method == "candidate_bank" else "parent"
                report["methods"][key]["requests"] = counts["attempts"]
        if (run/"result.json").exists() and status != "failed":
            report["status"] = "audit_pending"
    if audit_path is None or not Path(audit_path).exists():
        return report
    audit = read(audit_path)
    require(run is not None, "Final native audit requires copied run evidence")
    run = Path(run)
    result = verify_final_binding(plan, plan_sha256, run, audit)
    packets = {}
    for index, (method, key) in enumerate(METHODS):
        manifest, producer = audit["trials"][index], result["trials"][index]
        require(all(manifest[field] == producer[field] for field in ("method", "file", "sha256"))
                and producer.get("requests") == 35,
                "Audited trial manifest binding changed")
        path = (run/manifest["file"]).resolve()
        require(path.is_relative_to(run.resolve()) and digest(path) == manifest["sha256"], "Audited trial bytes/path changed")
        packet = read(path)
        packets[key] = packet
        values = audit["methods"][method]
        counts, metrics = values["counts"], values["metrics"]
        reasons = verify_outcomes(packet, plan["cohorts"][index], plan, method, counts)
        require(metrics == producer.get("metrics"), "Audited resource measurements changed")
        compute = dict(wall_s=metrics["wall_s"], cpu_s=metrics["cpu_s"], peak_rss_mib=metrics["peak_rss_mib"],
            cuda_peak_allocated_mib=metrics.get("torch_peak_allocated_mib"),
            cuda_peak_reserved_mib=metrics.get("torch_peak_reserved_mib"),
            sampled_device_occupied_mib=metrics.get("sampled_device_occupied_mib"),
            setup_wall_s=metrics.get("setup_wall_s"), score_wall_s=metrics.get("score_wall_s"),
            geometry_s=None, selection_s=0. if key == "parent" else native.get("preprocessing", {}).get("selection_wall_s"))
        report["methods"][key] = dict(requests=counts["requests"], planned_requests=35,
            captures=counts["captures"], clean=counts["clean"], rejected=counts["rejected"],
            released=counts["released"], rejection_reasons=reasons,
            outcome_breakdown=copy.deepcopy(counts.get("failures", {})), compute=compute)
    require(audit["rescored_releases"] == sum(m["released"] for m in report["methods"].values()),
            "Audited release denominator changed")
    cases = []
    for i, cell in enumerate(plan["cells"]):
        case = dict(cell_id=cell["id"], bucket_xy_m=cell["bucket_xy_m"],
                    action_changed=plan["selections"][i]["action_changed"])
        for key in ("parent", "bank"):
            row = packets[key]["rows"][i]
            case[key] = dict(capture=row["route_success"], clean=row["route_quality"] == 2,
                             rejection=row["invalid"], action=packets[key]["actions"][i])
        cases.append(case)
    report.update(status="passed", audit=dict(status="passed", sha256=digest(audit_path),
        rescored_releases=audit["rescored_releases"], full_rate_steps=audit["full_rate_steps"]), cases=cases,
        paired=dict(gains=sum(not c["parent"]["capture"] and c["bank"]["capture"] for c in cases),
                    losses=sum(c["parent"]["capture"] and not c["bank"]["capture"] for c in cases),
                    unchanged=sum(c["parent"]["capture"] == c["bank"]["capture"] for c in cases),
                    changed_actions=sum(c["action_changed"] for c in cases)),
        compute=dict(initialization_shared_wall_s=result["initialization_shared_wall_s"],
                     comparison_wall_s=result["comparison_wall_s"]),
        resource_scope=result["resource_scope"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--status", default="prepared", choices=("prepared", "queued", "running", "failed", "audit_pending"))
    parser.add_argument("--queue-reason")
    parser.add_argument("--error")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    report = build_report(plan, plan_sha256=digest(args.plan), run=args.run,
                          audit_path=args.audit, status=args.status, queue_reason=args.queue_reason, error=args.error)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    temp.replace(args.output)


if __name__ == "__main__":
    main()
