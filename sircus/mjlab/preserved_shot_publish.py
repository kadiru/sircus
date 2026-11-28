"""Compose read-only UI results from completed, independently audited studies.

This publisher never plans, runs physics, changes a planner, or restarts a service.
Development repeats and fresh first-drop results have separate denominators.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .benchmark_verify import contained, digest, read, sha
from .expanded_selector_publish import outcome
from .approach_comparison_publish import require, statistics
from .sweep_worker import atomic

VERSION = "sircus-preserved-shot-display-v1"
METHODS = ("current", "protected")
DISPLAY = {"current": "current", "protected": "preserved"}


def verified(study, audit_path, cohort):
    study, audit_path = Path(study), Path(audit_path)
    report, audit, contract = (read(p) for p in (study/"report.json", audit_path, study/"contract.json"))
    require(report.get("status") == "completed" and report.get("native_completed") is True,
            "Completed native study required")
    require(audit.get("version") == "sircus-preserved-shot-independent-audit-v1"
            and audit.get("status") == "passed" and audit.get("frozen_before_test") is True,
            "Independent frozen-study audit required")
    for key, name in (("report_sha256", "report.json"), ("evidence_manifest_sha256", "evidence-manifest.json"),
                      ("contract_sha256", "contract.json"), ("choices_sha256", "choices.json"),
                      ("native_started_sha256", "native-started.json")):
        require(audit.get(key) == sha(study/name), "Audit no longer binds " + name)
    require(contract.get("fingerprint") == digest({k:v for k,v in contract.items() if k != "fingerprint"}),
            "Frozen contract changed")
    require(contract.get("cohort") == cohort and contract.get("methods") == list(METHODS)
            and contract.get("physics_substeps") == 64 and contract.get("training_performed") is False,
            "Unexpected comparison cohort or protocol")
    manifest = read(study/"evidence-manifest.json")
    for name, checksum in manifest.items():
        require(sha(contained(study,name)) == checksum, "Changed evidence: " + name)
    for name in ("contract.json", "layouts.json", "plans.json", "selection.json", "choices.json", "rows.jsonl", "native-started.json"):
        require(manifest.get(name) == sha(study/name), "Unbound evidence: " + name)
    require(report.get("comparison") == audit.get("comparison"), "Independent summary differs")
    for phase in ("controls_pre", "controls_post"):
        gate = report.get("controls", {}).get(phase, {})
        require(gate.get("status") == "passed" and gate.get("board_first_passed") is True,
                "Both control blocks must pass")
    provenance = dict(report_sha256=sha(study/"report.json"), audit_sha256=sha(audit_path),
                      contract_sha256=sha(study/"contract.json"), manifest_sha256=sha(study/"evidence-manifest.json"))
    return study, report, audit, contract, provenance


def compose_cohort(study, audit_path, cohort):
    study, report, audit, contract, provenance = verified(study, audit_path, cohort)
    layouts = read(study/"layouts.json")["layouts"]
    slots = read(study/"choices.json")["slots"]
    plans = {(p["site"]["layout"]["id"], p["variant"]):p["site"] for p in read(study/"plans.json")}
    selections = {s["context_id"]:s for s in read(study/"selection.json")}
    rows = [json.loads(line) for line in (study/"rows.jsonl").read_text().splitlines() if line.strip()]
    native = {r["id"]:r for r in rows if r["adapter_directory"] != "native-controls"}
    repeats = 3 if cohort == "development" else 1
    count = 9 if cohort == "development" else 50
    require(contract["attempts_per_method_per_position"] == repeats and len(layouts) == count,
            "Separate development and fresh denominators required")
    require(len(slots) == count*repeats*2 and len({s["id"] for s in slots}) == len(slots),
            "Exact paired native slot inventory required")
    require(set(native) == {s["id"] for s in slots if s["request"] is not None}
            and len(rows)-len(native) == 18, "Native attempts and controls must be fully accounted")
    position_rows = []
    flat = {m:[] for m in METHODS}
    for layout in layouts:
        identity = layout["id"]
        selection = selections[identity]
        row = dict(id=identity, bucket_xy_m=layout["bucket_xy_m"])
        for method in METHODS:
            attempts = [s for s in slots if s["context_id"] == identity and s["method"] == method]
            require(len(attempts) == repeats, "Missing paired repeat")
            labels = [outcome(s,native) for s in attempts]
            flat[method].extend(labels)
            # The frozen rule records a fallback branch even if both planners
            # decline. Display only an actual retained checked command as a
            # fallback use, rather than claiming a shot exists in that case.
            retained_current = bool(selection["fallback_used"] and any(s["request"] is not None for s in attempts))
            elapsed = (plans[identity,"current"]["request_wall_s"] if method == "current"
                       else selection["timing"]["pair_wall_s"])
            if cohort == "development":
                row[DISPLAY[method]] = dict(attempts=repeats, captures=labels.count("capture"),
                    releases=repeats-labels.count("setup refused"), outcomes=labels, request_s=elapsed)
                if method == "protected":
                    row["preserved"]["used_fallback"] = sum(s["request"] is not None for s in attempts) if retained_current else 0
            else:
                row[DISPLAY[method]] = dict(outcome=labels[0], request_s=elapsed)
                if method == "protected":
                    row["preserved"]["used_fallback"] = retained_current
        position_rows.append(row)
    result = dict(status="completed", audit_status="passed", checks_count=audit["checks_count"],
        provenance=provenance, controls=dict(passed=18,total=18), physics_substeps=64,
        frozen_before_test=True, attempts_per_position=repeats,
        gpu_wait_s=report["gpu_wait_s"], gpu_lease_s=report["gpu_lease_s"], native_releases=len(rows))
    if cohort == "development":
        result["positions"] = position_rows
        return result
    methods = []
    for method in METHODS:
        labels = flat[method]
        methods.append(dict(id=DISPLAY[method], attempts=count,
            captures=labels.count("capture"), releases=count-labels.count("setup refused"),
            setup_rejections=labels.count("setup refused"), released_misses=labels.count("flight miss"),
            invalid_physics=labels.count("invalid physics"),
            request_s=statistics([r[DISPLAY[method]]["request_s"] for r in position_rows])))
    result.update(positions=count, methods=methods, results=position_rows,
        reserved_worker_window_s=statistics([s["timing"]["reserved_worker_window_s"] for s in selections.values()]))
    return result


def compose(development, development_audit, fresh, fresh_audit, diagnosis):
    diagnosis = read(Path(diagnosis))
    require(diagnosis.get("version") == "sircus-preserved-shot-diagnosis-v1"
            and diagnosis.get("audit_status") == "passed" and len(diagnosis.get("positions",[])) == 9,
            "Nine-case diagnosis required")
    dev = compose_cohort(development, development_audit, "development")
    new = compose_cohort(fresh, fresh_audit, "fresh")
    require({r["id"] for r in dev["positions"]} == {r["id"] for r in diagnosis["positions"]},
            "Development cases differ from diagnosed failures")
    require(not ({tuple(r["bucket_xy_m"]) for r in dev["positions"]}
                 & {tuple(r["bucket_xy_m"]) for r in new["results"]}), "Fresh positions overlap development")
    return dict(version=VERSION, status="completed", phase="completed", observed_at_unix=time.time(),
        diagnosis=diagnosis, development=dev, fresh=new,
        progress_text="Development repeats and the 50-position test are complete and independently audited.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development", "development-audit", "fresh", "fresh-audit", "diagnosis", "display-report"):
        parser.add_argument("--"+name, type=Path, required=True)
    a = parser.parse_args()
    section = compose(a.development,a.development_audit,a.fresh,a.fresh_audit,a.diagnosis)
    report = read(a.display_report)
    report["preserved_shot"] = section
    atomic(a.display_report,report)
    print(json.dumps(dict(status="composed",methods=section["fresh"]["methods"])))


if __name__ == "__main__":
    main()
