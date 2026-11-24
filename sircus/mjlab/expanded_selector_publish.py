"""Publish an audited expanded-library comparison without switching planners."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from .benchmark_verify import contained, digest, read, sha

VERSION = "sircus-expanded-selector-display-v1"
METHODS = ("current", "expanded")


def outcome(slot, native):
    if slot["request"] is None:
        return "setup refused"
    row = native[slot["id"]]
    if row.get("native_release") is not True or row.get("board_first_scalar_audit_passed") is not True:
        raise ValueError("Admitted slot needs one independently scored native release")
    score = row["board_first_score"]
    if score.get("scoring_version") != "sircus-board-first-native-capture-v4":
        raise ValueError("Expected explicit version-four scoring")
    if type(score.get("invalid_physics")) is not bool or type(score.get("route_success")) is not bool:
        raise ValueError("Expected boolean validity and capture")
    if score["invalid_physics"]:
        if score["route_success"]:
            raise ValueError("Invalid physics cannot capture")
        return "invalid physics"
    return "capture" if score["route_success"] else "flight miss"


def compose(study, audit_path):
    from .expanded_selector_study import summarize
    study, audit_path = Path(study), Path(audit_path)
    report, audit = read(study/"report.json"), read(audit_path)
    if (report.get("version"), report.get("status"), report.get("phase"),
        audit.get("version"), audit.get("status")) != (
        "sircus-expanded-selector-comparison-v1", "completed", "completed",
        "sircus-expanded-selector-independent-audit-v1", "passed"):
        raise ValueError("A completed comparison and bound independent audit are required")
    if (report.get("native_completed") is not True or audit.get("frozen_before_test") is not True
        or type(audit.get("checks_count")) is not int or audit["checks_count"] <= 0):
        raise ValueError("Complete frozen native evidence must be audited")
    for key, name in (("report_sha256", "report.json"), ("evidence_manifest_sha256", "evidence-manifest.json"),
                      ("contract_sha256", "contract.json"), ("choices_sha256", "choices.json"),
                      ("native_started_sha256", "native-started.json")):
        if audit.get(key) != sha(study/name):
            raise ValueError("Audit no longer binds " + name)
    manifest, contract = read(study/"evidence-manifest.json"), read(study/"contract.json")
    if contract.get("fingerprint") != digest({k:v for k,v in contract.items() if k != "fingerprint"}):
        raise ValueError("Contract changed")
    if (contract.get("methods"), contract.get("test_positions"), contract.get("physics_substeps"),
        contract.get("attempts_per_method_per_position"), contract.get("scoring_version")) != (
        list(METHODS), 50, 64, 1, "sircus-board-first-native-capture-v4"):
        raise ValueError("Expected current/expanded, fifty fresh positions, one attempt each, native64")
    choices = read(study/"choices.json")
    for name, expected in contract["frozen_inputs"].items():
        if sha(contained(study,name)) != expected or choices["bindings"].get(name) != expected:
            raise ValueError("Frozen input changed: " + name)
    for name in ("contract.json", "layouts.json", "plans.json", "choices.json", "rows.jsonl", "native-started.json",
                 "current-checkpoint.json", "expanded-checkpoint.json", "library-summary.json", "library-manifest.json"):
        if manifest.get(name) != sha(study/name):
            raise ValueError("Evidence changed: " + name)
    started = read(study/"native-started.json")
    expected = dict(choices["bindings"], **{"choices.json":sha(study/"choices.json")})
    if started.get("bindings") != expected:
        raise ValueError("Choices and model weights were not frozen before native")
    for name, expected_hash in expected.items():
        if sha(contained(study,name)) != expected_hash:
            raise ValueError("Native-start evidence changed: " + name)
    for phase in ("controls_pre", "controls_post"):
        gate = report.get("controls", {}).get(phase, {})
        if gate.get("status") != "passed" or gate.get("board_first_passed") is not True:
            raise ValueError("Both matched control blocks must pass")
    layouts = read(study/"layouts.json")["layouts"]
    slots, plans = choices["slots"], read(study/"plans.json")
    rows = [json.loads(line) for line in (study/"rows.jsonl").read_text().splitlines() if line.strip()]
    if (len(layouts) != 50 or len({r["id"] for r in layouts}) != 50
        or len({tuple(r["bucket_xy_m"]) for r in layouts}) != 50
        or any(len(r["bucket_xy_m"]) != 2 or not all(math.isfinite(v) for v in r["bucket_xy_m"]) for r in layouts)):
        raise ValueError("Expected fifty distinct finite placements")
    keyed = {(s["context_id"],s["method"]):s for s in slots}
    wanted = {(r["id"],m) for r in layouts for m in METHODS}
    if len(slots) != 100 or len(keyed) != 100 or set(keyed) != wanted or len({s["id"] for s in slots}) != 100:
        raise ValueError("Exactly one attempt per method and position is required")
    native = {r["id"]:r for r in rows if r["adapter_directory"] != "native-controls"}
    if len(rows)-len(native) != 18 or set(native) != {s["id"] for s in slots if s["request"] is not None}:
        raise ValueError("Admitted choices and eighteen controls must be fully accounted")
    comparison = summarize(layouts,slots,rows,plans)
    if report.get("comparison") != comparison or audit.get("comparison") != comparison:
        raise ValueError("Published counts differ from independently checked outcomes")
    results=[]
    for layout in layouts:
        result={"bucket_xy_m":layout["bucket_xy_m"]}
        for method in METHODS:
            slot=keyed[layout["id"],method]
            if slot["bucket_xy_m"] != layout["bucket_xy_m"]:
                raise ValueError("Attempt target differs from frozen position")
            result[method]=outcome(slot,native)
        results.append(result)
    methods=[]
    for method in METHODS:
        checkpoint=sha(study/(method+"-checkpoint.json"))
        if audit["checkpoint_sha256"][method] != checkpoint:
            raise ValueError("Audited checkpoint changed")
        m=comparison["methods"][method]
        methods.append(dict(id=method,label="Current planner" if method=="current" else "Expanded-library selector",
            checkpoint_sha256=checkpoint,attempts=50,captures=m["captures"],setup_rejections=m["setup_rejections"],
            released_misses=m["flight_misses"],invalid_physics=m["invalid_physics"],
            mean_decision_s=m["mean_planning_s"],mean_request_s=m["mean_request_s"]))
    # Count the actual deduplicated training records, not raw repeated drops.
    library_rows=[]
    for split in ("train", "validation"):
        records=[json.loads(line) for line in (study/(split+"-dataset.jsonl")).read_text().splitlines() if line.strip()]
        if any(r.get("split") != split for r in records) or len({r["dedup_key"] for r in records}) != len(records):
            raise ValueError("Expected declared, deduplicated training split")
        library_rows.append(dict(label="Training" if split=="train" else "Validation",
            positions=len({tuple(r["request"]["bucket_xy_m"]) for r in records}),settings=len(records),
            capturing_settings=sum((r.get("capture_probability") or 0)>0 for r in records)))
    old,new=(m["captures"] for m in methods)
    pair=comparison["paired"]
    return dict(version=VERSION,status="completed",phase="completed",audit_status="passed",positions=50,
        physics_substeps=64,frozen_before_test=True,methods=methods,results=results,library_rows=library_rows,
        training_text=("44 geometry/source-seed inputs → 64 → 32 hidden units → setup and capture scores. "
            "Supervised learning; repeated outcomes estimate each setting’s capture frequency without extra weight. "
            "Historical test shots reused here are development data for this new experiment; previous benchmark results remain unchanged."),
        conclusion=(f"Expanded selector: {new}/50; current planner: {old}/50 ({new-old:+d}). "
            f"{pair['expanded_only']} gained positions, {pair['current_only']} lost. The live planner is unchanged. "
            "This compares complete methods: the expanded method changes the library, learned weights and contact ranking."),
        timing_text=(f"Both use a {contract['planning_budget_s']:g}-second decision deadline. Request time also includes worker/protocol overhead. "
            f"All 18 control drops passed; {audit['checks_count']:,} independent checks. No test-drop feedback was used to change selections."),
        provenance=dict(report_sha256=sha(study/"report.json"),audit_sha256=sha(audit_path),
            manifest_sha256=sha(study/"evidence-manifest.json"),library_manifest_sha256=sha(study/"library-manifest.json")))


def publish(study, audit, display):
    study, audit, display=Path(study),Path(audit),Path(display)
    temporary=display.with_suffix(display.suffix+".expanded.tmp")
    if any(p.resolve().is_relative_to(study.resolve()) or p.resolve()==audit.resolve() for p in (display,temporary)):
        raise ValueError("Display cannot replace immutable evidence")
    section=compose(study,audit)
    data=read(display) if display.exists() else {}
    data["expanded_selector_study"]=section
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
    os.replace(temporary,display)
    return section


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",type=Path,required=True)
    parser.add_argument("--audit",type=Path,required=True)
    parser.add_argument("--display-report",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(publish(args.study,args.audit,args.display_report)["methods"]))


if __name__=="__main__":
    main()
