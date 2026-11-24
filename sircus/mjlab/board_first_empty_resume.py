"""Additive recovery for the timer-type failure before the first native step.

This cannot resume a partial simulation. The original source, contract, frozen
choices, trained weights and native-start receipt remain immutable. Only the
integer alarm fix and explicit empty-run recovery plumbing may differ.
"""
from __future__ import annotations

import json
from pathlib import Path

from .benchmark_verify import contained, digest, read, sha
from .corrected_grid import source_manifest

VERSION = "sircus-board-first-empty-native-resume-v1"
CHANGED_FILES = {
    "sircus/mjlab/board_first_selector_study.py",
    "sircus/mjlab/tests/test_board_first_selector.py",
}
ADDED_FILES = {"sircus/mjlab/board_first_empty_resume.py"}


def require_empty(output):
    """No rows, requests, native archives, controls or previous resumed attempt."""
    output = Path(output)
    for name in ("rows.jsonl", "timing.jsonl", "native-resumed.json"):
        if (output/name).exists():
            raise ValueError("Empty-only recovery refuses existing " + name)
    if any(output.rglob("*.npz")) or any((output/"requests").glob("*.json")):
        raise ValueError("Empty-only recovery refuses native arrays or request packets")
    if (output/"native-controls").exists() or any(output.glob("native-*/effective-options.json")):
        raise ValueError("Empty-only recovery refuses evidence of native initialization")
    report = read(output/"report.json")
    if (report.get("controls") or report.get("native_completed")
            or report.get("progress", {}).get("completed_releases") != 0
            or report.get("phase") not in {"prepared", "waiting_for_4090"}):
        raise ValueError("Empty-only recovery requires prepared choices and zero releases")


def source_changes(before, after):
    removed = set(before)-set(after)
    added = set(after)-set(before)
    changed = {name for name in before.keys() & after.keys() if before[name] != after[name]}
    if removed or not added <= ADDED_FILES or not changed <= CHANGED_FILES:
        raise ValueError("Recovery source migration includes changes beyond timer/resume plumbing")
    if "sircus/mjlab/board_first_selector_study.py" not in changed:
        raise ValueError("Recovery must include the timer type correction")
    return dict(changed=sorted(changed), added=sorted(added), removed=[])


def verify(output, execution_root):
    output, execution_root = Path(output), Path(execution_root)
    receipt = read(output/"execution-recovery.json")
    if receipt.get("version") != VERSION or receipt.get("fingerprint") != digest({k:v for k,v in receipt.items() if k!="fingerprint"}):
        raise ValueError("Recovery receipt identity changed")
    contract = read(output/"contract.json")
    for name, key in (("contract.json", "original_contract_sha256"),
                      ("native-started.json", "original_native_started_sha256"),
                      ("pre-native-failure-report.json", "original_report_sha256")):
        if sha(output/name) != receipt[key]:
            raise ValueError("Original recovery evidence changed: " + name)
    old_root = Path(receipt["original_source_root"])
    for name, expected in contract["source_sha256"].items():
        if sha(contained(old_root,name)) != expected:
            raise ValueError("Original producer source changed: " + name)
    actual = source_manifest(execution_root)
    if actual != receipt["execution_source_sha256"]:
        raise ValueError("Corrected execution source changed after recovery binding")
    changes = source_changes(contract["source_sha256"], actual)
    if changes != receipt["source_changes"] or receipt["gpu_cap_s"] != contract["gpu_cap_s"] or receipt["gpu_cap_s"] != 3600:
        raise ValueError("Recovery changed source scope or experiment budget")
    for name, expected in receipt["frozen_choices_bindings"].items():
        if sha(contained(output,name)) != expected:
            raise ValueError("Recovery changed a frozen choice/input: " + name)
    return receipt


def prepare(a):
    """Record the narrow migration; do not refit, replan, acquire a GPU or run."""
    output = a.output
    if a.original_source_root is None:
        raise ValueError("The unchanged original producer checkout is required")
    require_empty(output)
    contract = read(output/"contract.json")
    if contract.get("fingerprint") != digest({k:v for k,v in contract.items() if k!="fingerprint"}):
        raise ValueError("Original contract fingerprint changed")
    for name, expected in contract["source_sha256"].items():
        if sha(contained(a.original_source_root,name)) != expected:
            raise ValueError("Original producer source changed: " + name)
    started = read(output/"native-started.json")
    choices = read(output/"choices.json")
    bindings = dict(choices["bindings"], **{"choices.json":sha(output/"choices.json")})
    planned = {method:[s["request"] for s in choices["slots"] if s["method"]==method and s["request"] is not None]
               for method in contract["methods"]}
    if started != dict(bindings=bindings, requests_sha256=digest(planned)):
        raise ValueError("Original start receipt differs from exact frozen commands")
    for name, expected in bindings.items():
        if sha(contained(output,name)) != expected:
            raise ValueError("Frozen choice changed before recovery: " + name)
    root = Path(__file__).resolve().parents[2]
    current = source_manifest(root)
    changed = source_changes(contract["source_sha256"],current)
    old_report = (output/"report.json").read_bytes()
    with (output/"pre-native-failure-report.json").open("xb") as stream:
        stream.write(old_report)
    receipt = dict(version=VERSION, reason="signal.alarm requires integer seconds; original3600.0 failed before native engine creation",
                  original_contract_sha256=sha(output/"contract.json"), original_native_started_sha256=sha(output/"native-started.json"),
                  original_report_sha256=sha(output/"pre-native-failure-report.json"),
                  original_source_root=str(a.original_source_root.resolve()), execution_source_sha256=current,
                  source_changes=changed, frozen_choices_bindings=bindings, requests_sha256=digest(planned),
                  no_native_evidence_before_resume=True, completed_releases_before_resume=0, gpu_cap_s=3600,
                  policy="No refit, re-preparation, changes to physics/scoring/planner or replacement of original receipts")
    receipt["fingerprint"] = digest(receipt)
    with (output/"execution-recovery.json").open("x") as stream:
        json.dump(receipt,stream,indent=2)
    verify(output,root)
    from .board_first_selector_study import publish
    report = read(output/"report.json")
    report.update(phase="prepared", execution_recovery_sha256=sha(output/"execution-recovery.json"))
    publish(a,report)


def start(a, expected):
    """Exclusively mark one resumed attempt, preserving the original start file."""
    require_empty(a.output)
    receipt = verify(a.output,Path(__file__).resolve().parents[2])
    if read(a.output/"native-started.json") != expected:
        raise ValueError("Resumed commands differ from original pre-native start")
    resumed = dict(version=VERSION, execution_recovery_sha256=sha(a.output/"execution-recovery.json"),
                   original_native_started_sha256=sha(a.output/"native-started.json"),
                   requests_sha256=receipt["requests_sha256"], bindings=expected["bindings"])
    with (a.output/"native-resumed.json").open("x") as stream:
        json.dump(resumed,stream,indent=2)
