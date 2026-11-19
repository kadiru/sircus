"""Export audited replay metrics without importing a simulator or raw arrays.

Run only after replay650_verify has passed. This is a byte-binding check and
small publication exporter, not a replacement for its full native-trace audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re


STUDY_VERSION = "sircus-calibrated-650-replay-v1"
AUDIT_VERSION = "sircus-calibrated-650-independent-audit-v2"
VERSION = "sircus-calibrated-650-curation-v1"
STEPS = (64, 128)
GAPS = (.3768, .3808, .3848, .3868, .3888)
STAT_FIELDS = ("attempts", "native_releases", "captures", "invalid", "clean", "whole_entries", "max_retention_s")
SCOPE = ("Fresh calibrated simulated-contact engineering replay of fixed held endpoints. "
         "Historical shallow-study outcomes retain their original scoring definition. "
         "Three nominal copies per configuration/resolution are not fresh-noise robustness, "
         "RL training, held-out performance, full-path validation, or physical validation.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read(path):
    def invalid(value):
        raise ValueError("Non-finite JSON constant: " + value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def relative_file(root, name):
    require(isinstance(name, str) and not Path(name).is_absolute(), "Absolute evidence filename")
    path = (root / name).resolve()
    require(path.is_relative_to(root) and path.is_file(), "Missing or escaping evidence file")
    return path


def hash_map(value):
    require(isinstance(value, dict) and bool(value), "Missing nonempty evidence hash map")
    for name, value_hash in value.items():
        require(isinstance(name, str) and not Path(name).is_absolute() and ".." not in Path(name).parts,
                "Unsafe provenance filename")
        require(isinstance(value_hash, str) and re.fullmatch(r"[0-9a-f]{64}", value_hash), "Invalid SHA256")
    return dict(value)


def number(value):
    require(type(value) in (float, int) and math.isfinite(value), "Expected finite number")
    return value


def identifier(value):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]+", value), "Unsafe publication identifier")
    return value


def stats(value):
    result = {}
    for key in STAT_FIELDS:
        result[key] = number(value[key])
        require(result[key] >= 0, "Negative outcome count")
        if key != "max_retention_s":
            require(type(result[key]) is int, "Outcome denominator must be an integer")
    require(result["captures"] <= result["native_releases"] <= result["attempts"], "Invalid outcome denominators")
    require(result["clean"] <= result["captures"] and result["invalid"] <= result["attempts"], "Invalid subcounts")
    return result


def verified_inputs(study, audit_path):
    """Verify immutable publication inputs before creating any output files."""
    study = Path(study).resolve()
    report, spec, audit = (read(p) for p in (study / "report.json", study / "contract.json", audit_path))
    require(report.get("version") == spec.get("version") == STUDY_VERSION, "Unexpected study version")
    require(report.get("status") == report.get("phase") == "completed", "Study is not completed")
    require(audit.get("version") == AUDIT_VERSION and audit.get("status") == "passed", "Full independent audit must pass")
    checks = audit.get("checks")
    require(isinstance(checks, list) and bool(checks) and type(audit.get("checks_count")) is int
            and audit["checks_count"] == len(checks)
            and all(isinstance(c, dict) and c.get("passed") is True and isinstance(c.get("label"), str) for c in checks),
            "Audit needs nonempty, fully passed checks with an exact count")
    require(audit.get("failures") == [], "Audit failures present")
    require(audit.get("report_sha256") == sha(study / "report.json"), "Audit/report bytes differ")
    require(audit.get("evidence_manifest_sha256") == sha(study / "evidence-manifest.json"), "Audit/manifest bytes differ")
    unsigned = dict(spec)
    fp = unsigned.pop("fingerprint", None)
    require(fp == digest(unsigned) == report.get("contract_fingerprint") == audit.get("contract_fingerprint"),
            "Contract fingerprint binding differs")
    manifest = hash_map(read(study / "evidence-manifest.json"))
    require(report.get("evidence_sha256") == manifest, "Report/manifest binding differs")
    require("contract.json" in manifest and manifest["contract.json"] == sha(study / "contract.json"), "Contract bytes not manifest-bound")
    for name, expected in manifest.items():
        require(sha(relative_file(study, name)) == expected, "Changed evidence bytes: " + name)
    require(report["settings"] == spec["settings"], "Report/settings binding differs")
    require(report.get("gate") == audit.get("gate"), "Independent gate differs")
    require(audit.get("requested_trials") == report["counts"]["attempts"]
            and audit.get("native_releases") == report["counts"]["native_releases"], "Audit/report denominators differ")
    return study, report, spec, audit, manifest


def curate(study, audit_path, output):
    study, report, spec, audit, manifest = verified_inputs(study, audit_path)
    output = Path(output).resolve()
    require(not output.is_relative_to(study), "Curated output must be outside frozen evidence")
    settings, gate = report["settings"], report["gate"]
    require(settings["substeps"] == list(STEPS) and settings["gaps_m"] == list(GAPS)
            and settings["repeats_per_group"] == 3 and settings["max_requests"] == 48, "Unexpected declared finite protocol")
    gate_keys = ("status", "positive_control", "negative_control", "candidate", "physics_valid", "oracle_agreement", "max_drift_m", "window_opened")
    require(set(gate) == set(gate_keys) and gate["status"] in ("passed", "failed"), "Incomplete final gate")
    require(type(gate["window_opened"]) is bool and gate["window_opened"] == (gate["status"] == "passed"), "Gate/window mismatch")
    for key in gate_keys[1:6]:
        require(type(gate[key]) is bool, "Non-boolean gate predicate")
    if gate["max_drift_m"] is not None:
        number(gate["max_drift_m"])
    rows, groups = [], []
    require(len(report["groups"]) == 8, "All eight predeclared groups must be retained")
    require(len({g["id"] for g in report["groups"]}) == 8, "Duplicate declared group")
    for group in report["groups"]:
        role = identifier(group["role"])
        require(role in ("positive_control", "negative_control", "candidate", "window"), "Unexpected group role")
        v, bucket = group["values"], group["bucket_xy_m"]
        require(len(v) == 5 and len(bucket) == 2 and group["board_xy_m"] == v[:2], "Incomplete configuration")
        for value in [*v, *bucket]:
            number(value)
        require(type(group["eligible"]) is bool, "Missing endpoint eligibility")
        cfg = dict(id=identifier(group["id"]), role=role, source_record_id=identifier(group["source_record_id"]),
                   source_record_sha256=group["source_record_sha256"], board_xy_m=v[:2], bucket_xy_m=bucket,
                   board_height_m=v[2], tilt_deg=v[3], gap_m=v[4],
                   release_center_height_m=number(group["release_center_height_m"]), distance_m=number(group["distance_m"]),
                   geometry_eligible=group["eligible"], by_substeps={})
        hash_map({"source_record": cfg["source_record_sha256"]})
        sensitivity = group["sensitivity"]
        cfg["paired_sensitivity"] = {k:sensitivity[k] for k in (
            "paired_attempts", "capture_agreement", "nominal_position_delta_350ms_max_m")}
        require(type(sensitivity["paired_attempts"]) is int and type(sensitivity["capture_agreement"]) is int
                and 0 <= sensitivity["capture_agreement"] <= sensitivity["paired_attempts"] <= 3,
                "Invalid matched-resolution sensitivity denominator")
        if sensitivity["nominal_position_delta_350ms_max_m"] is not None:
            require(number(sensitivity["nominal_position_delta_350ms_max_m"]) >= 0, "Negative paired drift")
        for step in STEPS:
            result = stats(group["by_substeps"][str(step)])
            expected = 3 if role != "window" or gate["window_opened"] else 0
            require(result["attempts"] == expected, "Executed/skipped conditional denominator differs")
            status = "skipped_gate_closed" if expected == 0 else "executed_nominal"
            cfg["by_substeps"][str(step)] = dict(status=status, **result)
            rows.append(dict(group_id=cfg["id"], role=role, source_record_id=cfg["source_record_id"], substeps=step,
                status=status, geometry_eligible=cfg["geometry_eligible"], board_x_m=v[0], board_y_m=v[1],
                board_height_m=v[2], tilt_deg=v[3], release_gap_m=v[4], release_ball_center_height_m=cfg["release_center_height_m"],
                bucket_x_m=bucket[0], bucket_y_m=bucket[1], distance_m=cfg["distance_m"], **result,
                paired_attempts=sensitivity["paired_attempts"], paired_capture_agreement=sensitivity["capture_agreement"],
                max_position_delta_350ms_m=sensitivity["nominal_position_delta_350ms_max_m"]))
        groups.append(cfg)
    require([g["role"] for g in groups].count("window") == 5 and all([g["role"] for g in groups].count(r) == 1
            for r in ("positive_control", "negative_control", "candidate")), "Declared group roles differ")
    total = stats(report["counts"])
    for key in STAT_FIELDS[:-1]:
        require(sum(r[key] for r in rows) == total[key], "Aggregate group denominator differs: " + key)
    require(total["attempts"] == (48 if gate["window_opened"] else 18), "Conditional execution count differs")
    hash_map({"independent_auditor": audit["auditor_sha256"]})
    summary = dict(version=VERSION, study_version=STUDY_VERSION, study_id=identifier(study.name), status="completed_independently_audited",
        scope=SCOPE, counts=total, gate={k:gate[k] for k in gate_keys},
        declared_protocol=dict(substeps=list(STEPS), gaps_m=list(GAPS), repeats_per_group=3, maximum_requests=48,
                               window_planned_requests=30, nominal_repeats_only=True),
        window=dict(status="completed" if gate["window_opened"] else "skipped_gate_closed", opened=gate["window_opened"],
                    requested_trials=sum(r["attempts"] for r in rows if r["role"] == "window")),
        groups=groups,
        provenance=dict(contract_fingerprint=spec["fingerprint"], contract_sha256=sha(study / "contract.json"),
                        report_sha256=sha(study / "report.json"), independent_audit_sha256=sha(audit_path),
                        independent_auditor_sha256=audit["auditor_sha256"],
                        independent_audit_checks=audit["checks_count"], evidence_manifest_sha256=sha(study / "evidence-manifest.json"),
                        evidence_sha256=manifest, executed_source_sha256=hash_map(spec["source_sha256"]),
                        asset_sha256=hash_map(spec["asset_sha256"])),
        privacy="Configuration parameters, aggregate outcomes and evidence hashes only; no joints, sampled poses, trajectories, credentials or host paths.")
    require(not output.exists(), "Refusing to overwrite a previous curated export")
    output.mkdir(parents=True)
    write_json(output / "summary.json", summary)
    with (output / "configurations.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "checksums.json", {p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    return summary


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--audit", type=Path, help="Defaults to STUDY/audit-replay650.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = curate(args.study, args.audit or args.study / "audit-replay650.json", args.output)
    print(json.dumps(dict(status=summary["status"], counts=summary["counts"], window=summary["window"])))


if __name__ == "__main__":
    main()
