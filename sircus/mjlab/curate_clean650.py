"""Publish byte-bound clean650 metrics and requested-shot parameters only.

This stdlib-only exporter requires a completed independent audit. It verifies
its input bytes but does not replace native-contact rescoring or rerun physics.
Raw joints, sampled poses, link arrays and point trajectories never enter output.
"""

from __future__ import annotations

from collections import Counter
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re


VERSION = "sircus-clean650-curation-v1"
STUDY_VERSION = "sircus-clean650-height-tilt-gap-v1"
AUDIT_VERSION = "sircus-clean650-independent-audit-v1"
FINE_STUDY_VERSION = "sircus-clean650-height-tilt-gap-fine-v2"
RAISED_STUDY_VERSION = "sircus-clean650-height-tilt-gap-raised-v3"
VERSION_PAIRS = {STUDY_VERSION:AUDIT_VERSION,FINE_STUDY_VERSION:"sircus-clean650-independent-audit-v2",
                 RAISED_STUDY_VERSION:"sircus-clean650-independent-audit-v3"}
STEPS = (64, 128)
PHASES = ("controls_pre", "coarse", "refinement", "verification", "controls_post")
STAT_FIELDS = ("attempts", "native_releases", "captures", "invalid", "clean", "whole_entries",
               "max_retention_s", "rim_assisted", "geometry_rejections")
SCOPE = ("Fixed650mm DEVELOPMENT search and matched nominal engineering verification, using "
         "calibrated simulated contacts and held endpoints. Historical results keep their original "
         "scene/scoring definitions. No noise qualification, new PPO, held-out generalization, "
         "grasp/transit, actuator, perception or physical validation.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            value.update(chunk)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read(path):
    def invalid(value):
        raise ValueError("Non-finite JSON constant: "+value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def identifier(value):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]+", value), "Unsafe publication identifier")
    return value


def number(value, *, optional=False):
    if optional and value is None:
        return None
    require(type(value) in (float, int) and math.isfinite(value), "Expected finite numerical metric")
    return value


def boolean(value):
    require(type(value) is bool, "Expected explicit boolean")
    return value


def hash_map(value):
    require(isinstance(value, dict) and bool(value), "Nonempty hash map required")
    for name, expected in value.items():
        require(isinstance(name, str) and not Path(name).is_absolute() and ".." not in Path(name).parts,
                "Unsafe provenance filename")
        require(isinstance(expected, str) and re.fullmatch(r"[a-f0-9]{64}", expected), "Invalid SHA256")
    return dict(value)


def relative_file(root, name):
    require(isinstance(name, str) and not Path(name).is_absolute() and ".." not in Path(name).parts, "Unsafe evidence filename")
    result = (root/name).resolve()
    require(result.is_relative_to(root) and result.is_file(), "Missing or escaping evidence file")
    return result


def captured(row):
    return bool(row.get("native_release") and row.get("route_success") and not row.get("invalid") and not row.get("invalid_physics"))


def statistics(rows):
    return dict(attempts=len(rows), native_releases=sum(bool(r.get("native_release")) for r in rows),
                captures=sum(captured(r) for r in rows), invalid=sum(bool(r.get("invalid") or r.get("invalid_physics")) for r in rows),
                clean=sum(captured(r) and r.get("route_quality") == 2 for r in rows),
                whole_entries=sum(bool(r.get("native_release") and r.get("whole_ball_entry")) for r in rows),
                max_retention_s=max((float(r.get("max_retention_s", 0.)) for r in rows), default=0.),
                rim_assisted=sum(captured(r) and r.get("rim_band_contact") is True for r in rows),
                geometry_rejections=sum(not bool(r.get("native_release")) for r in rows))


def checked_statistics(declared, rows):
    result = statistics(rows)
    for key, expected in result.items():
        require(declared.get(key) == expected, "Outcome accounting differs: "+key)
    return result


def verified_inputs(study, audit_path):
    study, audit_path = Path(study).resolve(), Path(audit_path).resolve()
    report, spec, audit = read(study/"report.json"), read(study/"contract.json"), read(audit_path)
    require(report.get("version") == spec.get("version") and spec.get("version") in VERSION_PAIRS, "Unexpected study version")
    require(report.get("status") == report.get("phase") == "completed", "Study is not completed")
    require(spec.get("evidence_kind") == "native_calibrated_clean650_sweep", "Unexpected evidence kind")
    require(audit.get("version") == VERSION_PAIRS[spec['version']] and audit.get("status") == "passed", "Full independent audit must pass")
    checks = audit.get("checks")
    require(isinstance(checks, list) and bool(checks) and type(audit.get("checks_count")) is int
            and audit["checks_count"] == len(checks)
            and all(isinstance(c, dict) and c.get("passed") is True and isinstance(c.get("label"), str) for c in checks),
            "Nonempty complete strictly passed audit checks required")
    require(audit.get("failures") == [], "Independent audit failures present")
    require(audit.get("report_sha256") == sha(study/"report.json"), "Audit/report bytes differ")
    require(audit.get("evidence_manifest_sha256") == sha(study/"evidence-manifest.json"), "Audit/manifest bytes differ")
    unsigned = dict(spec); fp = unsigned.pop("fingerprint", None)
    require(fp == digest(unsigned) == report.get("contract_fingerprint") == audit.get("contract_fingerprint"), "Contract fingerprint binding differs")
    manifest = hash_map(read(study/"evidence-manifest.json"))
    require(report.get("evidence_sha256") == manifest and manifest.get("contract.json") == sha(study/"contract.json"), "Report/contract manifest binding differs")
    for name, expected in manifest.items():
        require(sha(relative_file(study, name)) == expected, "Changed evidence bytes: "+name)
    actual = {str(p.relative_to(study)) for p in study.rglob("*") if p.is_file()
              and p.name not in ("report.json", "evidence-manifest.json") and not p.name.startswith("audit")}
    require(set(manifest) == actual, "Evidence manifest does not cover every saved artifact")
    require(report["settings"] == spec["settings"], "Executed and published settings differ")
    require(report.get("gate") == audit.get("gate"), "Independent gate differs")
    rows = [json.loads(line) for line in (study/"records.jsonl").read_text().splitlines() if line]
    require(len(rows) == report["counts"]["attempts"] == audit.get("requested_trials"), "Requested trial accounting differs")
    require(sum(bool(r.get("native_release")) for r in rows) == audit.get("native_releases"), "Native release accounting differs")
    return study, report, spec, audit, manifest, rows


def settings(value,version=STUDY_VERSION):
    require(version in VERSION_PAIRS,"Unexpected frozen profile")
    fine=version==FINE_STUDY_VERSION
    raised=version==RAISED_STUDY_VERSION
    expected = dict(coarse_requests=28 if raised else 76 if fine else 361, refinement_seeds=3, refinement_max_requests=81,
                    verification_sources=5, verification_dedup_decimals=9, repeats_per_resolution=3, substeps=list(STEPS), max_requests=163 if raised else 211 if fine else 496,
                    minimum_clearance_m=.005, max_drift_350ms_m=.005)
    for key, target in expected.items():
        require(value.get(key) == target, "Unexpected frozen protocol setting: "+key)
    grid = value["grid"]
    require(grid.get("heights_m") == ([.40,.42,.44] if raised else [.375,.38,.385] if fine else [.22,.26,.30,.34,.38,.42])
            and grid.get("tilts_deg") == ([26.,27.,28.] if raised else [27.,27.5,28.,28.5,29.] if fine else [16.,20.,24.,28.,32.,36.])
            and grid.get("gaps_m") == ([.3268,.3468,.3668] if raised else [.38,.3868,.39,.395,.40] if fine else [.20,.24,.28,.32,.36,.40,.44,.48,.52,.56])
            and grid.get("reference") == [.38,28.,.3868], "Unexpected frozen grid")
    offsets = (dict(height_m=[-.005,0.,.005],tilt_deg=[-.5,0.,.5],gap_m=[-.0025,0.,.0025]) if raised else
               dict(height_m=[-.005,0.,.005],tilt_deg=[-.25,0.,.25],gap_m=[-.0025,0.,.0025]) if fine else
               dict(height_m=[-.02,0.,.02], tilt_deg=[-2.,0.,2.], gap_m=[-.02,0.,.02]))
    require(value.get("refinement_offsets") == offsets, "Unexpected adaptive offsets")
    return dict(**expected, grid={k:list(grid[k]) for k in ("heights_m", "tilts_deg", "gaps_m", "reference")},
                refinement_offsets=offsets, nominal_only=True, new_ppo=False, noise_study=False)


def geometry(group):
    action, bucket = group["values"], group["bucket_xy_m"]
    require(isinstance(action, list) and len(action) == 5 and isinstance(bucket, list) and len(bucket) == 2, "Incomplete configured geometry")
    for value in [*action, *bucket]: number(value)
    require(group["board_xy_m"] == action[:2] and group["board_height_m"] == action[2]
            and group["tilt_deg"] == action[3] and group["gap_m"] == action[4], "Configured geometry fields disagree")
    require(abs(group["distance_m"]-math.dist(bucket, action[:2])) <= 1e-12, "Configured bucket distance differs")
    require(type(group["eligible"]) is bool, "Explicit endpoint eligibility required")
    hash_map({"source_record":group["source_record_sha256"]})
    return dict(id=identifier(group["id"]), phase=identifier(group["phase"]), role=identifier(group["role"]),
                source_record_id=identifier(group["source_record_id"]), source_record_sha256=group["source_record_sha256"],
                board_xy_m=list(action[:2]), bucket_xy_m=list(bucket), board_height_m=action[2], tilt_deg=action[3], gap_m=action[4],
                release_center_height_m=number(group["release_center_height_m"]), distance_m=number(group["distance_m"]),
                eligible=group["eligible"])


def checked_gate(raw):
    result = dict(status=identifier(raw["status"]),
                  positive_control=boolean(raw["positive_control"]), negative_control=boolean(raw["negative_control"]),
                  physics_valid=boolean(raw["physics_valid"]), oracle_agreement=boolean(raw["oracle_agreement"]))
    require(result["status"] in ("passed", "failed", "not_run"), "Unexpected control status")
    return result


def curate(study, audit_path, output):
    study, report, spec, audit, manifest, records = verified_inputs(study, audit_path)
    output = Path(output).resolve()
    require(not output.is_relative_to(study) and not output.exists(), "New publication directory outside evidence required")
    declared = settings(spec["settings"],spec['version'])
    require(12 <= len(records) <= declared["max_requests"], "Finite attempt budget exceeded")
    groups = {g["id"]:g for g in report["groups"]}
    require(len(groups) == len(report["groups"]), "Duplicate group identity")
    configurations = {key:geometry(value) for key, value in groups.items()}
    require(len({r["id"] for r in records}) == len(records), "Duplicate requested trial ID")
    rows = []
    for record in records:
        cfg = configurations[record["group_id"]]
        require(record["values"] == groups[record["group_id"]]["values"] and record["bucket_xy_m"] == cfg["bucket_xy_m"], "Actual request geometry differs from its group")
        require(record["phase"] in PHASES and record["substeps"] in STEPS, "Unexpected actual phase or resolution")
        require(record["role"] == cfg["role"] and record["eligible"] is cfg["eligible"], "Requested group identity differs")
        radius = number(spec["scene"]["ball"]["radius_m"])
        require(abs(cfg["release_center_height_m"]-(cfg["board_height_m"]+cfg["gap_m"]+radius)) <= 1e-12, "Configured release height differs")
        invalid = record.get("invalid")
        require(invalid is None or isinstance(invalid, str), "Unexpected invalid-reason type")
        if invalid is not None: identifier(invalid)
        native = boolean(record["native_release"])
        require(type(record["repeat_index"]) is int and 0 <= record["repeat_index"] <= 2, "Invalid nominal repeat index")
        margin = number(record.get("minimum_mouth_clearance_mm"), optional=True)
        retention = number(record["max_retention_s"])
        require(0 <= retention <= 1.+1e-10, "Invalid retained duration")
        rim = record.get("rim_band_contact")
        require(rim is None or type(rim) is bool, "Unexpected rim observation")
        rows.append(dict(id=identifier(record["id"]), phase=record["phase"], group_id=identifier(record["group_id"]),
            role=record["role"], substeps=record["substeps"], repeat_index=record["repeat_index"],
            source_record_id=cfg["source_record_id"], board_x_m=cfg["board_xy_m"][0], board_y_m=cfg["board_xy_m"][1],
            board_height_m=cfg["board_height_m"], tilt_deg=cfg["tilt_deg"], release_gap_m=cfg["gap_m"],
            release_ball_center_height_m=cfg["release_center_height_m"], bucket_x_m=cfg["bucket_xy_m"][0], bucket_y_m=cfg["bucket_xy_m"][1],
            distance_m=cfg["distance_m"], eligible=cfg["eligible"], native_release=native,
            capture=captured(record), clean=captured(record) and record.get("route_quality") == 2,
            rim_band_contact=rim, whole_ball_entry=boolean(record["whole_ball_entry"]), invalid=invalid,
            invalid_physics=boolean(record["invalid_physics"]), max_retention_s=retention, minimum_mouth_clearance_mm=margin))
    counts = checked_statistics(report["counts"], records)
    phase_counts = dict(Counter(r["phase"] for r in records))
    substep_counts = dict(Counter(str(r["substeps"]) for r in records))
    require(report["counts"]["phase_counts"] == phase_counts and report["counts"]["substep_counts"] == substep_counts,
            "Phase/resolution accounting differs")
    control_gates = {key:checked_gate(report["control_gates"][key]) for key in ("controls_pre", "controls_post")}
    if control_gates["controls_pre"]["status"] != "passed":
        require(phase_counts == {"controls_pre":12} and control_gates["controls_post"]["status"] == "not_run", "Search ran despite failed initial controls")
    else:
        require(phase_counts.get("controls_pre") == phase_counts.get("controls_post") == 12 and phase_counts.get("coarse") == declared['coarse_requests']
                and phase_counts.get("refinement",0) <= 81 and phase_counts.get("refinement",0)%27 == 0
                and phase_counts.get("verification",0) <= 30 and phase_counts.get("verification",0)%6 == 0,
                "Adaptive or control request budget differs")
    controls = []
    for control in report["controls"]:
        require(control["phase"] in ("controls_pre", "controls_post") and control["substeps"] in STEPS, "Invalid control cohort")
        selected = [r for r in records if r["phase"] == control["phase"] and r["group_id"] == control["group_id"] and r["substeps"] == control["substeps"]]
        controls.append(dict(phase=control["phase"], group_id=identifier(control["group_id"]), role=identifier(control["role"]),
                             substeps=control["substeps"], stats=checked_statistics(control["stats"], selected)))
    require(len(controls) == 8, "Every before/after control cohort must remain visible")
    lookup = {r["id"]:r for r in records}
    top_settings = []
    for result in report["top_settings"]:
        source = lookup[result["id"]]
        require(source["phase"] in ("coarse", "refinement") and result["group_id"] == source["group_id"], "Top setting is not an actual development record")
        top_settings.append(dict(record_id=identifier(source["id"]), group_id=identifier(source["group_id"]),
            configuration=configurations[source["group_id"]], substeps=source["substeps"], stats=checked_statistics(result["stats"], [source]),
            minimum_mouth_clearance_mm=number(source.get("minimum_mouth_clearance_mm"), optional=True),
            closest_opening_distance_m=number(source.get("closest_opening_distance_m"), optional=True)))
    verified = []
    for result in report["verification"]:
        cfg = configurations[result["id"]]
        source = lookup[cfg["source_record_id"]]
        selected = [r for r in records if r["group_id"] == cfg["id"]]
        require(cfg["source_record_sha256"] == digest(source), "Verification source provenance differs")
        by_step = {str(s):checked_statistics(result["by_substeps"][str(s)], [r for r in selected if r["substeps"] == s]) for s in STEPS}
        qualified = result["qualification"]
        q = {k:boolean(qualified[k]) for k in ("clean_both", "physics_valid", "oracle_agreement", "qualified")}
        q.update(status=identifier(qualified["status"]), max_drift_m=number(qualified.get("max_drift_m"), optional=True))
        require(q["qualified"] == (q["status"] == "qualified"), "Qualification status differs")
        if q["qualified"]:
            require(all(by_step[str(s)]["attempts"] == by_step[str(s)]["clean"] == 3 for s in STEPS)
                    and captured(source) and source["route_quality"] == 2 and q["clean_both"] and q["physics_valid"] and q["oracle_agreement"]
                    and q["max_drift_m"] is not None and 0 <= q["max_drift_m"] <= .005
                    and all(g["status"] == "passed" for g in control_gates.values()), "Qualified setting lacks matched clean/control evidence")
        sensitivity = result["sensitivity"]
        require(type(sensitivity["paired_attempts"]) is int and type(sensitivity["capture_agreement"]) is int
                and 0 <= sensitivity["capture_agreement"] <= sensitivity["paired_attempts"] <= 3, "Invalid paired sensitivity counts")
        paired_drift = number(sensitivity["nominal_position_delta_350ms_max_m"], optional=True)
        require(paired_drift is None or paired_drift >= 0, "Invalid paired sensitivity drift")
        require(paired_drift == q["max_drift_m"], "Qualification and sensitivity drift differ")
        verified.append(dict(configuration=cfg, by_substeps=by_step, qualification=q,
            sensitivity=dict(paired_attempts=sensitivity["paired_attempts"], capture_agreement=sensitivity["capture_agreement"],
                             nominal_position_delta_350ms_max_m=paired_drift)))
    gate = report["gate"]
    selected_gate = dict(status=identifier(gate["status"]), controls_pre_passed=boolean(gate["controls_pre_passed"]),
                         controls_post_passed=boolean(gate["controls_post_passed"]), qualified_settings=gate["qualified_settings"],
                         more_ppo_started=boolean(gate["more_ppo_started"]))
    require(selected_gate["more_ppo_started"] is False and selected_gate["qualified_settings"] == sum(v["qualification"]["qualified"] for v in verified),
            "Final qualification count or PPO boundary differs")
    require(type(selected_gate["qualified_settings"]) is int and 0 <= selected_gate["qualified_settings"] <= 5
            and selected_gate["controls_pre_passed"] == (control_gates["controls_pre"]["status"] == "passed")
            and selected_gate["controls_post_passed"] == (control_gates["controls_post"]["status"] == "passed"), "Final control/count fields differ")
    expected_gate = ("blocked_controls" if not selected_gate["controls_pre_passed"] or not selected_gate["controls_post_passed"]
                     else "passed" if selected_gate["qualified_settings"] else "failed")
    require(selected_gate["status"] == expected_gate, "Final gate status differs")
    auditor = hash_map({"independent_auditor":audit["auditor_sha256"]})
    summary = dict(version=VERSION, study_version=spec['version'], study_id=identifier(study.name), status="completed_independently_audited",
        scope=SCOPE, settings=declared, counts=dict(**counts, phase_counts=phase_counts, substep_counts=substep_counts), gate=selected_gate,
        control_gates=control_gates, controls=controls, top_settings=top_settings, verification=verified,
        declared_groups=len(groups), executed_groups=len({r["group_id"] for r in records}),
        provenance=dict(contract_fingerprint=spec["fingerprint"], contract_sha256=sha(study/"contract.json"), report_sha256=sha(study/"report.json"),
                        independent_audit_sha256=sha(audit_path), independent_audit_version=VERSION_PAIRS[spec['version']],
                        independent_auditor_sha256=auditor["independent_auditor"], independent_audit_checks=audit["checks_count"],
                        evidence_manifest_sha256=sha(study/"evidence-manifest.json"), evidence_sha256=manifest,
                        executed_source_sha256=hash_map(spec["source_sha256"]), asset_sha256=hash_map(spec["asset_sha256"]),
                        calibration_sha256=hash_map({k:spec["calibration"][k] for k in ("calibration_sha256", "verification_sha256", "score_config_sha256")})),
        privacy="Configured shot parameters, outcome metrics and hashes only; no joints, links, sampled poses, point trajectories, credentials or host paths.")
    identity = audit.get("auditor_identity", {})
    dependencies = identity.get("dependencies_sha256")
    if dependencies:
        summary["provenance"]["auditor_dependencies_sha256"] = hash_map(dependencies)
    output.mkdir(parents=True)
    write_json(output/"summary.json", summary)
    with (output/"requested-shots.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    write_json(output/"checksums.json", {p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    return summary


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--audit", type=Path, help="Defaults to STUDY/audit-clean650.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = curate(args.study, args.audit or args.study/"audit-clean650.json", args.output)
    print(json.dumps(dict(status=result["status"], counts=result["counts"], gate=result["gate"])))


if __name__ == "__main__": main()
