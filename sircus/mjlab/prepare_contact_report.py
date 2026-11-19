"""Append versioned contact/transit evidence to a separate read-only UI report."""

import argparse
import copy
import datetime
import json
from pathlib import Path

from .prepare_expanded_report import optional_json, select


def verification(report, audit):
    if audit is None:
        return dict(status="pending", checks=None)
    fingerprint = report.get("contract_fingerprint")
    checks = audit.get("checks")
    matched = bool(fingerprint) and audit.get("contract_fingerprint") == fingerprint
    valid_checks = isinstance(checks, list) and bool(checks) and all(
        isinstance(row, dict) and row.get("passed") is True for row in checks)
    passed = matched and valid_checks and audit.get("failures") == [] and (
        audit.get("status") == "passed" or audit.get("passed") is True)
    return dict(status="passed" if passed else "failed", checks=len(checks) if matched and valid_checks else 0,
        identity_matched=matched)


def configuration(row, condition, radius):
    values = row.get("values")
    xy = row.get("bucket_xy_m")
    if not isinstance(values, list) or len(values) != 5 or not isinstance(xy, list) or len(xy) != 2:
        return None
    return dict(id=row.get("id", "unknown"), condition=condition, board_xyz_m=values[:3],
        tilt_deg=values[3], drop_gap_m=values[4],
        release_center_m=[values[0], values[1], values[2] + radius + values[4]],
        bucket_xy_m=xy, **select(row, ("q10", "q14")))


def compact_contact(study):
    raw = optional_json(study / "report.json")
    if raw is None:
        return None
    value = select(raw, ("version", "status", "scope", "contract_fingerprint", "elapsed_s"))
    contract = optional_json(study / "contract.json")
    if contract is None:
        raise ValueError("Contact evidence requires its scene contract")
    radius = contract["scene"]["ball"]["radius_m"]
    value.update(no_training=True, total_attempts=raw.get("records"), groups=[], configurations=[])
    for group in raw.get("groups", []):
        for family in group.get("family_results", []):
            identity = f'{raw.get("version", "unknown")}:{group["id"]}:{family["family"]}'
            model_label = "Ideal fixed board" if group.get("board_mode") == "kinematic" else "Gravity fix" if group.get("gravity_mode") == "passive_only" else "Original"
            requests = family.get("independent_requests", family.get("attempts", 0) // 2)
            family_label = {"nominal": "identical nominal", "original_final_development": f"old {requests} × 2 resets", "fresh_perturbation": f"new {requests} × 2 resets"}.get(family["family"], family["family"])
            value["groups"].append(dict(id=identity, label=f'{model_label} · {240 * group["substeps"]:,} Hz · {family_label}',
                family=family["family"], stage="confirmation" if raw.get("version") == "sircus-contact-fresh-confirmation-v1" else "diagnostic",
                protocol=select(group, ("board_mode", "gravity_mode", "substeps")),
                **select(family, ("attempts", "released", "captures", "clean", "invalid", "board_bounces")),
                max_joint_excursion_rad=family.get("maximum_integrated_arm_hinge_excursion_rad"),
                reset_order=group.get("reset_order"),
                # The first diagnostic snapshot measured arm AND finger q. Keep
                # its mixed-unit maximum out of a field labeled joint radians.
                excursion_scope="Arm hinge radians only" if family.get("maximum_integrated_arm_hinge_excursion_rad") is not None else "Initial diagnostic includes hinge radians and finger metres; omitted here"))
    source = raw.get("source", {})
    if config := configuration(source, "historical nominal; fixed before this contact diagnosis", radius):
        value["configurations"].append(config)
    try:
        plan = json.loads((study / "plan.json").read_text())
        if not isinstance(plan, list) or not all(isinstance(row, dict) for row in plan):
            raise ValueError("Invalid contact plan")
    except FileNotFoundError:
        plan = []
    seen = {tuple(source.get("values", []))}
    for row in plan:
        key = tuple(row.get("values", []))
        if key in seen:
            continue
        seen.add(key)
        row = {"bucket_xy_m": source.get("bucket_xy_m"), **row}
        if config := configuration(row, row.get("family", "unknown"), radius):
            value["configurations"].append(config)
    notes = optional_json(study / "interpretation.json") or {}
    value.update(select(notes, ("diagnosis", "conclusion", "recommended_protocol", "limitations")))
    return value


def compact_transit(directory):
    report = optional_json(directory / "report.json")
    if report is None:
        return None
    previews = optional_json(directory / "preview.json") or {}
    paths = {row.get("source_id"): row for row in previews.get("paths", [])}
    routes = []
    preview_budget = 1  # One illustrative route; do not duplicate large FK traces in every UI refresh.
    for trial in report.get("trials", []):
        phases = trial.get("phases", [])
        failed = next((p for p in phases if not p.get("passed")), {})
        frames = []
        if trial.get("passed") and preview_budget:
            for i, frame in enumerate(paths.get(trial.get("source_id"), {}).get("frames", [])):
                frames.append(dict(time=float(i) * .15,
                    board_center=frame.get("board_center_m"),
                    **select(frame, ("phase", "board_quaternion_wxyz", "link_positions10", "link_positions14", "ball_center_m"))))
            if frames:
                preview_budget -= 1
        routes.append(dict(id=trial.get("source_id", "unknown"), accepted=trial.get("passed"),
            path_samples=trial.get("checked_joint_configurations"),
            min_clearance_mm=(trial["minimum_clearance_m"] * 1000 if isinstance(trial.get("minimum_clearance_m"), (int, float)) else None),
            bucket_xy_m=trial.get("bucket_xy_m"),
            reason=("Specified serialized route; native geometry only" if trial.get("passed") else
                failed.get("reason") or trial.get("initial", {}).get("reason") or "No verified route"),
            limitations=trial.get("limitations", []),
            frames=frames, preview_time_scope="Illustrative pacing of sampled native FK; no actuator timing claim"))
    passed = sum(row["accepted"] is True for row in routes)
    return dict(version=report.get("version"), scope=f'{report.get("scope", "CPU geometry")} · {passed}/{len(routes)} complete cycles passed', routes=routes)


def prepare(base_display, contact_study, output, transit=None, retest=None, audit=None):
    base_display, contact_study, output = map(Path, (base_display, contact_study, output))
    protected = [contact_study, *([Path(transit)] if transit else []), *([Path(retest)] if retest else [])]
    if output.resolve() == base_display.resolve() or any(output.resolve().is_relative_to(p.resolve()) for p in protected):
        raise ValueError("Output must be separate from source studies and base display")
    base = optional_json(base_display)
    if base is None:
        raise ValueError("A valid existing base report is required")
    value = copy.deepcopy(base)
    contact = compact_contact(contact_study)
    value["contact_repeatability_evaluation"] = contact
    if contact is not None:
        if transit:
            contact["transit"] = compact_transit(Path(transit))
        if retest:
            final = compact_contact(Path(retest))
            if final:
                confirmation_audit = optional_json(Path(retest) / "audit-confirm.json")
                final["verification"] = verification(final, confirmation_audit)
                contact["retest"] = select(final, ("version", "status", "scope", "groups", "conclusion", "verification"))
                contact["groups"].extend(final.get("groups", []))
                previous = {(tuple(row["board_xyz_m"]), row["tilt_deg"], row["drop_gap_m"], tuple(row["bucket_xy_m"])) for row in contact["configurations"]}
                for row in final.get("configurations", []):
                    key = (tuple(row["board_xyz_m"]), row["tilt_deg"], row["drop_gap_m"], tuple(row["bucket_xy_m"]))
                    if key not in previous:
                        contact["configurations"].append(row); previous.add(key)
                contact["total_attempts"] = (contact.get("total_attempts") or 0) + (final.get("total_attempts") or 0)
                if final.get("conclusion"):
                    contact["conclusion"] = final["conclusion"]
        audit_file = Path(audit) if audit else contact_study / "audit-contact-final.json"
        audit_value = optional_json(audit_file)
        contact["verification"] = dict(**verification(contact, audit_value), evidence_file=audit_file.name)
        if contact.get("retest"):
            other = contact["retest"]["verification"]
            contact["verification"]["diagnostic_checks"] = contact["verification"]["checks"]
            contact["verification"]["confirmation_checks"] = other.get("checks")
            if other.get("status") != "passed":
                contact["verification"]["status"] = other.get("status", "pending")
            if contact["verification"]["checks"] is not None and other.get("checks") is not None:
                contact["verification"]["checks"] += other["checks"]
        contact["display_updated_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-display", "contact-study", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--transit", type=Path)
    parser.add_argument("--retest", type=Path)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    prepare(args.base_display, args.contact_study, args.output, args.transit, args.retest, args.audit)
