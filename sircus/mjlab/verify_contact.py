"""Independent CPU audit of saved contact diagnosis evidence, never dynamics.

Run from an auditor snapshot with --source identifying the *executed* native
snapshot. Later bookkeeping changes in a working checkout are not substituted.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .verify_expansion import Audit, digest, sha256

VERSION = "sircus-contact-evidence-audit-v1"
STUDY_VERSION = "sircus-contact-repeatability-study-v1"
SOURCE_ID = "stable-shot-0004711"
GPU_UUID = "GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4"
SCORER_KEYS = ("success", "complete_success", "board_bounce", "whole_ball_entry",
    "quality_score", "route_success", "route_quality", "single_bounce_success",
    "board_contact_episodes", "external_contact_steps", "physics_steps", "entry_class",
    "bucket_contact_before_entry", "rim_band_contact", "exited_after_entry")
NUMERIC_SCORER_KEYS = ("max_retention_s", "closest_opening_distance_m",
    "minimum_mouth_clearance_mm", "peak_bucket_force_before_entry_N")


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def family_stats(rows):
    return dict(attempts=len(rows), released=sum(bool(r["native_release"]) for r in rows),
        captures=sum(bool(r["route_success"]) for r in rows),
        clean=sum(r["route_quality"] == 2 for r in rows),
        board_bounces=sum(bool(r["board_bounce"]) for r in rows),
        invalid=sum(bool(r["invalid"]) for r in rows),
        max_integrated_joint_excursion_rad=max(r["max_integrated_joint_excursion_rad"] for r in rows),
        peak_joint_velocity_rad_s=max(r["peak_integrated_joint_velocity_rad_s"] for r in rows))


def audit_records(spec, report, rows, prior_rows):
    a = Audit()
    payload = dict(spec); fp = payload.pop("fingerprint", None)
    a.check("contract_fingerprint", digest(payload) == fp == report.get("contract_fingerprint"))
    a.check("completed_version", spec.get("version") == report.get("version") == STUDY_VERSION
        and report.get("status") == "completed")
    lookup = {r["id"]: r for r in prior_rows}
    source = lookup.get(SOURCE_ID)
    a.check("exact_original_source", source is not None and source == spec.get("source") == report.get("source")
        and spec.get("source_id") == report.get("source_id") == SOURCE_ID)
    plan = spec["plan"]
    ids = [p["id"] for p in plan]
    a.check("unique_plan_ids", len(set(ids)) == len(ids) and len(ids) > 0)
    families = Counter(p["family"] for p in plan)
    a.check("declared_plan_partition", families.get("nominal", 0) >= 2
        and families.get("original_final_development") == 20 and families.get("fresh_perturbation", 0) > 0
        and set(families) == {"nominal", "original_final_development", "fresh_perturbation"})
    a.check("all_plan_geometry_qualified", all(p.get("geometry_eligible") is True and p.get("geometry_reason") == 0 for p in plan))
    historical = [r for r in prior_rows if r.get("phase") == "final_confirmation" and r.get("confirmation_source_id") == SOURCE_ID]
    retained = [p for p in plan if p["family"] == "original_final_development"]
    a.check("all_original_twenty_preserved", len(historical) == len(retained) == 20
        and [r["id"] for r in historical] == [p["id"] for p in retained]
        and all(all(p[k] == r[k] for k in ("values", "q10", "q14")) and p.get("historical_success") == r["route_success"]
            for p, r in zip(retained, historical)))
    a.check("nominal_exact_saved_joints", source is not None and all(
        all(p[k] == source[k] for k in ("values", "q10", "q14")) for p in plan if p["family"] == "nominal"))
    a.check("fixed_bucket_for_every_query", source is not None and all(p.get("bucket_xy_m") == source["bucket_xy_m"] for p in plan))
    a.check("fresh_seed_noise_protocol", spec.get("seed") == 20261124 and spec.get("noise_std") == [.001, .001, .002, .35, .002])
    if source is not None:
        rng = np.random.default_rng(20261124)
        expected = [np.asarray(source["values"]) + rng.normal(0, [.001, .001, .002, .35, .002])
                    for _ in range(families.get("fresh_perturbation", 0))]
        fresh = [p for p in plan if p["family"] == "fresh_perturbation"]
        a.check("independently_regenerated_fresh_requests", len(fresh) > 0 and all(
            p["id"] == f"fresh-{i:02d}" and np.array_equal(p["values"], e) and "historical_success" not in p
            for i, (p, e) in enumerate(zip(fresh, expected))))
    variants = spec["variants"]
    expected_variants = [dict(id=f"{mode}-{gravity}-{s}", board_mode=mode, gravity_mode=gravity, substeps=s)
        for mode, gravity in (("articulated", "legacy"), ("articulated", "passive_only"), ("kinematic", "passive_only"))
        for s in (8, 16, 32)]
    a.check("full_declared_factorial", variants == expected_variants and spec.get("reset_repeats") == 2)
    expected_keys = Counter((v["id"], reset, p["id"]) for v in variants for reset in (0, 1) for p in plan)
    a.check("no_missing_duplicate_or_extra_outcomes", Counter((r["variant"], r["reset"], r["source_plan_id"]) for r in rows) == expected_keys)
    a.check("exact_record_denominator", report.get("records") == len(rows) == len(variants) * 2 * len(plan))
    plans = {p["id"]: p for p in plan}; protocols = {v["id"]: {k: v[k] for k in ("board_mode", "gravity_mode", "substeps")} for v in variants}
    bad = []
    for row in rows:
        p = plans.get(row.get("source_plan_id"))
        if p is None or row.get("id") != f'{row["variant"]}-reset-{row["reset"]}-{p["id"]}' or row.get("family") != p["family"]:
            bad.append(row["id"]); continue
        if not all(row.get(k) == p[k] for k in ("bucket_xy_m", "values", "q10", "q14")) or row.get("contact_protocol") != protocols.get(row["variant"]):
            bad.append(row["id"])
    a.check("native_requests_and_cached_joint_identity", not bad, bad[:10])
    a.check("native_release_and_scalar_audit_accounting", all(r.get("native_release") == (not bool(r["invalid"]))
        and r.get("scalar_score_audit_passed") is True and not r["invalid"] for r in rows))
    a.check("unchanged_complete_capture_definition", all(
        r["complete_success"] == r["route_success"]
        and (not r["route_success"] or (r["success"] and r["board_bounce"] and r["whole_ball_entry"]
            and r["max_retention_s"] >= 1 - 1e-9 and r["external_contact_steps"] == 0))
        and r["route_quality"] == (r["quality_score"] if r["route_success"] else 0)
        and r["scoring_version"] == "sircus-isaac-fixed-shot-v5" and r["route_version"] == "sircus-board-route-v1" for r in rows))
    groups = {g["id"]: g for g in report["groups"]}
    a.check("all_reported_variants_present_once", len(groups) == len(report["groups"]) == len(variants) and set(groups) == set(protocols))
    for variant in variants:
        key = variant["id"]; group = groups.get(key, {})
        selected = [r for r in rows if r["variant"] == key]
        a.check(key + "_group_counts", group.get("attempts") == len(selected)
            and group.get("captures") == sum(r["route_success"] for r in selected)
            and all(group.get(k) == v for k, v in variant.items()))
        claims = {f["family"]: f for f in group.get("family_results", [])}
        for family in families:
            records = [r for r in selected if r["family"] == family]
            a.check(key + "_" + family + "_statistics", len(records) == 2 * families[family]
                and all(claims.get(family, {}).get(k) == value for k, value in family_stats(records).items()))
    return a


def replay_scores(audit, scene, rows, arrays, variant, reset):
    from .contract import ScoreStream, FORCE_NAMES
    shape = all(k in arrays and arrays[k].shape == (960, len(rows), width)
        and np.isfinite(arrays[k]).all() for k, width in (("position_m", 3), ("velocity_m_s", 3), ("forces_N", 6)))
    audit.check(f"{variant}-{reset}_saved_array_shapes", shape)
    if not shape:
        return
    audit.check(f"{variant}-{reset}_nonnegative_attributed_forces", bool((arrays["forces_N"] >= 0).all()))
    mismatches = []
    for index, row in enumerate(rows):
        score = ScoreStream(scene, row["bucket_xy_m"], row["board_normal"])
        for p, v, f in zip(arrays["position_m"][:, index], arrays["velocity_m_s"][:, index], arrays["forces_N"][:, index]):
            score.step(p, v, dict(zip(FORCE_NAMES, f)))
            if score.finished:
                break
        actual = score.result()
        numeric_matches = all((actual[k] is None and row[k] is None) or
            (actual[k] is not None and row[k] is not None and np.isclose(actual[k], row[k], atol=1e-3, rtol=1e-5))
            for k in NUMERIC_SCORER_KEYS)
        if any(actual[k] != row[k] for k in SCORER_KEYS) or not numeric_matches:
            mismatches.append(row["id"])
    audit.check(f"{variant}-{reset}_independent_scalar_rescore", not mismatches, mismatches[:10])


def verify(study, source, prior_study, assets, gravity_report, *, rescore=True):
    study, source, prior_study, assets = map(Path, (study, source, prior_study, assets))
    spec, report = read(study / "contract.json"), read(study / "report.json")
    rows, prior_rows = lines(study / "results.jsonl"), lines(prior_study / "shots.jsonl")
    audit = audit_records(spec, report, rows, prior_rows)
    audit.check("standalone_plan_identity", read(study / "plan.json") == spec["plan"])
    audit.check("unchanged_original_scene_and_prior_files", read(prior_study / "contract.json")["scene"] == spec["scene"]
        and sha256(prior_study / "contract.json") == spec["prior_contract_sha256"]
        and sha256(prior_study / "shots.jsonl") == spec["prior_records_sha256"])
    provenance = read(study / "execution-provenance.json")
    manifest = provenance["source_sha256"]
    required = {"sircus/mjlab/" + n for n in ("contact_model.py", "contact_env.py", "contact_study.py", "model.py", "contract.py", "learning_score.py", "bank_study.py", "bank_geometry.py")}
    required |= {"sircus/isaac/" + n for n in ("geometry.py", "rl_contract.py", "entry_quality.py", "board_route.py")}
    audit.check("critical_executed_sources_manifested", required <= set(manifest))
    audit.check("explicit_4090_frozen_execution", provenance.get("gpu_uuid") == GPU_UUID
        and provenance.get("source_snapshot_frozen") is True and Path(provenance.get("source_directory", "")).resolve() == source.resolve())
    for name, expected in manifest.items():
        path = (source / name).resolve()
        audit.check("executed_source:" + name, path.is_relative_to(source.resolve()) and path.is_file() and sha256(path) == expected)
    parent_report = read(prior_study / "report.json")
    prior_contract = read(prior_study / "contract.json")
    parent_study = Path(prior_contract["parent_study_directory"])
    original_provenance = read(parent_study / "artifact_provenance.json")
    audit.check("original_parent_provenance_association", original_provenance["checkpoint_sha256"] == prior_contract["parent_checkpoint_sha256"]
        and original_provenance["contract_fingerprint"] == prior_contract["parent_contract_fingerprint"])
    # Physics/scorer source files used by the old study remain byte-identical;
    # new intervention modules are separately pinned above.
    for key in ("model.py", "learning_score.py", "contract.py"):
        matched = [v for k, v in parent_report["source_sha256"].items() if Path(k).name == key]
        actual = [v for k, v in manifest.items() if Path(k).name == key]
        audit.check("historical_source_preserved:" + key, len(matched) == len(actual) == 1 and matched == actual)
    auditor_root = Path(__file__).resolve().parents[2]
    for name, expected in original_provenance["source_sha256"].items():
        audit.check("historical_shared_source:" + name, manifest.get(name) == expected)
        audit.check("auditor_shared_source:" + name, sha256(auditor_root / name) == expected)
    for name, expected in original_provenance["asset_sha256"].items():
        path = (assets.parent / name).resolve()
        audit.check("original_asset:" + name, path.is_relative_to(assets.parent.resolve()) and path.is_file() and sha256(path) == expected)
    gravity = read(gravity_report)
    audit.check("independent_gravity_report_identity", gravity.get("status") == "passed"
        and gravity.get("model_source_sha256") == manifest.get("sircus/mjlab/model.py")
        and gravity.get("contract_sha256") == spec["prior_contract_sha256"]
        and gravity.get("records_sha256") == spec["prior_records_sha256"]
        and len(gravity.get("cases", [])) == 3 and all(c.get("passed") is True for c in gravity["cases"]))
    for name, expected in gravity["assets_sha256"].items():
        path = Path(name)
        audit.check("native_asset:" + path.name, path.is_relative_to(assets.parent) and path.is_file() and sha256(path) == expected)
    # Static full-pose FK, joint limits and collision checks independently verify
    # every distinct requested endpoint, including freshly solved continuation.
    from .verify_stable import audit_strict_endpoints
    audit_strict_endpoints(audit, assets, spec, [dict(p, native_release=True) for p in spec["plan"]])
    artifacts = {n.name: sha256(n) for n in study.iterdir() if n.suffix in (".json", ".jsonl", ".npz") and not n.name.startswith("audit-")}
    records_by_key = {(r["variant"], r["reset"], r["source_plan_id"]): r for r in rows}
    score_count = 0
    for variant in spec["variants"]:
        key = variant["id"]; saved = []
        paired_rows = []
        for reset in (0, 1):
            file = study / f"{key}-reset-{reset}.npz"
            with np.load(file, allow_pickle=False) as data:
                arrays = {name: data[name].copy() for name in data.files}
            selected = [records_by_key[(key, reset, p["id"])] for p in spec["plan"]]
            if rescore:
                replay_scores(audit, spec["scene"], selected, arrays, key, reset); score_count += len(selected)
            saved.append(arrays); paired_rows.append(selected)
        difference = np.abs(saved[0]["position_m"] - saved[1]["position_m"])
        per_query = difference.max(axis=(0, 2))
        claimed = next(g for g in report["groups"] if g["id"] == key)["reset_order"]
        expected = dict(max_position_difference_m=float(per_query.max()), identical_trajectories=int((per_query == 0).sum()),
            queries=len(per_query), capture_disagreements=sum(x["route_success"] != y["route_success"] for x, y in zip(*paired_rows)))
        audit.check(key + "_reported_full_horizon_reset_comparison", claimed == expected)
    failures = [c for c in audit.checks if not c["passed"]]
    return dict(version=VERSION, status="passed" if not failures else "failed", checks=audit.checks,
        checks_count=len(audit.checks), failures=failures, independently_rescored_throws=score_count,
        contract_fingerprint=spec["fingerprint"], executed_source_directory=str(source),
        executed_contact_env_sha256=manifest["sircus/mjlab/contact_env.py"],
        evidence_sha256=artifacts, gravity_report_sha256=sha256(gravity_report),
        limitations=["Stored-array scalar rescoring and static FK are independent; native ball dynamics are not rerun.",
            "The first diagnostic's excursion field includes finger translation as well as hinge rotation; do not label its maximum as purely radians.",
            "Full-horizon reset differences include post-terminal movement; capture disagreements are reported separately.",
            "This is development diagnosis; no checkpoint was trained or promoted."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "source", "prior-study", "assets", "gravity-report", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.study, args.source, args.prior_study, args.assets, args.gravity_report)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({k: result[k] for k in ("status", "checks_count", "independently_rescored_throws")}))
    if result["failures"]:
        raise RuntimeError("Contact evidence audit failed: " + str([x["name"] for x in result["failures"]]))


if __name__ == "__main__":
    main()
