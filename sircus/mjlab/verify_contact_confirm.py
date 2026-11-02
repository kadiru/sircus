"""Independent CPU verification of fresh contact confirmation and singleton checks."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .verify_contact import Audit, digest, sha256, read, lines, replay_scores, SOURCE_ID, GPU_UUID

VERSION = "sircus-contact-confirmation-evidence-audit-v1"


def audit_records(spec, report, rows, original):
    a = Audit()
    payload = dict(spec); fp = payload.pop("fingerprint", None)
    a.check("contract_fingerprint", digest(payload) == fp == report.get("contract_fingerprint"))
    a.check("completed_version", spec.get("version") == report.get("version") == "sircus-contact-fresh-confirmation-v1"
        and report.get("status") == "completed")
    a.check("unchanged_original_nominal_source", original == spec.get("source") == report.get("source")
        and original["id"] == spec.get("source_id") == SOURCE_ID)
    a.check("new_seed_and_same_noise", spec.get("seed") == 20261125 and spec.get("noise_std") == [.001, .001, .002, .35, .002])
    plan = spec["plan"]; nominal = [p for p in plan if p["family"] == "nominal"]
    fresh = [p for p in plan if p["family"] == "fresh_perturbation"]
    a.check("nominal_and_fresh_partition", len(nominal) == 16 and len(fresh) == 100 and len(plan) == 116
        and len({p["id"] for p in plan}) == 116 and plan == nominal + fresh)
    a.check("unchanged_nominal_requests_and_joints", all(p["values"] == original["values"]
        and p["q10"] == original["q10"] and p["q14"] == original["q14"] for p in nominal))
    rng = np.random.default_rng(20261125)
    regenerated = [np.asarray(original["values"]) + rng.normal(0, [.001, .001, .002, .35, .002]) for _ in range(100)]
    a.check("independently_regenerated_100_requests", len(fresh) == 100 and all(
        p["id"] == f"fresh-{i:03d}" and np.array_equal(p["values"], v)
        for i, (p, v) in enumerate(zip(fresh, regenerated))))
    old_rng = np.random.default_rng(20261124)
    old = {tuple(np.asarray(original["values"]) + old_rng.normal(0, [.001, .001, .002, .35, .002])) for _ in range(20)}
    a.check("fresh_requests_disjoint_from_diagnostic", old.isdisjoint(tuple(p["values"]) for p in fresh))
    a.check("all_endpoint_rejects_retained_and_zero", all(p.get("geometry_eligible") is True and p.get("geometry_reason") == 0 for p in plan))
    expected_variants = [dict(id=f"articulated-{gravity}-{steps}", board_mode="articulated", gravity_mode=gravity, substeps=steps)
        for gravity, steps in (("legacy", 8), ("passive_only", 32), ("passive_only", 64))]
    a.check("matched_legacy_and_corrected_timestep_protocol", spec["variants"] == expected_variants and spec["reset_repeats"] == 2)
    regular = [r for r in rows if r["reset"] != "singleton"]
    singles = [r for r in rows if r["reset"] == "singleton"]
    expected = Counter((v["id"], reset, p["id"]) for v in expected_variants for reset in (0, 1) for p in plan)
    a.check("all_696_paired_outcomes_present_once", Counter((r["variant"], r["reset"], r["source_plan_id"]) for r in regular) == expected)
    singleton_keys = {(v["id"], p["id"]) for v in expected_variants[1:] for p in (plan[0], plan[len(nominal)])}
    a.check("four_separate_singleton_diagnostics", len(singles) == 4 and {(r["variant"], r["source_plan_id"]) for r in singles} == singleton_keys)
    a.check("exact_700_total_denominator", report["records"] == len(rows) == 700)
    lookup = {p["id"]: p for p in plan}; variants = {v["id"]: v for v in expected_variants}
    bad = []
    for row in rows:
        p = lookup.get(row["source_plan_id"]); v = variants.get(row["variant"])
        if p is None or v is None:
            bad.append(row["id"]); continue
        suffix = f'singleton-{p["id"]}' if row["reset"] == "singleton" else f'reset-{row["reset"]}-{p["id"]}'
        if row["id"] != f'{row["variant"]}-{suffix}' or row["bucket_xy_m"] != original["bucket_xy_m"]:
            bad.append(row["id"]); continue
        if not all(row[k] == p[k] for k in ("values", "q10", "q14")) or row["contact_protocol"] != {k: v[k] for k in ("board_mode", "gravity_mode", "substeps")}:
            bad.append(row["id"])
    a.check("same_requests_and_joint_branches_every_protocol", not bad, bad[:10])
    a.check("release_and_scalar_parity_every_record", all(r["native_release"] == (not bool(r["invalid"]))
        and not r["invalid"] and r.get("scalar_score_audit_passed") is True for r in rows))
    groups = {g["id"]: g for g in report["groups"]}
    a.check("three_unique_reported_groups", len(groups) == len(report["groups"]) == 3 and set(groups) == set(variants))
    for v in expected_variants:
        records = [r for r in regular if r["variant"] == v["id"]]; group = groups.get(v["id"], {})
        a.check(v["id"] + "_group_counts", group.get("attempts") == len(records) == 232
            and group.get("captures") == sum(r["route_success"] for r in records))
        claims = {f["family"]: f for f in group.get("family_results", [])}
        for family, independent in (("nominal", 16), ("fresh_perturbation", 100)):
            rr = [r for r in records if r["family"] == family]
            expected_counts = dict(attempts=2 * independent, independent_requests=independent,
                captures=sum(r["route_success"] for r in rr), clean=sum(r["route_quality"] == 2 for r in rr),
                invalid=sum(bool(r["invalid"]) for r in rr), board_bounces=sum(r["board_bounce"] for r in rr),
                maximum_integrated_arm_hinge_excursion_rad=max(r["max_integrated_joint_excursion_rad"] for r in rr))
            a.check(v["id"] + "_" + family + "_statistics", all(claims.get(family, {}).get(k) == val for k, val in expected_counts.items()))
    a.check("singleton_report_outcomes", len(report.get("singletons", [])) == 4 and all(
        any(c["variant"] == r["variant"] and c["source_plan_id"] == r["source_plan_id"] and c["capture"] == r["route_success"]
            for c in report["singletons"]) for r in singles))
    return a


def differences(first, second, rows1, rows2):
    """Independent array reconstruction; all maximum values are componentwise."""
    delta = np.abs(first["position_m"] - second["position_m"])
    full = delta.max(axis=(0, 2)); prefix = []; before_bucket = []
    for i, (r1, r2) in enumerate(zip(rows1, rows2)):
        stop = min(r1["physics_steps"], r2["physics_steps"])
        prefix.append(float(delta[:stop, i].max()))
        first_contacts = [np.flatnonzero(data["forces_N"][:, i, 1] > .01) for data in (first, second)]
        stop = min([int(c[0]) for c in first_contacts if len(c)] + [stop])
        before_bucket.append(float(delta[:max(1, stop), i].max()))
    return dict(max_full_horizon_difference_m=float(full.max()), max_common_scored_prefix_difference_m=max(prefix),
        max_before_first_bucket_contact_difference_m=max(before_bucket),
        capture_disagreements=sum(r1["route_success"] != r2["route_success"] for r1, r2 in zip(rows1, rows2)),
        exactly_identical_trajectories=int((full == 0).sum()), queries=len(rows1))


def verify(study, source, prior_study, assets):
    study, source, prior_study, assets = map(Path, (study, source, prior_study, assets))
    spec, report = read(study / "contract.json"), read(study / "report.json")
    rows = lines(study / "results.jsonl")
    original = next(r for r in lines(prior_study / "shots.jsonl") if r["id"] == SOURCE_ID)
    a = audit_records(spec, report, rows, original)
    a.check("prior_records_hash", sha256(prior_study / "shots.jsonl") == spec["prior_records_sha256"])
    a.check("standalone_plan_identity", read(study / "plan.json") == spec["plan"])
    provenance = read(study / "execution-provenance.json")
    a.check("explicit_4090_frozen_source", provenance.get("gpu_uuid") == GPU_UUID and provenance.get("source_snapshot_frozen") is True
        and Path(provenance.get("source_directory", "")).resolve() == source.resolve())
    for name, expected in provenance["source_sha256"].items():
        path = (source / name).resolve()
        a.check("executed_source:" + name, path.is_relative_to(source.resolve()) and path.is_file() and sha256(path) == expected)
    for name, expected in spec["source_sha256"].items():
        a.check("contract_executed_module:" + name, Path(name).name == name and sha256(source / "sircus/mjlab" / name) == expected)
    previous = read(prior_study / "contract.json")
    parent = read(Path(previous["parent_study_directory"]) / "artifact_provenance.json")
    a.check("parent_identity", parent["checkpoint_sha256"] == previous["parent_checkpoint_sha256"]
        and parent["contract_fingerprint"] == previous["parent_contract_fingerprint"])
    auditor_root = Path(__file__).resolve().parents[2]
    for name, expected in parent["source_sha256"].items():
        a.check("unchanged_shared_source:" + name, provenance["source_sha256"].get(name) == sha256(auditor_root / name) == expected)
    for name, expected in parent["asset_sha256"].items():
        path = (assets.parent / name).resolve()
        a.check("unchanged_asset:" + name, path.is_relative_to(assets.parent.resolve()) and path.is_file() and sha256(path) == expected)
    a.check("unchanged_provisional_scene", previous["scene"] == spec["scene"])
    from .verify_stable import audit_strict_endpoints
    audit_strict_endpoints(a, assets, spec, [dict(p, native_release=True, bucket_xy_m=original["bucket_xy_m"]) for p in spec["plan"]])
    by_key = {(r["variant"], r["reset"], r["source_plan_id"]): r for r in rows}
    arrays_by_key = {}; rows_by_key = {}; score_count = 0
    plan = spec["plan"]
    for variant in spec["variants"]:
        key = variant["id"]
        for reset in (0, 1):
            with np.load(study / f"{key}-reset-{reset}.npz", allow_pickle=False) as data:
                raw = {name: data[name].copy() for name in data.files}
            rr = [by_key[(key, reset, p["id"])] for p in plan]
            a.check(f"{key}-{reset}_native_query_arrays", np.array_equal(raw["q10"], [p["q10"] for p in plan])
                and np.array_equal(raw["q14"], [p["q14"] for p in plan]) and np.array_equal(raw["requested_values"], [p["values"] for p in plan])
                and raw["eligible"].shape == (len(plan),) and raw["eligible"].all())
            replay_scores(a, spec["scene"], rr, raw, key, reset); score_count += len(rr)
            a.check(f"{key}-{reset}_350ms_velocity_rows", all(np.array_equal(r["ballistic_velocity_at_350ms_m_s"], raw["velocity_m_s"][83, i]) for i, r in enumerate(rr)))
            arrays_by_key[(key, reset)] = raw; rows_by_key[(key, reset)] = rr
        claimed = next(g for g in report["groups"] if g["id"] == key)["reset_order"]
        expected = differences(arrays_by_key[(key, 0)], arrays_by_key[(key, 1)], rows_by_key[(key, 0)], rows_by_key[(key, 1)])
        a.check(key + "_reset_metrics", all(claimed.get(k) == v for k, v in expected.items()))
    for row in [r for r in rows if r["reset"] == "singleton"]:
        with np.load(study / (row["id"] + ".npz"), allow_pickle=False) as data:
            raw = {name: data[name].copy() for name in data.files}
        replay_scores(a, spec["scene"], [row], raw, row["variant"], row["source_plan_id"]); score_count += 1
        index = next(i for i, p in enumerate(plan) if p["id"] == row["source_plan_id"])
        batch = {k: v[:, index:index + 1] for k, v in arrays_by_key[(row["variant"], 0)].items() if k in ("forces_N", "position_m", "velocity_m_s")}
        expected = differences(batch, raw, [rows_by_key[(row["variant"], 0)][index]], [row])
        claimed = next(x for x in report["singletons"] if x["variant"] == row["variant"] and x["source_plan_id"] == row["source_plan_id"])["batch_comparison"]
        a.check(row["id"] + "_comparison", all(claimed.get(k) == v for k, v in expected.items()))
    corrected = [v for v in spec["variants"] if v["gravity_mode"] == "passive_only"]
    for first, second in zip(corrected, corrected[1:]):
        a1, a2 = arrays_by_key[(first["id"], 0)], arrays_by_key[(second["id"], 0)]
        velocity = np.linalg.norm(a1["velocity_m_s"][83] - a2["velocity_m_s"][83], axis=1)
        position = np.linalg.norm(a1["position_m"][83] - a2["position_m"][83], axis=1)
        claimed = next(x for x in report["timestep_comparisons"] if x["first"] == first["id"] and x["second"] == second["id"])
        a.check("timestep_ballistic_comparison", claimed["ballistic_velocity_difference_at_350ms_max_m_s"] == float(velocity.max())
            and claimed["ballistic_velocity_difference_at_350ms_mean_m_s"] == float(velocity.mean())
            and claimed["ballistic_position_difference_at_350ms_max_m"] == float(position.max()))
    failed = [c for c in a.checks if not c["passed"]]
    return dict(version=VERSION, status="passed" if not failed else "failed", checks_count=len(a.checks),
        checks=a.checks, failures=failed, independently_rescored_throws=score_count,
        executed_source_directory=str(source), contract_fingerprint=spec["fingerprint"],
        evidence_sha256={p.name: sha256(p) for p in study.iterdir() if p.suffix in (".json", ".jsonl", ".npz") and not p.name.startswith("audit-")},
        limitations=["CPU rescoring and static FK verify saved evidence, not a new native dynamics execution.",
            "There are 100 independent jitter requests per protocol, replayed twice; these are not 200 independent scenes.",
            "Only the frozen nominal configuration and small local perturbations are covered; no all-table or training claim."])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "source", "prior-study", "assets", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args(); result = verify(args.study, args.source, args.prior_study, args.assets)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({k: result[k] for k in ("status", "checks_count", "independently_rescored_throws")}))
    if result["failures"]:
        raise RuntimeError("Confirmation audit failed: " + str([x["name"] for x in result["failures"]]))


if __name__ == "__main__":
    main()
