"""Independent CPU audit of finite missing-six search and confirmation evidence."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .verify_expansion import audit_records as original_checks, digest, sha256

EXPECTED_TARGETS = {f"old-test-{i:02d}" for i in (8, 9, 12, 23, 35, 41)}


def audit_records(spec, report, rows, bank, original, seeds, prior_rows):
    audit = original_checks(spec, report, rows, bank, original)
    # The historical auditor fixes its original expand-shot namespace. Retain
    # every scoring/provenance check, and replace that single naming check with
    # this study's exact, independently checked far-shot namespace.
    audit.checks = [
        check for check in audit.checks if check["name"] != "sequential_trial_ids"
    ]
    audit.check(
        "sequential_far_trial_ids",
        [r["id"] for r in rows] == [f"far-shot-{i:07d}" for i in range(len(rows))],
    )
    audit.check(
        "fixed_six_targets", set(spec["selected_target_ids"]) == EXPECTED_TARGETS
    )
    audit.check(
        "full_twelve_remain_development", len(spec["development_layouts"]) == 12
    )
    audit.check(
        "unchanged_qualification",
        spec["confirmations_per_candidate"] == 10
        and spec["minimum_confirmation_rate"] == 0.8,
    )
    budget = spec["native_per_target"]
    audit.check("bounded_native_budget", 1024 <= budget <= 2048)
    counts = Counter(r["target_id"] for r in rows if r["phase"] == "search")
    audit.check(
        "only_six_get_fresh_trials",
        set(counts) <= EXPECTED_TARGETS
        and all(count <= budget for count in counts.values()),
    )
    audit.check(
        "no_fresh_confirmation_on_other_six",
        all(r["target_id"] in EXPECTED_TARGETS for r in rows),
    )
    prior_lookup = {row["id"]: row for row in prior_rows}
    audit.check(
        "frozen_prior_seed_records",
        spec["selected_prior_seed_ids"] == [s["id"] for s in seeds]
        and spec["selected_prior_seed_records_sha256"] == [digest(s) for s in seeds]
        and all(prior_lookup.get(s["id"]) == s for s in seeds),
    )
    audit.check(
        "prior_seeds_are_proposals_only",
        all(
            s["phase"] == "search"
            and s["source_phase"] == "targeted_development"
            and s["target_id"] in EXPECTED_TARGETS
            and not s["invalid"]
            and s["board_bounce"]
            for s in seeds
        ),
    )
    fresh_lookup = {r["id"]: r for r in rows}
    seed_lookup = {s["id"]: s for s in seeds}
    associations = []
    for row in rows:
        sid = row.get("seed_source_id")
        if sid is None:
            continue
        source = seed_lookup.get(sid, fresh_lookup.get(sid))
        if (
            source is None
            or source["target_id"] != row["target_id"]
            or source["phase"] != "search"
            or digest(source) != row.get("seed_source_record_sha256")
            or (
                sid in fresh_lookup
                and int(sid.rsplit("-", 1)[1]) >= int(row["id"].rsplit("-", 1)[1])
            )
        ):
            associations.append(row["id"])
    audit.check("local_seed_geometry_provenance", not associations, associations[:10])
    for template in bank["templates"][len(original["templates"]) :]:
        source = fresh_lookup.get(template["source_trial_id"])
        audit.check(
            template["source_target_id"] + "_joined_source_provenance",
            source is not None
            and template.get("source_record_sha256") == digest(source)
            and template.get("source_study_contract_fingerprint") == spec["fingerprint"]
            and template.get("confirmation_study_contract_fingerprint")
            == spec["fingerprint"],
        )
    audit.check(
        "inherited_attempts_not_added",
        report["native_search_attempts"] == sum(counts.values())
        and report["prior_native_search_attempts"]
        == spec["prior_native_search_attempts"]
        and report["prior_confirmation_attempts"]
        == spec["prior_confirmation_attempts"],
    )
    return audit


def verify(study, prior_search, old_bank_study, source=None, assets=None):
    study, prior_search, old_bank_study = map(
        Path, (study, prior_search, old_bank_study)
    )
    read = lambda path: json.loads(path.read_text())
    lines = lambda path: [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    spec, report = read(study / "contract.json"), read(study / "report.json")
    rows = lines(study / "shots.jsonl")
    bank, original = read(study / "expanded-bank.json"), read(
        old_bank_study / "proposal-bank.json"
    )
    seeds, prior_rows = lines(study / "source-near-misses.jsonl"), lines(
        prior_search / "shots.jsonl"
    )
    audit = audit_records(spec, report, rows, bank, original, seeds, prior_rows)
    prior_spec, prior_report = read(prior_search / "contract.json"), read(
        prior_search / "report.json"
    )
    payload = dict(prior_spec)
    fp = payload.pop("fingerprint")
    audit.check(
        "prior_scene_and_contract",
        digest(payload)
        == fp
        == spec["prior_search_contract_fingerprint"]
        == prior_report["contract_fingerprint"]
        and prior_spec["scene"] == spec["scene"]
        and prior_spec["development_layouts"] == spec["development_layouts"],
    )
    audit.check(
        "source_files_frozen",
        sha256(prior_search / "report.json") == spec["prior_search_report_sha256"]
        and sha256(prior_search / "shots.jsonl") == spec["prior_search_records_sha256"]
        and sha256(old_bank_study / "proposal-bank.json")
        == spec["old_bank_file_sha256"]
        and sha256(old_bank_study / "geometry-test.npz")
        == spec["old_test_geometry_sha256"],
    )
    audit.check(
        "bank_artifact_hash",
        sha256(study / "expanded-bank.json") == report["expanded_bank_sha256"]
        and read(study / "proposal-bank.json") == bank,
    )
    audit.check(
        "near_miss_artifact_hash",
        sha256(study / "source-near-misses.jsonl")
        == report["source_near_misses_sha256"],
    )
    geometry_bad, cache = [], {}
    for row in rows:
        name = row.get("geometry_cache_file", "")
        if (
            Path(name).name != name
            or not name.startswith("geometry-native-")
            or not name.endswith(".npz")
        ):
            geometry_bad.append(row["id"])
            continue
        if name not in cache:
            with np.load(study / name, allow_pickle=False) as values:
                cache[name] = {key: values[key].copy() for key in values.files}
        values = cache[name]
        i = row["geometry_cache_row"]
        good = (
            0 <= i < len(values["eligible"])
            and np.array_equal(values["bucket"][i], row["bucket_xy_m"])
            and np.array_equal(values["values"][i], row["values"])
            and bool(values["eligible"][i]) == row["geometry_eligible"]
        )
        if not row["invalid"]:
            good = (
                good
                and bool(values["eligible"][i])
                and np.array_equal(values["q10"][i], row["q10"])
                and np.array_equal(values["q14"][i], row["q14"])
            )
        elif not values["eligible"][i]:
            good = good and not row["native_release"]
        if not good:
            geometry_bad.append(row["id"])
    audit.check("exact_cached_native_branches", not geometry_bad, geometry_bad[:10])
    bad_hashes = [
        name
        for name, sha in report.get("artifacts_sha256", {}).items()
        if Path(name).name != name
        or not (study / name).is_file()
        or sha256(study / name) != sha
    ]
    audit.check("all_saved_artifact_hashes", not bad_hashes, bad_hashes)
    if source is not None:
        source = Path(source)
        bad = [
            name
            for name, sha in report["source_sha256"].items()
            if not (source / name).is_file() or sha256(source / name) != sha
        ]
        audit.check("immutable_source_hashes", not bad, bad)
    fk = None
    if assets is not None:
        from .verify_learning import audit_fk

        accepted = [
            dict(r, locks=dict(board=None, dropper=None))
            for r in rows
            if not r["invalid"]
        ]
        fk = audit_fk(Path(assets), spec, accepted, audit)
    return dict(
        version="sircus-far-development-audit-v1",
        passed=all(c["passed"] for c in audit.checks),
        checks=audit.checks,
        records=len(rows),
        contract_fingerprint=spec["fingerprint"],
        independent_fk=fk,
        warnings=[
            "Finite development search, not held-out evaluation or proof of impossibility.",
            "Prior near-misses guide geometry only; captures still require unchanged native route scoring and8/10fresh perturbed confirmations.",
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    for name in ("prior-search", "old-bank-study"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--output", type=Path)
    a = parser.parse_args()
    result = verify(a.study, a.prior_search, a.old_bank_study, a.source, a.assets)
    text = json.dumps(result, indent=2, allow_nan=False)
    if a.output:
        a.output.write_text(text + "\n")
    print(text)
    raise SystemExit(0 if result["passed"] else 1)
