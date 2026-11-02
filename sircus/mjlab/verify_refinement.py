"""CPU audit of fresh confirmations against immutable prior captured sources."""

import argparse
import json
from pathlib import Path

import numpy as np

from .verify_expansion import Audit, digest, sha256, statistics


def audit_records(spec, report, rows, sources, bank, original, prior_rows):
    a = Audit()
    payload = dict(spec)
    fp = payload.pop("fingerprint")
    a.check(
        "contract_fingerprint", digest(payload) == fp == report["contract_fingerprint"]
    )
    payload = dict(bank)
    bank_sha = payload.pop("sha256")
    a.check(
        "expanded_bank_fingerprint",
        digest(payload) == bank_sha == report["proposal_bank_sha256"],
    )
    payload = dict(original)
    old_sha = payload.pop("sha256")
    a.check(
        "original_bank_fingerprint",
        digest(payload) == old_sha == spec["old_bank_sha256"],
    )
    n = len(original["templates"])
    a.check("original_prefix_unchanged", bank["templates"][:n] == original["templates"])
    a.check("completed", report["status"] == report["phase"] == "completed")
    a.check(
        "only_fresh_confirmations_counted",
        report["native_search_attempts"] == 0
        and report["confirmation_attempts"]
        == report["records"]
        == len(rows)
        == len(sources) * 10,
    )
    a.check(
        "qualification_unchanged",
        spec["confirmations_per_candidate"] == 10
        and spec["minimum_confirmation_rate"] == 0.8,
    )
    a.check(
        "sequential_fresh_ids",
        [r["id"] for r in rows] == [f"refine-shot-{i:07d}" for i in range(len(rows))],
    )
    a.check(
        "selected_source_hashes",
        spec["selected_source_ids"] == [s["id"] for s in sources]
        and spec["selected_source_records_sha256"] == [digest(s) for s in sources],
    )
    old = {r["id"]: r for r in prior_rows}
    a.check(
        "exact_original_source_records", all(old.get(s["id"]) == s for s in sources)
    )
    a.check(
        "only_actual_prior_development_captures",
        all(
            s["phase"] == "search"
            and s["source_phase"] == "targeted_development"
            and s["route_success"]
            and not s["invalid"]
            for s in sources
        ),
    )
    targets = {t["id"]: t for t in spec["development_layouts"]}
    a.check(
        "twelve_development_targets",
        len(targets) == 12
        and all(t["role"] == "targeted_development" for t in targets.values()),
    )
    a.check(
        "development_provenance",
        bank["development_layouts"]
        == spec["development_layouts"]
        == report["development_layouts"],
    )
    lookup = {s["id"]: s for s in sources}
    confirmations = {s["id"]: [] for s in sources}
    associations, bad_bounds, bad_capture, bad_pose = [], [], [], []
    for row in rows:
        source = lookup.get(row.get("confirmation_source_id"))
        if (
            not source
            or row["phase"] != "confirmation"
            or row["target_id"] != source["target_id"]
            or row["bound_phase"] != source["bound_phase"]
            or row["bucket_xy_m"] != source["bucket_xy_m"]
            or row["source_phase"] != "targeted_development"
            or row["source_study_contract_fingerprint"]
            != spec["prior_search_contract_fingerprint"]
        ):
            associations.append(row["id"])
            continue
        confirmations[source["id"]].append(row)
        bounds = spec[row["bound_phase"] + "_bounds"]
        value = np.asarray(row["values"])
        if (
            value.shape != (5,)
            or not np.isfinite(value).all()
            or not (
                (value >= np.asarray(bounds["low"]) - 1e-9)
                & (value <= np.asarray(bounds["high"]) + 1e-9)
            ).all()
        ):
            bad_bounds.append(row["id"])
        if row["route_success"] and (
            row["invalid"]
            or not row["board_bounce"]
            or not row["whole_ball_entry"]
            or row["max_retention_s"] < 1 - 1e-9
            or row["external_contact_steps"]
            or row["board_contact_episodes"] < 1
        ):
            bad_capture.append(row["id"])
        if not row["invalid"]:
            pose = row.get("pose_gate", {})
            if (
                not pose
                or pose.get("board_position_error_mm", 1e6) > 2
                or pose.get("board_normal_error_deg", 1e6) > 0.5
                or pose.get("dropper_error_mm", 1e6) > 2
                or row.get("endpoint_collision_pairs")
            ):
                bad_pose.append(row["id"])
    a.check("source_associations", not associations, associations[:10])
    a.check(
        "ten_per_source_invalids_stay_in_denominator",
        all(len(rs) == 10 for rs in confirmations.values()),
    )
    a.check("bounds", not bad_bounds, bad_bounds[:10])
    a.check("capture_rule", not bad_capture, bad_capture[:10])
    a.check("native_endpoint_gates", not bad_pose, bad_pose[:10])
    additions = bank["templates"][n:]
    a.check(
        "one_promotion_per_target",
        len({t["source_target_id"] for t in additions}) == len(additions),
    )
    a.check(
        "addition_counts",
        len(additions)
        == report["additions"]
        == report["expanded_templates"] - report["original_templates"],
    )
    promoted = {t["source_target_id"]: t for t in additions}
    for template in additions:
        source = lookup.get(template["source_trial_id"])
        rs = confirmations.get(template["source_trial_id"], [])
        captures = sum(bool(r["route_success"]) and not r["invalid"] for r in rs)
        a.check(
            template["source_target_id"] + "_actual_8_of_10",
            source is not None
            and len(rs) == 10
            and captures >= 8
            and template["confirmation"]
            == dict(
                attempts=10,
                captures=captures,
                invalid=sum(bool(r["invalid"]) for r in rs),
            ),
        )
        if source:
            v, xy = np.asarray(source["values"]), np.asarray(source["bucket_xy_m"])
            a.check(
                template["source_target_id"] + "_exact_geometry_provenance",
                np.allclose(
                    template["relative_board_xy_m"], v[:2] - xy, atol=1e-12, rtol=0
                )
                and template["height_m"] == v[2]
                and template["tilt_deg"] == v[3]
                and template["gap_m"] == v[4]
                and template["source_record_sha256"] == digest(source)
                and template["source_study_contract_fingerprint"]
                == spec["prior_search_contract_fingerprint"]
                and template["confirmation_study_contract_fingerprint"] == fp
                and template["development_phase"] == source["bound_phase"]
                and template["action_bounds_version"]
                == spec[source["bound_phase"] + "_bounds"]["version"],
            )
    for tid in targets:
        eligible = []
        for s in sources:
            rs = confirmations[s["id"]]
            count = sum(bool(r["route_success"]) and not r["invalid"] for r in rs)
            if s["target_id"] == tid and len(rs) == 10 and count >= 8:
                margin = s.get("minimum_mouth_clearance_mm")
                margin = margin if margin is not None and np.isfinite(margin) else -1e6
                eligible.append((count, margin, s["route_quality"], s["id"]))
        expected = max(eligible)[-1] if eligible else None
        a.check(
            tid + "_promotion_selection",
            (promoted[tid]["source_trial_id"] if tid in promoted else None) == expected,
        )
    a.stats(
        "fresh_confirmation_statistics",
        report["confirmation_statistics"],
        statistics(rows),
    )
    return a


def verify(study, prior_search, old_bank_study, source=None):
    study, prior_search, old_bank_study = (
        Path(study),
        Path(prior_search),
        Path(old_bank_study),
    )
    read = lambda path: json.loads(path.read_text())
    lines = lambda path: [
        json.loads(x) for x in path.read_text().splitlines() if x.strip()
    ]
    spec, report = read(study / "contract.json"), read(study / "report.json")
    rows, sources = lines(study / "shots.jsonl"), lines(study / "source-search.jsonl")
    a = audit_records(
        spec,
        report,
        rows,
        sources,
        read(study / "expanded-bank.json"),
        read(old_bank_study / "proposal-bank.json"),
        lines(prior_search / "shots.jsonl"),
    )
    for name, expected in (
        ("report.json", spec["prior_search_report_sha256"]),
        ("shots.jsonl", spec["prior_search_records_sha256"]),
    ):
        a.check("prior_" + name + "_hash", sha256(prior_search / name) == expected)
    a.check(
        "selected_source_file_hash",
        sha256(study / "source-search.jsonl") == report["source_search_sha256"],
    )
    a.check(
        "expanded_bank_file_hash",
        sha256(study / "expanded-bank.json") == report["expanded_bank_sha256"],
    )
    a.check(
        "alias_identity",
        read(study / "expanded-bank.json") == read(study / "proposal-bank.json"),
    )
    if source is not None:
        source = Path(source)
        bad = [
            name
            for name, expected in report["source_sha256"].items()
            if not (source / name).is_file() or sha256(source / name) != expected
        ]
        a.check("immutable_source_hashes", not bad, bad)
    return dict(
        version="sircus-refinement-evidence-audit-v1",
        passed=all(c["passed"] for c in a.checks),
        checks=a.checks,
        records=len(rows),
        retained_sources=len(sources),
        warnings=[
            "Converted development layouts are not a held-out test.",
            "Prior sources and fresh confirmation counts are kept separate.",
            "Clipped Gaussian geometry perturbations do not establish real-world reliability.",
        ],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("study", type=Path)
    p.add_argument("--prior-search", type=Path, required=True)
    p.add_argument("--old-bank-study", type=Path, required=True)
    p.add_argument("--source", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    result = verify(a.study, a.prior_search, a.old_bank_study, a.source)
    text = json.dumps(result, indent=2, allow_nan=False)
    if a.output:
        a.output.write_text(text + "\n")
    print(text)
    raise SystemExit(0 if result["passed"] else 1)
