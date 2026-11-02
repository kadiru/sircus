"""Independent CPU evidence audit for targeted bank expansion, never a GPU job."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(row):
    if row["invalid"]:
        return -1.0
    shaped = 0.15 * float(row["board_bounce"]) + 0.5 * np.exp(
        -row["closest_opening_distance_m"] / 0.12
    )
    if row["external_contact_steps"]:
        return float(shaped - 0.5)
    return float(
        shaped
        + 0.5 * min(row["max_retention_s"], 1.0)
        + 3 * row["route_success"]
        + 0.3 * (row["route_quality"] == 2)
        + 0.1 * row["single_bounce_success"]
    )


def statistics(rows):
    return dict(
        attempts=len(rows),
        captures=sum(bool(r["route_success"]) and not r["invalid"] for r in rows),
        clean=sum(r["route_quality"] == 2 and not r["invalid"] for r in rows),
        invalid=sum(bool(r["invalid"]) for r in rows),
        board_bounces=sum(bool(r["board_bounce"]) and not r["invalid"] for r in rows),
        mean_reward=float(np.mean([score(r) for r in rows])) if rows else 0.0,
    )


class Audit:
    def __init__(self):
        self.checks = []

    def check(self, name, passed, detail=None):
        self.checks.append(dict(name=name, passed=bool(passed), detail=detail))

    def stats(self, name, actual, expected):
        self.check(
            name,
            all(
                k in actual and np.isclose(actual[k], v, atol=1e-9, rtol=1e-9)
                for k, v in expected.items()
            ),
            dict(actual=actual, expected=expected),
        )


def audit_records(spec, report, rows, bank, original):
    a = Audit()
    p = dict(spec)
    fp = p.pop("fingerprint")
    a.check("contract_fingerprint", digest(p) == fp == report["contract_fingerprint"])
    a.check(
        "completed",
        report.get("status") == "completed" and report.get("phase") == "completed",
    )
    p = dict(original)
    old_sha = p.pop("sha256")
    a.check(
        "original_bank_fingerprint",
        digest(p) == old_sha == spec["old_bank_sha256"] == report["old_bank_sha256"],
    )
    p = dict(bank)
    bank_sha = p.pop("sha256")
    a.check(
        "expanded_bank_fingerprint",
        digest(p) == bank_sha == report["proposal_bank_sha256"],
    )
    n = len(original["templates"])
    a.check(
        "original_template_prefix_unchanged",
        bank["templates"][:n] == original["templates"],
    )
    a.check(
        "bank_parent_identity",
        bank["parent_bank_sha256"] == old_sha
        and bank["expansion_contract_fingerprint"] == fp,
    )
    a.check(
        "development_provenance",
        bank["development_layouts"]
        == spec["development_layouts"]
        == report["development_layouts"]
        and all(
            t["role"] == "targeted_development" for t in spec["development_layouts"]
        ),
    )
    targets = {t["id"]: t for t in spec["development_layouts"]}
    a.check("target_ids_unique", len(targets) == len(spec["development_layouts"]))
    a.check(
        "sequential_trial_ids",
        [r["id"] for r in rows] == [f"expand-shot-{i:07d}" for i in range(len(rows))],
    )
    a.check("records_count", len(rows) == report["records"])
    phases = Counter(r["phase"] for r in rows)
    a.check("native_search_count", phases["search"] == report["native_search_attempts"])
    a.check(
        "confirmation_count", phases["confirmation"] == report["confirmation_attempts"]
    )
    a.check("only_development_phases", set(phases) <= set(("search", "confirmation")))
    a.check("geometry_denominator", report["geometry_proposals"] >= phases["search"])
    bad_targets = []
    bad_bounds = []
    bad_scores = []
    bad_poses = []
    lookup = {r["id"]: r for r in rows}
    for r in rows:
        target = targets.get(r["target_id"])
        if (
            not target
            or r.get("source_phase") != "targeted_development"
            or not np.allclose(
                r["bucket_xy_m"], target["bucket_xy_m"], atol=1e-12, rtol=0
            )
        ):
            bad_targets.append(r["id"])
        bounds = spec.get(r["bound_phase"] + "_bounds")
        v = np.asarray(r.get("values", []))
        if (
            not bounds
            or v.shape != (5,)
            or not np.isfinite(v).all()
            or not (
                (v >= np.asarray(bounds["low"]) - 1e-9)
                & (v <= np.asarray(bounds["high"]) + 1e-9)
            ).all()
        ):
            bad_bounds.append(r["id"])
        if r["route_success"] and (
            r["invalid"]
            or not r["board_bounce"]
            or not r["whole_ball_entry"]
            or r["max_retention_s"] < 1 - 1e-9
            or r["external_contact_steps"]
            or r["board_contact_episodes"] < 1
        ):
            bad_scores.append(r["id"])
        if not r["invalid"]:
            gate = r.get("pose_gate", {})
            if (
                not gate
                or gate.get("board_position_error_mm", 1e6) > 2
                or gate.get("board_normal_error_deg", 1e6) > 0.5
                or gate.get("dropper_error_mm", 1e6) > 2
                or r.get("endpoint_collision_pairs")
            ):
                bad_poses.append(r["id"])
    a.check(
        "all_trial_targets_are_declared_development", not bad_targets, bad_targets[:10]
    )
    a.check("all_actions_inside_their_declared_bounds", not bad_bounds, bad_bounds[:10])
    a.check("capture_rule_invariants", not bad_scores, bad_scores[:10])
    a.check("native_pose_and_contact_gates", not bad_poses, bad_poses[:10])
    confirms = {}
    association_bad = []
    for r in rows:
        if r["phase"] != "confirmation":
            continue
        source = lookup.get(r.get("confirmation_source_id"))
        if (
            not source
            or source["phase"] != "search"
            or source["target_id"] != r["target_id"]
            or source["bound_phase"] != r["bound_phase"]
        ):
            association_bad.append(r["id"])
        confirms.setdefault(r.get("confirmation_source_id"), []).append(r)
    a.check(
        "confirmation_source_association", not association_bad, association_bad[:10]
    )
    expected_sources = []
    for target in targets:
        for phase in ("old", "wide"):
            attempts = [
                r
                for r in rows
                if r["phase"] == "search"
                and r["target_id"] == target
                and r["bound_phase"] == phase
            ]
            if "native_per_target_per_phase" in spec:
                a.check(
                    target + "_" + phase + "_finite_native_budget",
                    len(attempts) <= spec["native_per_target_per_phase"],
                )
            selected = sorted(
                [
                    r
                    for r in rows
                    if r["phase"] == "search"
                    and r["target_id"] == target
                    and r["bound_phase"] == phase
                    and r["route_success"]
                    and not r["invalid"]
                ],
                key=lambda r: (
                    bool(r["route_success"] and not r["invalid"]),
                    int(r["route_quality"]),
                    score(r),
                ),
                reverse=True,
            )[: spec["top_candidates"]]
            expected_sources += [r["id"] for r in selected]
    a.check("top_captured_sources_confirmed", set(confirms) == set(expected_sources))
    a.check(
        "independent_confirmation_denominators",
        all(len(rs) == spec["confirmations_per_candidate"] for rs in confirms.values()),
    )
    additions = bank["templates"][n:]
    a.check(
        "addition_counts",
        len(additions)
        == report["additions"]
        == report["expanded_templates"] - report["original_templates"],
    )
    a.check(
        "one_promotion_per_target",
        len(set(t["source_target_id"] for t in additions)) == len(additions),
    )
    by_target = {}
    for template in additions:
        tid = template["source_target_id"]
        by_target[tid] = template
        source = lookup.get(template["source_trial_id"])
        rs = confirms.get(template["source_trial_id"], [])
        captures = sum(bool(r["route_success"]) and not r["invalid"] for r in rs)
        invalid = sum(bool(r["invalid"]) for r in rs)
        source_ok = bool(
            source
            and source["phase"] == "search"
            and source["target_id"] == tid
            and source["route_success"]
            and not source["invalid"]
        )
        a.check(
            tid + "_source_is_actual_development_capture",
            source_ok and template["source_phase"] == "targeted_development",
        )
        a.check(
            tid + "_confirmation",
            len(rs) == spec["confirmations_per_candidate"]
            and captures / len(rs) >= spec["minimum_confirmation_rate"]
            and template["confirmation"]
            == dict(attempts=len(rs), captures=captures, invalid=invalid),
            template["confirmation"],
        )
        if source:
            value = source["values"]
            xy = source["bucket_xy_m"]
            a.check(
                tid + "_geometry_from_measured_source",
                np.allclose(
                    template["relative_board_xy_m"],
                    np.asarray(value[:2]) - xy,
                    atol=1e-12,
                    rtol=0,
                )
                and template["height_m"] == value[2]
                and template["tilt_deg"] == value[3]
                and template["gap_m"] == value[4],
            )
            a.check(
                tid + "_bounds_and_phase_provenance",
                template["development_phase"] == source["bound_phase"]
                and template["action_bounds_version"]
                == spec[source["bound_phase"] + "_bounds"]["version"],
            )
    for cell in report["cells"]:
        tid = cell["id"]
        search = [r for r in rows if r["phase"] == "search" and r["target_id"] == tid]
        a.stats(tid + "_search_stats", cell["search"], statistics(search))
        a.check(
            tid + "_reported_promotion", cell.get("promotion") == by_target.get(tid)
        )
        if tid in by_target and by_target[tid]["development_phase"] == "old":
            a.check(
                tid + "_old_success_not_widened",
                not any(
                    r["target_id"] == tid and r["bound_phase"] == "wide" for r in rows
                ),
            )
        eligible = []
        for r in search:
            rs = confirms.get(r["id"], [])
            captured = sum(bool(x["route_success"]) and not x["invalid"] for x in rs)
            if (
                r["route_success"]
                and not r["invalid"]
                and len(rs) == spec["confirmations_per_candidate"]
                and captured / len(rs) >= spec["minimum_confirmation_rate"]
            ):
                eligible.append((r, captured))
        old_eligible = [x for x in eligible if x[0]["bound_phase"] == "old"]
        considered = old_eligible or [
            x for x in eligible if x[0]["bound_phase"] == "wide"
        ]
        expected = (
            max(
                considered,
                key=lambda x: (
                    x[1],
                    bool(x[0]["route_success"]),
                    int(x[0]["route_quality"]),
                    score(x[0]),
                ),
            )[0]["id"]
            if considered
            else None
        )
        a.check(
            tid + "_promotion_selected_from_measured_confirmations",
            (by_target[tid]["source_trial_id"] if tid in by_target else None)
            == expected,
        )
    wide = any(t["development_phase"] == "wide" for t in additions)
    expected_bounds = spec["wide_bounds"] if wide else spec["old_bounds"]
    a.check(
        "expanded_action_domain",
        bank["action_bounds"] == report["bounds"] == expected_bounds,
    )
    a.stats(
        "all_native_search_stats",
        report["native_search_statistics"],
        statistics([r for r in rows if r["phase"] == "search"]),
    )
    a.stats(
        "all_confirmation_stats",
        report["confirmation_statistics"],
        statistics([r for r in rows if r["phase"] == "confirmation"]),
    )
    return a


def verify(study, old_bank_study, source=None, assets=None):
    study, old_bank_study = Path(study), Path(old_bank_study)
    spec = json.loads((study / "contract.json").read_text())
    report = json.loads((study / "report.json").read_text())
    rows = [
        json.loads(line) for line in (study / "shots.jsonl").read_text().splitlines()
    ]
    bank = json.loads((study / "expanded-bank.json").read_text())
    original = json.loads((old_bank_study / "proposal-bank.json").read_text())
    a = audit_records(spec, report, rows, bank, original)
    a.check(
        "expanded_bank_file_hash",
        sha256(study / "expanded-bank.json") == report["expanded_bank_sha256"],
    )
    a.check(
        "proposal_alias_unchanged",
        json.loads((study / "proposal-bank.json").read_text()) == bank,
    )
    a.check(
        "original_bank_file_hash",
        sha256(old_bank_study / "proposal-bank.json") == spec["old_bank_file_sha256"],
    )
    a.check(
        "original_geometry_file_hash",
        sha256(old_bank_study / "geometry-test.npz")
        == spec["old_test_geometry_sha256"],
    )
    old_report = json.loads((old_bank_study / "report.json").read_text())
    a.check(
        "original_study_identity",
        old_report["contract_fingerprint"]
        == spec["old_bank_study_fingerprint"]
        == report["old_bank_study_fingerprint"],
    )
    geometry = np.load(old_bank_study / "geometry-test.npz")
    indices = np.flatnonzero(~geometry["mask"].any(1))
    a.check(
        "exact_original_twelve_reclassified",
        len(indices) == 12
        and [t["original_test_index"] for t in spec["development_layouts"]]
        == indices.tolist()
        and all(
            np.array_equal(t["bucket_xy_m"], geometry["bucket"][i])
            for t, i in zip(spec["development_layouts"], indices)
        ),
    )
    if source is not None:
        source = Path(source)
        mismatches = [
            name
            for name, expected in report["source_sha256"].items()
            if not (source / name).is_file() or sha256(source / name) != expected
        ]
        a.check("immutable_source_hashes", not mismatches, mismatches)
    fk = None
    if assets is not None:
        from .verify_learning import audit_fk

        accepted = [
            dict(r, locks=dict(board=None, dropper=None))
            for r in rows
            if not r["invalid"]
        ]
        fk = audit_fk(Path(assets), spec, accepted, a)
    return dict(
        version="sircus-expansion-evidence-audit-v1",
        passed=all(x["passed"] for x in a.checks),
        checks=a.checks,
        contract_fingerprint=spec["fingerprint"],
        records=len(rows),
        phase_counts=dict(Counter(r["phase"] for r in rows)),
        independent_fk=fk,
        warnings=[
            "Former test failures are development targets here, not a held-out generalization result.",
            "This audits records, scoring invariants and provenance; it does not independently rerun ball dynamics or prove global impossibility.",
            "Confirmations are clipped Gaussian geometry perturbations under unchanged simulation materials, not physical/domain-randomized reliability.",
        ],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("study", type=Path)
    p.add_argument("--old-bank-study", type=Path, required=True)
    p.add_argument("--source", type=Path)
    p.add_argument(
        "--assets",
        type=Path,
        help="Generated URDF directory for independent accepted-endpoint CPU FK",
    )
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    r = verify(a.study, a.old_bank_study, a.source, a.assets)
    text = json.dumps(r, indent=2, allow_nan=False)
    if a.output:
        a.output.write_text(text + "\n")
    print(text)
    raise SystemExit(0 if r["passed"] else 1)
