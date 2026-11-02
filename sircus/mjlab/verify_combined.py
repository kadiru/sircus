"""Read-only CPU audit of joined development teachers and each native child."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from .verify_expansion import Audit, digest, sha256, statistics


def audit_records(spec, report, bank, original, children, prior_report):
    a = Audit()
    payload = dict(spec)
    fp = payload.pop("fingerprint")
    a.check(
        "combination_contract_fingerprint",
        digest(payload) == fp == report["contract_fingerprint"],
    )
    payload = dict(bank)
    bfp = payload.pop("sha256")
    a.check(
        "combined_bank_fingerprint",
        digest(payload) == bfp == report["proposal_bank_sha256"],
    )
    payload = dict(original)
    oldfp = payload.pop("sha256")
    a.check(
        "original_bank_fingerprint", digest(payload) == oldfp == spec["old_bank_sha256"]
    )
    n = len(original["templates"])
    a.check("original_prefix_unchanged", bank["templates"][:n] == original["templates"])
    a.check(
        "bank_parent",
        bank["parent_bank_sha256"] == oldfp
        and bank["expansion_contract_fingerprint"] == fp,
    )
    a.check(
        "completed_combination_has_zero_new_native_trials",
        report["status"] == report["phase"] == "completed"
        and report["native_search_attempts"]
        == report["confirmation_attempts"]
        == report["records"]
        == 0,
    )
    a.check(
        "two_children_required",
        set(children) == {"refinement", "far"}
        and {c["role"] for c in spec["child_artifacts"]} == set(children),
    )
    a.check(
        "qualification_unchanged",
        spec["confirmations_per_candidate"] == 10
        and spec["minimum_confirmation_rate"] == 0.8,
    )
    targets = {t["id"]: t for t in spec["development_layouts"]}
    a.check(
        "twelve_development_targets",
        len(targets) == 12
        and all(t["role"] == "targeted_development" for t in targets.values()),
    )
    a.check(
        "development_provenance",
        spec["development_layouts"]
        == report["development_layouts"]
        == bank["development_layouts"],
    )
    expected_candidates = []
    for role, child in children.items():
        c, cr, cb = child["spec"], child["report"], child["bank"]
        identity = (
            "scene",
            "parent_contract_fingerprint",
            "parent_checkpoint_sha256",
            "old_bank_study_fingerprint",
            "old_bank_sha256",
            "old_bank_file_sha256",
            "old_test_geometry_sha256",
            "development_layouts",
            "old_bounds",
            "wide_bounds",
            "prior_search_contract_fingerprint",
            "prior_search_report_sha256",
            "prior_search_records_sha256",
        )
        a.check(
            role + "_identity_and_completion",
            all(c[k] == spec[k] for k in identity)
            and cr["status"] == cr["phase"] == "completed"
            and cb["templates"][:n] == original["templates"]
            and c["confirmations_per_candidate"] == 10
            and c["minimum_confirmation_rate"] == 0.8,
        )
        lookup = {r["id"]: r for r in child["rows"]}
        for index, t in enumerate(cb["templates"][n:], n):
            source = lookup.get(t["source_trial_id"])
            good = bool(
                source
                and source["route_success"]
                and not source["invalid"]
                and source["phase"] == "search"
                and t["source_phase"] == "targeted_development"
                and source["target_id"] == t["source_target_id"]
                and t["source_record_sha256"] == digest(source)
            )
            conf = t["confirmation"]
            a.check(
                role + "_" + str(index) + "_actual_qualified_source",
                good
                and conf["attempts"] == 10
                and 8 <= conf["captures"] <= 10
                and 0 <= conf["invalid"] <= 10 - conf["captures"],
            )
            if not source:
                continue
            margin = source.get("minimum_mouth_clearance_mm")
            margin = (
                float(margin) if margin is not None and np.isfinite(margin) else -1e6
            )
            expected = copy.deepcopy(t)
            expected.update(
                combined_child_role=role,
                combined_child_contract_fingerprint=c["fingerprint"],
                combined_child_bank_sha256=cb["sha256"],
                combined_child_template_index=index,
                combined_source_mouth_clearance_mm=margin,
            )
            expected_candidates.append(
                (
                    t["source_target_id"],
                    conf["captures"],
                    margin,
                    role,
                    source["id"],
                    expected,
                )
            )
    expected = []
    for tid in targets:
        candidates = [x for x in expected_candidates if x[0] == tid]
        if candidates:
            expected.append(max(candidates, key=lambda x: x[1:5])[-1])
    a.check("exact_confirmed_winners_and_provenance", bank["templates"][n:] == expected)
    a.check(
        "addition_counts",
        len(expected)
        == report["additions"]
        == report["expanded_templates"] - report["original_templates"],
    )
    bounds = (
        spec["wide_bounds"]
        if any(t["development_phase"] == "wide" for t in expected)
        else spec["old_bounds"]
    )
    a.check("action_bounds", report["bounds"] == bank["action_bounds"] == bounds)
    search = prior_report["native_search_attempts"]
    confirmations = prior_report["confirmation_attempts"]
    expected_effort = dict(
        prior_search_attempts=search,
        prior_confirmation_attempts=confirmations,
        refinement_confirmation_attempts=children["refinement"]["report"][
            "confirmation_attempts"
        ],
        far_search_attempts=children["far"]["report"]["native_search_attempts"],
        far_confirmation_attempts=children["far"]["report"]["confirmation_attempts"],
        native_search_attempts_total=search
        + sum(c["report"]["native_search_attempts"] for c in children.values()),
        native_confirmation_attempts_total=confirmations
        + sum(c["report"]["confirmation_attempts"] for c in children.values()),
    )
    a.check(
        "distinct_effort_without_prior_double_count",
        report["effort"] == expected_effort
        and report["shared_prior_effort"]
        == dict(
            native_search_attempts=search,
            confirmation_attempts=confirmations,
            counted_once=True,
        )
        and report["total_distinct_native_effort"]
        == dict(
            native_search_attempts=expected_effort["native_search_attempts_total"],
            confirmation_attempts=expected_effort["native_confirmation_attempts_total"],
        ),
    )
    a.check(
        "child_effort_exact",
        all(
            next(e for e in report["child_effort"] if e["role"] == role)
            == dict(
                role=role,
                native_search_attempts=c["report"]["native_search_attempts"],
                confirmation_attempts=c["report"]["confirmation_attempts"],
                prior_native_search_attempts=c["report"][
                    "prior_native_search_attempts"
                ],
                prior_confirmation_attempts=c["report"]["prior_confirmation_attempts"],
                additions=c["report"]["additions"],
            )
            for role, c in children.items()
        ),
    )
    cells = {c["id"]: c for c in report["cells"]}
    a.check("complete_cell_ids", set(cells) == set(targets))
    for tid in targets:
        cell = cells[tid]
        prior = next(c["search"] for c in prior_report["cells"] if c["id"] == tid)
        far = next(
            c["search"] for c in children["far"]["report"]["cells"] if c["id"] == tid
        )
        cumulative = {
            k: prior[k] + far[k]
            for k in ("attempts", "captures", "clean", "invalid", "board_bounces")
        }
        cumulative["mean_reward"] = (
            (
                prior["mean_reward"] * prior["attempts"]
                + far["mean_reward"] * far["attempts"]
            )
            / cumulative["attempts"]
            if cumulative["attempts"]
            else 0.0
        )
        a.stats(tid + "_distinct_search_counts", cell["search"], cumulative)
        fresh = [
            r
            for c in children.values()
            for r in c["rows"]
            if r["phase"] == "confirmation" and r["target_id"] == tid
        ]
        a.stats(
            tid + "_fresh_confirmation_counts",
            cell["fresh_confirmation"],
            statistics(fresh),
        )
        a.check(
            tid + "_promotion_report",
            cell["promotion"]
            == next((x for x in expected if x["source_target_id"] == tid), None),
        )
    return a


def verify(study, source=None, assets=None):
    from .combine_expansions import load_child
    from .verify_refinement import verify as verify_refinement
    from .verify_far import verify as verify_far

    study = Path(study)
    read = lambda path: json.loads(path.read_text())
    spec, report, bank = (
        read(study / "contract.json"),
        read(study / "report.json"),
        read(study / "expanded-bank.json"),
    )
    children, child_audits = {}, {}
    for mapping in spec["child_artifacts"]:
        role = mapping["role"]
        dep = mapping["audit_dependencies"]
        child = load_child(
            role,
            mapping["directory"],
            dep["prior_search"],
            dep["old_bank_study"],
            dep["source"],
        )
        if child["mapping"] != mapping:
            raise RuntimeError("Recorded child path/checksum mapping changed: " + role)
        children[role] = child
        if role == "refinement":
            checked = verify_refinement(
                mapping["directory"],
                dep["prior_search"],
                dep["old_bank_study"],
                source=dep["source"],
            )
        else:
            checked = verify_far(
                mapping["directory"],
                dep["prior_search"],
                dep["old_bank_study"],
                source=dep["source"],
                assets=assets,
            )
        child_audits[role] = checked
    dep = spec["child_artifacts"][0]["audit_dependencies"]
    original = read(Path(dep["old_bank_study"]) / "proposal-bank.json")
    prior_report = read(Path(dep["prior_search"]) / "report.json")
    a = audit_records(spec, report, bank, original, children, prior_report)
    for role, checked in child_audits.items():
        a.check(role + "_independent_child_audit", checked["passed"])
    a.check(
        "expanded_bank_file_hash",
        sha256(study / "expanded-bank.json") == report["expanded_bank_sha256"],
    )
    a.check("bank_alias", read(study / "proposal-bank.json") == bank)
    if source is not None:
        source = Path(source)
        bad = [
            name
            for name, expected in report["source_sha256"].items()
            if not (source / name).is_file() or sha256(source / name) != expected
        ]
        a.check("immutable_join_source_hashes", not bad, bad)
    return dict(
        version="sircus-combined-evidence-audit-v1",
        passed=all(c["passed"] for c in a.checks),
        checks=a.checks,
        child_audits={
            role: dict(passed=r["passed"], checks=len(r["checks"]))
            for role, r in child_audits.items()
        },
        warnings=[
            "Joining banks creates zero native trials; shared prior effort is counted once.",
            "All twelve formerly unresolved positions are development, not held-out tests.",
            "A source's 8/10 development confirmation is not a policy generalization claim.",
        ],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("study", type=Path)
    p.add_argument("--source", type=Path)
    p.add_argument("--assets", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    result = verify(a.study, a.source, a.assets)
    text = json.dumps(result, indent=2, allow_nan=False)
    if a.output:
        a.output.write_text(text + "\n")
    print(text)
    raise SystemExit(0 if result["passed"] else 1)
