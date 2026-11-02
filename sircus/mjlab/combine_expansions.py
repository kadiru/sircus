"""Join completed, confirmed development banks without inventing new trials."""

import argparse
import copy
import datetime
import json
from pathlib import Path

import numpy as np

from .expand_bank import atomic, expanded_bank, sha256
from .learning_contract import digest, statistics

VERSION = "sircus-confirmed-bank-combination-v1"
CHILD_VERSIONS = dict(
    refinement="sircus-targeted-bank-refinement-v1",
    far="sircus-far-development-search-v1",
)


def load_child(role, path, prior_search, old_bank_study, source):
    """Pin every required child artifact and its immutable audit dependencies."""
    path = Path(path).resolve()
    spec = json.loads((path / "contract.json").read_text())
    report = json.loads((path / "report.json").read_text())
    bank = json.loads((path / "expanded-bank.json").read_text())
    payload = dict(spec)
    fp = payload.pop("fingerprint")
    if digest(payload) != fp or report.get("contract_fingerprint") != fp:
        raise RuntimeError(role + " contract fingerprint changed")
    if spec.get("version") != CHILD_VERSIONS[role]:
        raise RuntimeError(role + " has an unsupported child contract")
    if report.get("status") != "completed" or report.get("phase") != "completed":
        raise RuntimeError(role + " must finish before banks can be joined")
    payload = dict(bank)
    bank_sha = payload.pop("sha256")
    if digest(payload) != bank_sha or report.get("proposal_bank_sha256") != bank_sha:
        raise RuntimeError(role + " bank fingerprint changed")
    if sha256(path / "expanded-bank.json") != report["expanded_bank_sha256"]:
        raise RuntimeError(role + " bank file changed")
    if (
        sha256(Path(prior_search) / "report.json") != spec["prior_search_report_sha256"]
        or sha256(Path(prior_search) / "shots.jsonl")
        != spec["prior_search_records_sha256"]
    ):
        raise RuntimeError(role + " prior evidence dependency changed")
    source = Path(source)
    for name, expected in report["source_sha256"].items():
        if (
            Path(name).name != name
            or not (source / name).is_file()
            or sha256(source / name) != expected
        ):
            raise RuntimeError(role + " immutable source dependency changed: " + name)
    names = {
        "contract.json",
        "report.json",
        "expanded-bank.json",
        "proposal-bank.json",
        "shots.jsonl",
    }
    for name in ("source-search.jsonl", "source-near-misses.jsonl"):
        if (path / name).is_file():
            names.add(name)
    for name, expected in report.get("artifacts_sha256", {}).items():
        if Path(name).name != name or sha256(path / name) != expected:
            raise RuntimeError(role + " declared artifact changed: " + name)
        names.add(name)
    files = {name: sha256(path / name) for name in sorted(names)}
    records = []
    for name in ("source-search.jsonl", "source-near-misses.jsonl", "shots.jsonl"):
        if name in names:
            records += [
                json.loads(line)
                for line in (path / name).read_text().splitlines()
                if line.strip()
            ]
    mapping = dict(
        role=role,
        directory=str(path),
        paths={
            "contract": "contract.json",
            "report": "report.json",
            "bank": "expanded-bank.json",
            "records": "shots.jsonl",
        },
        files_sha256=files,
        contract_fingerprint=fp,
        proposal_bank_sha256=bank_sha,
        audit_dependencies=dict(
            prior_search=str(Path(prior_search).resolve()),
            old_bank_study=str(Path(old_bank_study).resolve()),
            source=str(Path(source).resolve()),
        ),
    )
    return dict(spec=spec, report=report, bank=bank, rows=records, mapping=mapping)


def combine(original, children, prior_report):
    """Validate child teachers, then resolve duplicate targets from evidence."""
    if set(children) != set(CHILD_VERSIONS):
        raise ValueError("Both completed refinement and far children are required")
    specs = [children[r]["spec"] for r in sorted(children)]
    reference = specs[0]
    for spec in specs:
        for key in (
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
        ):
            if spec[key] != reference[key]:
                raise ValueError("Child identity mismatch: " + key)
        if (
            spec["confirmations_per_candidate"] != 10
            or spec["minimum_confirmation_rate"] != 0.8
        ):
            raise ValueError(
                "Child qualification must retain ten attempts / eight captures"
            )
        if spec["old_bank_sha256"] != original["sha256"]:
            raise ValueError("Child belongs to a different original bank")
    targets = {t["id"]: t for t in reference["development_layouts"]}
    if len(targets) != 12 or any(
        t["role"] != "targeted_development" for t in targets.values()
    ):
        raise ValueError("The original twelve targets must remain explicit development")
    n = len(original["templates"])
    candidates = []
    for role, child in sorted(children.items()):
        if (
            child["report"]["status"] != "completed"
            or child["report"]["phase"] != "completed"
        ):
            raise ValueError("Cannot join a running child")
        if child["bank"]["templates"][:n] != original["templates"]:
            raise ValueError("Child modified the original template prefix")
        lookup = {r["id"]: r for r in child["rows"]}
        for index, template in enumerate(child["bank"]["templates"][n:], n):
            tid = template["source_target_id"]
            source = lookup.get(template["source_trial_id"])
            conf = template["confirmation"]
            if (
                tid not in targets
                or template.get("source_phase") != "targeted_development"
                or not source
                or not source.get("route_success")
                or source.get("invalid")
                or source.get("phase") != "search"
                or source.get("target_id") != tid
            ):
                raise ValueError(
                    "Child promotion lacks its actual captured development source"
                )
            if (
                conf["attempts"] != 10
                or not 8 <= conf["captures"] <= 10
                or not 0 <= conf["invalid"] <= 10 - conf["captures"]
            ):
                raise ValueError("Child promotion does not pass unchanged 8/10")
            measured = [
                r
                for r in child["rows"]
                if r.get("phase") == "confirmation"
                and r.get("confirmation_source_id") == source["id"]
            ]
            actual = dict(
                attempts=len(measured),
                captures=sum(
                    bool(r["route_success"]) and not r["invalid"] for r in measured
                ),
                invalid=sum(bool(r["invalid"]) for r in measured),
            )
            if conf != actual:
                raise ValueError(
                    "Child promotion confirmation counters differ from actual records"
                )
            v, xy = np.asarray(source["values"]), np.asarray(source["bucket_xy_m"])
            if (
                not np.allclose(
                    template["relative_board_xy_m"], v[:2] - xy, atol=1e-12, rtol=0
                )
                or template["height_m"] != v[2]
                or template["tilt_deg"] != v[3]
                or template["gap_m"] != v[4]
                or template.get("source_record_sha256") != digest(source)
            ):
                raise ValueError("Child promotion geometry/source hash changed")
            margin = source.get("minimum_mouth_clearance_mm")
            margin = (
                float(margin) if margin is not None and np.isfinite(margin) else -1e6
            )
            addition = copy.deepcopy(template)
            addition.update(
                combined_child_role=role,
                combined_child_contract_fingerprint=child["spec"]["fingerprint"],
                combined_child_bank_sha256=child["bank"]["sha256"],
                combined_child_template_index=index,
                combined_source_mouth_clearance_mm=margin,
            )
            candidates.append(
                (tid, conf["captures"], margin, role, source["id"], addition)
            )
    winners = []
    for tid in targets:
        eligible = [c for c in candidates if c[0] == tid]
        if eligible:
            winners.append(max(eligible, key=lambda c: c[1:5])[-1])
    spec = {
        key: copy.deepcopy(reference[key])
        for key in (
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
    }
    spec.update(
        version=VERSION,
        confirmations_per_candidate=10,
        minimum_confirmation_rate=0.8,
        child_artifacts=[
            copy.deepcopy(children[r]["mapping"]) for r in sorted(children)
        ],
        selection="One teacher per development target: highest confirmed capture count, then measured source mouth clearance; deterministic role/source-ID tie break",
        scope="Artifact-only combination of separately completed native development studies; no new native throws, held-out evaluation, or inferred success labels",
    )
    spec["fingerprint"] = digest(spec)
    bounds = (
        spec["wide_bounds"]
        if any(t["development_phase"] == "wide" for t in winners)
        else spec["old_bounds"]
    )
    bank = expanded_bank(original, winners, spec, bounds)
    efforts = [
        dict(
            role=role,
            native_search_attempts=int(c["report"]["native_search_attempts"]),
            confirmation_attempts=int(c["report"]["confirmation_attempts"]),
            prior_native_search_attempts=int(
                c["report"].get("prior_native_search_attempts", 0)
            ),
            prior_confirmation_attempts=int(
                c["report"].get("prior_confirmation_attempts", 0)
            ),
            additions=int(c["report"]["additions"]),
        )
        for role, c in sorted(children.items())
    ]
    prior_counts = {
        (x["prior_native_search_attempts"], x["prior_confirmation_attempts"])
        for x in efforts
    }
    if len(prior_counts) != 1:
        raise ValueError("Children disagree about their shared prior effort")
    prior_search, prior_confirmation = next(iter(prior_counts))
    if (
        prior_search != prior_report["native_search_attempts"]
        or prior_confirmation != prior_report["confirmation_attempts"]
    ):
        raise ValueError("Shared prior effort does not match its frozen report")

    def cell(t):
        tid = t["id"]
        prior = next(c for c in prior_report["cells"] if c["id"] == tid)["search"]
        far = next(c for c in children["far"]["report"]["cells"] if c["id"] == tid)[
            "search"
        ]
        search = {
            k: prior[k] + far[k]
            for k in ("attempts", "captures", "clean", "invalid", "board_bounces")
        }
        search["mean_reward"] = (
            (
                prior["mean_reward"] * prior["attempts"]
                + far["mean_reward"] * far["attempts"]
            )
            / search["attempts"]
            if search["attempts"]
            else 0.0
        )
        fresh = [
            r
            for c in children.values()
            for r in c["rows"]
            if r["phase"] == "confirmation" and r["target_id"] == tid
        ]
        return dict(
            t,
            promotion=next((x for x in winners if x["source_target_id"] == tid), None),
            search=search,
            search_scope="Shared prior v2 native search counted once plus fresh far search",
            fresh_confirmation=statistics(fresh),
            fresh_confirmation_scope="Fresh refinement and far confirmations; excludes shared prior confirmations",
            child_evidence={
                role: next((x for x in c["report"]["cells"] if x["id"] == tid), None)
                for role, c in children.items()
            },
        )

    per_role = {e["role"]: e for e in efforts}
    effort = dict(
        prior_search_attempts=prior_search,
        prior_confirmation_attempts=prior_confirmation,
        refinement_confirmation_attempts=per_role["refinement"][
            "confirmation_attempts"
        ],
        far_search_attempts=per_role["far"]["native_search_attempts"],
        far_confirmation_attempts=per_role["far"]["confirmation_attempts"],
        native_search_attempts_total=prior_search
        + sum(x["native_search_attempts"] for x in efforts),
        native_confirmation_attempts_total=prior_confirmation
        + sum(x["confirmation_attempts"] for x in efforts),
    )
    report = dict(
        status="completed",
        phase="completed",
        contract_fingerprint=spec["fingerprint"],
        old_bank_study_fingerprint=spec["old_bank_study_fingerprint"],
        old_bank_sha256=original["sha256"],
        development_layouts=spec["development_layouts"],
        bounds=bounds,
        original_templates=n,
        expanded_templates=len(bank["templates"]),
        additions=len(winners),
        proposal_bank_sha256=bank["sha256"],
        native_search_attempts=0,
        confirmation_attempts=0,
        records=0,
        shared_prior_effort=dict(
            native_search_attempts=prior_search,
            confirmation_attempts=prior_confirmation,
            counted_once=True,
        ),
        child_effort=efforts,
        effort=effort,
        total_distinct_native_effort=dict(
            native_search_attempts=prior_search
            + sum(x["native_search_attempts"] for x in efforts),
            confirmation_attempts=prior_confirmation
            + sum(x["confirmation_attempts"] for x in efforts),
        ),
        cells=[cell(t) for t in spec["development_layouts"]],
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope=spec["scope"],
    )
    return spec, report, bank


def run(a):
    old_path = a.old_bank_study / "proposal-bank.json"
    original = json.loads(old_path.read_text())
    children = {
        role: load_child(
            role,
            getattr(a, role + "_study"),
            a.prior_search,
            a.old_bank_study,
            getattr(a, role + "_source"),
        )
        for role in CHILD_VERSIONS
    }
    prior_report = json.loads((a.prior_search / "report.json").read_text())
    spec, report, bank = combine(original, children, prior_report)
    if sha256(old_path) != spec["old_bank_file_sha256"]:
        raise RuntimeError("Original bank file changed")
    a.output.mkdir(parents=True, exist_ok=False)
    atomic(a.output / "contract.json", spec)
    atomic(a.output / "expanded-bank.json", bank)
    atomic(a.output / "proposal-bank.json", bank)
    report["expanded_bank_sha256"] = sha256(a.output / "expanded-bank.json")
    report["source_sha256"] = {
        p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")
    }
    atomic(a.output / "report.json", report)
    print(
        json.dumps(
            dict(
                output=str(a.output),
                additions=report["additions"],
                total_distinct_native_effort=report["total_distinct_native_effort"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "refinement-study",
        "far-study",
        "refinement-source",
        "far-source",
        "prior-search",
        "old-bank-study",
        "output",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    run(p.parse_args())
