"""Fresh robustness confirmation of high-clearance, diverse development shots.

Prior measured sources remain separate from this run's native trials. This is
finite development selection, not a held-out policy evaluation or a new search
of bucket positions that have no captured source.
"""

import argparse
import copy
import datetime
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np

from .expand_bank import atomic, expanded_bank, qualified_template, sha256
from .learning_contract import digest, statistics
from .learning_solve import validate_artifacts

VERSION = "sircus-targeted-bank-refinement-v1"
SEED = 20261108
NOISE = [0.001, 0.001, 0.002, 0.35, 0.002]


def clearance(row):
    value = row.get("minimum_mouth_clearance_mm")
    return float(value) if value is not None and np.isfinite(value) else -1e6


def select_sources(rows, targets, bounds, limit=20):
    """Deterministic 75% measured mouth margin / 25% action diversity selection.

    Only prior actual route captures qualify. Action normalization is fixed by
    the declared union bounds; repeated settings never consume another slot.
    No additional success labels or held-out coordinates are consulted.
    """
    if not 1 <= limit <= 40:
        raise ValueError("Finite selection limit must be between 1 and 40")
    target_ids = {t["id"] for t in targets}
    lo, hi = np.asarray(bounds["low"]), np.asarray(bounds["high"])
    selected = []
    for target in sorted(target_ids):
        pool = []
        seen = set()
        for row in sorted(rows, key=lambda r: (-clearance(r), r["id"])):
            if (
                row.get("phase") != "search"
                or row.get("target_id") != target
                or row.get("source_phase") != "targeted_development"
                or row.get("invalid")
                or not row.get("route_success")
            ):
                continue
            value = np.asarray(row["values"], dtype=float)
            if value.shape != (5,) or not np.isfinite(value).all():
                raise ValueError("A captured source has malformed actions")
            key = tuple(np.round(value, 10))
            if key not in seen:
                seen.add(key)
                pool.append(row)
        chosen = []
        while pool and len(chosen) < limit:
            if not chosen:
                pick = 0
            else:
                prior = np.asarray([r["values"] for r in chosen])
                scores = []
                for row in pool:
                    distance = np.linalg.norm(
                        (prior - row["values"]) / (hi - lo), axis=1
                    ).min() / np.sqrt(5)
                    # A whole-ball center clearance of 50 mm saturates this
                    # acquisition score; this does not change the shot scorer.
                    margin = np.clip(clearance(row) / 50, 0, 1)
                    scores.append(0.75 * margin + 0.25 * min(distance, 1))
                pick = int(np.argmax(scores))
            chosen.append(pool.pop(pick))
        selected.extend(chosen)
    return selected


def make_promotions(sources, confirmations, spec):
    """Pick one measured robust teacher per target; do not weaken 8/10."""
    result = []
    for target in spec["development_layouts"]:
        qualified = []
        for source in sources:
            if source["target_id"] != target["id"]:
                continue
            rows = confirmations.get(source["id"], [])
            if len(rows) != spec["confirmations_per_candidate"]:
                continue
            template = qualified_template(
                source,
                rows,
                target["id"],
                spec[source["bound_phase"] + "_bounds"],
                spec["minimum_confirmation_rate"],
            )
            if template:
                template.update(
                    source_study_contract_fingerprint=spec[
                        "prior_search_contract_fingerprint"
                    ],
                    source_record_sha256=digest(source),
                    confirmation_study_contract_fingerprint=spec["fingerprint"],
                )
                qualified.append((source, template))
        if qualified:
            source, template = max(
                qualified,
                key=lambda pair: (
                    pair[1]["confirmation"]["captures"],
                    clearance(pair[0]),
                    pair[0]["route_quality"],
                    pair[0]["id"],
                ),
            )
            result.append(template)
    return result


def run(a):
    _, parent_checkpoint, parent_sha, parent_fp = validate_artifacts(
        a.parent_study, a.assets
    )
    prior = a.prior_search
    prior_spec = json.loads((prior / "contract.json").read_text())
    prior_report = json.loads((prior / "report.json").read_text())
    p = dict(prior_spec)
    fp = p.pop("fingerprint")
    if digest(p) != fp or prior_report["contract_fingerprint"] != fp:
        raise RuntimeError("Prior search contract changed")
    if prior_report.get("status") != "completed":
        raise RuntimeError("Prior native development search must be complete")
    if (
        prior_spec["parent_checkpoint_sha256"] != parent_sha
        or prior_spec["parent_contract_fingerprint"] != parent_fp
        or prior_spec["scene"] != parent_checkpoint["contract"]["scene"]
    ):
        raise RuntimeError("Prior development search belongs to another parent scene")
    all_rows = [
        json.loads(line) for line in (prior / "shots.jsonl").read_text().splitlines()
    ]
    if len(all_rows) != prior_report["records"]:
        raise RuntimeError("Prior native record count changed")
    old_path = a.old_bank_study / "proposal-bank.json"
    original = json.loads(old_path.read_text())
    if sha256(old_path) != prior_spec["old_bank_file_sha256"]:
        raise RuntimeError("Preserved original bank changed")
    targets = prior_spec["development_layouts"]
    sources = select_sources(
        all_rows, targets, prior_spec["wide_bounds"], a.sources_per_target
    )
    if not sources:
        raise RuntimeError("No actual prior native capture can be confirmed")
    scene = prior_spec["scene"]
    spec = copy.deepcopy(prior_spec)
    spec.pop("fingerprint")
    # Remove previous budget fields so inherited and fresh effort cannot be
    # mistaken for one native search or an identical source-selection rule.
    for key in (
        "native_per_target_per_phase",
        "proposal_rounds_per_phase",
        "proposals_per_target_round",
        "top_candidates",
    ):
        spec.pop(key, None)
    spec.update(
        version=VERSION,
        seed=SEED,
        prior_search_contract_fingerprint=fp,
        prior_search_report_sha256=sha256(prior / "report.json"),
        prior_search_records_sha256=sha256(prior / "shots.jsonl"),
        prior_native_search_attempts=prior_report["native_search_attempts"],
        prior_confirmation_attempts=prior_report["confirmation_attempts"],
        selected_source_ids=[r["id"] for r in sources],
        selected_source_records_sha256=[digest(r) for r in sources],
        sources_per_target=a.sources_per_target,
        source_selection="Deterministic captured-only acquisition: 75% clipped measured mouth clearance plus 25% normalized minimum action distance; first source has greatest margin; exact action duplicates removed",
        confirmations_per_candidate=10,
        minimum_confirmation_rate=0.8,
        confirmation_noise_std=NOISE,
        confirmation_distribution="Fresh independent Gaussian action perturbations clipped to each source's original declared action bounds; fresh CPU IK; geometry rejects stay in denominator",
        gpu_wait_timeout_s=a.gpu_wait_timeout,
        scope="Finite robustness continuation on previously measured development captures; inherited throws reported separately; no fresh search of uncaptured targets or held-out evaluation",
    )
    spec["fingerprint"] = digest(spec)
    a.output.mkdir(parents=True, exist_ok=False)
    atomic(a.output / "contract.json", spec)
    with (a.output / "source-search.jsonl").open("w") as out:
        for row in sources:
            out.write(json.dumps(row, allow_nan=False) + "\n")
    report = dict(
        status="running",
        phase="initializing",
        contract_fingerprint=spec["fingerprint"],
        old_bank_study_fingerprint=spec["old_bank_study_fingerprint"],
        old_bank_sha256=original["sha256"],
        development_layouts=targets,
        prior_native_search_attempts=prior_report["native_search_attempts"],
        prior_confirmation_attempts=prior_report["confirmation_attempts"],
        retained_source_captures=len(sources),
        source_search_sha256=sha256(a.output / "source-search.jsonl"),
        native_search_attempts=0,
        confirmation_attempts=0,
        geometry_proposals=0,
        additions=0,
        cells=[],
    )
    started = time.monotonic()
    records, confirmations = [], {}

    def publish(phase):
        promotions = make_promotions(sources, confirmations, spec)
        report.update(
            phase=phase,
            elapsed_s=time.monotonic() - started,
            timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            additions=len(promotions),
        )
        by_target = {t["source_target_id"]: t for t in promotions}
        report["cells"] = [
            dict(
                t,
                retained_source_captures=sum(
                    s["target_id"] == t["id"] for s in sources
                ),
                fresh_confirmation=statistics(
                    [r for r in records if r["target_id"] == t["id"]]
                ),
                promotion=by_target.get(t["id"]),
                unresolved_reason=(
                    None
                    if t["id"] in by_target
                    else "No independently confirmed source from this finite continuation"
                ),
            )
            for t in targets
        ]
        atomic(a.output / "report.json", report)
        print(
            json.dumps(
                dict(
                    phase=phase,
                    fresh_confirmations=len(records),
                    additions=len(promotions),
                    elapsed_s=round(report["elapsed_s"], 1),
                )
            ),
            flush=True,
        )

    from .expand_geometry import audit_values
    from .expanded_env import ExpandedEnvironment
    from .bank_study import run_bank_rows
    from .gpu_slot import gpu_slot

    last_wait_notice = -float("inf")

    def waiting(info):
        nonlocal last_wait_notice
        report["waiting_for_4090"] = info
        if info["wait_s"] - last_wait_notice >= 30:
            publish("waiting-for-4090")
            last_wait_notice = info["wait_s"]

    rng = np.random.default_rng(SEED)
    packets = []
    for source in sources:
        bounds = spec[source["bound_phase"] + "_bounds"]
        values = np.clip(
            np.asarray(source["values"]) + rng.normal(0, NOISE, (10, 5)),
            bounds["low"],
            bounds["high"],
        )
        packets.extend(dict(source=source, value=v) for v in values)
    with gpu_slot(a.gpu_lock, timeout_s=a.gpu_wait_timeout, notify=waiting):
        report.pop("waiting_for_4090", None)
        publish("gpu-acquired")
        import torch

        torch.set_num_threads(2)
        environment = ExpandedEnvironment(
            scene, a.assets, n=a.batch, bounds=spec["wide_bounds"]
        )
        for phase in ("old", "wide"):
            selected = [p for p in packets if p["source"]["bound_phase"] == phase]
            for offset in range(0, len(selected), a.batch):
                packet = selected[offset : offset + a.batch]
                xy = np.asarray([p["source"]["bucket_xy_m"] for p in packet])
                values = np.asarray([p["value"] for p in packet])
                mask, detail = audit_values(
                    scene,
                    a.assets,
                    xy,
                    values[:, None, :],
                    bounds=spec[phase + "_bounds"],
                )
                rows = run_bank_rows(
                    environment,
                    xy,
                    values,
                    detail["q10"][:, 0],
                    detail["q14"][:, 0],
                    mask[:, 0],
                    record=False,
                )
                with (a.output / "shots.jsonl").open("a") as out:
                    for row, p in zip(rows, packet):
                        source = p["source"]
                        row.update(
                            id=f"refine-shot-{len(records):07d}",
                            phase="confirmation",
                            bound_phase=phase,
                            target_id=source["target_id"],
                            values=p["value"].tolist(),
                            source_phase="targeted_development",
                            confirmation_source_id=source["id"],
                            source_study_contract_fingerprint=fp,
                        )
                        records.append(row)
                        confirmations.setdefault(source["id"], []).append(row)
                        out.write(json.dumps(row, allow_nan=False) + "\n")
                report["confirmation_attempts"] += len(rows)
                report["geometry_proposals"] += len(values)
                publish("confirmation-" + phase)
    additions = make_promotions(sources, confirmations, spec)
    bounds = (
        spec["wide_bounds"]
        if any(t["development_phase"] == "wide" for t in additions)
        else spec["old_bounds"]
    )
    bank = expanded_bank(original, additions, spec, bounds)
    atomic(a.output / "expanded-bank.json", bank)
    atomic(a.output / "proposal-bank.json", bank)
    report.update(
        status="completed",
        records=len(records),
        expanded_bank_sha256=sha256(a.output / "expanded-bank.json"),
        proposal_bank_sha256=bank["sha256"],
        bounds=bounds,
        original_templates=len(original["templates"]),
        expanded_templates=len(bank["templates"]),
        native_search_statistics=statistics([]),
        confirmation_statistics=statistics(records),
        versions={
            n: importlib.metadata.version(n)
            for n in ("mjlab", "mujoco", "mujoco-warp", "torch", "warp-lang")
        },
        source_sha256={p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
    )
    publish("completed")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "parent-study",
        "prior-search",
        "old-bank-study",
        "assets",
        "output",
        "gpu-lock",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--sources-per-target", type=int, default=20)
    p.add_argument("--gpu-wait-timeout", type=float, default=1800)
    a = p.parse_args()
    if not (
        1 <= a.batch <= 256
        and 1 <= a.sources_per_target <= 40
        and 1 <= a.gpu_wait_timeout <= 7200
    ):
        p.error("Finite batch, source selection and GPU wait budgets required")
    run(a)
