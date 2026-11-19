"""Targeted native development search for holes in a frozen relative-shot bank.

The former test failures become DEVELOPMENT coordinates in this experiment.
No translated source success label is reused. A new template must capture in a
native rollout and pass separately seeded, perturbed native confirmation.
"""

import argparse
import copy
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np

from .learning_contract import digest, reward, statistics
from .learning_solve import validate_artifacts
from .learning_search import VERSION as BANK_VERSION

VERSION = "sircus-targeted-bank-expansion-v1"
SEED = 20261107


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(
            value,
            indent=2,
            allow_nan=False,
            default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist(),
        )
    )
    temp.replace(path)


def development_targets(bucket, mask):
    """Declare former empty-mask test coordinates as development, never test."""
    xy, mask = np.asarray(bucket), np.asarray(mask, dtype=bool)
    if xy.ndim != 2 or xy.shape[1] != 2 or mask.ndim != 2 or len(mask) != len(xy):
        raise ValueError("Inconsistent original test geometry arrays")
    return [
        dict(
            id=f"old-test-{i:02d}",
            original_test_index=i,
            bucket_xy_m=xy[i].tolist(),
            role="targeted_development",
            reason="Original finite bank had no endpoint-feasible candidate",
        )
        for i in range(len(xy))
        if not mask[i].any()
    ]


def candidate_rank(row):
    return (
        bool(row["route_success"] and not row["invalid"]),
        int(row["route_quality"]),
        reward(row),
    )


def qualified_template(source, confirmations, target_id, bounds, minimum_rate=0.8):
    """Return a measured template, or None; no propagation of a source label."""
    if source.get("invalid") or not source.get("route_success"):
        return None
    attempts = len(confirmations)
    captures = sum(bool(r["route_success"]) and not r["invalid"] for r in confirmations)
    if not attempts or captures / attempts < minimum_rate:
        return None
    if any(r.get("confirmation_source_id") != source["id"] for r in confirmations):
        raise ValueError("Confirmation belongs to a different frozen source")
    value, xy = np.asarray(source["values"]), np.asarray(source["bucket_xy_m"])
    return dict(
        source_trial_id=source["id"],
        source_cell_id=target_id,
        source_target_id=target_id,
        source_phase="targeted_development",
        development_phase=source["bound_phase"],
        action_bounds_version=bounds["version"],
        relative_board_xy_m=(value[:2] - xy).tolist(),
        height_m=float(value[2]),
        tilt_deg=float(value[3]),
        gap_m=float(value[4]),
        source_route_quality=int(source["route_quality"]),
        confirmation=dict(
            attempts=attempts,
            captures=int(captures),
            invalid=sum(bool(r["invalid"]) for r in confirmations),
        ),
        qualification=f"development_confirmation_{captures}_of_{attempts}",
    )


def expanded_bank(original, additions, contract, bounds):
    """Preserve old slots exactly and append explicitly qualified DEV templates."""
    payload = copy.deepcopy(original)
    recorded = payload.pop("sha256")
    if original.get("version") != BANK_VERSION or digest(payload) != recorded:
        raise ValueError("Original bank fingerprint changed")
    if any(t.get("source_phase") != "targeted_development" for t in additions):
        raise ValueError("New sources must be explicitly labeled development")
    for template in additions:
        c = template["confirmation"]
        if not c["attempts"] or c["captures"] <= 0 or c["captures"] > c["attempts"]:
            raise ValueError("A template lacks measured confirmation")
    payload["templates"] += copy.deepcopy(additions)
    payload.update(
        parent_bank_sha256=recorded,
        expansion_contract_fingerprint=contract["fingerprint"],
        action_bounds=copy.deepcopy(bounds),
        source_scope="Original TRAIN teacher prefix preserved; additional sources use former unresolved TEST coordinates explicitly reclassified as DEVELOPMENT. No query success label is inherited.",
        qualification="Original qualifications retained; each addition has independent native perturbed-confirmation counts",
        development_layouts=copy.deepcopy(contract["development_layouts"]),
    )
    payload["sha256"] = digest(payload)
    return payload


def run(a):
    parent_report, parent_cp, parent_sha, parent_fp = validate_artifacts(
        a.parent_study, a.assets
    )
    old_report = json.loads((a.old_bank_study / "report.json").read_text())
    if old_report.get("status") != "completed" or not old_report.get(
        "frozen_evaluation"
    ):
        raise RuntimeError("The original frozen bank study must be complete")
    if old_report["parent_checkpoint_sha256"] != parent_sha:
        raise RuntimeError("Original bank belongs to another parent checkpoint")
    old_bank_path = a.old_bank_study / "proposal-bank.json"
    original = json.loads(old_bank_path.read_text())
    if original["sha256"] != old_report["proposal_bank_sha256"]:
        raise RuntimeError("Original bank identity changed")
    geometry_path = a.old_bank_study / "geometry-test.npz"
    expected = old_report.get("artifacts_sha256", {}).get(geometry_path.name)
    if expected != sha256(geometry_path):
        raise RuntimeError("Original frozen test geometry changed")
    geometry = np.load(geometry_path)
    targets = development_targets(geometry["bucket"], geometry["mask"])
    if len(targets) != 12:
        raise RuntimeError(
            f"Expected the frozen twelve unresolved positions; found {len(targets)}"
        )
    from .expanded_actions import OLD_BOUNDS, WIDE_BOUNDS

    scene = parent_cp["contract"]["scene"]
    spec = dict(
        version=VERSION,
        seed=SEED,
        scene=scene,
        parent_contract_fingerprint=parent_fp,
        parent_checkpoint_sha256=parent_sha,
        old_bank_study_fingerprint=old_report["contract_fingerprint"],
        old_bank_sha256=original["sha256"],
        old_bank_file_sha256=sha256(old_bank_path),
        old_test_geometry_sha256=sha256(geometry_path),
        development_layouts=targets,
        target_role="The old twelve test failures are development targets here; future evaluation must use new untouched layouts",
        old_bounds=OLD_BOUNDS,
        wide_bounds=WIDE_BOUNDS,
        native_per_target_per_phase=a.native_per_target,
        proposal_rounds_per_phase=a.proposal_rounds,
        proposals_per_target_round=a.proposals,
        confirmations_per_candidate=a.confirmations,
        top_candidates=a.top,
        minimum_confirmation_rate=a.minimum_confirmation_rate,
        gpu_wait_timeout_s=a.gpu_wait_timeout,
        confirmation_noise_std=[0.001, 0.001, 0.002, 0.35, 0.002],
        confirmation_distribution="Independent Gaussian action perturbations clipped to active bounds; geometry rejections remain failures",
        physics="Unchanged mjlab1920Hz/240Hz shared capture scorer; rigid held endpoints; same scene and materials",
        scope="Finite development search and expanded action bank; not new held-out policy performance or proof of impossibility",
    )
    spec["fingerprint"] = digest(spec)
    a.output.mkdir(parents=True, exist_ok=False)
    atomic(a.output / "contract.json", spec)
    report = dict(
        status="running",
        phase="initializing",
        contract_fingerprint=spec["fingerprint"],
        old_bank_study_fingerprint=old_report["contract_fingerprint"],
        old_bank_sha256=original["sha256"],
        development_layouts=targets,
        cells=[],
        native_search_attempts=0,
        confirmation_attempts=0,
        geometry_proposals=0,
        additions=0,
    )
    started = time.monotonic()
    records, searched, confirmation = [], {t["id"]: [] for t in targets}, {}
    promoted = {}
    by_id = {t["id"]: t for t in targets}
    rng = np.random.default_rng(SEED)

    def publish(phase):
        report.update(
            phase=phase,
            elapsed_s=time.monotonic() - started,
            timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
        report["cells"] = [
            dict(
                target,
                search=statistics(searched[target["id"]]),
                promotion=promoted.get(target["id"]),
                unresolved_reason=(
                    None
                    if target["id"] in promoted
                    else "No qualifying source found by the finite search yet"
                ),
            )
            for target in targets
        ]
        atomic(a.output / "report.json", report)
        print(
            json.dumps(
                dict(
                    phase=phase,
                    native_search_attempts=report["native_search_attempts"],
                    confirmation_attempts=report["confirmation_attempts"],
                    additions=len(promoted),
                    elapsed_s=round(report["elapsed_s"], 1),
                )
            ),
            flush=True,
        )

    def save(rows, metadata):
        with (a.output / "shots.jsonl").open("a") as out:
            for row, meta in zip(rows, metadata):
                row.update(
                    meta,
                    id=f"expand-shot-{len(records):07d}",
                    source_phase="targeted_development",
                )
                records.append(row)
                out.write(json.dumps(row, allow_nan=False) + "\n")
        return rows

    from .expand_geometry import proposals, audit_values
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

    # GPU work is serialized with the existing native jobs. No other process is
    # stopped, signaled, moved, or reconfigured by this finite experiment.
    with gpu_slot(a.gpu_lock, timeout_s=a.gpu_wait_timeout, notify=waiting):
        report.pop("waiting_for_4090", None)
        publish("gpu-acquired")
        import torch

        torch.set_num_threads(2)
        for phase, bounds in (("old", OLD_BOUNDS), ("wide", WIDE_BOUNDS)):
            active = [t for t in targets if t["id"] not in promoted]
            if not active:
                break
            environment = ExpandedEnvironment(scene, a.assets, n=a.batch, bounds=bounds)
            scheduled = {t["id"]: 0 for t in active}
            queue = []

            def flush(force=False):
                while len(queue) >= a.batch or (force and queue):
                    packet = queue[: a.batch]
                    del queue[: len(packet)]
                    xy = np.asarray([p["xy"] for p in packet])
                    values = np.asarray([p["value"] for p in packet])
                    rows = run_bank_rows(
                        environment,
                        xy,
                        values,
                        np.asarray([p["q10"] for p in packet]),
                        np.asarray([p["q14"] for p in packet]),
                        np.ones(len(packet), dtype=bool),
                    )
                    meta = [
                        dict(
                            phase="search",
                            bound_phase=phase,
                            target_id=p["target_id"],
                            requested_values=p["value"],
                        )
                        for p in packet
                    ]
                    rows = save(rows, meta)
                    for row in rows:
                        searched[row["target_id"]].append(row)
                    report["native_search_attempts"] += len(rows)
                    publish("search-" + phase)

            for round_index in range(a.proposal_rounds):
                pending = [
                    t for t in active if scheduled[t["id"]] < a.native_per_target
                ]
                if not pending:
                    break
                xy = np.repeat(
                    np.asarray([t["bucket_xy_m"] for t in pending]), a.proposals, axis=0
                )
                values = []
                for target in pending:
                    old = searched[target["id"]]
                    best = max(old, key=reward)["values"] if old else None
                    values.append(
                        proposals(
                            np.repeat([target["bucket_xy_m"]], a.proposals, axis=0),
                            rng,
                            scene,
                            stage=(
                                "refine"
                                if best is not None and round_index >= 4
                                else "broad"
                            ),
                            best=best,
                            bounds=bounds,
                        )
                    )
                values = np.concatenate(values)
                mask, detail = audit_values(
                    scene, a.assets, xy, values[:, None, :], bounds=bounds
                )
                report["geometry_proposals"] += len(values)
                for j, target in enumerate(pending):
                    for local in np.flatnonzero(
                        mask[j * a.proposals : (j + 1) * a.proposals, 0]
                    ):
                        if scheduled[target["id"]] >= a.native_per_target:
                            break
                        i = j * a.proposals + int(local)
                        queue.append(
                            dict(
                                target_id=target["id"],
                                xy=target["bucket_xy_m"],
                                value=values[i].tolist(),
                                q10=detail["q10"][i, 0],
                                q14=detail["q14"][i, 0],
                            )
                        )
                        scheduled[target["id"]] += 1
                flush()
                publish("geometry-" + phase)
            flush(True)

            # Freeze top captured settings, then collect independent perturbed
            # confirmation with geometry rejects retained in the denominator.
            pending_confirmation = []
            for target in active:
                candidates = sorted(
                    [
                        r
                        for r in searched[target["id"]]
                        if r["bound_phase"] == phase
                        and r["route_success"]
                        and not r["invalid"]
                    ],
                    key=candidate_rank,
                    reverse=True,
                )[: a.top]
                for candidate in candidates:
                    values = np.asarray(candidate["values"]) + rng.normal(
                        0, spec["confirmation_noise_std"], (a.confirmations, 5)
                    )
                    values = np.clip(
                        values, np.asarray(bounds["low"]), np.asarray(bounds["high"])
                    )
                    for value in values:
                        pending_confirmation.append(
                            dict(target_id=target["id"], source=candidate, value=value)
                        )
            for offset in range(0, len(pending_confirmation), a.batch):
                packet = pending_confirmation[offset : offset + a.batch]
                xy = np.asarray([by_id[p["target_id"]]["bucket_xy_m"] for p in packet])
                values = np.asarray([p["value"] for p in packet])
                mask, detail = audit_values(
                    scene, a.assets, xy, values[:, None, :], bounds=bounds
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
                for row, value in zip(rows, values):
                    row.setdefault("values", value.tolist())
                rows = save(
                    rows,
                    [
                        dict(
                            phase="confirmation",
                            bound_phase=phase,
                            target_id=p["target_id"],
                            confirmation_source_id=p["source"]["id"],
                        )
                        for p in packet
                    ],
                )
                for row in rows:
                    confirmation.setdefault(row["confirmation_source_id"], []).append(
                        row
                    )
                report["confirmation_attempts"] += len(rows)
                publish("confirmation-" + phase)
            for target in active:
                qualified = []
                for source in searched[target["id"]]:
                    if source["bound_phase"] != phase:
                        continue
                    template = qualified_template(
                        source,
                        confirmation.get(source["id"], []),
                        target["id"],
                        bounds,
                        a.minimum_confirmation_rate,
                    )
                    if template:
                        qualified.append((template, source))
                if qualified:
                    template, source = max(
                        qualified,
                        key=lambda pair: (
                            pair[0]["confirmation"]["captures"],
                            candidate_rank(pair[1]),
                        ),
                    )
                    promoted[target["id"]] = template
            report["additions"] = len(promoted)
            publish("phase-complete-" + phase)
            del environment
            torch.cuda.empty_cache()
    final_bounds = (
        WIDE_BOUNDS
        if any(t["development_phase"] == "wide" for t in promoted.values())
        else OLD_BOUNDS
    )
    bank = expanded_bank(original, list(promoted.values()), spec, final_bounds)
    atomic(a.output / "expanded-bank.json", bank)
    atomic(a.output / "proposal-bank.json", bank)
    report.update(
        status="completed",
        records=len(records),
        expanded_bank_sha256=sha256(a.output / "expanded-bank.json"),
        proposal_bank_sha256=bank["sha256"],
        bounds=final_bounds,
        original_templates=len(original["templates"]),
        expanded_templates=len(bank["templates"]),
        native_search_statistics=statistics(
            [r for r in records if r["phase"] == "search"]
        ),
        confirmation_statistics=statistics(
            [r for r in records if r["phase"] == "confirmation"]
        ),
        versions={
            n: importlib.metadata.version(n)
            for n in ("mjlab", "mujoco", "mujoco-warp", "torch", "warp-lang")
        },
        source_sha256={p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
    )
    publish("completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-study", "old-bank-study", "assets", "output", "gpu-lock"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--native-per-target", type=int, default=256)
    parser.add_argument("--proposals", type=int, default=128)
    parser.add_argument("--proposal-rounds", type=int, default=24)
    parser.add_argument("--confirmations", type=int, default=10)
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--minimum-confirmation-rate", type=float, default=0.8)
    parser.add_argument("--gpu-wait-timeout", type=float, default=1800)
    args = parser.parse_args()
    if not (
        1 <= args.batch <= 256
        and 1 <= args.native_per_target <= 2048
        and 1 <= args.proposals <= 512
        and 1 <= args.proposal_rounds <= 128
        and 3 <= args.confirmations <= 100
        and 1 <= args.top <= 8
        and 0.5 <= args.minimum_confirmation_rate <= 1
        and 1 <= args.gpu_wait_timeout <= 7200
    ):
        parser.error("Finite positive search/confirmation budgets required")
    run(args)
