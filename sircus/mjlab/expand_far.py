"""Bounded fresh native exploration for six still-uncaptured DEV positions.

Prior near-misses guide proposals only. A table/robot-assisted entry is never a
capture teacher. Native physics/scoring remain unchanged, and promotion still
requires a real source capture plus at least eight of ten fresh confirmations.
"""

import argparse
import copy
import datetime
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np

from .expand_bank import (
    atomic,
    candidate_rank,
    expanded_bank,
    qualified_template,
    sha256,
)
from .expanded_actions import OLD_BOUNDS, WIDE_BOUNDS
from .expand_geometry import maximum_gap, outer_geometry_mask
from .learning_contract import digest, statistics
from .learning_solve import validate_artifacts

VERSION = "sircus-far-development-search-v1"
SEED = 20261109
MISSING_INDICES = (8, 9, 12, 23, 35, 41)
NOISE = [0.001, 0.001, 0.002, 0.35, 0.002]


def proposal_rank(row):
    """Acquisition prior only; never a replacement for the qualified scorer.

    A miss eventually landing on the table is not discarded: the native stream
    records external contacts until termination. Robot/unattributed contact is
    penalized strongly, while final table contacts receive a small penalty.
    """
    if row.get("invalid") or not row.get("board_bounce"):
        return -10.0
    peaks = row.get("contact_peaks_N", {})
    distance = float(row.get("closest_opening_distance_m", 2.0))
    if not np.isfinite(distance):
        return -10.0
    return float(
        0.5 * np.exp(-max(distance, 0) / 0.12)
        + 0.1 * bool(row.get("whole_ball_entry"))
        + 0.1 * min(float(row.get("max_retention_s", 0)), 1)
        - 0.8 * (float(peaks.get("robot", 0)) > 0.01)
        - 0.8 * (float(peaks.get("unattributed", 0)) > 0.01)
        - 0.04 * (float(peaks.get("table", 0)) > 0.01)
        - 0.05 * (float(peaks.get("floor", 0)) > 0.01)
    )


def select_seeds(rows, target_id, limit=16):
    """Deterministic high-acquisition/diverse measured near-miss seeds."""
    if not 1 <= limit <= 32:
        raise ValueError("Seed limit must be between 1 and 32")
    low, high = np.asarray(WIDE_BOUNDS["low"]), np.asarray(WIDE_BOUNDS["high"])
    pool, seen, chosen = [], set(), []
    for row in sorted(rows, key=lambda r: (-proposal_rank(r), r["id"])):
        if (
            row.get("phase") != "search"
            or row.get("target_id") != target_id
            or row.get("invalid")
        ):
            continue
        if row.get("source_phase") != "targeted_development" or not row.get(
            "board_bounce"
        ):
            continue
        value = np.asarray(row.get("values"), dtype=float)
        if value.shape != (5,) or not np.isfinite(value).all():
            raise ValueError("Malformed native near-miss action")
        if ((value < low - 1e-9) | (value > high + 1e-9)).any():
            continue
        key = tuple(np.round(value, 9))
        if key not in seen:
            seen.add(key)
            pool.append(row)
    while pool and len(chosen) < limit:
        if not chosen:
            index = 0
        else:
            prior = np.asarray([r["values"] for r in chosen])
            scores = [
                proposal_rank(row)
                + 0.15
                * min(
                    1,
                    float(
                        np.linalg.norm(
                            (prior - row["values"]) / (high - low), axis=1
                        ).min()
                    ),
                )
                for row in pool
            ]
            index = int(np.argmax(scores))
        chosen.append(pool.pop(index))
    return chosen


def far_proposals(bucket_xy, count, rng, scene, seeds=(), *, round_index=0):
    """Mix full-height space-filling, target-relative ballistic and local seeds.

    The old global proposal prior restricted board height below400mm. This one
    explicitly samples the full120–600mm domain,16azimuth sectors around the
    target, and the complete8–60degree tilt interval. Exact CPU/native gates
    retain authority; geometric/ballistic assumptions are not success labels.
    """
    if not 1 <= count <= 2048:
        raise ValueError("Finite proposal count1..2048 required")
    xy = np.asarray(bucket_xy, dtype=float)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError("Finite bucket XY required")
    low, high = np.asarray(WIDE_BOUNDS["low"]), np.asarray(WIDE_BOUNDS["high"])
    values = rng.uniform(low, high, (count, 5))
    # Balanced low/mid/high board heights and low/high tilt bins, independently
    # shuffled. This avoids collapsing all proposals onto the old low-board prior.
    unit_h = (rng.permutation(count) + rng.random(count)) / count
    unit_t = (rng.permutation(count) + rng.random(count)) / count
    values[:, 2] = low[2] + unit_h * (high[2] - low[2])
    values[:, 3] = low[3] + unit_t * (high[3] - low[3])
    kinds = np.arange(count) % 3
    rng.shuffle(kinds)
    relative = kinds == 1
    n_relative = int(relative.sum())
    azimuth = (
        2 * np.pi * (rng.integers(0, 16, n_relative) + rng.random(n_relative)) / 16
    )
    distance = rng.uniform(0.25, 1.20, n_relative)
    values[relative, :2] = (
        xy - np.column_stack((np.cos(azimuth), np.sin(azimuth))) * distance[:, None]
    )
    distance = np.linalg.norm(xy - values[:, :2], axis=1)
    angle = np.deg2rad(values[:, 3])
    restitution = rng.uniform(0.55, 0.90, count)
    horizontal = (1 + restitution) * np.sin(angle) * np.cos(angle)
    vertical = (1 + restitution) * np.cos(angle) ** 2 - 1
    center_at_entry = (
        scene["bucket"]["assumed_bottom_thickness_m"]
        + scene["bucket"]["assumed_interior_depth_m"]
        - scene["ball"]["radius_m"]
    )
    denominator = (
        4
        * horizontal**2
        * (values[:, 2] - center_at_entry + vertical / horizontal * distance)
    )
    ballistic = distance**2 / np.maximum(denominator, 0.015)
    cap = np.minimum(high[4], maximum_gap(values[:, :3], scene))
    # Half ballistic, quarter uniform gap, quarter high gap: low-restitution
    # approximations must not erase high-energy or short-hop possibilities.
    mode = rng.random(count)
    values[:, 4] = np.minimum(ballistic * rng.uniform(0.70, 1.35, count), cap)
    uniform = (mode >= 0.5) & (mode < 0.75)
    values[uniform, 4] = low[4] + rng.random(uniform.sum()) * np.maximum(
        cap[uniform] - low[4], 0
    )
    high_gap = mode >= 0.75
    values[high_gap, 4] = cap[high_gap] * rng.uniform(0.85, 1, high_gap.sum())
    meta = [
        dict(
            proposal_family="target-relative" if kind == 1 else "full-height-global",
            seed_source_id=None,
            round_index=round_index,
        )
        for kind in kinds
    ]
    if seeds:
        local = np.flatnonzero(kinds == 2)
        # Multiple scales remain present in every round, rather than annealing
        # to a single brittle v2 near-miss configuration.
        scales = np.asarray(
            [
                [0.006, 0.006, 0.006, 0.75, 0.008],
                [0.020, 0.025, 0.025, 2.5, 0.030],
                [0.055, 0.070, 0.065, 7.0, 0.075],
            ]
        )
        for i in local:
            source = seeds[int(rng.integers(len(seeds)))]
            values[i] = np.asarray(source["values"]) + rng.normal(
                0, scales[int(rng.integers(3))]
            )
            meta[i].update(
                proposal_family="measured-near-miss-local",
                seed_source_id=source["id"],
                seed_source_record_sha256=digest(source),
            )
    values = np.nan_to_num(np.clip(values, low, high), neginf=low[4], posinf=high[4])
    return values, meta


def load_prior(a):
    _, cp, parent_sha, parent_fp = validate_artifacts(a.parent_study, a.assets)
    prior_spec = json.loads((a.prior_search / "contract.json").read_text())
    prior_report = json.loads((a.prior_search / "report.json").read_text())
    payload = dict(prior_spec)
    fp = payload.pop("fingerprint")
    if digest(payload) != fp or prior_report.get("contract_fingerprint") != fp:
        raise RuntimeError("Prior search contract changed")
    if prior_report.get("status") != "completed":
        raise RuntimeError("Prior native search must complete")
    if (
        prior_spec["parent_checkpoint_sha256"] != parent_sha
        or prior_spec["parent_contract_fingerprint"] != parent_fp
        or prior_spec["scene"] != cp["contract"]["scene"]
    ):
        raise RuntimeError("Prior search belongs to another parent scene")
    rows = [
        json.loads(line)
        for line in (a.prior_search / "shots.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if len(rows) != prior_report["records"]:
        raise RuntimeError("Prior native record count changed")
    targets = prior_spec["development_layouts"]
    if len(targets) != 12 or len({t["id"] for t in targets}) != 12:
        raise RuntimeError("Expected all twelve declared development targets")
    selected = [t for t in targets if t["original_test_index"] in MISSING_INDICES]
    if len(selected) != 6 or any(t["role"] != "targeted_development" for t in targets):
        raise RuntimeError("The six missing development targets changed")
    if any(
        r.get("phase") == "search"
        and r.get("target_id") in {t["id"] for t in selected}
        and r.get("route_success")
        and not r.get("invalid")
        for r in rows
    ):
        raise RuntimeError(
            "A selected target already has a qualified source; use confirmation continuation instead"
        )
    original_path = a.old_bank_study / "proposal-bank.json"
    original = json.loads(original_path.read_text())
    original_payload = dict(original)
    original_sha = original_payload.pop("sha256")
    old_report = json.loads((a.old_bank_study / "report.json").read_text())
    if (
        sha256(original_path) != prior_spec["old_bank_file_sha256"]
        or original["sha256"] != prior_spec["old_bank_sha256"]
        or digest(original_payload) != original_sha
        or old_report["contract_fingerprint"]
        != prior_spec["old_bank_study_fingerprint"]
        or sha256(a.old_bank_study / "geometry-test.npz")
        != prior_spec["old_test_geometry_sha256"]
    ):
        raise RuntimeError("Original bank identity changed")
    seeds = [r for t in selected for r in select_seeds(rows, t["id"], a.seed_sources)]
    return prior_spec, prior_report, rows, original, selected, seeds


def run(a):
    prior_spec, prior_report, prior_rows, original, selected, seed_sources = load_prior(
        a
    )
    scene = prior_spec["scene"]
    spec = copy.deepcopy(prior_spec)
    spec.pop("fingerprint")
    for key in (
        "native_per_target_per_phase",
        "proposal_rounds_per_phase",
        "proposals_per_target_round",
    ):
        spec.pop(key, None)
    spec.update(
        version=VERSION,
        seed=SEED,
        prior_search_contract_fingerprint=prior_spec["fingerprint"],
        prior_search_report_sha256=sha256(a.prior_search / "report.json"),
        prior_search_records_sha256=sha256(a.prior_search / "shots.jsonl"),
        selected_target_ids=[t["id"] for t in selected],
        native_per_target=a.native_per_target,
        proposal_rounds=a.proposal_rounds,
        proposals_per_target_round=a.proposals,
        seed_sources_per_target=a.seed_sources,
        selected_prior_seed_ids=[r["id"] for r in seed_sources],
        selected_prior_seed_records_sha256=[digest(r) for r in seed_sources],
        prior_native_search_attempts=prior_report["native_search_attempts"],
        prior_confirmation_attempts=prior_report["confirmation_attempts"],
        acquisition_prior="0.5exp(-opening_distance/0.12)+0.1entry+0.1clipped_retention-0.8robot_contact-0.8unattributed-0.04table-0.05floor; proposal-only, unchanged final scorer",
        proposal_mixture="Equal full-height global,16-sector target-relative, and multiscale diverse measured near-miss local; full120–600mm height and8–60degree tilt domains",
        top_candidates=a.top,
        confirmations_per_candidate=10,
        minimum_confirmation_rate=0.8,
        confirmation_noise_std=NOISE,
        gpu_wait_timeout_s=a.gpu_wait_timeout,
        confirmation_distribution="Fresh independent Gaussian perturbations clipped to WIDE bounds; fresh CPU IK; every geometry rejection remains in the denominator",
        ik_limitation="The unchanged multistart CPU IK seed depends on batch slot; identical requested actions may produce different fresh branch/feasibility outcomes. Cached q is reused within each native packet, not across fresh perturbation audits. No guard or IK setting is relaxed here.",
        scope="Finite new native development search of six uncaptured positions; no inherited native trials, no held-out inference, no global feasibility claim",
    )
    spec["fingerprint"] = digest(spec)
    a.output.mkdir(parents=True, exist_ok=False)
    atomic(a.output / "contract.json", spec)
    (a.output / "shots.jsonl").write_text("")
    with (a.output / "source-near-misses.jsonl").open("w") as out:
        for row in seed_sources:
            out.write(json.dumps(row, allow_nan=False) + "\n")
    report = dict(
        status="running",
        phase="initializing",
        contract_fingerprint=spec["fingerprint"],
        old_bank_study_fingerprint=spec["old_bank_study_fingerprint"],
        old_bank_sha256=original["sha256"],
        development_layouts=spec["development_layouts"],
        selected_target_ids=spec["selected_target_ids"],
        prior_native_search_attempts=prior_report["native_search_attempts"],
        prior_confirmation_attempts=prior_report["confirmation_attempts"],
        retained_proposal_sources=len(seed_sources),
        source_near_misses_sha256=sha256(a.output / "source-near-misses.jsonl"),
        native_search_attempts=0,
        confirmation_attempts=0,
        geometry_proposals=0,
        additions=0,
        cells=[],
    )
    started = time.monotonic()
    records, confirmations, promotions = [], {}, {}
    searched = {t["id"]: [] for t in selected}
    cache_index = 0

    def publish(phase):
        report.update(
            phase=phase,
            elapsed_s=time.monotonic() - started,
            timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            additions=len(promotions),
        )
        report["cells"] = [
            dict(
                t,
                search=statistics(searched.get(t["id"], [])),
                searched_in_this_continuation=t["id"] in searched,
                promotion=promotions.get(t["id"]),
                unresolved_reason=(
                    None
                    if t["id"] in promotions
                    else (
                        "No 8/10 qualified source in this finite missing-target search"
                        if t["id"] in searched
                        else "Not searched here; prior captured sources are handled by separate confirmation continuation"
                    )
                ),
            )
            for t in spec["development_layouts"]
        ]
        atomic(a.output / "report.json", report)
        print(
            json.dumps(
                dict(
                    phase=phase,
                    native_search_attempts=report["native_search_attempts"],
                    confirmation_attempts=report["confirmation_attempts"],
                    additions=len(promotions),
                    elapsed_s=round(report["elapsed_s"], 1),
                )
            ),
            flush=True,
        )

    publish("initializing")
    from .bank_study import run_bank_rows
    from .expand_geometry import audit_values
    from .expanded_env import ExpandedEnvironment
    from .gpu_slot import gpu_slot

    def waiting(info):
        report["waiting_for_4090"] = info
        publish("waiting-for-4090")

    def save_packet(environment, packet, *, phase, audited=None):
        nonlocal cache_index
        xy = np.asarray([p["bucket_xy_m"] for p in packet])
        values = np.asarray([p["values"] for p in packet])
        if audited is None:
            mask, detail = audit_values(
                scene, a.assets, xy, values[:, None], bounds=WIDE_BOUNDS
            )
            report["geometry_proposals"] += len(values)
            q10, q14, eligible = detail["q10"][:, 0], detail["q14"][:, 0], mask[:, 0]
        else:
            q10, q14 = np.asarray([p["q10"] for p in packet]), np.asarray(
                [p["q14"] for p in packet]
            )
            eligible = np.ones(len(packet), dtype=bool)
        cache_name = f"geometry-native-{cache_index:04d}.npz"
        cache_index += 1
        np.savez_compressed(
            a.output / cache_name,
            bucket=xy,
            values=values,
            q10=q10,
            q14=q14,
            eligible=eligible,
        )
        rows = run_bank_rows(environment, xy, values, q10, q14, eligible, record=False)
        with (a.output / "shots.jsonl").open("a") as out:
            for i, (row, p) in enumerate(zip(rows, packet)):
                row.update(
                    id=f"far-shot-{len(records):07d}",
                    phase=phase,
                    bound_phase="wide",
                    target_id=p["target_id"],
                    values=p["values"],
                    source_phase="targeted_development",
                    geometry_cache_file=cache_name,
                    geometry_cache_row=i,
                    geometry_eligible=bool(eligible[i]),
                )
                for name in (
                    "confirmation_source_id",
                    "proposal_family",
                    "seed_source_id",
                    "seed_source_record_sha256",
                    "round_index",
                ):
                    if name in p:
                        row[name] = p[name]
                records.append(row)
                out.write(json.dumps(row, allow_nan=False) + "\n")
        return rows

    rng = np.random.default_rng(SEED)
    queue = []
    with gpu_slot(a.gpu_lock, timeout_s=a.gpu_wait_timeout, notify=waiting):
        report.pop("waiting_for_4090", None)
        publish("gpu-acquired")
        import torch

        torch.set_num_threads(2)
        env = ExpandedEnvironment(scene, a.assets, n=a.batch, bounds=WIDE_BOUNDS)
        scheduled = {t["id"]: 0 for t in selected}

        def flush(force=False):
            while len(queue) >= a.batch or (force and queue):
                packet = queue[: a.batch]
                del queue[: len(packet)]
                rows = save_packet(env, packet, phase="search", audited=True)
                for row in rows:
                    searched[row["target_id"]].append(row)
                report["native_search_attempts"] += len(rows)
                publish("native-search")

        for round_index in range(a.proposal_rounds):
            for target in selected:
                tid = target["id"]
                if scheduled[tid] >= a.native_per_target:
                    continue
                seeds = select_seeds(seed_sources + searched[tid], tid, a.seed_sources)
                values, metadata = far_proposals(
                    target["bucket_xy_m"],
                    a.proposals,
                    rng,
                    scene,
                    seeds,
                    round_index=round_index,
                )
                xy = np.repeat([target["bucket_xy_m"]], len(values), axis=0)
                coarse = outer_geometry_mask(
                    scene, xy, values[:, None], bounds=WIDE_BOUNDS
                )
                ids = np.flatnonzero((coarse[0] & coarse[1])[:, 0])
                report["geometry_proposals"] += len(values)
                if not len(ids):
                    publish("cpu-geometry")
                    continue
                mask, detail = audit_values(
                    scene, a.assets, xy[ids], values[ids, None], bounds=WIDE_BOUNDS
                )
                for j in np.flatnonzero(mask[:, 0]):
                    if scheduled[tid] >= a.native_per_target:
                        break
                    i = int(ids[j])
                    queue.append(
                        dict(
                            target_id=tid,
                            bucket_xy_m=target["bucket_xy_m"],
                            values=values[i].tolist(),
                            q10=detail["q10"][j, 0],
                            q14=detail["q14"][j, 0],
                            **metadata[i],
                        )
                    )
                    scheduled[tid] += 1
                flush()
                publish("cpu-geometry")
            flush(True)
            if all(n >= a.native_per_target for n in scheduled.values()):
                break

        captured = []
        for target in selected:
            captured.extend(
                sorted(
                    [
                        r
                        for r in searched[target["id"]]
                        if r["route_success"] and not r["invalid"]
                    ],
                    key=candidate_rank,
                    reverse=True,
                )[: a.top]
            )
        packet = []
        for source in captured:
            values = np.clip(
                np.asarray(source["values"]) + rng.normal(0, NOISE, (10, 5)),
                WIDE_BOUNDS["low"],
                WIDE_BOUNDS["high"],
            )
            packet.extend(
                dict(
                    target_id=source["target_id"],
                    bucket_xy_m=source["bucket_xy_m"],
                    values=value.tolist(),
                    confirmation_source_id=source["id"],
                )
                for value in values
            )
        for offset in range(0, len(packet), a.batch):
            rows = save_packet(
                env, packet[offset : offset + a.batch], phase="confirmation"
            )
            for row in rows:
                confirmations.setdefault(row["confirmation_source_id"], []).append(row)
            report["confirmation_attempts"] += len(rows)
            publish("fresh-confirmation")
        for target in selected:
            qualified = []
            for source in searched[target["id"]]:
                rows = confirmations.get(source["id"], [])
                if len(rows) != 10:
                    continue
                template = qualified_template(
                    source, rows, target["id"], WIDE_BOUNDS, 0.8
                )
                if template:
                    template.update(
                        source_record_sha256=digest(source),
                        source_study_contract_fingerprint=spec["fingerprint"],
                        confirmation_study_contract_fingerprint=spec["fingerprint"],
                    )
                    qualified.append((template, source))
            if qualified:
                promotions[target["id"]] = max(
                    qualified,
                    key=lambda pair: (
                        pair[0]["confirmation"]["captures"],
                        candidate_rank(pair[1]),
                    ),
                )[0]
    bounds = WIDE_BOUNDS if promotions else OLD_BOUNDS
    bank = expanded_bank(original, list(promotions.values()), spec, bounds)
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
        native_search_statistics=statistics(
            [r for r in records if r["phase"] == "search"]
        ),
        confirmation_statistics=statistics(
            [r for r in records if r["phase"] == "confirmation"]
        ),
        versions={
            name: importlib.metadata.version(name)
            for name in ("mjlab", "mujoco", "mujoco-warp", "torch", "warp-lang")
        },
        source_sha256={p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        artifacts_sha256={
            p.name: sha256(p)
            for p in a.output.iterdir()
            if p.is_file() and p.name != "report.json"
        },
    )
    if sha256(a.prior_search / "shots.jsonl") != spec["prior_search_records_sha256"]:
        raise RuntimeError("Prior development records changed during this run")
    publish("completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "parent-study",
        "prior-search",
        "old-bank-study",
        "assets",
        "output",
        "gpu-lock",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--native-per-target", type=int, default=1536)
    parser.add_argument("--proposal-rounds", type=int, default=96)
    parser.add_argument("--proposals", type=int, default=768)
    parser.add_argument("--seed-sources", type=int, default=16)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--gpu-wait-timeout", type=float, default=1800)
    a = parser.parse_args()
    if not (
        1 <= a.batch <= 256
        and 1024 <= a.native_per_target <= 2048
        and 1 <= a.proposal_rounds <= 128
        and 32 <= a.proposals <= 2048
        and 1 <= a.seed_sources <= 32
        and 1 <= a.top <= 20
        and 0 < a.gpu_wait_timeout <= 86400
    ):
        parser.error(
            "Finite missing-target/native/proposal/confirmation budgets required"
        )
    run(a)
