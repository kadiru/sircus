"""Fresh native comparison after explicitly converting old unresolved tests to DEV.

The twelve targeted positions may influence the expanded proposal bank and new
training, but never count as new held-out evidence. A new spatial test compares
old and expanded banks with uniform and frozen learned selection on identical
queries. No ball-outcome search occurs during policy evaluation.
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

from .bank_study import atomic, mask_hash, run_bank_rows
from .learning_contract import SEED, digest, reward, statistics
from .learning_solve import validate_artifacts
from .bank_solve import validate_bank
from .learning_search import VERSION as BANK_VERSION
from .expanded_actions import OLD_BOUNDS, checked_bounds
from .gpu_slot import gpu_slot

VERSION = "sircus-mjlab-expanded-bank-ppo-v2"
CACHE_STRATEGY = "stable-candidate-id-shared-prefix-v1"
TRAIN_SEED, VALIDATION_SEED, TEST_SEED = SEED + 1101, SEED + 1102, SEED + 1103
MIN_SPLIT_DISTANCE_M = 0.005
PRIMARY_METHODS = ("old_uniform", "old_learned", "expanded_uniform", "expanded_learned")


def shared_prefix_slots(old_metadata, expanded_metadata):
    """Map stable candidate IDs despite interleaved new template/rotation slots."""
    old = [item["candidate_id"] for item in old_metadata]
    expanded = [item["candidate_id"] for item in expanded_metadata]
    if len(set(old)) != len(old) or len(set(expanded)) != len(expanded):
        raise RuntimeError("Candidate IDs must be unique within each bank")
    lookup = {value: i for i, value in enumerate(expanded)}
    if not set(old).issubset(lookup):
        raise RuntimeError("Expanded bank lost an original stable candidate ID")
    shared = np.asarray([lookup[value] for value in old], dtype=int)
    added = np.asarray(
        [i for i, value in enumerate(expanded) if value not in set(old)], dtype=int
    )
    return shared, added


def copy_shared_prefix(destination, expanded_metadata, original, old_metadata):
    """Reuse exact branches/masks, including failures; no fresh IK for old slots."""
    shared, added = shared_prefix_slots(old_metadata, expanded_metadata)
    if not np.array_equal(
        destination["bucket"], original["bucket"]
    ) or not np.array_equal(destination["values"][:, shared], original["values"]):
        raise RuntimeError("Stable candidate IDs refer to different requested geometry")
    for key in ("mask", "q10", "q14", "reasons"):
        destination[key][:, shared] = original[key]
    return shared, added


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def coordinates(values):
    """Normalize old DEV records or plain XY arrays without changing positions."""
    result = []
    for item in values:
        point = (
            item.get("bucket_xy_m", item.get("bucket_xy"))
            if isinstance(item, dict)
            else item
        )
        point = np.asarray(point, dtype=float)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError("Expected finite two-coordinate bucket positions")
        result.append(point.tolist())
    return result


def _draw(count, seed, excluded):
    rng = np.random.default_rng(seed)
    chosen = []
    forbidden = np.asarray(excluded, dtype=float).reshape(-1, 2)
    for _ in range(100000):
        if len(chosen) == count:
            return chosen
        point = rng.uniform([0.1, -0.65], [1.24, 0.25])
        previous = np.asarray(chosen, dtype=float).reshape(-1, 2)
        if (
            len(forbidden)
            and np.linalg.norm(forbidden - point, axis=1).min() < MIN_SPLIT_DISTANCE_M
        ):
            continue
        if (
            len(previous)
            and np.linalg.norm(previous - point, axis=1).min() < MIN_SPLIT_DISTANCE_M
        ):
            continue
        chosen.append(point.tolist())
    raise RuntimeError("Could not construct the bounded disjoint spatial split")


def new_splits(
    grid, development, historical, random_train=128, validation_count=30, test_count=100
):
    grid, development, historical = (
        coordinates(grid),
        coordinates(development),
        coordinates(historical),
    )
    if (
        len(grid) != 35
        or len(development) != 12
        or len({tuple(p) for p in grid + development}) != 47
    ):
        raise ValueError(
            "Expected 35 training-grid and 12 distinct converted-development positions"
        )
    fixed = grid + development
    train = fixed + _draw(random_train, TRAIN_SEED, historical + fixed)
    validation = _draw(validation_count, VALIDATION_SEED, historical + train)
    test = _draw(test_count, TEST_SEED, historical + train + validation)
    merged = train + validation + test
    if len({tuple(p) for p in merged}) != len(merged):
        raise RuntimeError("New study spatial split overlap")
    return dict(train=train, validation=validation, test=test, development=development)


def validate_expansion(expansion_study, old_bank, old_report):
    """Bind new proposals to native development evidence, never source success labels."""
    study = Path(expansion_study)
    report = json.loads((study / "report.json").read_text())
    contract = json.loads((study / "contract.json").read_text())
    payload = dict(contract)
    recorded_fp = payload.pop("fingerprint")
    if digest(payload) != recorded_fp:
        raise RuntimeError("Expansion contract fingerprint changed")
    if report.get("status") != "completed":
        raise RuntimeError("Native bank expansion must finish before retraining")
    path = study / "expanded-bank.json"
    if sha256(path) != report.get("expanded_bank_sha256"):
        raise RuntimeError("Expanded proposal artifact checksum changed")
    bank = json.loads(path.read_text())
    payload = dict(bank)
    bank_fp = payload.pop("sha256")
    if bank.get("version") != BANK_VERSION or digest(payload) != bank_fp:
        raise RuntimeError("Expanded relative bank fingerprint changed")
    if report.get("proposal_bank_sha256") != bank_fp:
        raise RuntimeError("Expansion report names another proposal bank")
    if (
        contract.get("old_bank_study_fingerprint")
        != old_report.get("contract_fingerprint")
        or contract.get("old_bank_sha256") != old_bank["sha256"]
    ):
        raise RuntimeError("Expansion does not identify the preserved old bank/study")
    if bank["templates"][: len(old_bank["templates"])] != old_bank["templates"]:
        raise RuntimeError("Original proposal prefix was modified")
    if len(bank["templates"]) <= len(old_bank["templates"]):
        raise RuntimeError(
            "No new confirmed templates were added; retraining an unchanged bank is not claimed"
        )
    dev = coordinates(
        contract.get("development_layouts", report.get("development_layouts", []))
    )
    if len(dev) != 12 or len({tuple(p) for p in dev}) != 12:
        raise RuntimeError(
            "Expansion must identify the twelve old-test development targets"
        )
    # The expansion driver owns detailed confirmation auditing. Do not silently
    # reinterpret unconfirmed or evaluation-derived rows as new training teachers.
    minimum_rate = float(contract.get("minimum_confirmation_rate", 1.0))
    expected_attempts = int(contract.get("confirmations_per_candidate", 0))
    if not 0.5 <= minimum_rate <= 1 or expected_attempts < 3:
        raise RuntimeError("Expansion lacks a declared native confirmation protocol")
    for template in bank["templates"][len(old_bank["templates"]) :]:
        if template.get("source_phase") != "targeted_development":
            raise RuntimeError(
                "New templates must disclose their converted-development provenance"
            )
        confirmation = template.get("confirmation", {})
        n, captures = confirmation.get("attempts", 0), confirmation.get("captures", 0)
        invalid = confirmation.get("invalid", 0)
        if (
            n != expected_attempts
            or not 0 <= invalid <= n
            or not 0 <= captures <= n
            or captures + invalid > n
            or captures / n < minimum_rate
        ):
            raise RuntimeError(
                "A promoted template fails the declared native confirmation threshold"
            )
        if not template.get("source_trial_id") or not template.get("source_target_id"):
            raise RuntimeError("Expanded teacher is missing native source identifiers")
    bounds = bank.get("action_bounds", OLD_BOUNDS)
    checked_bounds(bounds)
    return report, contract, bank, dev, bounds


def uniform_indices(mask, quantiles):
    """One common uniform quantile per query, then frozen across nominal repeats."""
    mask = np.asarray(mask, dtype=bool)
    q = np.asarray(quantiles, dtype=float)
    if mask.ndim != 2 or q.shape != (len(mask),) or not ((q >= 0) & (q < 1)).all():
        raise ValueError("Expected masks and one unit-interval quantile per query")
    result = np.zeros(len(mask), dtype=np.int64)
    for i, legal in enumerate(mask):
        ids = np.flatnonzero(legal)
        if len(ids):
            result[i] = ids[min(len(ids) - 1, int(q[i] * len(ids)))]
    return result


def enriched_stats(rows):
    result = statistics(rows)
    eligible = [r for r in rows if r.get("eligible_bank_endpoint")]
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row["bucket_xy_m"]), []).append(row)
    result.update(
        native_releases=sum(bool(r.get("native_release")) for r in rows),
        eligible_attempts=len(eligible),
        conditional_capture_rate=(
            (sum(bool(r["route_success"]) for r in eligible) / len(eligible))
            if eligible
            else None
        ),
        positions_with_capture=sum(
            any(r["route_success"] for r in group) for group in groups.values()
        ),
        positions_all_repeats_captured=sum(
            all(r["route_success"] for r in group) for group in groups.values()
        ),
        unresolved_positions=sum(
            not any(r.get("eligible_bank_endpoint") for r in group)
            for group in groups.values()
        ),
    )
    return result


def paired_comparison(first, second, *, seed=TEST_SEED + 91):
    """Layout-paired descriptive bootstrap; repeated trials are not independent units."""

    def rates(rows):
        groups = {}
        for row in rows:
            groups.setdefault(tuple(row["bucket_xy_m"]), []).append(
                bool(row["route_success"])
            )
        return {k: float(np.mean(v)) for k, v in groups.items()}

    a, b = rates(first), rates(second)
    if set(a) != set(b) or not a:
        raise ValueError("Paired methods must use the same nonempty query set")
    delta = np.array([a[k] - b[k] for k in sorted(a)])
    rng = np.random.default_rng(seed)
    draws = delta[rng.integers(0, len(delta), (2000, len(delta)))].mean(1)
    return dict(
        layouts=len(delta),
        capture_rate_difference=float(delta.mean()),
        descriptive_95pct_layout_bootstrap=np.quantile(draws, [0.025, 0.975]).tolist(),
        improved_layouts=int((delta > 0).sum()),
        worsened_layouts=int((delta < 0).sum()),
        tied_layouts=int((delta == 0).sum()),
        scope="Descriptive paired bootstrap over held-out positions; nominal repeats stay grouped",
    )


def run(a):
    parent_report, parent_cp, parent_sha, parent_fp = validate_artifacts(
        a.parent_study, a.assets
    )
    old_report, old_cp = validate_bank(a.old_bank_study, parent_report, parent_cp)
    old_bank = old_cp["bank"]
    expansion_report, expansion_contract, bank, development, bounds = (
        validate_expansion(a.expansion_study, old_bank, old_report)
    )
    if bank["sha256"] == old_bank["sha256"]:
        raise RuntimeError("Expanded and old banks are identical")
    historical = []
    for contract in (parent_cp["contract"], old_cp["contract"]):
        for key in ("train_layouts", "validation_layouts", "test_layouts"):
            historical.extend(contract[key])
    pools = new_splits(
        parent_cp["contract"]["train_layouts"],
        development,
        historical,
        random_train=a.random_train,
        validation_count=a.validation_count,
        test_count=a.test_count,
    )
    a.output.mkdir(parents=True, exist_ok=False)
    changed_bounds = (
        bounds["low"] != OLD_BOUNDS["low"] or bounds["high"] != OLD_BOUNDS["high"]
    )
    methods = list(PRIMARY_METHODS) + (
        ["old_uniform_expanded_bounds"] if changed_bounds else []
    )
    spec = dict(
        version=VERSION,
        seed=TRAIN_SEED,
        train_seed=TRAIN_SEED,
        validation_seed=VALIDATION_SEED,
        test_seed=TEST_SEED,
        min_split_distance_m=MIN_SPLIT_DISTANCE_M,
        parent_contract_fingerprint=parent_fp,
        parent_checkpoint_sha256=parent_sha,
        old_bank_contract_fingerprint=old_report["contract_fingerprint"],
        old_bank_checkpoint_sha256=old_report["checkpoint_sha256"],
        expansion_contract_fingerprint=expansion_contract["fingerprint"],
        expansion_report_sha256=sha256(a.expansion_study / "report.json"),
        expanded_bank_artifact_sha256=sha256(a.expansion_study / "expanded-bank.json"),
        proposal_bank_sha256=bank["sha256"],
        old_proposal_bank_sha256=old_bank["sha256"],
        scene=parent_cp["contract"]["scene"],
        action_bounds=bounds,
        native_environment="sircus.mjlab.expanded_env.ExpandedEnvironment",
        policy_class="expanded_relative_bank_categorical_v1",
        cache_strategy=CACHE_STRATEGY,
        cache_contract="Strict original-prefix candidates audited once per split/layout. Widened prefix preserves exact strict mask/q10/q14/reasons for every action within OLD_BOUNDS, including failures; only outside-OLD actions receive wider audit. Expanded prefix copies active old-bank cache by stable candidate_id; only added template slots receive separate IK.",
        train_layouts=pools["train"],
        validation_layouts=pools["validation"],
        test_layouts=pools["test"],
        development_layouts=development,
        historical_layouts=coordinates(historical),
        batch_size=a.batch,
        ppo_updates=a.updates,
        validation_every=a.validate_every,
        test_repeats=a.test_repeats,
        evaluation_methods=methods,
        actor="Unchanged shared 9->64->64->1 candidate scorer; masked categorical",
        critic="Unchanged query XY 2->64->64->1",
        initialization="Warm-start selected old-bank actor/critic, fresh Adam optimizer; uniform baselines use zero actor logits",
        lr=0.0003,
        epochs=6,
        clip=0.2,
        entropy_coef=0.001,
        value_coef=0.5,
        grad_norm=1.0,
        checkpoint_selection="Validation captures first, then mean shaped reward; test never selects",
        baseline_protocol="One common seeded uniform quantile per test query; each chosen action frozen across nominal repeats. Learned choices deterministic argmax.",
        learning="Actual on-policy terminal PPO on geometry-eligible training contexts; no GAE/bootstrap; unresolved layouts retained in complete evaluation denominator",
        development_note="Twelve previously unresolved OLD test positions are explicitly converted to development and may shape the bank and training; never relabeled as new held-out results",
        fairness="All comparisons share scene, scorer and native integration. Same stable candidate IDs use bit-identical masks/joint branches/reasons across bank sizes; widened-domain controls preserve strict failures and branches for the overlapping action domain. New template slots are audited separately. Expanded learned versus uniform selection shares the exact full cache.",
        scope="Simulation-only privileged-geometry endpoint configuration selection; no grasping, transit, perception, exhaustive coverage or physical validation",
    )
    spec["fingerprint"] = digest(spec)
    atomic(a.output / "contract.json", spec)
    atomic(a.output / "proposal-bank.json", bank)
    atomic(a.output / "old-proposal-bank.json", old_bank)
    atomic(a.output / "splits.json", pools)
    report = dict(
        status="running",
        phase="geometry-audit",
        contract_fingerprint=spec["fingerprint"],
        proposal_bank_sha256=bank["sha256"],
        old_proposal_bank_sha256=old_bank["sha256"],
        parent_checkpoint_sha256=parent_sha,
        old_bank_checkpoint_sha256=old_report["checkpoint_sha256"],
        train_episodes=0,
        native_train_releases=0,
        updates=0,
        learning_curve=[],
        frozen_evaluation=None,
        geometry={},
        cache_strategy=CACHE_STRATEGY,
        scope=spec["scope"],
        development_note=spec["development_note"],
    )
    started = time.monotonic()

    def publish(phase):
        report.update(
            phase=phase,
            elapsed_s=time.monotonic() - started,
            timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
        atomic(a.output / "report.json", report)
        print(
            json.dumps(
                {
                    k: report[k]
                    for k in ("phase", "updates", "train_episodes", "elapsed_s")
                }
            ),
            flush=True,
        )

    publish("geometry-audit")

    from .bank_geometry import build_candidates, audit_candidates
    from .expand_geometry import audit_values

    data, metadata = {}, {}

    def store(label, arrays, meta, **cache_info):
        data[label] = arrays
        metadata[label] = meta
        np.savez_compressed(a.output / ("geometry-" + label + ".npz"), **arrays)
        atomic(a.output / ("candidate-metadata-" + label + ".json"), meta)
        report["geometry"][label] = dict(
            queries=len(arrays["bucket"]),
            candidates_per_query=arrays["values"].shape[1],
            eligible_candidates=int(arrays["mask"].sum()),
            queries_with_candidate=int(arrays["mask"].any(1).sum()),
            unresolved_queries=int((~arrays["mask"].any(1)).sum()),
            cache_strategy=CACHE_STRATEGY,
            **cache_info,
        )
        publish("geometry-audit")

    def prepare_strict(label, layouts):
        xy = np.asarray(layouts)
        values, meta = build_candidates(old_bank, xy, spec["scene"])
        arrays = dict(bucket=xy, values=values)
        lists = {k: [] for k in ("mask", "q10", "q14", "reasons")}
        for offset in range(0, len(xy), a.audit_chunk):
            stop = min(offset + a.audit_chunk, len(xy))
            args = (spec["scene"], a.assets, xy[offset:stop], values[offset:stop])
            mask, detail = audit_candidates(*args)
            lists["mask"].append(mask)
            for key in ("q10", "q14", "reasons"):
                lists[key].append(detail[key])
            report["geometry_progress"] = dict(
                group=label,
                completed=stop,
                total=len(xy),
                candidates_per_query=values.shape[1],
            )
            publish("geometry-audit")
        arrays.update({k: np.concatenate(v) for k, v in lists.items()})
        store(label, arrays, meta, newly_audited_slots=int(np.prod(values.shape[:2])))

    def prepare_wide(label, source_label):
        original = data[source_label]
        arrays = {key: value.copy() for key, value in original.items()}
        low, high = np.asarray(OLD_BOUNDS["low"]), np.asarray(OLD_BOUNDS["high"])
        within = (arrays["values"] >= low - 1e-9).all(-1) & (
            arrays["values"] <= high + 1e-9
        ).all(-1)
        # Strict in-domain failures stay failures. Wider bounds cannot trigger
        # another multistart branch lottery for any overlapping requested shot.
        for offset in range(0, len(arrays["bucket"]), a.audit_chunk):
            stop = min(offset + a.audit_chunk, len(arrays["bucket"]))
            qi, ci = np.where(~within[offset:stop])
            qi += offset
            if len(qi):
                mask, detail = audit_values(
                    spec["scene"],
                    a.assets,
                    arrays["bucket"][qi],
                    arrays["values"][qi, ci][:, None],
                    bounds=bounds,
                )
                arrays["mask"][qi, ci] = mask[:, 0]
                for key in ("q10", "q14", "reasons"):
                    arrays[key][qi, ci] = detail[key][:, 0]
            report["geometry_progress"] = dict(
                group=label,
                completed=stop,
                total=len(arrays["bucket"]),
                candidates_per_query=arrays["values"].shape[1],
            )
            publish("geometry-audit")
        store(
            label,
            arrays,
            metadata[source_label],
            strict_overlap_source=source_label,
            strict_overlap_slots=int(within.sum()),
            newly_audited_slots=int((~within).sum()),
        )

    def prepare_expanded(label, source_label, layouts):
        xy = np.asarray(layouts)
        values, meta = build_candidates(bank, xy, spec["scene"])
        shape = values.shape[:2]
        arrays = dict(
            bucket=xy,
            values=values,
            mask=np.zeros(shape, dtype=bool),
            q10=np.zeros((*shape, 6)),
            q14=np.zeros((*shape, 6)),
            reasons=np.ones(shape, dtype=data[source_label]["reasons"].dtype),
        )
        shared, added = copy_shared_prefix(
            arrays, meta, data[source_label], metadata[source_label]
        )
        for offset in range(0, len(xy), a.audit_chunk):
            stop = min(offset + a.audit_chunk, len(xy))
            if len(added):
                mask, detail = audit_values(
                    spec["scene"],
                    a.assets,
                    xy[offset:stop],
                    values[offset:stop][:, added],
                    bounds=bounds,
                )
                arrays["mask"][offset:stop, added] = mask
                for key in ("q10", "q14", "reasons"):
                    arrays[key][offset:stop, added] = detail[key]
            report["geometry_progress"] = dict(
                group=label,
                completed=stop,
                total=len(xy),
                candidates_per_query=values.shape[1],
            )
            publish("geometry-audit")
        store(
            label,
            arrays,
            meta,
            shared_prefix_source=source_label,
            shared_prefix_candidates=len(shared),
            newly_audited_candidates=len(added),
            newly_audited_slots=len(xy) * len(added),
        )

    for label in ("train", "validation", "test", "development"):
        prepare_strict("old-" + label, pools[label])
        active_source = "old-" + label
        if changed_bounds:
            active_source = "old-wide-" + label
            prepare_wide(active_source, "old-" + label)
        prepare_expanded("expanded-" + label, active_source, pools[label])
    if a.prepare_only:
        report["status"] = "geometry_prepared"
        publish("geometry-prepared")
        return
    eligible_train = np.flatnonzero(data["expanded-train"]["mask"].any(1))
    if not len(eligible_train):
        raise RuntimeError("Expanded bank has no endpoint-eligible training contexts")

    # No GPU is touched until all CPU audits finish and ownership is acquired.
    def waiting(info):
        report["gpu_queue"] = info
        publish("waiting-for-4090")

    with gpu_slot(
        a.gpu_lock, timeout_s=getattr(a, "gpu_wait_timeout", 1800), notify=waiting
    ):
        report["gpu_queue"] = {"status": "acquired"}
        publish("native-start")
        import torch
        from .bank_policy import Policy, FEATURE_NAMES, predict, update, weight_hash
        from .expanded_env import ExpandedEnvironment

        torch.set_num_threads(2)
        torch.manual_seed(TRAIN_SEED)
        rng = np.random.default_rng(TRAIN_SEED)
        env = ExpandedEnvironment(spec["scene"], a.assets, n=a.batch, bounds=bounds)
        policy = Policy().to(env.device)
        policy.load_state_dict(old_cp["state"])
        warm = copy.deepcopy(policy.state_dict())
        current = copy.deepcopy(warm)
        uniform = Policy().to(env.device)
        uniform_state = copy.deepcopy(uniform.state_dict())
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.0003)
        report["train_sampling"] = dict(
            rule="Uniform over endpoint-eligible expanded-bank training layouts",
            declared_contexts=len(pools["train"]),
            eligible_contexts=len(eligible_train),
            excluded_contexts=len(pools["train"]) - len(eligible_train),
        )
        records = []

        def tensors(group, ids):
            return (
                torch.tensor(
                    group["bucket"][ids], device=env.device, dtype=torch.float32
                ),
                torch.tensor(
                    group["values"][ids], device=env.device, dtype=torch.float32
                ),
                torch.tensor(group["mask"][ids], device=env.device, dtype=torch.bool),
            )

        def trial(
            group_name, ids, indices, phase, *, logp=None, state_hash=None, record=False
        ):
            group = data[group_name]
            ids = np.asarray(ids, dtype=int)
            indices = np.asarray(indices, dtype=int)
            eligible = group["mask"][ids].any(1)
            if not group["mask"][ids[eligible], indices[eligible]].all():
                raise RuntimeError("A selected candidate is masked out")
            rows = run_bank_rows(
                env,
                group["bucket"][ids],
                group["values"][ids, indices],
                group["q10"][ids, indices],
                group["q14"][ids, indices],
                eligible,
                record=record,
            )
            first = len(records)
            for i, row in enumerate(rows):
                row.update(
                    id=f"expanded-shot-{first+i:07d}",
                    phase=phase,
                    geometry_group=group_name,
                    query_index=int(ids[i]),
                    candidate_index=int(indices[i]),
                    candidate_metadata=(
                        metadata[group_name][int(indices[i])] if eligible[i] else None
                    ),
                    mask_sha256=mask_hash(group["mask"][ids[i]]),
                    eligible_bank_endpoint=bool(eligible[i]),
                    behavior_logp=(
                        float(logp[i]) if logp is not None and eligible[i] else None
                    ),
                    behavior_policy_weight_sha256=state_hash,
                )
            records.extend(rows)
            with (a.output / "shots.jsonl").open("a") as stream:
                for row in rows:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
            return rows

        def evaluate(
            state, group_name, phase, *, uniform_choice=False, repeats=1, record=False
        ):
            policy.load_state_dict(state)
            group = data[group_name]
            count = len(group["bucket"])
            picks, logps = [], []
            quantiles = np.random.default_rng(
                TEST_SEED + 71 if "test" in group_name else VALIDATION_SEED + 71
            ).random(count)
            for offset in range(0, count, a.batch):
                ids = np.arange(offset, min(offset + a.batch, count))
                tx, tv, tm = tensors(group, ids)
                with torch.no_grad():
                    dist, _ = policy.distribution(tx, tv, tm)
                    selection = (
                        uniform_indices(group["mask"][ids], quantiles[ids])
                        if uniform_choice
                        else dist.probs.argmax(-1).cpu().numpy()
                    )
                    lp = (
                        dist.log_prob(torch.tensor(selection, device=env.device))
                        .cpu()
                        .numpy()
                    )
                picks.extend(selection.tolist())
                logps.extend(lp.tolist())
            allrows = []
            for rep in range(repeats):
                for offset in range(0, count, a.batch):
                    ids = np.arange(offset, min(offset + a.batch, count))
                    rr = trial(
                        group_name,
                        ids,
                        np.asarray(picks)[ids],
                        f"{phase}-{rep}",
                        logp=np.asarray(logps)[ids],
                        state_hash=weight_hash(policy),
                        record=record,
                    )
                    allrows.extend(rr)
            return allrows

        initial_val = evaluate(warm, "expanded-validation", "validation-initial")
        selected, best_update = copy.deepcopy(warm), 0
        best_key = (
            statistics(initial_val)["captures"],
            statistics(initial_val)["mean_reward"],
        )
        report["initial_validation"] = enriched_stats(initial_val)
        for name, state in (("update-000.pt", warm), ("uniform.pt", uniform_state)):
            torch.save(
                dict(
                    state=state,
                    contract=spec,
                    bank=bank,
                    weight_sha256=(
                        weight_hash(policy) if state is warm else weight_hash(uniform)
                    ),
                    policy_class="expanded_relative_bank_categorical_v1",
                ),
                a.output / name,
            )
        publish("ppo")
        for iteration in range(a.updates):
            policy.load_state_dict(current)
            ids = rng.choice(eligible_train, a.batch, replace=True)
            tx, tv, tm = tensors(data["expanded-train"], ids)
            before = weight_hash(policy)
            with torch.no_grad():
                dist, _ = policy.distribution(tx, tv, tm)
                indices = dist.sample()
                lp = dist.log_prob(indices)
            rows = trial(
                "expanded-train",
                ids,
                indices.cpu().numpy(),
                "train-ppo",
                logp=lp.cpu().numpy(),
                state_hash=before,
            )
            returns = torch.tensor(
                [reward(r) for r in rows], device=env.device, dtype=torch.float32
            )
            change = update(policy, optimizer, tx, tv, tm, indices, lp, returns)
            current = copy.deepcopy(policy.state_dict())
            report["train_episodes"] += len(rows)
            report["native_train_releases"] += sum(r["native_release"] for r in rows)
            report["updates"] = iteration + 1
            curve = dict(
                update=iteration + 1, **enriched_stats(rows), weight_change=change
            )
            report["learning_curve"].append(curve)
            if (iteration + 1) % a.validate_every == 0 or iteration + 1 == a.updates:
                val = evaluate(
                    current, "expanded-validation", f"validation-update-{iteration+1}"
                )
                curve["validation"] = enriched_stats(val)
                key = (statistics(val)["captures"], statistics(val)["mean_reward"])
                if key > best_key:
                    selected, best_update, best_key = (
                        copy.deepcopy(current),
                        iteration + 1,
                        key,
                    )
            policy.load_state_dict(current)
            torch.save(
                dict(
                    state=current,
                    contract=spec,
                    bank=bank,
                    update=iteration + 1,
                    weight_sha256=weight_hash(policy),
                    policy_class="expanded_relative_bank_categorical_v1",
                ),
                a.output / f"update-{iteration+1:03d}.pt",
            )
            publish("ppo")
        policy.load_state_dict(selected)
        selected_hash = weight_hash(policy)
        checkpoint = dict(
            state=selected,
            contract=spec,
            bank=bank,
            policy_class="expanded_relative_bank_categorical_v1",
            selected_update=best_update,
            weight_sha256=selected_hash,
            feature_names=list(FEATURE_NAMES),
        )
        torch.save(checkpoint, a.output / "selected.pt")
        report.update(
            selected_update=best_update,
            checkpoint_sha256=sha256(a.output / "selected.pt"),
            policy_weight_sha256=selected_hash,
        )
        publish("frozen-evaluation")
        # Freeze all policies and banks before any new test outcomes are observed.
        evaluation, test_rows = {}, {}
        methods_to_run = [
            ("old_uniform", uniform_state, "old-test", True),
            ("old_learned", old_cp["state"], "old-test", False),
            ("expanded_uniform", uniform_state, "expanded-test", True),
            ("expanded_learned", selected, "expanded-test", False),
        ]
        if changed_bounds:
            methods_to_run.append(
                ("old_uniform_expanded_bounds", uniform_state, "old-wide-test", True)
            )
        for method, state, group, random in methods_to_run:
            rr = evaluate(
                state,
                group,
                "test-" + method,
                uniform_choice=random,
                repeats=a.test_repeats,
                record=method == "expanded_learned",
            )
            test_rows[method] = rr
            evaluation[method] = dict(
                **enriched_stats(rr),
                layouts=len(pools["test"]),
                repeats=a.test_repeats,
                policy_weight_sha256=weight_hash(policy),
                action_protocol=(
                    "One frozen uniform quantile"
                    if random
                    else "Frozen deterministic argmax"
                ),
                search_at_inference=False,
                bank_sha256=(
                    old_bank["sha256"] if method.startswith("old_") else bank["sha256"]
                ),
                action_bounds=OLD_BOUNDS if group == "old-test" else bounds,
            )
            publish("frozen-evaluation")
        report["paired_comparisons"] = dict(
            learning_vs_uniform=paired_comparison(
                test_rows["expanded_learned"], test_rows["expanded_uniform"]
            ),
            expanded_uniform_vs_old_uniform=paired_comparison(
                test_rows["expanded_uniform"], test_rows["old_uniform"]
            ),
            expanded_learned_vs_old_learned=paired_comparison(
                test_rows["expanded_learned"], test_rows["old_learned"]
            ),
        )
        # Targeted twelve are developmental diagnostics, never added to test scores.
        development_results = {}
        for method, state, group, random in (
            ("old_uniform", uniform_state, "old-development", True),
            ("old_learned", old_cp["state"], "old-development", False),
            ("expanded_uniform", uniform_state, "expanded-development", True),
            ("expanded_learned", selected, "expanded-development", False),
        ):
            rr = evaluate(
                state,
                group,
                "development-" + method,
                uniform_choice=random,
                repeats=a.test_repeats,
            )
            development_results[method] = dict(
                **enriched_stats(rr),
                layouts=12,
                repeats=a.test_repeats,
                scope="Converted OLD test positions; targeted development only",
            )
        policy.load_state_dict(selected)
        if (
            weight_hash(policy) != selected_hash
            or sha256(a.old_bank_study / "selected.pt")
            != old_report["checkpoint_sha256"]
        ):
            raise RuntimeError("A frozen checkpoint changed during evaluation")
        if (
            sha256(a.expansion_study / "expanded-bank.json")
            != spec["expanded_bank_artifact_sha256"]
        ):
            raise RuntimeError("Expanded bank changed during training/evaluation")
        report["development_evaluation"] = development_results
        report["frozen_evaluation"] = dict(
            **evaluation["expanded_learned"],
            trials=evaluation["expanded_learned"]["attempts"],
            comparison=evaluation,
            scope="New untouched continuous tabletop positions; nominal repeats; no policy/search updates",
        )
        report["test_rows"] = test_rows["expanded_learned"]
        report["records"] = len(records)
        report["versions"] = {
            k: importlib.metadata.version(k)
            for k in ("mjlab", "mujoco", "mujoco-warp", "torch", "warp-lang")
        }
        report["source_sha256"] = {
            p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")
        }
        report["artifacts_sha256"] = {
            p.name: sha256(p)
            for p in a.output.iterdir()
            if p.is_file() and p.name != "report.json"
        }
        report["status"] = "completed"
        publish("completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in (
        "parent-study",
        "old-bank-study",
        "expansion-study",
        "assets",
        "output",
        "gpu-lock",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--updates", type=int, default=32)
    parser.add_argument("--validate-every", type=int, default=4)
    parser.add_argument("--test-repeats", type=int, default=3)
    parser.add_argument("--random-train", type=int, default=128)
    parser.add_argument("--validation-count", type=int, default=30)
    parser.add_argument("--test-count", type=int, default=100)
    parser.add_argument("--audit-chunk", type=int, default=4)
    parser.add_argument("--gpu-wait-timeout", type=float, default=1800)
    parser.add_argument("--prepare-only", action="store_true")
    a = parser.parse_args()
    if (
        not 1 <= a.batch <= 256
        or not 1 <= a.updates <= 128
        or not 1 <= a.validate_every <= a.updates
        or not 1 <= a.test_repeats <= 10
    ):
        parser.error("Finite positive training/evaluation budgets required")
    if (
        not 0 <= a.random_train <= 512
        or not 1 <= a.validation_count <= 100
        or not 1 <= a.test_count <= 300
        or not 1 <= a.audit_chunk <= 16
    ):
        parser.error("Finite bounded spatial/geometry budgets required")
    if not np.isfinite(a.gpu_wait_timeout) or not 0 < a.gpu_wait_timeout <= 86400:
        parser.error("GPU queue timeout must be in (0, 86400] seconds")
    run(a)
