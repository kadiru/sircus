"""Read-only CPU audit of expanded-bank development and fresh frozen evaluation.

No native simulator, CUDA initialization, rollout search or service is used.
Stored evidence supports reproducibility, not an independent rerun of dynamics.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .verify_bank import _mask_hash, materialize
from .verify_learning import Audit, digest, file_hash, state_hash, statistics

VERSION = "sircus-mjlab-expanded-bank-audit-v1"
STUDY_VERSION = "sircus-mjlab-expanded-bank-ppo-v2"
CACHE_STRATEGY = "stable-candidate-id-shared-prefix-v1"
METHODS = ("old_uniform", "old_learned", "expanded_uniform", "expanded_learned")
OLD_LOW = [0.32, -0.30, 0.18, 12.0, 0.065]
OLD_HIGH = [1.01, 0.25, 0.53, 48.0, 0.52]
TRAIN_SEED, VALIDATION_SEED, TEST_SEED = 20262204, 20262205, 20262206


def _fingerprint(value):
    payload = dict(value)
    recorded = payload.pop("fingerprint")
    return digest(payload) == recorded


def _bank_hash(bank):
    payload = dict(bank)
    recorded = payload.pop("sha256")
    return digest(payload) == recorded


def _separated(first, second, minimum):
    a, b = np.asarray(first), np.asarray(second)
    return (
        not len(a)
        or not len(b)
        or bool(
            (np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1) >= minimum).all()
        )
    )


def _own_separation(values, minimum):
    p = np.asarray(values)
    if len(p) < 2:
        return True
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    return bool((d[np.triu_indices(len(p), 1)] >= minimum).all())


def _draw(count, seed, forbidden, minimum):
    """Independent finite reconstruction of the declared rejection-sampled split."""
    rng, points = np.random.default_rng(seed), []
    for _ in range(100000):
        if len(points) == count:
            return points
        point = rng.uniform([0.1, -0.65], [1.24, 0.25]).tolist()
        if _separated([point], forbidden + points, minimum):
            points.append(point)
    raise RuntimeError("Declared split could not be independently reconstructed")


def audit_splits(spec, a):
    pools = {k: spec[k + "_layouts"] for k in ("train", "validation", "test")}
    dev, hist = spec["development_layouts"], spec["historical_layouts"]
    minimum = spec["min_split_distance_m"]
    a.check("contract_fingerprint", _fingerprint(spec))
    a.check(
        "declared_seeded_split_protocol",
        minimum == 0.005
        and (spec["train_seed"], spec["validation_seed"], spec["test_seed"])
        == (TRAIN_SEED, VALIDATION_SEED, TEST_SEED),
    )
    a.check(
        "twelve_old_tests_explicitly_development",
        len(dev) == 12
        and len({tuple(p) for p in dev}) == 12
        and pools["train"][35:47] == dev
        and all(p in hist for p in dev),
    )
    all_points = sum(pools.values(), [])
    a.check(
        "spatial_split_unique", len({tuple(p) for p in all_points}) == len(all_points)
    )
    a.check(
        "new_test_protocol_100_by_3",
        len(pools["test"]) == 100 and spec["test_repeats"] == 3,
    )
    for label in ("validation", "test"):
        forbidden = (
            hist + pools["train"] + (pools["validation"] if label == "test" else [])
        )
        a.check(
            label + "_fresh_5mm_separation",
            _separated(pools[label], forbidden, minimum)
            and _own_separation(pools[label], minimum),
        )
    a.check(
        "new_random_train_fresh_5mm_separation",
        _separated(pools["train"][47:], hist + pools["train"][:47], minimum)
        and _own_separation(pools["train"][47:], minimum),
    )
    expected_train = pools["train"][:47] + _draw(
        len(pools["train"]) - 47, TRAIN_SEED, hist + pools["train"][:47], minimum
    )
    expected_validation = _draw(
        len(pools["validation"]), VALIDATION_SEED, hist + expected_train, minimum
    )
    expected_test = _draw(
        len(pools["test"]),
        TEST_SEED,
        hist + expected_train + expected_validation,
        minimum,
    )
    a.check(
        "independent_seeded_split_reconstruction",
        pools["train"] == expected_train
        and pools["validation"] == expected_validation
        and pools["test"] == expected_test,
    )
    a.check(
        "finite_tabletop_positions",
        all(
            np.asarray(p).shape == (2,)
            and np.isfinite(p).all()
            and 0.1 <= p[0] <= 1.24
            and -0.65 <= p[1] <= 0.25
            for p in all_points
        ),
    )
    return pools


def audit_geometry(spec, report, banks, data, metadata, a):
    a.check(
        "shared_prefix_contract_version",
        spec.get("version") == STUDY_VERSION
        and spec.get("cache_strategy") == CACHE_STRATEGY,
    )
    bounds = spec["action_bounds"]
    low, high = np.asarray(bounds["low"]), np.asarray(bounds["high"])
    a.check(
        "explicit_versioned_bounds",
        bool(bounds.get("version"))
        and low.shape == high.shape == (5,)
        and np.isfinite([low, high]).all()
        and (high > low).all(),
    )
    wide = bounds["low"] != OLD_LOW or bounds["high"] != OLD_HIGH
    a.check(
        "native_bounds_adapter_declared",
        spec["native_environment"] == "sircus.mjlab.expanded_env.ExpandedEnvironment",
    )
    for key, bank in banks.items():
        expected = spec[
            "old_proposal_bank_sha256" if key == "old" else "proposal_bank_sha256"
        ]
        a.check(
            key + "_bank_hash",
            _bank_hash(bank)
            and bank["sha256"]
            == expected
            == report[
                "old_proposal_bank_sha256" if key == "old" else "proposal_bank_sha256"
            ],
        )
    old_count = len(banks["old"]["templates"])
    a.check(
        "old_template_prefix_preserved",
        banks["expanded"]["templates"][:old_count] == banks["old"]["templates"]
        and len(banks["expanded"]["templates"]) > old_count,
    )
    splits = ("train", "validation", "test", "development")
    expected_groups = {"expanded-" + k for k in splits} | {"old-" + k for k in splits}
    if wide:
        expected_groups |= {"old-wide-" + k for k in splits}
    a.check(
        "geometry_groups_complete",
        set(data) == expected_groups == set(metadata) == set(report["geometry"]),
    )
    for label, group in data.items():
        split = label.split("-")[-1]
        xy = spec[split + "_layouts"]
        bank = banks["old" if label.startswith("old-") else "expanded"]
        values, expected_metadata = materialize(bank, xy)
        n, k = values.shape[:2]
        a.check(
            label + "_materialization",
            np.array_equal(group["bucket"], xy)
            and group["values"].shape == values.shape
            and np.allclose(group["values"], values, atol=1e-12, rtol=0)
            and metadata[label] == expected_metadata,
        )
        a.check(
            label + "_shapes",
            group["mask"].shape == group["reasons"].shape == (n, k)
            and group["q10"].shape == group["q14"].shape == (n, k, 6),
        )
        a.check(
            label + "_mask_reasons_finite",
            group["mask"].dtype == np.bool_
            and np.array_equal(group["mask"], group["reasons"] == 0)
            and np.isin(group["reasons"], range(7)).all()
            and all(
                np.isfinite(group[x]).all() for x in ("bucket", "values", "q10", "q14")
            ),
        )
        lo, hi = (
            (np.asarray(OLD_LOW), np.asarray(OLD_HIGH))
            if label.startswith("old-") and not label.startswith("old-wide-")
            else (low, high)
        )
        selected = group["values"][group["mask"]]
        a.check(
            label + "_eligible_declared_bounds",
            bool(((selected >= lo - 1e-9) & (selected <= hi + 1e-9)).all()),
        )
        g = report["geometry"][label]
        expected = dict(
            queries=n,
            candidates_per_query=k,
            eligible_candidates=int(group["mask"].sum()),
            queries_with_candidate=int(group["mask"].any(1).sum()),
            unresolved_queries=int((~group["mask"].any(1)).sum()),
        )
        a.check(label + "_counts", all(g.get(k) == v for k, v in expected.items()))
    audit_shared_prefix(spec, data, metadata, report, a)
    return wide


def audit_shared_prefix(spec, data, metadata, report, a):
    """Original semantic slots keep identical audited branches under each envelope."""
    wide = (
        spec["action_bounds"]["low"] != OLD_LOW
        or spec["action_bounds"]["high"] != OLD_HIGH
    )
    for split in ("train", "validation", "test", "development"):
        target = "expanded-" + split
        source = ("old-wide-" if wide else "old-") + split
        if source not in data or target not in data:
            a.check(
                split + "_shared_prefix_cache_present",
                False,
                dict(source=source, target=target),
            )
            continue
        source_meta, target_meta = metadata[source], metadata[target]
        slots = {m["candidate_id"]: i for i, m in enumerate(target_meta)}
        missing = [
            m["candidate_id"] for m in source_meta if m["candidate_id"] not in slots
        ]
        a.check(
            split + "_shared_prefix_slot_ids",
            not missing and len(slots) == len(target_meta),
            missing,
        )
        if missing:
            continue
        ids = np.asarray([slots[m["candidate_id"]] for m in source_meta])
        bad = [
            key
            for key in ("values", "mask", "q10", "q14", "reasons")
            if not np.array_equal(data[source][key], data[target][key][:, ids])
        ]
        a.check(
            split + "_shared_prefix_exact_arrays",
            not bad,
            dict(source=source, fields_different=bad),
        )
        summary = report["geometry"][target]
        a.check(
            split + "_shared_prefix_summary",
            summary.get("shared_prefix_source") == source
            and summary.get("shared_prefix_candidates") == len(source_meta)
            and summary.get("newly_audited_candidates")
            == len(target_meta) - len(source_meta),
        )
        if wide:
            strict = "old-" + split
            if strict not in data:
                a.check(split + "_strict_overlap_cache_present", False)
                continue
            values = data[strict]["values"]
            within = (
                (values >= np.asarray(OLD_LOW) - 1e-9)
                & (values <= np.asarray(OLD_HIGH) + 1e-9)
            ).all(2)
            bad = [
                key
                for key in ("values", "mask", "q10", "q14", "reasons")
                if not np.array_equal(
                    data[strict][key][within], data[source][key][within]
                )
            ]
            a.check(
                split + "_wide_preserves_strict_overlap",
                not bad,
                dict(fields_different=bad, overlap_slots=int(within.sum())),
            )
            summary = report["geometry"][source]
            a.check(
                split + "_strict_overlap_summary",
                summary.get("strict_overlap_source") == strict
                and summary.get("strict_overlap_slots") == int(within.sum())
                and summary.get("newly_audited_slots") == int((~within).sum()),
            )


def _extra_stats(rows):
    eligible = [r for r in rows if r["eligible_bank_endpoint"]]
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row["bucket_xy_m"]), []).append(row)
    return dict(
        native_releases=sum(bool(r["native_release"]) for r in rows),
        eligible_attempts=len(eligible),
        conditional_capture_rate=(
            sum(bool(r["route_success"]) for r in eligible) / len(eligible)
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
            not any(r["eligible_bank_endpoint"] for r in group)
            for group in groups.values()
        ),
    )


def _stats(a, name, rows, actual):
    a.stats(name, statistics(rows), actual)
    expected = _extra_stats(rows)
    a.check(
        name + "_eligibility_and_positions",
        all(
            k in actual
            and (
                v == actual[k]
                if v is None
                else np.isclose(v, actual[k], atol=1e-8, rtol=1e-8)
            )
            for k, v in expected.items()
        ),
    )


def audit_rows(spec, report, rows, data, metadata, a):
    a.check(
        "study_completed", report.get("status") == report.get("phase") == "completed"
    )
    a.check(
        "report_contract_identity",
        report["contract_fingerprint"] == spec["fingerprint"],
    )
    a.check(
        "sequential_trial_ids",
        len(rows) == report["records"]
        and [r["id"] for r in rows]
        == [f"expanded-shot-{i:07d}" for i in range(len(rows))],
    )
    batch, updates, repeats = (
        spec["batch_size"],
        spec["ppo_updates"],
        spec["test_repeats"],
    )
    grouped = {}
    for row in rows:
        grouped.setdefault(row["phase"], []).append(row)
    train = grouped.get("train-ppo", [])
    curves = report["learning_curve"]
    a.check(
        "training_budget",
        len(train) == batch * updates == report["train_episodes"]
        and len(curves) == updates == report["updates"],
    )
    a.check(
        "training_releases",
        sum(r["native_release"] for r in train) == report["native_train_releases"],
    )
    eligible = data["expanded-train"]["mask"].any(1)
    sampling = report["train_sampling"]
    a.check(
        "eligible_training_denominator",
        sampling["declared_contexts"] == len(eligible)
        and sampling["eligible_contexts"] == int(eligible.sum())
        and sampling["excluded_contexts"] == int((~eligible).sum()),
    )
    rng = np.random.default_rng(TRAIN_SEED)
    seeded_queries = np.concatenate(
        [
            rng.choice(np.flatnonzero(eligible), batch, replace=True)
            for _ in range(updates)
        ]
    )
    a.check(
        "seeded_training_query_sequence",
        np.array_equal(seeded_queries, [r["query_index"] for r in train]),
    )
    validation = [(0, "validation-initial-0")] + [
        (i, f"validation-update-{i}-0")
        for i in range(1, updates + 1)
        if i % spec["validation_every"] == 0 or i == updates
    ]
    allowed = {"train-ppo"} | {label for _, label in validation}
    methods = list(METHODS) + (
        ["old_uniform_expanded_bounds"] if "old-wide-test" in data else []
    )
    a.check("evaluation_method_contract", spec["evaluation_methods"] == methods)
    for prefix, tags in (("test", methods), ("development", METHODS)):
        allowed |= {
            f"{prefix}-{method}-{rep}" for method in tags for rep in range(repeats)
        }
    a.check("known_phase_labels", set(grouped) == allowed)
    best_update, best_key = 0, None
    for update, label in validation:
        rr = grouped.get(label, [])
        a.check(
            label + "_query_order",
            [r["query_index"] for r in rr]
            == list(range(len(spec["validation_layouts"]))),
        )
        actual = (
            report["initial_validation"]
            if update == 0
            else curves[update - 1]["validation"]
        )
        _stats(a, label + "_statistics", rr, actual)
        key = statistics(rr)["captures"], statistics(rr)["mean_reward"]
        if best_key is None or key > best_key:
            best_update, best_key = update, key
    for i, curve in enumerate(curves):
        a.check(f"update_{i+1}_number", curve["update"] == i + 1)
        _stats(a, f"update_{i+1}_statistics", train[i * batch : (i + 1) * batch], curve)
    a.check(
        "validation_only_checkpoint_selection", report["selected_update"] == best_update
    )
    tests, devs = {}, {}
    for prefix, tags, target in (
        ("test", methods, tests),
        ("development", METHODS, devs),
    ):
        layouts = spec[("test" if prefix == "test" else "development") + "_layouts"]
        for method in tags:
            rr = [r for r in rows if r["phase"].startswith(f"{prefix}-{method}-")]
            target[method] = rr
            a.check(
                prefix + "_" + method + "_all_queries_retained",
                len(rr) == len(layouts) * repeats
                and all(
                    [
                        r["bucket_xy_m"]
                        for r in rr[i * len(layouts) : (i + 1) * len(layouts)]
                    ]
                    == layouts
                    for i in range(repeats)
                ),
            )
            actual = (
                report["frozen_evaluation"]["comparison"][method]
                if prefix == "test"
                else report["development_evaluation"][method]
            )
            _stats(a, prefix + "_" + method + "_statistics", rr, actual)
            a.check(
                prefix + "_" + method + "_frozen_repeats",
                all(
                    [r["candidate_index"] for r in rr[: len(layouts)]]
                    == [
                        r["candidate_index"]
                        for r in rr[i * len(layouts) : (i + 1) * len(layouts)]
                    ]
                    for i in range(repeats)
                ),
            )
            a.check(
                prefix + "_" + method + "_metadata",
                actual["layouts"] == len(layouts) and actual["repeats"] == repeats,
            )
            if prefix == "test":
                a.check(
                    method + "_no_search_declared",
                    actual.get("search_at_inference") is False,
                )
                bank_key = (
                    "old_proposal_bank_sha256"
                    if method.startswith("old_")
                    else "proposal_bank_sha256"
                )
                expected_bounds = (
                    spec["action_bounds"]
                    if method.startswith("expanded_")
                    or method == "old_uniform_expanded_bounds"
                    else dict(
                        low=OLD_LOW, high=OLD_HIGH, version="original-shot-bounds-v1"
                    )
                )
                a.check(
                    method + "_bank_and_bounds_identity",
                    actual.get("bank_sha256") == spec[bank_key]
                    and actual.get("action_bounds") == expected_bounds,
                )
    test_at = [i for i, r in enumerate(rows) if r["phase"].startswith("test-")]
    train_val_at = [
        i
        for i, r in enumerate(rows)
        if r["phase"] == "train-ppo" or r["phase"].startswith("validation-")
    ]
    a.check(
        "all_training_validation_precede_new_test",
        bool(test_at) and bool(train_val_at) and max(train_val_at) < min(test_at),
    )
    _stats(
        a,
        "top_level_selected_statistics",
        tests["expanded_learned"],
        report["frozen_evaluation"],
    )
    a.check(
        "reported_test_rows_exact",
        report["test_rows"] == tests["expanded_learned"]
        and report["frozen_evaluation"]["trials"] == len(tests["expanded_learned"]),
    )
    bad = []
    radius = spec["scene"]["ball"]["radius_m"]
    for row in rows:
        group_name = row.get("geometry_group")
        group = data.get(group_name)
        if group is None:
            bad.append((row["id"], "unknown_geometry_group"))
            continue
        query, index = row["query_index"], row["candidate_index"]
        if (
            not 0 <= query < len(group["bucket"])
            or not 0 <= index < group["mask"].shape[1]
        ):
            bad.append((row["id"], "index"))
            continue
        phase = row["phase"]
        expected_group = (
            "expanded-train"
            if phase == "train-ppo"
            else (
                "expanded-validation"
                if phase.startswith("validation-")
                else (
                    "old-wide-test"
                    if phase.startswith("test-old_uniform_expanded_bounds-")
                    else (
                        "old-"
                        if phase.startswith(("test-old_", "development-old_"))
                        else "expanded-"
                    )
                    + ("test" if phase.startswith("test-") else "development")
                )
            )
        )
        mask, value = group["mask"][query], group["values"][query, index]
        eligible = bool(mask.any())
        if (
            group_name != expected_group
            or row["bucket_xy_m"] != group["bucket"][query].tolist()
            or row["mask_sha256"] != _mask_hash(mask)
            or row["eligible_bank_endpoint"] != eligible
        ):
            bad.append((row["id"], "query_mask_or_group"))
        if not eligible:
            if (
                row["native_release"]
                or row["route_success"]
                or row["invalid"] != "no_eligible_bank_endpoint"
                or row["candidate_metadata"] is not None
                or row["behavior_logp"] is not None
            ):
                bad.append((row["id"], "all_invalid_query"))
            continue
        if not mask[index] or row["candidate_metadata"] != metadata[group_name][index]:
            bad.append((row["id"], "candidate_slot"))
        for key, expected in (
            ("values", value),
            ("q10", group["q10"][query, index]),
            ("q14", group["q14"][query, index]),
            ("release_ball_center_m", value[:3] + [0, 0, radius + value[4]]),
        ):
            if not np.allclose(row.get(key, []), expected, atol=1e-12, rtol=0):
                bad.append((row["id"], key))
        route = bool(
            row["success"]
            and row["board_contact_episodes"] >= 1
            and not row["external_contact_steps"]
            and not row["invalid"]
        )
        if (
            row["route_success"] != route
            or row["complete_success"] != route
            or row["native_release"] != (not bool(row["invalid"]))
        ):
            bad.append((row["id"], "native_route_score"))
        if row["success"] and not (
            row["board_bounce"]
            and row["whole_ball_entry"]
            and row["max_retention_s"] >= 1 - 1e-8
        ):
            bad.append((row["id"], "whole_ball_one_second_rule"))
        if row["native_release"]:
            gate = row.get("pose_gate", {})
            if (
                gate.get("board_position_error_mm", 1e6) > 2
                or gate.get("board_normal_error_deg", 1e6) > 0.5
                or gate.get("dropper_error_mm", 1e6) > 2
                or row.get("endpoint_collision_pairs", ["missing"])
            ):
                bad.append((row["id"], "native_pose_contact_gate"))
            if row.get("bucket_collider_error_m", 1e6) > 1e-5:
                bad.append((row["id"], "bucket_collision_transform"))
    a.check("row_geometry_branch_and_score_evidence", not bad, bad[:20])
    for label, first, second in (
        ("learning_vs_uniform", "expanded_learned", "expanded_uniform"),
        ("expanded_uniform_vs_old_uniform", "expanded_uniform", "old_uniform"),
        ("expanded_learned_vs_old_learned", "expanded_learned", "old_learned"),
    ):

        def rates(rr):
            groups = {}
            for row in rr:
                groups.setdefault(tuple(row["bucket_xy_m"]), []).append(
                    bool(row["route_success"])
                )
            return {key: float(np.mean(values)) for key, values in groups.items()}

        first_rates, second_rates = rates(tests[first]), rates(tests[second])
        delta = np.asarray(
            [first_rates[k] - second_rates[k] for k in sorted(first_rates)]
        )
        rng = np.random.default_rng(TEST_SEED + 91)
        draws = delta[rng.integers(0, len(delta), (2000, len(delta)))].mean(1)
        expected = dict(
            layouts=len(delta),
            capture_rate_difference=float(delta.mean()),
            descriptive_95pct_layout_bootstrap=np.quantile(
                draws, [0.025, 0.975]
            ).tolist(),
            improved_layouts=int((delta > 0).sum()),
            worsened_layouts=int((delta < 0).sum()),
            tied_layouts=int((delta == 0).sum()),
        )
        actual = report.get("paired_comparisons", {}).get(label, {})
        a.check(
            label + "_paired_statistics",
            all(
                k in actual and np.allclose(actual[k], v, atol=1e-10, rtol=0)
                for k, v in expected.items()
            ),
        )
    return dict(
        train=train,
        validation=validation,
        grouped=grouped,
        tests=tests,
        development=devs,
    )


def audit_checkpoints(study, spec, report, records, data, old_cp, a):
    import torch
    from .bank_policy import Policy, FEATURE_NAMES

    torch.set_num_threads(2)
    bank = json.loads((study / "proposal-bank.json").read_text())
    load = lambda name: torch.load(study / name, map_location="cpu", weights_only=True)
    checkpoints = [load(f"update-{i:03d}.pt") for i in range(spec["ppo_updates"] + 1)]
    hashes = [state_hash(cp["state"]) for cp in checkpoints]
    for i, cp in enumerate(checkpoints):
        a.check(
            f"checkpoint_{i}_identity",
            cp["contract"] == spec
            and cp["bank"] == bank
            and cp["weight_sha256"] == hashes[i]
            and cp["policy_class"] == "expanded_relative_bank_categorical_v1"
            and (i == 0 or cp["update"] == i),
        )
    selected, uniform = load("selected.pt"), load("uniform.pt")
    selected_hash, uniform_hash = state_hash(selected["state"]), state_hash(
        uniform["state"]
    )
    a.check(
        "selected_checkpoint_identity",
        selected_hash
        == hashes[report["selected_update"]]
        == selected["weight_sha256"]
        == report["policy_weight_sha256"]
        and selected["contract"] == spec
        and selected["bank"] == bank
        and selected["selected_update"] == report["selected_update"]
        and selected["policy_class"] == "expanded_relative_bank_categorical_v1"
        and selected["feature_names"] == list(FEATURE_NAMES)
        and file_hash(study / "selected.pt") == report["checkpoint_sha256"],
    )
    a.check(
        "uniform_checkpoint_identity",
        uniform["contract"] == spec
        and uniform["bank"] == bank
        and uniform["weight_sha256"] == uniform_hash
        and uniform["policy_class"] == "expanded_relative_bank_categorical_v1"
        and bool((uniform["state"]["actor.4.weight"] == 0).all())
        and bool((uniform["state"]["actor.4.bias"] == 0).all()),
    )
    if old_cp is not None:
        a.check(
            "warm_start_exact_old_selected_weights",
            hashes[0] == state_hash(old_cp["state"]),
        )
    else:
        a.warnings.append(
            "Old checkpoint unavailable; warm-start and old learned predictions are missing."
        )
    errors = []

    def predictions(rows, cp, label, rule=None):
        bad = []
        policy = Policy()
        policy.load_state_dict(cp["state"])
        policy.eval()
        expected_hash = state_hash(cp["state"])
        # Bounded batches avoid allocating a full 300-query candidate-feature tensor.
        for start in range(0, len(rows), 32):
            chunk = rows[start : start + 32]
            group = data[chunk[0]["geometry_group"]]
            ids = np.asarray([r["query_index"] for r in chunk])
            indices = np.asarray([r["candidate_index"] for r in chunk])
            with torch.no_grad():
                dist, eligible = policy.distribution(
                    torch.tensor(group["bucket"][ids], dtype=torch.float32),
                    torch.tensor(group["values"][ids], dtype=torch.float32),
                    torch.tensor(group["mask"][ids]),
                )
                lp = dist.log_prob(torch.tensor(indices)).numpy()
                picks = dist.probs.argmax(-1).numpy()
            if rule == "uniform":
                seed = (
                    TEST_SEED + 71
                    if "test" in chunk[0]["geometry_group"]
                    else VALIDATION_SEED + 71
                )
                quantiles = np.random.default_rng(seed).random(len(group["bucket"]))
                for i, q in enumerate(ids):
                    legal = np.flatnonzero(group["mask"][q])
                    picks[i] = (
                        legal[min(len(legal) - 1, int(quantiles[q] * len(legal)))]
                        if len(legal)
                        else 0
                    )
            for i, row in enumerate(chunk):
                if row["behavior_policy_weight_sha256"] != expected_hash:
                    bad.append((row["id"], "weight_identity"))
                if eligible[i]:
                    e = abs(float(lp[i]) - row["behavior_logp"])
                    errors.append(e)
                    if e > 2e-5:
                        bad.append((row["id"], "behavior_logp"))
                if rule and picks[i] != indices[i]:
                    bad.append((row["id"], "frozen_prediction"))
        a.check(label, not bad, bad[:10])

    for i, curve in enumerate(report["learning_curve"]):
        change = curve["weight_change"]
        a.check(
            f"ppo_{i+1}_weight_chain",
            change["before"] == hashes[i]
            and change["after"] == hashes[i + 1]
            and hashes[i] != hashes[i + 1]
            and change["optimizer_rows"] == spec["batch_size"]
            and change["excluded_all_invalid"] == 0,
        )
        a.check(
            f"ppo_{i+1}_finite_epoch_evidence",
            len(change["epochs"]) == spec["epochs"]
            and all(np.isfinite(list(e.values())).all() for e in change["epochs"]),
        )
        predictions(
            records["train"][i * spec["batch_size"] : (i + 1) * spec["batch_size"]],
            checkpoints[i],
            f"ppo_{i+1}_preupdate_logp",
        )
    for update, label in records["validation"]:
        predictions(
            records["grouped"][label],
            checkpoints[update],
            label + "_frozen_argmax",
            "argmax",
        )
    states = dict(
        old_uniform=uniform,
        old_learned=old_cp,
        expanded_uniform=uniform,
        expanded_learned=selected,
        old_uniform_expanded_bounds=uniform,
    )
    for prefix, groups in (
        ("test", records["tests"]),
        ("development", records["development"]),
    ):
        for method, rr in groups.items():
            cp = states[method]
            if cp is None:
                continue
            predictions(
                rr,
                cp,
                prefix + "_" + method + "_frozen_predictions",
                "uniform" if "uniform" in method else "argmax",
            )
            if prefix == "test":
                a.check(
                    method + "_reported_frozen_weight",
                    report["frozen_evaluation"]["comparison"][method][
                        "policy_weight_sha256"
                    ]
                    == state_hash(cp["state"]),
                )
    return dict(
        update_weight_sha256=hashes,
        selected_weight_sha256=selected_hash,
        maximum_logp_error=max(errors, default=0.0),
    )


def verify_promotion(
    study, old_bank_study, *, prior_search=None, source=None, assets=None
):
    """Dispatch by frozen evidence protocol; refinement has a separate ancestry."""
    version = json.loads((Path(study) / "contract.json").read_text())["version"]
    if version == "sircus-targeted-bank-expansion-v1":
        from .verify_expansion import verify as verify_expansion

        return verify_expansion(study, old_bank_study, source=source, assets=assets)
    if version == "sircus-targeted-bank-refinement-v1":
        if prior_search is None:
            raise ValueError(
                "Refinement promotion verification requires --prior-search"
            )
        from .verify_refinement import verify as verify_refinement

        return verify_refinement(study, prior_search, old_bank_study, source=source)
    if version == "sircus-confirmed-bank-combination-v1":
        from .verify_combined import verify as verify_combined

        return verify_combined(study, source=source, assets=assets)
    raise ValueError("Unsupported native promotion evidence protocol: " + version)


def audit_ancestry(
    spec,
    report,
    banks,
    a,
    *,
    parent_study=None,
    old_bank_study=None,
    expansion_study=None,
    prior_search=None,
    expansion_source=None,
    source=None,
    assets=None,
):
    old_cp = None
    for role, path in (("parent", parent_study), ("old_bank", old_bank_study)):
        if path is None:
            a.warnings.append(
                role + " study unavailable; ancestry verification missing."
            )
            continue
        path = Path(path)
        ancestor = json.loads((path / "contract.json").read_text())
        a.check(
            role + "_contract_and_checkpoint_preserved",
            _fingerprint(ancestor)
            and ancestor["fingerprint"] == spec[role + "_contract_fingerprint"]
            and file_hash(path / "selected.pt") == spec[role + "_checkpoint_sha256"],
        )
        a.check(role + "_scene_geometry_unchanged", ancestor["scene"] == spec["scene"])
        original_report = json.loads((path / "report.json").read_text())
        critical = (
            "model.py",
            "run.py",
            "contract.py",
            "learning_contract.py",
            "learning_env.py",
            "learning_ik.py",
            "learning_score.py",
            "learning_reach.py",
            "bank_policy.py",
        )
        a.check(
            role + "_native_physics_scoring_sources_preserved",
            all(
                report["source_sha256"].get(k) == v
                for k, v in original_report["source_sha256"].items()
                if k in critical
            ),
        )
        if role == "parent":
            a.check(
                "parent_training_grid_preserved",
                spec["train_layouts"][:35] == ancestor["train_layouts"],
            )
        else:
            import torch

            old_cp = torch.load(
                path / "selected.pt", map_location="cpu", weights_only=True
            )
            g = np.load(path / "geometry-test.npz", allow_pickle=False)
            expected_dev = g["bucket"][~g["mask"].any(1)].tolist()
            a.check(
                "old_twelve_empty_masks_exactly_converted_to_development",
                expected_dev == spec["development_layouts"],
            )
            a.check("old_bank_artifact_unchanged", old_cp["bank"] == banks["old"])
    if expansion_study is not None:
        path = Path(expansion_study)
        expansion = json.loads((path / "contract.json").read_text())
        a.check(
            "completed_expansion_artifact_bound",
            _fingerprint(expansion)
            and expansion["fingerprint"] == spec["expansion_contract_fingerprint"]
            and file_hash(path / "report.json") == spec["expansion_report_sha256"]
            and file_hash(path / "expanded-bank.json")
            == spec["expanded_bank_artifact_sha256"]
            and json.loads((path / "expanded-bank.json").read_text())
            == banks["expanded"],
        )
        if old_bank_study is not None:
            try:
                result = verify_promotion(
                    path,
                    old_bank_study,
                    prior_search=prior_search,
                    source=expansion_source,
                    assets=assets,
                )
                a.check(
                    "independent_native_promotion_audit",
                    result["passed"],
                    [c for c in result["checks"] if not c["passed"]],
                )
            except (ValueError, FileNotFoundError) as e:
                a.check("independent_native_promotion_audit", False, str(e))
            if expansion_source is None:
                a.warnings.append(
                    "Expansion source snapshot not supplied; promotion records are audited but its exact executed source bytes are missing."
                )
    else:
        a.warnings.append(
            "Expansion study missing; independent native promotion audit missing."
        )
    if parent_study is not None and source is not None and assets is not None:
        from .learning_solve import validate_artifacts

        try:
            validate_artifacts(
                parent_study, assets, repository=Path(source).resolve().parents[1]
            )
            a.check("original_shared_scorer_and_model_asset_bytes", True)
        except Exception as e:
            a.check("original_shared_scorer_and_model_asset_bytes", False, str(e))
    else:
        a.warnings.append(
            "Shared scorer/generated asset bytes require --parent-study, --source and --assets; current verification is missing."
        )
    return old_cp


def verify(
    study,
    *,
    source=None,
    parent_study=None,
    old_bank_study=None,
    expansion_study=None,
    prior_search=None,
    expansion_source=None,
    assets=None,
    checkpoints=True,
):
    study = Path(study)
    read = lambda name: json.loads((study / name).read_text())
    spec, report = read("contract.json"), read("report.json")
    banks = dict(
        old=read("old-proposal-bank.json"), expanded=read("proposal-bank.json")
    )
    data = {
        p.stem.removeprefix("geometry-"): dict(np.load(p, allow_pickle=False))
        for p in study.glob("geometry-*.npz")
    }
    metadata = {
        p.stem.removeprefix("candidate-metadata-"): json.loads(p.read_text())
        for p in study.glob("candidate-metadata-*.json")
    }
    rows = [
        json.loads(line)
        for line in (study / "shots.jsonl").read_text().splitlines()
        if line.strip()
    ]
    a = Audit()
    pools = audit_splits(spec, a)
    audit_geometry(spec, report, banks, data, metadata, a)
    records = audit_rows(spec, report, rows, data, metadata, a)
    a.check(
        "split_file_identity",
        read("splits.json") == dict(**pools, development=spec["development_layouts"]),
    )
    required = {
        "contract.json",
        "splits.json",
        "proposal-bank.json",
        "old-proposal-bank.json",
        "shots.jsonl",
        "selected.pt",
        "uniform.pt",
    }
    required |= {f"geometry-{k}.npz" for k in data} | {
        f"candidate-metadata-{k}.json" for k in data
    }
    required |= {f"update-{i:03d}.pt" for i in range(spec["ppo_updates"] + 1)}
    hashes = report["artifacts_sha256"]
    a.check("critical_artifact_manifest_complete", required.issubset(hashes))
    bad = [
        name
        for name, sha in hashes.items()
        if Path(name).name != name
        or not (study / name).is_file()
        or file_hash(study / name) != sha
    ]
    a.check("all_artifact_checksums", not bad, bad)
    if source is not None:
        source = Path(source)
        bad = [
            name
            for name, sha in report["source_sha256"].items()
            if Path(name).name != name
            or not (source / name).is_file()
            or file_hash(source / name) != sha
        ]
        required_source = {
            "expanded_study.py",
            "expanded_env.py",
            "expanded_actions.py",
            "expand_geometry.py",
            "bank_policy.py",
            "bank_geometry.py",
            "model.py",
            "learning_score.py",
        }
        a.check(
            "executed_source_bytes",
            not bad and required_source.issubset(report["source_sha256"]),
            bad,
        )
    else:
        a.warnings.append(
            "Executed source not supplied; exact code verification missing."
        )
    old_cp = audit_ancestry(
        spec,
        report,
        banks,
        a,
        parent_study=parent_study,
        old_bank_study=old_bank_study,
        expansion_study=expansion_study,
        prior_search=prior_search,
        expansion_source=expansion_source,
        source=source,
        assets=assets,
    )
    result = dict(
        version=VERSION,
        contract_fingerprint=spec["fingerprint"],
        records=len(rows),
        phase_counts=dict(Counter(r["phase"] for r in rows)),
    )
    if checkpoints:
        result["checkpoints"] = audit_checkpoints(
            study, spec, report, records, data, old_cp, a
        )
    else:
        a.warnings.append("Checkpoint/action reconstruction explicitly disabled.")
    if assets is not None:
        from .verify_learning import audit_fk

        accepted = [
            dict(row, locks=dict(board=None, dropper=None))
            for row in rows
            if row["native_release"]
        ]
        result["independent_fk"] = audit_fk(Path(assets), spec, accepted, a)
    else:
        a.warnings.append(
            "Generated URDF assets not supplied; independent accepted-endpoint joint-limit/FK verification missing."
        )
    a.warnings.extend(
        [
            "Previously unresolved twelve positions are development, not new held-out evidence.",
            "Nominal repeats establish numerical repeatability, not perturbation robustness or exhaustive tabletop coverage.",
            "Recorded checkpoint changes and pre-update probabilities are verified; optimizer gradients and GPU stochastic samples are not independently replayed.",
            "Stored identities/order support frozen evaluation but cannot independently prove wall-clock chronology or exclude unrecorded external experiments.",
            "No native ball dynamics rerun or collision-free arm transit certification occurs in this CPU audit.",
        ]
    )
    result.update(
        passed=all(c["passed"] for c in a.checks), checks=a.checks, warnings=a.warnings
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    for name in (
        "source",
        "parent-study",
        "old-bank-study",
        "expansion-study",
        "prior-search",
        "expansion-source",
        "assets",
        "output",
    ):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--records-only", action="store_true")
    args = parser.parse_args()
    result = verify(
        args.study,
        source=args.source,
        parent_study=args.parent_study,
        old_bank_study=args.old_bank_study,
        expansion_study=args.expansion_study,
        prior_search=args.prior_search,
        expansion_source=args.expansion_source,
        assets=args.assets,
        checkpoints=not args.records_only,
    )
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    raise SystemExit(not result["passed"])
