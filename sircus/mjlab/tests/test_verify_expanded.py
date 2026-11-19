"""Synthetic corruption tests for the independent CPU expanded-study auditor."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from sircus.mjlab.expanded_study import new_splits, enriched_stats, paired_comparison
from sircus.mjlab.learning_contract import splits
from sircus.mjlab.verify_bank import materialize, _mask_hash
from sircus.mjlab.verify_learning import Audit, digest, state_hash, file_hash
from sircus.mjlab.verify_expanded import (
    METHODS,
    OLD_LOW,
    OLD_HIGH,
    TRAIN_SEED,
    VALIDATION_SEED,
    TEST_SEED,
    audit_splits,
    audit_geometry,
    audit_rows,
    audit_checkpoints,
    verify_promotion,
    audit_shared_prefix,
    STUDY_VERSION,
    CACHE_STRATEGY,
)


def fixture():
    original = splits()
    dev = [[0.12 + 0.03 * i, -0.51] for i in range(12)]
    historical = sum(original.values(), []) + dev
    pools = new_splits(
        original["train"], dev, historical, random_train=0, validation_count=4
    )
    teacher = dict(
        source_trial_id="old-0",
        relative_board_xy_m=[0, 0.2],
        height_m=0.35,
        tilt_deg=25,
        gap_m=0.2,
    )
    old = dict(version="sircus-relative-search-bank-v1", templates=[teacher])
    old["sha256"] = digest(old)
    expanded = dict(
        version=old["version"],
        templates=[teacher, dict(teacher, source_trial_id="development-0", gap_m=0.22)],
    )
    expanded["sha256"] = digest(expanded)
    banks = dict(old=old, expanded=expanded)
    spec = dict(
        version=STUDY_VERSION,
        cache_strategy=CACHE_STRATEGY,
        scene=dict(ball=dict(radius_m=0.0375)),
        batch_size=2,
        ppo_updates=1,
        validation_every=1,
        test_repeats=3,
        min_split_distance_m=0.005,
        epochs=6,
        train_seed=TRAIN_SEED,
        validation_seed=VALIDATION_SEED,
        test_seed=TEST_SEED,
        historical_layouts=historical,
        action_bounds=dict(
            low=OLD_LOW, high=OLD_HIGH, version="original-shot-bounds-v1"
        ),
        native_environment="sircus.mjlab.expanded_env.ExpandedEnvironment",
        proposal_bank_sha256=expanded["sha256"],
        old_proposal_bank_sha256=old["sha256"],
        evaluation_methods=list(METHODS),
        **{k + "_layouts": v for k, v in pools.items()},
    )
    spec["fingerprint"] = digest(spec)
    data, metadata, geometry = {}, {}, {}
    for bank_name, labels in (
        ("expanded", ("train", "validation", "test", "development")),
        ("old", ("train", "validation", "test", "development")),
    ):
        for label in labels:
            key = bank_name + "-" + label
            xy = pools[label]
            values, meta = materialize(banks[bank_name], xy)
            n, k = values.shape[:2]
            mask = np.zeros((n, k), dtype=bool)
            within = (
                (values >= np.asarray(OLD_LOW)) & (values <= np.asarray(OLD_HIGH))
            ).all(2)
            for i in range(n):
                legal = np.flatnonzero(within[i])
                if len(legal):
                    mask[i, legal[0]] = True
            if label == "test":
                mask[-1] = False
            data[key] = dict(
                bucket=np.asarray(xy),
                values=values,
                mask=mask,
                reasons=np.where(mask, 0, 1).astype(np.uint8),
                q10=np.zeros((n, k, 6)),
                q14=np.zeros((n, k, 6)),
            )
            metadata[key] = meta
            geometry[key] = dict(
                queries=n,
                candidates_per_query=k,
                eligible_candidates=int(mask.sum()),
                queries_with_candidate=int(mask.any(1).sum()),
                unresolved_queries=int((~mask.any(1)).sum()),
            )
    for split in ("train", "validation", "test", "development"):
        key = "expanded-" + split
        geometry[key].update(
            shared_prefix_source="old-" + split,
            shared_prefix_candidates=len(metadata["old-" + split]),
            newly_audited_candidates=len(metadata[key]) - len(metadata["old-" + split]),
        )
    rows = []

    def add(label, query, phase):
        group = data[label]
        eligible = bool(group["mask"][query].any())
        index = int(np.flatnonzero(group["mask"][query])[0]) if eligible else 0
        value = group["values"][query, index]
        row = dict(
            id=f"expanded-shot-{len(rows):07d}",
            phase=phase,
            geometry_group=label,
            query_index=query,
            candidate_index=index,
            candidate_metadata=metadata[label][index] if eligible else None,
            bucket_xy_m=group["bucket"][query].tolist(),
            mask_sha256=_mask_hash(group["mask"][query]),
            eligible_bank_endpoint=eligible,
            behavior_logp=0.0 if eligible else None,
            behavior_policy_weight_sha256="synthetic",
            native_release=eligible,
            invalid=None if eligible else "no_eligible_bank_endpoint",
            route_success=eligible,
            route_quality=2 if eligible else 0,
            board_bounce=eligible,
            success=eligible,
            complete_success=eligible,
            whole_ball_entry=eligible,
            max_retention_s=1.0 if eligible else 0.0,
            closest_opening_distance_m=0.01 if eligible else 2.0,
            external_contact_steps=0,
            single_bounce_success=eligible,
            board_contact_episodes=1 if eligible else 0,
        )
        if eligible:
            row.update(
                values=value.tolist(),
                q10=[0.0] * 6,
                q14=[0.0] * 6,
                release_ball_center_m=(value[:3] + [0, 0, 0.0375 + value[4]]).tolist(),
                pose_gate=dict(
                    board_position_error_mm=0.0,
                    board_normal_error_deg=0.0,
                    dropper_error_mm=0.0,
                ),
                endpoint_collision_pairs=[],
                bucket_collider_error_m=0.0,
            )
        rows.append(row)
        return row

    initial = [
        add("expanded-validation", i, "validation-initial-0")
        for i in range(len(pools["validation"]))
    ]
    train_ids = np.random.default_rng(TRAIN_SEED).choice(
        np.flatnonzero(data["expanded-train"]["mask"].any(1)), 2, replace=True
    )
    train = [add("expanded-train", int(i), "train-ppo") for i in train_ids]
    val = [
        add("expanded-validation", i, "validation-update-1-0")
        for i in range(len(pools["validation"]))
    ]
    comparisons, development = {}, {}
    for prefix, target, split in (
        ("test", comparisons, "test"),
        ("development", development, "development"),
    ):
        for method in METHODS:
            label = ("old-" if method.startswith("old_") else "expanded-") + split
            target[method] = [
                add(label, i, f"{prefix}-{method}-{rep}")
                for rep in range(3)
                for i in range(len(pools[split]))
            ]
    report = dict(
        status="completed",
        phase="completed",
        contract_fingerprint=spec["fingerprint"],
        proposal_bank_sha256=expanded["sha256"],
        old_proposal_bank_sha256=old["sha256"],
        geometry=geometry,
        records=len(rows),
        train_episodes=2,
        native_train_releases=2,
        updates=1,
        selected_update=0,
        initial_validation=enriched_stats(initial),
        learning_curve=[
            dict(update=1, **enriched_stats(train), validation=enriched_stats(val))
        ],
        train_sampling=dict(
            declared_contexts=len(pools["train"]),
            eligible_contexts=int(data["expanded-train"]["mask"].any(1).sum()),
            excluded_contexts=int((~data["expanded-train"]["mask"].any(1)).sum()),
        ),
        development_evaluation={
            k: dict(**enriched_stats(v), layouts=12, repeats=3)
            for k, v in development.items()
        },
        frozen_evaluation=dict(
            **enriched_stats(comparisons["expanded_learned"]),
            trials=300,
            comparison={
                k: dict(
                    **enriched_stats(v),
                    layouts=100,
                    repeats=3,
                    search_at_inference=False,
                    bank_sha256=(
                        old["sha256"] if k.startswith("old_") else expanded["sha256"]
                    ),
                    action_bounds=spec["action_bounds"],
                )
                for k, v in comparisons.items()
            },
        ),
        test_rows=comparisons["expanded_learned"],
        paired_comparisons=dict(
            learning_vs_uniform=paired_comparison(
                comparisons["expanded_learned"], comparisons["expanded_uniform"]
            ),
            expanded_uniform_vs_old_uniform=paired_comparison(
                comparisons["expanded_uniform"], comparisons["old_uniform"]
            ),
            expanded_learned_vs_old_learned=paired_comparison(
                comparisons["expanded_learned"], comparisons["old_learned"]
            ),
        ),
    )
    return spec, report, banks, data, metadata, rows


class ExpandedAuditTests(unittest.TestCase):
    def setUp(self):
        self.spec, self.report, self.banks, self.data, self.metadata, self.rows = (
            fixture()
        )

    def audit(self):
        a = Audit()
        audit_splits(self.spec, a)
        audit_geometry(self.spec, self.report, self.banks, self.data, self.metadata, a)
        records = audit_rows(
            self.spec, self.report, self.rows, self.data, self.metadata, a
        )
        return a, records

    def test_positive_complete_evidence_and_unresolved_denominator(self):
        a, _ = self.audit()
        self.assertTrue(
            all(c["passed"] for c in a.checks), [c for c in a.checks if not c["passed"]]
        )
        self.assertEqual(self.report["frozen_evaluation"]["attempts"], 300)
        self.assertGreater(self.report["frozen_evaluation"]["unresolved_positions"], 0)

    def test_fresh_test_rejects_old_development_overlap_and_near_overlap(self):
        for point in (
            self.spec["development_layouts"][0],
            (np.asarray(self.spec["train_layouts"][0]) + [0.001, 0]).tolist(),
        ):
            spec = copy.deepcopy(self.spec)
            spec["test_layouts"][0] = point
            a = Audit()
            audit_splits(spec, a)
            self.assertFalse(
                next(
                    c["passed"]
                    for c in a.checks
                    if c["name"] == "test_fresh_5mm_separation"
                )
            )

    def test_old_prefix_and_mask_corruption_are_rejected(self):
        self.banks["expanded"]["templates"][0] = dict(
            self.banks["expanded"]["templates"][0], gap_m=0.21
        )
        self.data["old-test"]["reasons"][0, 0] = 0
        a, _ = self.audit()
        self.assertFalse(
            next(
                c["passed"]
                for c in a.checks
                if c["name"] == "old_template_prefix_preserved"
            )
        )
        self.assertTrue(
            any(not c["passed"] for c in a.checks if "mask_reasons" in c["name"])
        )

    def test_cached_branch_and_score_retention_corruption(self):
        row = next(r for r in self.rows if r["native_release"])
        row["q14"][0] = 0.01
        row["max_retention_s"] = 0.9
        a, _ = self.audit()
        failed = next(
            c for c in a.checks if c["name"] == "row_geometry_branch_and_score_evidence"
        )
        self.assertFalse(failed["passed"])
        self.assertTrue(any(reason == "q14" for _, reason in failed["detail"]))
        self.assertTrue(
            any(
                reason == "whole_ball_one_second_rule" for _, reason in failed["detail"]
            )
        )

    def test_shared_original_slot_branch_cannot_change_with_bank_size(self):
        # The same semantic ID occupies different numeric columns after expansion.
        source = "old-test"
        target = "expanded-test"
        source_id = self.metadata[source][2]["candidate_id"]
        target_id = next(
            i
            for i, m in enumerate(self.metadata[target])
            if m["candidate_id"] == source_id
        )
        self.assertNotEqual(2, target_id)
        self.data[target]["q14"][0, target_id, 0] = 0.01
        a = Audit()
        audit_shared_prefix(self.spec, self.data, self.metadata, self.report, a)
        c = next(c for c in a.checks if c["name"] == "test_shared_prefix_exact_arrays")
        self.assertFalse(c["passed"])
        self.assertEqual(c["detail"]["fields_different"], ["q14"])

    def test_shared_prefix_uses_wide_envelope_cache_when_declared(self):
        from sircus.mjlab.expanded_actions import WIDE_BOUNDS

        self.spec["action_bounds"] = WIDE_BOUNDS
        for split in ("train", "validation", "test", "development"):
            self.data["old-wide-" + split] = copy.deepcopy(self.data["old-" + split])
            self.metadata["old-wide-" + split] = copy.deepcopy(
                self.metadata["old-" + split]
            )
            self.report["geometry"]["expanded-" + split]["shared_prefix_source"] = (
                "old-wide-" + split
            )
            values = self.data["old-" + split]["values"]
            within = (
                (values >= np.asarray(OLD_LOW) - 1e-9)
                & (values <= np.asarray(OLD_HIGH) + 1e-9)
            ).all(2)
            self.report["geometry"]["old-wide-" + split] = dict(
                strict_overlap_source="old-" + split,
                strict_overlap_slots=int(within.sum()),
                newly_audited_slots=int((~within).sum()),
            )
        # Only out-of-domain strict slots may differ in the wider envelope.
        values = self.data["old-test"]["values"]
        within = (
            (values >= np.asarray(OLD_LOW) - 1e-9)
            & (values <= np.asarray(OLD_HIGH) + 1e-9)
        ).all(2)
        i, j = np.argwhere(~within)[0]
        self.data["old-test"]["q10"][i, j, 0] = 0.5
        a = Audit()
        audit_shared_prefix(self.spec, self.data, self.metadata, self.report, a)
        self.assertTrue(all(c["passed"] for c in a.checks), a.checks)
        i, j = np.argwhere(within)[0]
        self.data["old-wide-test"]["q14"][i, j, 0] = 0.1
        candidate_id = self.metadata["old-wide-test"][j]["candidate_id"]
        expanded_j = next(
            k
            for k, m in enumerate(self.metadata["expanded-test"])
            if m["candidate_id"] == candidate_id
        )
        self.data["expanded-test"]["q14"][i, expanded_j, 0] = 0.1
        a = Audit()
        audit_shared_prefix(self.spec, self.data, self.metadata, self.report, a)
        self.assertFalse(
            next(
                c["passed"]
                for c in a.checks
                if c["name"] == "test_wide_preserves_strict_overlap"
            )
        )

    def test_all_invalid_query_cannot_release_or_be_dropped(self):
        row = next(
            r
            for r in self.rows
            if r["phase"].startswith("test-") and not r["eligible_bank_endpoint"]
        )
        row["native_release"] = True
        a, _ = self.audit()
        self.assertFalse(
            next(
                c["passed"]
                for c in a.checks
                if c["name"] == "row_geometry_branch_and_score_evidence"
            )
        )
        self.rows.remove(row)
        a, _ = self.audit()
        self.assertTrue(
            any(
                not c["passed"] for c in a.checks if "all_queries_retained" in c["name"]
            )
        )

    def test_mislabeling_new_test_as_development_fails(self):
        row = next(r for r in self.rows if r["phase"] == "test-expanded_learned-0")
        row["geometry_group"] = "expanded-development"
        a, _ = self.audit()
        self.assertFalse(
            next(
                c["passed"]
                for c in a.checks
                if c["name"] == "row_geometry_branch_and_score_evidence"
            )
        )

    def test_promotion_dispatch_requires_refinement_prior_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "contract.json").write_text(
                json.dumps(dict(version="sircus-targeted-bank-refinement-v1"))
            )
            with self.assertRaisesRegex(ValueError, "prior-search"):
                verify_promotion(path, "old")
            expected = dict(passed=True, checks=[])
            with mock.patch(
                "sircus.mjlab.verify_refinement.verify", return_value=expected
            ) as runner:
                self.assertEqual(
                    verify_promotion(
                        path, "old", prior_search="prior", source="source"
                    ),
                    expected,
                )
                runner.assert_called_once_with(path, "prior", "old", source="source")
            (path / "contract.json").write_text(
                json.dumps(dict(version="sircus-targeted-bank-expansion-v1"))
            )
            with mock.patch(
                "sircus.mjlab.verify_expansion.verify", return_value=expected
            ) as runner:
                self.assertEqual(
                    verify_promotion(path, "old", assets="assets"), expected
                )
                runner.assert_called_once_with(
                    path, "old", source=None, assets="assets"
                )
            (path / "contract.json").write_text(
                json.dumps(dict(version="sircus-confirmed-bank-combination-v1"))
            )
            with mock.patch(
                "sircus.mjlab.verify_combined.verify", return_value=expected
            ) as runner:
                self.assertEqual(
                    verify_promotion(
                        path, "old", source="join-source", assets="assets"
                    ),
                    expected,
                )
                runner.assert_called_once_with(
                    path, source="join-source", assets="assets"
                )

    @unittest.skipUnless(importlib.util.find_spec("torch"), "CPU torch optional")
    def test_actual_checkpoint_chain_and_frozen_predictions(self):
        import torch
        from sircus.mjlab.bank_policy import Policy, update, FEATURE_NAMES
        from sircus.mjlab.verify_learning import reward

        a, records = self.audit()
        policy = Policy()
        warm = copy.deepcopy(policy.state_dict())
        h0 = state_hash(warm)
        ids = np.asarray([r["query_index"] for r in records["train"]])
        g = self.data["expanded-train"]
        xy = torch.tensor(g["bucket"][ids], dtype=torch.float32)
        values = torch.tensor(g["values"][ids], dtype=torch.float32)
        mask = torch.tensor(g["mask"][ids])
        indices = torch.tensor([r["candidate_index"] for r in records["train"]])
        with torch.no_grad():
            dist, _ = policy.distribution(xy, values, mask)
            lp = dist.log_prob(indices)
        change = update(
            policy,
            torch.optim.Adam(policy.parameters(), lr=0.0003),
            xy,
            values,
            mask,
            indices,
            lp,
            torch.tensor([reward(r) for r in records["train"]], dtype=torch.float32),
        )
        current = copy.deepcopy(policy.state_dict())
        h1 = state_hash(current)
        self.report["learning_curve"][0]["weight_change"] = change
        for row in self.rows:
            row["behavior_policy_weight_sha256"] = (
                h1 if row["phase"].startswith("validation-update-") else h0
            )
        for result in self.report["frozen_evaluation"]["comparison"].values():
            result["policy_weight_sha256"] = h0
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "proposal-bank.json").write_text(json.dumps(self.banks["expanded"]))
            common = dict(
                contract=self.spec,
                bank=self.banks["expanded"],
                policy_class="expanded_relative_bank_categorical_v1",
            )
            torch.save(
                dict(**common, state=warm, weight_sha256=h0), path / "update-000.pt"
            )
            torch.save(
                dict(**common, state=current, weight_sha256=h1, update=1),
                path / "update-001.pt",
            )
            torch.save(
                dict(**common, state=warm, weight_sha256=h0), path / "uniform.pt"
            )
            torch.save(
                dict(
                    **common,
                    state=warm,
                    weight_sha256=h0,
                    selected_update=0,
                    feature_names=list(FEATURE_NAMES),
                ),
                path / "selected.pt",
            )
            self.report.update(
                policy_weight_sha256=h0,
                checkpoint_sha256=file_hash(path / "selected.pt"),
            )
            a = Audit()
            audit_checkpoints(
                path, self.spec, self.report, records, self.data, dict(state=warm), a
            )
            self.assertTrue(
                all(c["passed"] for c in a.checks),
                [c for c in a.checks if not c["passed"]],
            )
            self.rows[0]["behavior_logp"] = 5.0
            a = Audit()
            audit_checkpoints(
                path, self.spec, self.report, records, self.data, dict(state=warm), a
            )
            self.assertTrue(
                any(
                    not c["passed"]
                    for c in a.checks
                    if "logp" in c["name"] or "argmax" in c["name"]
                )
            )


if __name__ == "__main__":
    unittest.main()
