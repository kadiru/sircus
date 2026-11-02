"""Finite-bank evidence corruption checks; no GPU or simulator required."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sircus.mjlab.verify_bank import (
    _mask_hash,
    audit_geometry,
    audit_rows,
    audit_checkpoints,
    materialize,
)
from sircus.mjlab.verify_learning import (
    Audit,
    digest,
    statistics,
    state_hash,
    file_hash,
)


def fixture():
    bank = dict(
        version="sircus-relative-search-bank-v1",
        templates=[
            dict(
                source_trial_id="teacher-0",
                source_cell_id="train-0",
                relative_board_xy_m=[0, 0.2],
                height_m=0.35,
                tilt_deg=25,
                gap_m=0.2,
                qualification="train_confirmation_3_of_3",
            )
        ],
    )
    bank["sha256"] = digest(bank)
    pools = dict(
        train=[[0.6, -0.4], [0.8, -0.4]],
        validation=[[0.7, -0.5]],
        test=[[0.55, -0.55], [0.85, -0.55], [1.2, -0.6]],
    )
    spec = dict(
        scene=dict(ball=dict(radius_m=0.0375)),
        batch_size=2,
        ppo_updates=1,
        test_repeats=3,
        proposal_bank_sha256=bank["sha256"],
        **{k + "_layouts": v for k, v in pools.items()},
    )
    spec["fingerprint"] = digest(spec)
    data = {}
    geometry = {}
    for label, xy in pools.items():
        values, metadata = materialize(bank, xy)
        n, k = values.shape[:2]
        mask = np.zeros((n, k), dtype=np.bool_)
        mask[:, 0] = True
        if label == "test":
            mask[-1] = False
        data[label] = dict(
            bucket=np.asarray(xy),
            values=values,
            mask=mask,
            reasons=np.where(mask, 0, 1).astype(np.uint8),
            q10=np.zeros((n, k, 6)),
            q14=np.zeros((n, k, 6)),
        )
        geometry[label] = dict(
            queries=n,
            candidates_per_query=k,
            eligible_candidates=int(mask.sum()),
            queries_with_candidate=int(mask.any(1).sum()),
            unresolved_queries=int((~mask.any(1)).sum()),
        )
    rows = []

    def add(split, query, phase):
        group = data[split]
        eligible = bool(group["mask"][query].any())
        value = group["values"][query, 0]
        row = dict(
            id=f"bank-shot-{len(rows):07d}",
            phase=phase,
            bucket_xy_m=group["bucket"][query].tolist(),
            query_index=query,
            candidate_index=0,
            candidate_metadata=metadata[0] if eligible else None,
            mask_sha256=_mask_hash(group["mask"][query]),
            eligible_bank_endpoint=eligible,
            behavior_logp=0.0 if eligible else None,
            behavior_policy_weight_sha256="synthetic",
            native_release=eligible,
            invalid=None if eligible else "no_eligible_bank_endpoint",
            board_bounce=eligible,
            whole_ball_entry=eligible,
            route_success=eligible,
            route_quality=2 if eligible else 0,
            success=eligible,
            complete_success=eligible,
            closest_opening_distance_m=0.01 if eligible else 2.0,
            max_retention_s=1.0 if eligible else 0.0,
            single_bounce_success=eligible,
            external_contact_steps=0,
            board_contact_episodes=1 if eligible else 0,
        )
        if eligible:
            row.update(
                values=value.tolist(),
                q10=[0.0] * 6,
                q14=[0.0] * 6,
                release_ball_center_m=(value[:3] + [0, 0, 0.0375 + value[4]]).tolist(),
            )
        rows.append(row)
        return row

    initial_validation = [add("validation", 0, "validation-selected-0")]
    train = [add("train", i, "train-ppo") for i in range(2)]
    val = [add("validation", 0, "validation-selected-0")]
    comparisons = {}
    for tag in ("uniform", "selected", "parent-v2"):
        comparisons[tag] = [
            add("test", i, f"test-{tag}-{rep}") for rep in range(3) for i in range(3)
        ]
    summary = {
        label: dict(
            **statistics(comparisons[tag]),
            layouts=3,
            repeats=3,
            search_at_inference=False,
        )
        for tag, label in (
            ("uniform", "uniform_valid_bank"),
            ("selected", "selected_bank"),
            ("parent-v2", "parent_v2"),
        )
    }
    report = dict(
        status="completed",
        phase="completed",
        contract_fingerprint=spec["fingerprint"],
        proposal_bank_sha256=bank["sha256"],
        records=len(rows),
        geometry=geometry,
        train_episodes=2,
        native_train_releases=2,
        updates=1,
        train_sampling=dict(
            declared_contexts=2, eligible_contexts=2, excluded_contexts=0
        ),
        learning_curve=[
            dict(update=1, **statistics(train), validation=statistics(val))
        ],
        initial_validation=statistics(initial_validation),
        selected_update=0,
        frozen_evaluation=dict(
            **statistics(comparisons["selected"]), comparison=summary, trials=9
        ),
        test_rows=copy.deepcopy(comparisons["selected"]),
    )
    return spec, bank, metadata, data, report, rows


class BankEvidenceTests(unittest.TestCase):
    def audit(self, fixture_data):
        spec, bank, metadata, data, report, rows = fixture_data
        result = Audit()
        audit_geometry(spec, bank, metadata, data, report, result)
        audit_rows(spec, report, rows, metadata, data, result)
        return result

    def test_complete_fixture_keeps_empty_queries_in_denominator(self):
        result = self.audit(fixture())
        self.assertTrue(
            all(c["passed"] for c in result.checks),
            [c for c in result.checks if not c["passed"]],
        )

    def test_candidate_materialization_cannot_change_or_drop_slots(self):
        f = fixture()
        f[3]["test"]["values"][0, 0, 0] += 0.01
        result = self.audit(f)
        self.assertFalse(
            next(
                c
                for c in result.checks
                if c["name"] == "test_candidate_materialization"
            )["passed"]
        )

    def test_mask_and_reason_tampering_is_detected(self):
        f = fixture()
        f[3]["test"]["reasons"][0, 0] = 3
        result = self.audit(f)
        self.assertFalse(
            next(c for c in result.checks if c["name"] == "test_mask_reason_identity")[
                "passed"
            ]
        )

    def test_recorded_joint_branch_must_match_precomputed_slot(self):
        f = fixture()
        row = next(r for r in f[5] if r["phase"] == "test-selected-0")
        row["q10"][0] += 0.01
        result = self.audit(f)
        self.assertFalse(
            next(
                c
                for c in result.checks
                if c["name"] == "stable_candidate_mask_and_cached_branch_evidence"
            )["passed"]
        )

    def test_empty_query_cannot_be_relabelled_as_successful_release(self):
        f = fixture()
        row = next(
            r
            for r in f[5]
            if r["phase"] == "test-selected-0" and not r["eligible_bank_endpoint"]
        )
        row["native_release"] = True
        row["route_success"] = True
        result = self.audit(f)
        self.assertFalse(
            next(
                c
                for c in result.checks
                if c["name"] == "stable_candidate_mask_and_cached_branch_evidence"
            )["passed"]
        )

    def test_extra_hidden_retry_is_detected(self):
        f = fixture()
        row = copy.deepcopy(next(r for r in f[5] if r["phase"] == "test-selected-0"))
        row["id"] = f"bank-shot-{len(f[5]):07d}"
        f[5].append(row)
        f[4]["records"] += 1
        result = self.audit(f)
        self.assertFalse(
            next(
                c
                for c in result.checks
                if c["name"] == "selected_bank_all_query_repeat_denominator"
            )["passed"]
        )

    def test_deleting_empty_test_query_is_detected(self):
        f = fixture()
        rows = f[5]
        index = next(
            i
            for i, r in enumerate(rows)
            if r["phase"] == "test-selected-0" and not r["eligible_bank_endpoint"]
        )
        del rows[index]
        f[4]["records"] -= 1
        for i, row in enumerate(rows):
            row["id"] = f"bank-shot-{i:07d}"
        result = self.audit(f)
        self.assertFalse(
            next(
                c
                for c in result.checks
                if c["name"] == "selected_bank_all_query_repeat_denominator"
            )["passed"]
        )

    def test_changed_source_teacher_provenance_breaks_fingerprint(self):
        f = fixture()
        f[1]["templates"][0]["source_trial_id"] = "test-teacher"
        result = self.audit(f)
        self.assertFalse(
            next(c for c in result.checks if c["name"] == "bank_fingerprint")["passed"]
        )


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None,
    "Pinned Ubuntu Torch environment required",
)
class BankCheckpointTests(unittest.TestCase):
    def test_recorded_behavior_logprob_is_checked_against_preupdate_checkpoint(self):
        import torch
        from sircus.mjlab.bank_policy import FEATURE_NAMES, Policy

        spec, bank, metadata, data, report, rows = fixture()
        spec["epochs"] = 1
        torch.manual_seed(12)
        initial = Policy()
        after = copy.deepcopy(initial)
        with torch.no_grad():
            after.actor[-1].bias.add_(0.1)
        h0, h1 = state_hash(initial.state_dict()), state_hash(after.state_dict())
        report["policy_weight_sha256"] = h0
        report["learning_curve"][0]["weight_change"] = dict(
            before=h0,
            after=h1,
            optimizer_rows=2,
            excluded_all_invalid=0,
            epochs=[dict(loss=1.0, gradient_norm=0.1)],
        )
        for row in rows:
            row["behavior_policy_weight_sha256"] = h0
        [r for r in rows if r["phase"] == "validation-selected-0"][-1][
            "behavior_policy_weight_sha256"
        ] = h1
        for label in ("uniform_valid_bank", "selected_bank"):
            report["frozen_evaluation"]["comparison"][label][
                "policy_weight_sha256"
            ] = h0
        report["frozen_evaluation"]["policy_weight_sha256"] = h0
        records = audit_rows(spec, report, rows, metadata, data, Audit())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "proposal-bank.json").write_text(json.dumps(bank))
            for i, policy in enumerate((initial, after)):
                torch.save(
                    dict(
                        state=policy.state_dict(),
                        contract=spec,
                        bank=bank,
                        policy_class="relative_bank_categorical",
                        weight_sha256=h0 if i == 0 else h1,
                        update=i,
                    ),
                    root / f"update-{i:03d}.pt",
                )
            torch.save(
                dict(
                    state=initial.state_dict(),
                    contract=spec,
                    bank=bank,
                    policy_class="relative_bank_categorical",
                    weight_sha256=h0,
                    selected_update=0,
                    feature_names=list(FEATURE_NAMES),
                ),
                root / "selected.pt",
            )
            report["checkpoint_sha256"] = file_hash(root / "selected.pt")
            audit = Audit()
            audit_checkpoints(root, spec, report, records, data, audit)
            self.assertTrue(
                all(c["passed"] for c in audit.checks),
                [c for c in audit.checks if not c["passed"]],
            )
            records["train"][0]["behavior_logp"] += 0.3
            audit = Audit()
            audit_checkpoints(root, spec, report, records, data, audit)
            self.assertFalse(
                next(c for c in audit.checks if c["name"] == "ppo_1_cpu_behavior_logp")[
                    "passed"
                ]
            )


if __name__ == "__main__":
    unittest.main()
