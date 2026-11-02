"""Synthetic evidence corruption checks, with no simulator or GPU dependency."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sircus.mjlab.verify_learning import (
    Audit,
    _observations,
    _project,
    audit_checkpoints,
    audit_records,
    digest,
    file_hash,
    state_hash,
    statistics,
    verify,
)

HAVE_TORCH = importlib.util.find_spec("torch") is not None


def fixture():
    train, validation, test = (
        [[0.6, -0.4], [0.8, -0.4]],
        [[0.7, -0.5]],
        [[0.55, -0.55], [0.85, -0.55]],
    )
    spec = dict(
        train_layouts=train,
        validation_layouts=validation,
        test_layouts=test,
        exploration_batches=2,
        batch_size=2,
        ppo_updates=1,
        scene=dict(ball=dict(radius_m=0.0375)),
        low=[0.32, -0.30, 0.18, 12.0, 0.065],
        high=[1.01, 0.25, 0.53, 48.0, 0.52],
    )
    spec["fingerprint"] = digest(spec)
    rows = []

    def add(phase, xy, success=True, **overrides):
        row = dict(
            id=f"shot-{len(rows):07d}",
            phase=phase,
            bucket_xy_m=xy,
            values=[0.65, -0.04, 0.36, 25, 0.24],
            release_ball_center_m=[0.65, -0.04, 0.6375],
            locks=dict(board=None, dropper=None),
            invalid=None,
            board_bounce=True,
            whole_ball_entry=success,
            success=success,
            complete_success=success,
            max_retention_s=1.0 if success else 0.0,
            closest_opening_distance_m=0.01 if success else 0.4,
            route_success=success,
            route_quality=2 if success else 0,
            single_bounce_success=success,
            board_contact_episodes=1,
            external_contact_steps=0,
            bucket_contact_before_entry=False,
            rim_band_contact=False,
            exited_after_entry=False,
            minimum_mouth_clearance_mm=2.0,
        )
        row.update(overrides)
        rows.append(row)
        return row

    exploration = [add("train-exploration", xy) for xy in train * 2]
    references = []
    for xy in train:
        for candidate in [r for r in exploration if r["bucket_xy_m"] == xy][:3]:
            for _ in range(3):
                references.append((candidate["id"], add("train-confirmation", xy)))
    warm_val = [add("validation-warm", xy) for xy in validation]
    ppo = [add("train-ppo", xy) for xy in train]
    val = [add("validation-1", xy) for xy in validation]
    grouped = {}
    for label in ("initial", "warm", "selected"):
        grouped[label] = [
            add("test-" + label, xy, label != "initial") for xy in test * 3
        ]
    grouped["selected"][0]["scalar_score_audit_passed"] = True
    locks = {}
    for mode in ("board", "dropper"):
        locks[mode] = []
        for xy, reference in zip(test, grouped["selected"]):
            lock = dict(board=None, dropper=None)
            lock[mode] = (
                reference["values"][:3]
                if mode == "board"
                else reference["release_ball_center_m"]
            )
            locks[mode].append(add("test-lock-" + mode, xy, locks=lock))
    catalogue = []
    for i, xy in enumerate(train):
        group = [r for r in exploration if r["bucket_xy_m"] == xy]
        best = group[0]
        repeats = [r for source, r in references if source == best["id"]]
        catalogue.append(
            dict(
                id=f"train-{i:02d}",
                bucket_xy_m=xy,
                trials=len(group),
                valid=len(group),
                captures=len(group),
                best=best,
                confirmation=statistics(repeats),
            )
        )
    comparisons = {
        label: dict(
            **statistics(grouped[label]),
            layouts=2,
            repeats=3,
            policy_weight_sha256="placeholder",
        )
        for label in grouped
    }
    comparisons.update(
        {mode + "_locked": dict(**statistics(locks[mode]), layouts=2) for mode in locks}
    )
    report = dict(
        status="completed",
        phase="completed",
        contract_fingerprint=spec["fingerprint"],
        records=len(rows),
        train_episodes=len(exploration) + len(references) + len(ppo),
        exploration=statistics(exploration),
        updates=1,
        learning_curve=[dict(update=1, **statistics(ppo), validation=statistics(val))],
        selected_update=0,
        imitation=dict(samples=6),
        warm_validation=statistics(warm_val),
        source_sha256={"fixture.py": "placeholder"},
        frozen_evaluation=dict(
            **statistics(grouped["selected"]),
            trials=len(grouped["selected"]),
            comparison=comparisons,
            search_at_inference=False,
        ),
    )
    report["coverage"] = dict(cells=catalogue)
    report["test_rows"] = grouped["selected"]
    return spec, report, rows, catalogue


class EvidenceAuditTests(unittest.TestCase):
    def test_complete_finite_fixture_passes(self):
        audit, _ = audit_records(*fixture())
        self.assertTrue(
            all(c["passed"] for c in audit.checks),
            [c for c in audit.checks if not c["passed"]],
        )

    def test_train_test_contamination_is_detected(self):
        spec, report, rows, catalogue = fixture()
        rows[0]["bucket_xy_m"] = spec["test_layouts"][0]
        audit, _ = audit_records(spec, report, rows, catalogue)
        self.assertFalse(
            next(c for c in audit.checks if c["name"] == "phase_layout_membership")[
                "passed"
            ]
        )

    def test_reported_captures_cannot_exceed_raw_evidence(self):
        spec, report, rows, catalogue = fixture()
        report["frozen_evaluation"]["comparison"]["selected"]["captures"] += 1
        audit, _ = audit_records(spec, report, rows, catalogue)
        self.assertFalse(
            next(c for c in audit.checks if c["name"] == "test_selected_statistics")[
                "passed"
            ]
        )

    def test_changed_confirmation_teacher_is_detected(self):
        spec, report, rows, catalogue = fixture()
        catalogue[0]["best"] = copy.deepcopy(catalogue[1]["best"])
        audit, _ = audit_records(spec, report, rows, catalogue)
        self.assertFalse(
            next(c for c in audit.checks if c["name"] == "train-00_teacher_selection")[
                "passed"
            ]
        )

    def test_broken_lock_reference_is_detected(self):
        spec, report, rows, catalogue = fixture()
        row = next(r for r in rows if r["phase"] == "test-lock-dropper")
        row["locks"]["dropper"] = [0.5, -0.2, 0.7]
        audit, _ = audit_records(spec, report, rows, catalogue)
        self.assertFalse(
            next(
                c for c in audit.checks if c["name"] == row["id"] + "_constraint_source"
            )["passed"]
        )

    def test_external_contact_cannot_be_kept_as_route_capture(self):
        spec, report, rows, catalogue = fixture()
        rows[-1]["external_contact_steps"] = 1
        audit, _ = audit_records(spec, report, rows, catalogue)
        self.assertFalse(
            next(
                c for c in audit.checks if c["name"] == "saved_scoring_label_invariants"
            )["passed"]
        )

    def test_ppo_lock_cannot_use_a_different_teacher(self):
        spec, report, rows, catalogue = fixture()
        row = next(r for r in rows if r["phase"] == "train-ppo")
        row["locks"]["board"] = [0.1, 0.2, 0.3]
        audit, _ = audit_records(spec, report, rows, catalogue)
        self.assertFalse(
            next(
                c
                for c in audit.checks
                if c["name"] == "ppo_constraints_anchored_to_train_only_teachers"
            )["passed"]
        )

    def test_records_only_source_audit_detects_modified_runtime_source(self):
        spec, report, rows, catalogue = fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "fixture.py").write_text("measured source\n")
            report["source_sha256"]["fixture.py"] = file_hash(root / "fixture.py")
            for name, value in (
                ("contract", spec),
                ("report", report),
                ("catalogue", catalogue),
                (
                    "splits",
                    {k: spec[k + "_layouts"] for k in ("train", "validation", "test")},
                ),
            ):
                (root / (name + ".json")).write_text(json.dumps(value))
            (root / "shots.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            self.assertTrue(verify(root, source=root, checkpoints=False)["passed"])
            (root / "fixture.py").write_text("changed source\n")
            self.assertFalse(verify(root, source=root, checkpoints=False)["passed"])

    def test_absent_constraint_masks_and_zero_coordinates_are_distinct(self):
        obs = _observations(
            [[0.6, -0.4], [0.6, -0.4]],
            [
                dict(board=None, dropper=None),
                dict(board=[0, 0, 0.3], dropper=[0, 0, 0.7]),
            ],
        )
        np.testing.assert_array_equal(obs[:, 2:4], [[0, 0], [1, 1]])
        projected = _project(
            np.tile([0.65, -0.04, 0.36, 25, 0.24], (2, 1)),
            [
                dict(board=None, dropper=None),
                dict(board=[0, 0, 0.3], dropper=[0, 0, 0.7]),
            ],
            0.0375,
        )
        np.testing.assert_array_equal(projected[1, :3], [0, 0, 0.3])
        self.assertAlmostEqual(projected[1, 4], 0.3625)


@unittest.skipUnless(HAVE_TORCH, "Pinned Ubuntu Torch environment required")
class CheckpointAuditTests(unittest.TestCase):
    def test_saved_actor_actions_and_weight_chain_detect_corruption(self):
        import torch
        from sircus.mjlab.learning_policy import Policy

        spec, report, rows, _ = fixture()
        policies = {}
        for i, label in enumerate(("initial", "warm", "selected")):
            torch.manual_seed(12)
            policy = Policy()
            with torch.no_grad():
                policy.actor[-1].bias.add_(i * 0.1)
            policies[label] = policy
        hashes = {k: state_hash(v.state_dict()) for k, v in policies.items()}
        report["selected_update"] = 1
        report["learning_curve"][0]["weight_change"] = dict(
            before=hashes["warm"],
            after=hashes["selected"],
            epochs=[dict(loss=1.0, gradient_norm=0.3)],
        )
        grouped = {
            phase: [r for r in rows if r["phase"] == phase]
            for phase in {r["phase"] for r in rows}
        }
        for label in ("initial", "warm", "selected"):
            for phase in ["test-" + label] + (
                ["test-lock-board", "test-lock-dropper"] if label == "selected" else []
            ):
                group = grouped[phase]
                if phase.startswith("test-lock-"):
                    mode = phase.removeprefix("test-lock-")
                    for row, ref in zip(group, grouped["test-selected"]):
                        row["locks"][mode] = (
                            ref["values"][:3]
                            if mode == "board"
                            else ref["release_ball_center_m"]
                        )
                locks = [r["locks"] for r in group]
                with torch.no_grad():
                    latent = (
                        policies[label]
                        .actor(
                            torch.tensor(
                                _observations([r["bucket_xy_m"] for r in group], locks)
                            )
                        )
                        .numpy()
                    )
                expected = (
                    np.asarray(spec["low"])
                    + (np.tanh(latent) + 1)
                    * (np.asarray(spec["high"]) - spec["low"])
                    / 2
                )
                expected = _project(expected, locks, 0.0375)
                for row, value in zip(group, expected):
                    row["values"] = value.tolist()
                    row["release_ball_center_m"] = [
                        value[0],
                        value[1],
                        value[2] + value[4] + 0.0375,
                    ]
            report["frozen_evaluation"]["comparison"][label]["policy_weight_sha256"] = (
                hashes[label]
            )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for label in ("initial", "warm", "selected", "latest"):
                name = "selected" if label == "latest" else label
                torch.save(
                    dict(
                        state=policies[name].state_dict(),
                        contract=spec,
                        updates=1,
                        selected_update=1,
                        weight_sha256=hashes[name],
                    ),
                    root / (label + ".pt"),
                )
            report["checkpoint_sha256"] = file_hash(root / "selected.pt")
            audit = Audit()
            audit_checkpoints(root, spec, report, grouped, audit)
            self.assertTrue(
                all(c["passed"] for c in audit.checks),
                [c for c in audit.checks if not c["passed"]],
            )
            grouped["test-selected"][0]["values"][0] += 0.02
            audit = Audit()
            audit_checkpoints(root, spec, report, grouped, audit)
            self.assertFalse(
                next(
                    c
                    for c in audit.checks
                    if c["name"] == "test-selected_cpu_frozen_actor_actions"
                )["passed"]
            )


if __name__ == "__main__":
    unittest.main()
