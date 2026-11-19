"""Meaningful evidence corruptions for gap-window fairness and scorer claims."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sircus.mjlab.contract import FORCE_NAMES, ScoreStream
from sircus.mjlab.verify_expansion import Audit, digest
from sircus.mjlab.window_contract import settings
from sircus.mjlab.window_verify import (audit_splits, audit_rows, audit_training_ids, audit_native,
    audit_evaluation, audit_geometry_evidence, audit_checkpoint_files, audit_batches, row_reward)


def result(a):
    return {c["name"]: c["passed"] for c in a.checks}


def fixture():
    spec = settings(); spec["board_branch_park_q10"] = [0.] * 6
    prepared = []
    for c in [x for values in spec["contexts"].values() for x in values]:
        row = dict(c, prepared=True, board_admitted=True, q14=[1.] * 6,
            board_world=np.eye(4).tolist())
        row["board_branch_sha256"] = digest(dict(version="sircus-fixed-board-gap-window-geometry-v1",
            bucket_xy_m=spec["bucket_xy_m"], board_world=row["board_world"], q14=row["q14"],
            park_q10=spec["board_branch_park_q10"], scene_sha256=digest(spec.get("scene", {}))))
        prepared.append(row)
    rows = []
    for role in ("train", "validation", "test"):
        for c in spec["contexts"][role]:
            p = next(x for x in prepared if x["id"] == c["id"])
            row = dict(id=c["id"] + "-shot", context_id=c["id"], role=role,
                values=[*spec["board_xy_m"], c["board_height_m"], c["tilt_deg"], .3],
                gap_m=.3, bucket_xy_m=spec["bucket_xy_m"], geometry_eligible=True,
                board_branch_sha256=p["board_branch_sha256"], q14=p["q14"].copy(), q10=[0.] * 6,
                minimum_clearance_lower_bound_m=.006, native_release=True, invalid=None,
                route_success=False, route_quality=0, single_bounce_success=False,
                whole_ball_entry=False, board_bounce=True, max_retention_s=0.,
                external_contact_steps=0, closest_opening_distance_m=.3)
            row["learning_reward"] = row_reward(row); rows.append(row)
    return spec, prepared, rows


class WindowEvidenceTests(unittest.TestCase):
    def test_valid_splits_and_supported_and_unsupported_contexts(self):
        spec, prepared, rows = fixture()
        p = prepared[0]; p.update(board_admitted=False, q14=None, board_branch_sha256=None)
        a = Audit(); audit_splits(a, spec, prepared)
        self.assertTrue(all(result(a).values()), result(a))

    def test_silent_unsupported_context_omission_is_rejected(self):
        spec, prepared, rows = fixture(); prepared.pop(0)
        a = Audit(); audit_splits(a, spec, prepared)
        self.assertFalse(result(a)["all_prepared_contexts_including_unsupported"])

    def test_heldout_geometry_overlap_rejected(self):
        spec, prepared, rows = fixture()
        spec["contexts"]["test"][0].update({k: spec["contexts"]["train"][0][k] for k in ("board_height_m", "tilt_deg")})
        a = Audit(); audit_splits(a, spec, prepared)
        self.assertFalse(result(a)["prespecified_24_6_8_disjoint_contexts"])

    def test_frozen_branch_identity_not_only_branch_label(self):
        spec, prepared, rows = fixture(); prepared[0]["q14"][0] += .1
        a = Audit(); audit_splits(a, spec, prepared)
        self.assertFalse(result(a)["frozen_context_geometry_and_branch_identity"])

    def test_valid_rows_keep_exact_geometry_role_and_reward(self):
        a = Audit(); audit_rows(a, *fixture())
        self.assertTrue(all(result(a).values()), result(a))

    def test_changed_q14_during_gap_sweep_rejected(self):
        spec, prepared, rows = fixture(); rows[0]["q14"][0] += .001
        a = Audit(); audit_rows(a, spec, prepared, rows)
        self.assertFalse(result(a)["every_gap_retains_frozen_q14_branch"])

    def test_validation_row_cannot_be_relabelled_training(self):
        spec, prepared, rows = fixture(); rows[-1]["role"] = "train"
        a = Audit(); audit_rows(a, spec, prepared, rows)
        self.assertFalse(result(a)["all_rows_keep_declared_split_roles"])

    def test_substituted_gap_or_bucket_rejected(self):
        spec, prepared, rows = fixture(); rows[0]["values"][4] += .01
        rows[1]["bucket_xy_m"] = [.2, -.6]
        a = Audit(); audit_rows(a, spec, prepared, rows)
        self.assertFalse(result(a)["fixed_bucket_xy_context_and_gap_identity"])

    def test_geometry_rejection_never_releases_or_captures(self):
        spec, prepared, rows = fixture(); rows[0]["geometry_eligible"] = False
        a = Audit(); audit_rows(a, spec, prepared, rows)
        self.assertFalse(result(a)["strict_release_and_capture_invariants"])

    def test_clearance_below_five_mm_and_changed_reward_rejected(self):
        spec, prepared, rows = fixture(); rows[0]["minimum_clearance_lower_bound_m"] = .0049
        rows[1]["learning_reward"] += 1
        a = Audit(); audit_rows(a, spec, prepared, rows)
        self.assertFalse(result(a)["strict_release_and_capture_invariants"])
        self.assertFalse(result(a)["unchanged_terminal_reward"])


class TrainingIdentityTests(unittest.TestCase):
    def fixture(self):
        shared = [f"shared-{i}" for i in range(504)]
        adaptive = [f"adaptive-{i}" for i in range(504)]
        sac = [f"sac-{i}" for i in range(504)]
        rows = [dict(id=i, role="train") for i in shared + adaptive + sac]
        rows += [dict(id="heldout", role="test"), dict(id="anchor", role="control")]
        checkpoints = []
        for method, extra in (("adaptive_surrogate", adaptive), ("sac", sac)):
            for n in (504, 672, 840, 1008):
                ids = (shared + extra)[:n]
                checkpoints.append(dict(method=method, training_requests=n,
                    training_record_ids=ids, training_record_ids_sha256=digest(ids)))
        return rows, checkpoints, dict(seed_ids=shared, adaptive_ids=adaptive, sac_ids=sac)

    def test_matched_budgets_and_shared_initial_dataset(self):
        rows, checkpoints, kwargs = self.fixture(); a = Audit()
        audit_training_ids(a, rows, checkpoints, **kwargs)
        self.assertTrue(all(result(a).values()), result(a))

    def test_same_count_cannot_hide_validation_or_other_method_leak(self):
        for bad_id in ("heldout", "anchor", "adaptive-0"):
            rows, checkpoints, kwargs = self.fixture(); checkpoint = checkpoints[-1]
            checkpoint["training_record_ids"][-1] = bad_id
            checkpoint["training_record_ids_sha256"] = digest(checkpoint["training_record_ids"])
            a = Audit(); audit_training_ids(a, rows, checkpoints, **kwargs)
            self.assertFalse(result(a)["checkpoints_train_only_exact_shared_then_method_prefix"])

    def test_dropped_invalid_training_attempt_cannot_shrink_budget(self):
        rows, checkpoints, kwargs = self.fixture(); rows.pop(800)
        a = Audit(); audit_training_ids(a, rows, checkpoints, **kwargs)
        self.assertFalse(result(a)["all_and_only_1512_unique_training_requests"])


class WindowNativeAuditTests(unittest.TestCase):
    def fixture(self):
        scene = json.loads((Path(__file__).resolve().parents[2] / "isaac/scene.json").read_text())
        positions = np.tile([.5, -.2, -.3], (960, 1, 1)); velocities = np.zeros_like(positions)
        forces = np.zeros((960, 1, 6))
        scorer = ScoreStream(scene, [.16, -.64], [0., 0., 1.])
        scorer.step(positions[0, 0], velocities[0, 0], dict(zip(FORCE_NAMES, forces[0, 0])))
        row = dict(scorer.result(), id="native", bucket_xy_m=[.16, -.64], board_normal=[0., 0., 1.],
            native_release=True, invalid=None,
            contact_protocol=dict(substeps=64, board_mode="articulated", gravity_mode="passive_only"))
        arrays = dict(record_ids=np.asarray(["native"]), position_m=positions, velocity_m_s=velocities, forces_N=forces)
        return scene, [row], arrays

    def test_all_saved_scorer_fields_rescore(self):
        a = Audit(); audit_native(a, *self.fixture(), "batch")
        self.assertTrue(all(result(a).values()), result(a))

    def test_native_record_substitution_and_legacy_physics_rejected(self):
        scene, rows, arrays = self.fixture(); arrays["record_ids"] = np.asarray(["another"])
        a = Audit(); audit_native(a, scene, rows, arrays, "batch")
        self.assertFalse(result(a)["batch:record_ids"])
        scene, rows, arrays = self.fixture(); rows[0]["contact_protocol"]["substeps"] = 8
        a = Audit(); audit_native(a, scene, rows, arrays, "batch")
        self.assertFalse(result(a)["batch:released_corrected_protocol"])

    def test_false_capture_and_altered_saved_trace_rejected(self):
        scene, rows, arrays = self.fixture(); rows[0]["route_success"] = True
        a = Audit(); audit_native(a, scene, rows, arrays, "batch")
        self.assertFalse(result(a)["batch:independent_scalar_rescore"])
        scene, rows, arrays = self.fixture(); arrays["position_m"][:, :, 2] = .5
        a = Audit(); audit_native(a, scene, rows, arrays, "batch")
        self.assertFalse(result(a)["batch:independent_scalar_rescore"])

    def test_native_cache_and_pre_outcome_plan_associations(self):
        scene, rows, arrays = self.fixture()
        row = rows[0]; row.update(eligible=True, geometry=dict(eligible=True),
            geometry_cache_file="geometry.json", geometry_cache_row=0,
            native_cache_file="native.npz", native_cache_row=0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez_compressed(root / "native.npz", **arrays)
            (root / "plan.json").write_text(json.dumps([dict(id=row["id"])]))
            (root / "geometry.json").write_text(json.dumps(dict(record_ids=[row["id"]], rows=[row["geometry"]])))
            batches = [dict(packet=0, record_ids=[row["id"]], native_record_ids=[row["id"]],
                native_cache_file="native.npz", plan_file="plan.json", geometry_cache_file="geometry.json")]
            (root / "batches.json").write_text(json.dumps(batches))
            a = Audit(); count = audit_batches(a, root, dict(scene=scene, batch=64), rows)
            self.assertEqual(count, 1); self.assertTrue(all(result(a).values()), result(a))
            row["native_cache_row"] = 2
            a = Audit(); audit_batches(a, root, dict(scene=scene, batch=64), rows)
            self.assertFalse(result(a)["batch-0:native_cache_row_association"])


class EvaluationAuditTests(unittest.TestCase):
    def fixture(self):
        settings_, prepared, base = fixture(); rows = []; checkpoints = []; selected = {}
        template = base[0]
        for method in ("adaptive_surrogate", "sac"):
            for milestone in (504, 672, 840, 1008):
                checkpoint = dict(method=method, milestone=milestone, sha256=method + str(milestone),
                    path=method + str(milestone), validation_record_ids=[])
                for c in settings_["contexts"]["validation"]:
                    for repeat in range(3):
                        name = f"{method}-{milestone}-{c['id']}-{repeat}"
                        row = dict(template, id=name, role="validation", method=method,
                            context_id=c["id"], milestone=milestone, source_checkpoint_sha256=checkpoint["sha256"])
                        rows.append(row); checkpoint["validation_record_ids"].append(name)
                checkpoints.append(checkpoint)
            selected[method] = checkpoints[-4].copy()  # All misses tie: earliest milestone wins.
        noise = np.random.default_rng(4).normal(0, .002, (8, 20))
        for method in ("adaptive_surrogate", "sac", "uniform_bounded"):
            for index, c in enumerate(settings_["contexts"]["test"]):
                for repeat in range(23):
                    row = dict(template, id=f"{method}-{c['id']}-{repeat}", role="test", method=method,
                        context_id=c["id"], nominal_gap_m=.3, gap_m=float(.3 + (0 if repeat < 3 else noise[index, repeat-3])),
                        repeat_index=repeat, phase="nominal" if repeat < 3 else "fresh",
                        source_checkpoint_sha256=selected[method]["sha256"] if method in selected else None)
                    rows.append(row)
        return settings_, rows, checkpoints, selected

    def test_validation_selection_and_paired_full_denominators(self):
        a = Audit(); audit_evaluation(a, *self.fixture())
        self.assertTrue(all(result(a).values()), result(a))

    def test_test_selected_checkpoint_cannot_be_swapped_or_best_test_selected(self):
        settings_, rows, checkpoints, selected = self.fixture()
        selected["sac"] = checkpoints[-1].copy()
        a = Audit(); audit_evaluation(a, settings_, rows, checkpoints, selected)
        self.assertFalse(result(a)["sac:checkpoint_selected_by_validation_only"])
        settings_, rows, checkpoints, selected = self.fixture()
        rows[-200]["source_checkpoint_sha256"] = "other"
        a = Audit(); audit_evaluation(a, settings_, rows, checkpoints, selected)
        self.assertFalse(result(a)["sac:test_uses_only_frozen_selected_checkpoint"])

    def test_omitted_failed_repeat_and_unpaired_noise_rejected(self):
        settings_, rows, checkpoints, selected = self.fixture(); rows.pop()
        a = Audit(); audit_evaluation(a, settings_, rows, checkpoints, selected)
        self.assertFalse(result(a)["uniform_bounded:all_8_test_contexts_and_184_requests"])
        settings_, rows, checkpoints, selected = self.fixture(); rows[-1]["gap_m"] += .01
        a = Audit(); audit_evaluation(a, settings_, rows, checkpoints, selected)
        self.assertFalse(result(a)["same_gap_noise_schedule_all_test_methods"])

    def test_native_oracle_gap_selection_within_test_rejected(self):
        settings_, rows, checkpoints, selected = self.fixture(); rows[-1]["nominal_gap_m"] = .4
        a = Audit(); audit_evaluation(a, settings_, rows, checkpoints, selected)
        self.assertFalse(result(a)["uniform_bounded:frozen_gap_three_nominal_twenty_fresh"])


class GeometryAndCheckpointTests(unittest.TestCase):
    def test_inline_cache_and_repeated_geometry_are_bound_to_requests(self):
        spec, prepared, rows = fixture(); row = rows[0]
        row["eligible"] = row.pop("geometry_eligible")
        keys = ("values", "bucket_xy_m", "q10", "q14", "board_branch_sha256", "eligible", "context_id")
        row["geometry"] = {k: copy.deepcopy(row[k]) for k in keys}
        row["geometry"]["geometry_version"] = "sircus-fixed-board-gap-window-geometry-v1"
        repeated = copy.deepcopy(row); repeated["id"] += "-repeat"
        a = Audit(); audit_geometry_evidence(a, prepared, [row, repeated])
        self.assertTrue(all(result(a).values()), result(a))
        repeated["geometry"]["q14"][0] += .01
        a = Audit(); audit_geometry_evidence(a, prepared, [row, repeated])
        self.assertFalse(result(a)["inline_geometry_matches_exact_executed_request"])
        self.assertFalse(result(a)["identical_requests_have_method_independent_geometry"])

    def test_actual_surrogate_file_cannot_claim_different_training_labels(self):
        from sircus.mjlab.window_learning import SuccessModel
        from sircus.mjlab.verify_expansion import sha256
        rows = [dict(id=f"train-{i}", role="train", values=[.47, -.28, .3, 36., gap],
            gap_m=gap, route_success=i == 1) for i, gap in enumerate([.1, .3, .5])]
        model = SuccessModel().fit(rows)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"; model.save(path, "contract")
            c = dict(method="adaptive_surrogate", milestone=504, path=path.name, sha256=sha256(path),
                training_record_ids=[r["id"] for r in rows])
            a = Audit(); audit_checkpoint_files(a, directory, "contract", settings(), rows, [c])
            self.assertTrue(all(result(a).values()), result(a))
            rows[1]["route_success"] = False
            a = Audit(); audit_checkpoint_files(a, directory, "contract", settings(), rows, [c])
            self.assertFalse(result(a)["adaptive_surrogate-504:actual_model_train_identity"])


if __name__ == "__main__":
    unittest.main()
