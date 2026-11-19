"""Converted-development provenance and new held-out study boundaries, CPU only."""

import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from sircus.mjlab.expanded_study import (
    MIN_SPLIT_DISTANCE_M,
    coordinates,
    enriched_stats,
    new_splits,
    paired_comparison,
    uniform_indices,
    validate_expansion,
    sha256,
    copy_shared_prefix,
)
from sircus.mjlab.learning_contract import digest, splits
from sircus.mjlab.bank_study import new_splits as old_splits, unresolved_row
from sircus.mjlab.learning_search import VERSION as BANK_VERSION
from sircus.mjlab import expanded_study
from sircus.mjlab.expanded_actions import OLD_BOUNDS, WIDE_BOUNDS


class ExpandedStudyTests(unittest.TestCase):
    def setUp(self):
        self.original = splits()
        self.old = old_splits(self.original["train"])
        self.dev = self.old["test"][:12]
        self.historical = [
            xy
            for pool in (self.original, self.old)
            for group in pool.values()
            for xy in group
        ]

    def test_new_splits_reclassify_only_targeted_positions_and_exclude_old_evidence(
        self,
    ):
        result = new_splits(self.original["train"], self.dev, self.historical)
        self.assertEqual(
            result, new_splits(self.original["train"], self.dev, self.historical)
        )
        self.assertEqual(
            {k: len(v) for k, v in result.items()},
            dict(train=175, validation=30, test=100, development=12),
        )
        self.assertEqual(result["train"][:35], self.original["train"])
        self.assertEqual(result["train"][35:47], self.dev)
        self.assertEqual(result["development"], self.dev)
        for label in ("validation", "test"):
            xy = np.asarray(result[label])
            forbidden = self.historical + result["train"]
            if label == "test":
                forbidden += result["validation"]
            distance = np.linalg.norm(
                xy[:, None, :] - np.asarray(forbidden)[None, :, :], axis=-1
            )
            self.assertGreaterEqual(float(distance.min()), MIN_SPLIT_DISTANCE_M)
        self.assertFalse(
            set(map(tuple, result["test"])) & set(map(tuple, self.old["test"]))
        )

    def test_uniform_quantiles_preserve_candidate_ids_and_empty_queries(self):
        mask = np.array([[0, 1, 0, 1], [1, 1, 1, 1], [0, 0, 0, 0]], dtype=bool)
        np.testing.assert_array_equal(
            uniform_indices(mask, [0.0, 0.999, 0.5]), [1, 3, 0]
        )
        np.testing.assert_array_equal(
            uniform_indices(mask, [0.99, 0.25, 0.1]), [3, 1, 0]
        )
        for q in ([1, 0, 0], [-0.1, 0, 0], [np.nan, 0, 0]):
            with self.assertRaises(ValueError):
                uniform_indices(mask, q)

    def test_finite_coordinate_validation(self):
        self.assertEqual(
            coordinates([dict(id="a", bucket_xy_m=[0.5, -0.4])]), [[0.5, -0.4]]
        )
        for bad in ([[1]], [[1, float("inf")]]):
            with self.assertRaises(ValueError):
                coordinates(bad)
        with self.assertRaises(ValueError):
            new_splits(self.original["train"], self.dev[:11], self.historical)

    def test_complete_denominators_and_layout_paired_comparison(self):
        missed = unresolved_row([0.5, -0.4])
        missed["eligible_bank_endpoint"] = False
        captured = dict(
            missed,
            invalid=None,
            route_success=True,
            route_quality=2,
            board_bounce=True,
            max_retention_s=1.0,
            closest_opening_distance_m=0.01,
            single_bounce_success=True,
            native_release=True,
            eligible_bank_endpoint=True,
            bucket_xy_m=[0.7, -0.2],
        )
        r = enriched_stats(
            [missed, copy.deepcopy(missed), captured, copy.deepcopy(captured)]
        )
        self.assertEqual(
            (
                r["attempts"],
                r["captures"],
                r["unresolved_positions"],
                r["positions_with_capture"],
            ),
            (4, 2, 1, 1),
        )
        self.assertEqual(r["conditional_capture_rate"], 1.0)
        same = paired_comparison([missed, captured], [missed, captured])
        self.assertEqual(same["capture_rate_difference"], 0.0)
        self.assertEqual(same["descriptive_95pct_layout_bootstrap"], [0.0, 0.0])
        with self.assertRaises(ValueError):
            paired_comparison([missed], [captured])

    def test_stable_ids_preserve_original_branches_when_template_count_shifts_slots(
        self,
    ):
        old_meta = [dict(candidate_id=i) for i in ("t0-r0", "t1-r0", "t0-r1", "t1-r1")]
        new_meta = [
            dict(candidate_id=i)
            for i in ("t0-r0", "t1-r0", "new-r0", "t0-r1", "t1-r1", "new-r1")
        ]
        source = dict(
            bucket=np.array([[0.7, -0.4]]),
            values=np.arange(20.0).reshape(1, 4, 5),
            mask=np.array([[True, False, True, False]]),
            q10=np.arange(24.0).reshape(1, 4, 6),
            q14=np.arange(24.0, 48.0).reshape(1, 4, 6),
            reasons=np.array([[0, 3, 0, 4]], dtype=np.uint8),
        )
        destination = dict(
            bucket=source["bucket"].copy(),
            values=np.zeros((1, 6, 5)),
            mask=np.ones((1, 6), bool),
            q10=np.full((1, 6, 6), -99.0),
            q14=np.full((1, 6, 6), -99.0),
            reasons=np.zeros((1, 6), np.uint8),
        )
        destination["values"][:, [0, 1, 3, 4]] = source["values"]
        shared, added = copy_shared_prefix(destination, new_meta, source, old_meta)
        np.testing.assert_array_equal(shared, [0, 1, 3, 4])
        np.testing.assert_array_equal(added, [2, 5])
        for key in ("mask", "q10", "q14", "reasons"):
            np.testing.assert_array_equal(destination[key][:, shared], source[key])
        self.assertTrue((destination["q10"][:, added] == -99).all())
        destination["values"][0, 0, 0] += 1
        with self.assertRaises(RuntimeError):
            copy_shared_prefix(destination, new_meta, source, old_meta)


class ExpansionProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old = dict(version=BANK_VERSION, templates=[dict(source_trial_id="old-0")])
        self.old["sha256"] = digest(self.old)
        teacher = dict(
            source_trial_id="expand-shot-1",
            source_target_id="old-test-01",
            source_phase="targeted_development",
            confirmation=dict(attempts=10, captures=10, invalid=0),
        )
        self.bank = dict(
            version=BANK_VERSION,
            templates=copy.deepcopy(self.old["templates"]) + [teacher],
        )
        self.contract = dict(
            development_layouts=[[0.2 + i * 0.02, -0.5] for i in range(12)],
            minimum_confirmation_rate=0.8,
            confirmations_per_candidate=10,
            old_bank_sha256=self.old["sha256"],
            old_bank_study_fingerprint="old-study-fp",
        )
        self.contract["fingerprint"] = digest(self.contract)
        self.report = dict(status="completed")
        self.write()

    def tearDown(self):
        self.tmp.cleanup()

    def write(self):
        self.bank.pop("sha256", None)
        self.bank["sha256"] = digest(self.bank)
        path = self.root / "expanded-bank.json"
        path.write_text(json.dumps(self.bank))
        self.report.update(
            expanded_bank_sha256=sha256(path), proposal_bank_sha256=self.bank["sha256"]
        )
        (self.root / "report.json").write_text(json.dumps(self.report))
        (self.root / "contract.json").write_text(json.dumps(self.contract))

    def test_verified_development_bank_is_separate_from_old_results(self):
        r, c, b, dev, bounds = validate_expansion(
            self.root, self.old, {"contract_fingerprint": "old-study-fp"}
        )
        self.assertEqual(len(dev), 12)
        self.assertEqual(len(b["templates"]), 2)
        self.assertEqual(b["templates"][0], self.old["templates"][0])

    def test_missing_confirmation_and_wrong_origin_are_rejected(self):
        for change in (
            dict(source_phase="test-selected"),
            dict(confirmation=dict(attempts=10, captures=7, invalid=0)),
        ):
            original = copy.deepcopy(self.bank["templates"][1])
            self.bank["templates"][1].update(change)
            self.write()
            with self.assertRaises(RuntimeError):
                validate_expansion(
                    self.root, self.old, {"contract_fingerprint": "old-study-fp"}
                )
            self.bank["templates"][1] = original

    def test_changed_prefix_or_bank_bytes_are_rejected(self):
        self.bank["templates"][0]["source_trial_id"] = "changed"
        self.write()
        with self.assertRaises(RuntimeError):
            validate_expansion(
                self.root, self.old, {"contract_fingerprint": "old-study-fp"}
            )
        self.bank["templates"][0]["source_trial_id"] = "old-0"
        self.write()
        with (self.root / "expanded-bank.json").open("a") as f:
            f.write(" ")
        with self.assertRaises(RuntimeError):
            validate_expansion(
                self.root, self.old, {"contract_fingerprint": "old-study-fp"}
            )

    def test_pending_search_cannot_enable_training(self):
        self.report["status"] = "running"
        self.write()
        with self.assertRaises(RuntimeError):
            validate_expansion(
                self.root, self.old, {"contract_fingerprint": "old-study-fp"}
            )


class GeometryPreparationTests(unittest.TestCase):
    """Exercise the complete CPU preparation path without creating any GPU context."""

    def prepare(self, root, *, prepare_only, slot):
        original = splits()
        old_layouts = old_splits(original["train"])
        fields = lambda values: dict(
            zip(
                ("train_layouts", "validation_layouts", "test_layouts"),
                (values["train"], values["validation"], values["test"]),
            )
        )
        parent_cp = dict(
            contract=dict(**fields(original), scene={"fake_fixture": True})
        )
        old_cp = dict(
            bank={"sha256": "old-bank", "templates": [{}]}, contract=fields(old_layouts)
        )
        old_report = dict(
            contract_fingerprint="old-contract", checkpoint_sha256="old-checkpoint"
        )
        expansion = root / "expansion"
        expansion.mkdir()
        (expansion / "report.json").write_text("{}")
        (expansion / "expanded-bank.json").write_text("{}")
        bank = dict(sha256="expanded-bank", templates=[{}, {}])
        a = SimpleNamespace(
            parent_study=root / "parent",
            old_bank_study=root / "old",
            expansion_study=expansion,
            assets=root / "assets",
            output=root / "result",
            random_train=3,
            validation_count=2,
            test_count=3,
            batch=8,
            updates=1,
            validate_every=1,
            test_repeats=2,
            audit_chunk=4,
            prepare_only=prepare_only,
            gpu_lock=root / "not-an-actual-gpu-lock",
            gpu_wait_timeout=1.0,
        )
        calls = []

        def build(candidate_bank, xy, scene):
            values, meta = [], []
            for rotation in range(3):
                for template in range(len(candidate_bank["templates"])):
                    value = [
                        0.31 if rotation == 2 else 0.65 + rotation * 0.01,
                        -0.04,
                        0.36,
                        25.0,
                        0.24 + template * 0.01,
                    ]
                    values.append(np.repeat([value], len(xy), axis=0))
                    meta.append(dict(candidate_id=f"t{template}-r{rotation}"))
            return np.stack(values, axis=1), meta

        def audit(scene, assets, xy, values, **kwargs):
            calls.append(kwargs.get("bounds", "old-native"))
            mask = np.ones(values.shape[:2], dtype=bool)
            if "bounds" not in kwargs:
                mask[:, 0] = False  # A strict in-bounds IK failure must stay failed.
                mask[values[:, :, 0] < OLD_BOUNDS["low"][0]] = False
            branch = (
                len(calls) * 100 + values.shape[1]
            )  # Deliberately slot/K-dependent.
            return mask, dict(
                q10=np.full((*mask.shape, 6), float(branch)),
                q14=np.full((*mask.shape, 6), float(-branch)),
                reasons=np.where(mask, 0, 3).astype(np.int16),
            )

        with contextlib.ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    expanded_study,
                    "validate_artifacts",
                    return_value=({}, parent_cp, "parent-sha", "parent-fp"),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    expanded_study, "validate_bank", return_value=(old_report, old_cp)
                )
            )
            stack.enter_context(
                mock.patch.object(
                    expanded_study,
                    "validate_expansion",
                    return_value=(
                        {},
                        dict(fingerprint="expansion-fp"),
                        bank,
                        old_layouts["test"][:12],
                        WIDE_BOUNDS,
                    ),
                )
            )
            stack.enter_context(
                mock.patch(
                    "sircus.mjlab.bank_geometry.build_candidates", side_effect=build
                )
            )
            stack.enter_context(
                mock.patch(
                    "sircus.mjlab.bank_geometry.audit_candidates", side_effect=audit
                )
            )
            stack.enter_context(
                mock.patch(
                    "sircus.mjlab.expand_geometry.audit_values", side_effect=audit
                )
            )
            slot_mock = stack.enter_context(
                mock.patch.object(expanded_study, "gpu_slot", side_effect=slot)
            )
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            expanded_study.run(a)
        return a, calls, slot_mock

    def test_prepare_only_writes_complete_frozen_geometry_without_gpu_acquisition(self):
        with tempfile.TemporaryDirectory() as directory:
            a, calls, slot = self.prepare(
                Path(directory),
                prepare_only=True,
                slot=AssertionError("GPU must not be acquired"),
            )
            slot.assert_not_called()
            report = json.loads((a.output / "report.json").read_text())
            contract = json.loads((a.output / "contract.json").read_text())
            self.assertEqual(report["status"], "geometry_prepared")
            self.assertEqual(len(report["geometry"]), 12)
            self.assertEqual(report["geometry"]["expanded-train"]["queries"], 50)
            self.assertIn("old_uniform_expanded_bounds", contract["evaluation_methods"])
            self.assertIn("old-native", calls)
            self.assertTrue(
                any(isinstance(c, dict) and c == WIDE_BOUNDS for c in calls)
            )
            self.assertEqual(len(list(a.output.glob("geometry-*.npz"))), 12)
            for cache in a.output.glob("geometry-*.npz"):
                with np.load(cache, allow_pickle=False) as arrays:
                    self.assertEqual(arrays["q10"].shape[:2], arrays["mask"].shape)
                    self.assertEqual(arrays["q14"].shape[:2], arrays["mask"].shape)
            self.assertEqual(contract["version"], "sircus-mjlab-expanded-bank-ppo-v2")
            self.assertEqual(
                contract["cache_strategy"], "stable-candidate-id-shared-prefix-v1"
            )
            for split in ("train", "validation", "test", "development"):
                with np.load(a.output / f"geometry-old-{split}.npz") as strict, np.load(
                    a.output / f"geometry-old-wide-{split}.npz"
                ) as wide, np.load(
                    a.output / f"geometry-expanded-{split}.npz"
                ) as expanded:
                    within = (strict["values"] >= OLD_BOUNDS["low"]).all(-1) & (
                        strict["values"] <= OLD_BOUNDS["high"]
                    ).all(-1)
                    for key in ("mask", "q10", "q14", "reasons"):
                        np.testing.assert_array_equal(
                            wide[key][within], strict[key][within]
                        )
                        np.testing.assert_array_equal(
                            expanded[key][:, [0, 2, 4]], wide[key]
                        )
                    self.assertFalse(strict["mask"][:, 0].any())
                    self.assertFalse(wide["mask"][:, 0].any())
                    self.assertTrue(
                        wide["mask"][:, 2].all(),
                        "new wider-domain slots may be audited",
                    )

    def test_queue_wait_is_reported_only_after_cpu_geometry_completes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            @contextlib.contextmanager
            def waiting_slot(path, timeout_s, notify):
                self.assertEqual(
                    len(list((root / "result").glob("geometry-*.npz"))), 12
                )
                notify(dict(status="waiting_for_4090", wait_s=timeout_s, reason=[123]))
                raise TimeoutError("Mock occupied 4090; no workload interrupted")
                yield

            with self.assertRaises(TimeoutError):
                self.prepare(root, prepare_only=False, slot=waiting_slot)
            report = json.loads((root / "result" / "report.json").read_text())
            self.assertEqual(report["phase"], "waiting-for-4090")
            self.assertEqual(report["gpu_queue"]["reason"], [123])
            self.assertEqual(report["train_episodes"], 0)


if __name__ == "__main__":
    unittest.main()
